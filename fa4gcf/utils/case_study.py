import os
import yaml

import torch
import numpy as np
import pandas as pd

import fa4gcf.utils as utils
import fa4gcf.data.utils as data_utils
from fa4gcf.data import Dataset
from fa4gcf.trainer import prepare_batched_data
from fa4gcf.evaluation import (
    Evaluator,
    Collector,
    compute_metric,
    get_scores,
    get_top_k
)


def extract_metrics_from_perturbed_edges(exp_info: dict,
                                         models=None,
                                         metrics=None,
                                         models_path='saved',
                                         policy_name=None,
                                         on_bad_models="error",
                                         remap=True,
                                         return_pref_data=False):
    models = ["NGCF", "LightGCN", "GCMC"] if models is None else models
    metrics = ["NDCG", "Recall", "Hit", "MRR"] if metrics is None else metrics
    cols = ['user_id', 'Epoch', '# Del Edges', 'Exp Loss', 'Metric',
            'Demo Group', 'Sens Attr', 'Model', 'Dataset', 'Value', 'Policy']

    exp_pref_data = {}
    model_files = list(os.scandir(models_path))
    df_data = {'train': [], 'valid': [], 'test': []}
    for mod in models:
        for meta, path_or_pert_edges in exp_info.items():
            if isinstance(meta, tuple):  # perturbed edges could also be dependent on sensitive attributes
                dset, s_attr = meta
            else:
                dset = meta
                s_attr = None

            try:
                model_file = [f.path for f in model_files if mod in f.name and dset.upper() in f.name][0]
            except IndexError:
                if on_bad_models == "ignore":
                    continue
                else:
                    raise ValueError(
                        f"in path `{models_path}` there is no file for model `{mod.upper()}` and dataset `{dset.upper()}`"
                    )
            exp_pref_data[mod] = {}

            checkpoint = torch.load(model_file)
            config = checkpoint['config']

            exp_dset = dset.replace('-1000', '') if '-1000' in dset else dset
            explainer_config_file = os.path.join(
                os.path.dirname(models_path), 'config', 'explainer', f'{exp_dset}_explainer.yaml'
            )
            with open(explainer_config_file, 'r', encoding='utf-8') as f:
                exp_file_content = f.read()
                explain_config_dict = yaml.load(exp_file_content, Loader=config.yaml_loader)
            config.final_config_dict.update(explain_config_dict)

            config['data_path'] = config['data_path'].replace('\\', os.sep)
            if mod == "SVD_GCN":
                # it could lead to an excessive amount of memory not used during evaluation
                config['user_coefficient'] = 0
                config['item_coefficient'] = 0

            dataset = Dataset(config)
            uid_field = dataset.uid_field
            iid_field = dataset.iid_field

            train_data, valid_data, test_data = utils.data_preparation(config, dataset)
            split_dict = dict(zip(['train', 'valid', 'test'], [train_data, valid_data, test_data]))

            if isinstance(path_or_pert_edges, np.ndarray):
                pert_edges = path_or_pert_edges
            else:
                pert_edges = np.load(path_or_pert_edges).T

            if remap:
                if callable(remap):
                    pert_edges = remap(pert_edges, dataset)
                else:
                    for i, field in enumerate([uid_field, iid_field]):
                        pert_edges[i] = [dataset.field2token_id[field][str(n)] for n in pert_edges[i]]
                    pert_edges[1] += dataset.user_num  # remap according to adjacency matrix

            user_data = torch.arange(train_data.dataset.user_num)[1:]  # id 0 is a padding in recbole
            pert_data = {
                split_name: data_utils.get_dataset_with_perturbed_edges(pert_edges, split.dataset)
                for split_name, split in split_dict.items()
            }

            mod_pref_data = {}
            args_pref_data = [config, checkpoint, pert_edges, dataset, train_data, valid_data, test_data]
            for split_name, split in split_dict.items():
                mod_pref_data[split_name] = pref_data_from_checkpoint_and_perturbed_edges(*args_pref_data, split=split_name)
                if return_pref_data:
                    exp_pref_data[mod][split_name] = mod_pref_data[split_name]

            config["metrics"] = metrics
            evaluator = Evaluator(config)
            for metric in metrics:
                if s_attr is None:
                    sens_attrs = [col for col in dataset.user_feat.columns if col != uid_field]
                else:
                    sens_attrs = [s_attr]

                for split_name, split in split_dict.items():
                    metric_data = compute_metric(
                        evaluator, pert_data[split_name], mod_pref_data[split_name], 'cf_topk_pred', metric.lower()
                    )[:, -1]

                    for s_attr in sens_attrs:
                        demo_group_map = dataset.field2id_token[s_attr]

                        df_data[split_name].extend(list(zip(*[
                            user_data.numpy(),
                            [-1] * len(user_data),
                            [pert_edges.shape[1]] * len(user_data),
                            [-1] * len(user_data),
                            [metric] * len(user_data),
                            [demo_group_map[dg] for dg in dataset.user_feat[s_attr][user_data].numpy()],
                            [s_attr.title()] * len(user_data),
                            [mod] * len(user_data),
                            [dset] * len(user_data),
                            metric_data,
                            [policy_name] * len(user_data),
                        ])))

    out = [pd.DataFrame(df_d, columns=cols) for df_d in df_data.values()]
    return (*out, exp_pref_data) if return_pref_data else tuple(out)


@torch.no_grad()
def pref_data_from_checkpoint(config,
                              checkpoint,
                              train_data,
                              eval_data):
    model = utils.get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    model.eval()

    user_data = torch.arange(train_data.dataset.user_num)[1:]  # id 0 is a padding in recbole
    batched_data = prepare_batched_data(user_data, eval_data)

    tot_item_num = train_data.dataset.item_num
    item_tensor = train_data.dataset.get_item_feature().to(model.device)

    model_scores = get_scores(model, batched_data, tot_item_num, item_tensor)
    _, model_topk_idx = get_top_k(model_scores, topk=10)
    model_topk_idx = model_topk_idx.detach().cpu().numpy()

    pref_data = pd.DataFrame(zip(user_data.numpy(), model_topk_idx), columns=['user_id', 'cf_topk_pred'])

    return pref_data


@torch.no_grad()
def pref_data_from_checkpoint_and_perturbed_edges(config,
                                                  checkpoint,
                                                  pert_edges,
                                                  dataset,
                                                  train_data,
                                                  valid_data,
                                                  test_data,
                                                  split='test'):
    train_dataset = data_utils.get_dataset_with_perturbed_edges(pert_edges, train_data.dataset)

    train_data, valid_data, test_data = data_utils.get_dataloader_with_perturbed_edges(
        pert_edges, config, dataset, train_data, valid_data, test_data
    )
    eval_data = dict(zip(['train', 'valid', 'test'], [train_data, valid_data, test_data]))[split]

    model = utils.get_model(config['model'])(config, train_dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    if hasattr(model, "restore_item_e"):
        model.restore_item_e = None
        model.restore_user_e = None
    model.eval()

    user_data = torch.arange(train_data.dataset.user_num)[1:]  # id 0 is a padding in recbole
    batched_data = prepare_batched_data(user_data, eval_data)

    tot_item_num = train_data.dataset.item_num
    item_tensor = train_data.dataset.get_item_feature().to(model.device)

    model_scores = get_scores(model, batched_data, tot_item_num, item_tensor)
    _, model_topk_idx = get_top_k(model_scores, topk=10)
    model_topk_idx = model_topk_idx.detach().cpu().numpy()

    pref_data = pd.DataFrame(zip(user_data.numpy(), model_topk_idx), columns=['user_id', 'cf_topk_pred'])

    return pref_data
