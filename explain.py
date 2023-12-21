import os
import re
import json
import yaml
import pickle
import logging
import inspect
import argparse

import wandb
import torch
import optuna
import pandas as pd

from recbole.data.dataloader import FullSortEvalDataLoader

import gnnuers.utils as utils
from gnnuers.explainers import DPBG, BaB


script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))


def get_base_exps_filepath(config,
                           config_id=-1,
                           model_name=None,
                           model_file="",
                           hyper=False):
    """
    return the filepath where explanations are saved
    :param config:
    :param config_id:
    :param model_name:
    :param model_file:
    :return:
    """
    epochs = config["cf_epochs"]
    model_name = model_name or config["model"]
    explainer = config["explainer"].lower()
    exp_type = "dp_explanations" if not hyper else "hyperoptimization"
    base_exps_file = os.path.join(script_path, "experiments", exp_type, config["dataset"], model_name, explainer)

    exp_metadata = config["sensitive_attribute"].lower() if 'consumer' in config["exp_metric"].lower() else ''
    exp_loss = config["exp_metric"] + "_loss"
    base_exps_file = os.path.join(base_exps_file, exp_loss, exp_metadata, f"epochs_{epochs}")

    if os.path.exists(base_exps_file):
        if config_id == -1:
            paths_c_ids = sorted(filter(str.isdigit, os.listdir(base_exps_file)), key=int)
            if len(paths_c_ids) == 0:
                config_id = 1
            else:
                int_paths_c_ids = list(map(int, paths_c_ids))
                candidates = set(range(1, max(int_paths_c_ids) + 1)) - set(int_paths_c_ids)
                config_id = str(min(candidates, default=max(int_paths_c_ids) + 1))

            for path_c in paths_c_ids:
                config_path = os.path.join(base_exps_file, path_c, "config.pkl")
                if os.path.exists(config_path):
                    with open(config_path, 'rb') as f:
                        _c = pickle.load(f)

                    if config.final_config_dict == _c.final_config_dict:
                        if model_file != "" and "perturbed" in model_file:
                            check_perturb = input("The explanations of the perturbed graph could overwrite the "
                                                  "explanations from which the perturbed graph was generated. Type "
                                                  "y/yes to confirm this outcome. Other inputs will assign a new id: ")
                            if check_perturb.lower() != "y" and check_perturb.lower() != "yes":
                                continue
                        config_id = os.path.join(base_exps_file, str(path_c))
                        break
                elif hyper:
                    config_id = os.path.join(base_exps_file, str(path_c))
                    break


        base_exps_file = os.path.join(base_exps_file, str(config_id))
    else:
        base_exps_file = os.path.join(base_exps_file, "1")

    if not os.path.exists(base_exps_file):
        os.makedirs(base_exps_file)

    with open(os.path.join(base_exps_file, "config.yaml"), 'w') as exp_file:
        exp_file.write(yaml.dump(config.final_config_dict, default_flow_style=False))

    return base_exps_file


def save_exps_df(base_exps_file, exps):
    """
    Saves the pandas dataframe representation of explanations
    :param base_exps_file:
    :param exps:
    :return:
    """
    data = [exp[:-3] + exp[-2:] for exp_list in exps.values() for exp in exp_list]
    data = list(map(lambda x: [x[0], *x[1:]], data))
    df = pd.DataFrame(
        data,
        columns=utils.EXPS_COLUMNS[:-3] + utils.EXPS_COLUMNS[-2:]
    )

    out_path = base_exps_file.split(os.sep)
    df.to_csv(
        f'{"_".join(out_path[out_path.index("explanations"):])}.csv',
        index=False
    )


def explain(config, model, _rec_data, _full_dataset, _train_data, _valid_data, _test_data, base_exps_file, **kwargs):
    """
    Function that explains, that is generates perturbed graphs.
    :param config:
    :param model:
    :param _train_dataset:
    :param _rec_data:
    :param _test_data:
    :param base_exps_file:
    :param kwargs:
    :return:
    """
    epochs = config['cf_epochs']
    topk = config['cf_topk']
    wandb_mode = kwargs.get("wandb_mode", "disabled")
    overwrite = kwargs.get("overwrite", False)

    exps_filename = os.path.join(base_exps_file, f"cf_data.pkl")
    users_order_file = os.path.join(base_exps_file, f"users_order.pkl")
    model_preds_file = os.path.join(base_exps_file, f"model_rec_test_preds.pkl")
    checkpoint_path = os.path.join(base_exps_file, "checkpoint.pth")

    if overwrite and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    user_source = _rec_data if _rec_data is not None else _test_data
    user_data = user_source.user_df[user_source.uid_field][torch.randperm(user_source.user_df.length)]

    with open(os.path.join(base_exps_file, "config.pkl"), 'wb') as config_file:
        pickle.dump(config, config_file)

    utils.wandb_init(
        config,
        **wandb_env_data,
        name="Explanation",
        job_type="train",
        group=f"{model.__class__.__name__}_{config['dataset']}_{config['sensitive_attribute'].title()}_epochs{config['cf_epochs']}_exp{os.path.basename(base_exps_file)}",
        mode=wandb_mode
    )
    # wandb.config.update({"exp": os.path.basename(base_exps_file)})

    explainer_model = {
        "dpbg": DPBG,
        "bab": BaB
    }.get(config["explainer"].lower(), DPBG)

    explainer = explainer_model(config, _train_data.dataset, _rec_data, model, dist=config['cf_dist'], **kwargs)
    explainer.set_checkpoint_path(checkpoint_path)
    exp, users_order, model_preds = explainer.explain(user_data, _full_dataset, _train_data, _valid_data, _test_data, epochs, topk=topk)

    with open(exps_filename, 'wb') as f:
        pickle.dump(exp, f)
    with open(users_order_file, 'wb') as f:
        pickle.dump(users_order, f)
    with open(model_preds_file, 'wb') as f:
        pickle.dump(model_preds, f)

    logging.getLogger().info(f"Saved explanations at path {base_exps_file}")


def optimize_explain(config, model, _train_dataset, _rec_data, _test_data, base_exps_file, **kwargs):
    """
    Function that explains, that is generates perturbed graphs.
    :param config:
    :param model:
    :param _train_dataset:
    :param _rec_data:
    :param _test_data:
    :param base_exps_file:
    :param kwargs:
    :return:
    """
    epochs = config['cf_epochs']
    topk = config['cf_topk']
    wandb_mode = kwargs.get("wandb_mode", "disabled")

    user_source = _rec_data if _rec_data is not None else _test_data
    user_data = user_source.user_df[user_source.uid_field][torch.randperm(user_source.user_df.length)]

    explainer_model = {
        "dpbg": DPBG,
        "bab": BaB
    }.get(config["explainer"].lower(), DPBG)

    exp_token = f"{model.__class__.__name__}_" + \
                f"{config['dataset']}_" + \
                f"{config['sensitive_attribute'].title()}_" + \
                f"epochs{config['cf_epochs']}_" + \
                f"exp{os.path.basename(base_exps_file)}"

    def objective(trial):
        wandb_config_keys = [
            'cf_learning_rate',
            'user_batch_exp',
            'cf_beta',
            'dropout_prob'
        ]

        config['cf_learning_rate'] = trial.suggest_int('cf_learning_rate', 1000, 10000)

        config['user_batch_exp'] = trial.suggest_int(
            'user_batch_exp',
            min(int(_test_data.dataset.user_num * 0.1), 32),
            min(int(_test_data.dataset.user_num * 0.33), 220)
        )

        config['cf_beta'] = trial.suggest_float('cf_beta', 0.01, 10.0)

        # config['dropout_prob'] = trial.suggest_float('dropout_prob', 0, 0.3)

        if config["explainer"].lower() == "bab":
            config['bab_min_del_edges'] = trial.suggest_int('bab_min_del_edges', 10, 200)
            config['bab_max_tries'] = trial.suggest_int('bab_max_tries', 50, 400)
            wandb_config_keys.extend(['bab_min_del_edges', 'bab_max_tries'])

        wandb_config = {k: config[k] for k in wandb_config_keys}

        run = utils.wandb_init(
            wandb_config,
            **wandb_env_data,
            policies=config['explainer_policies'],
            name=f"Explanation_trial{trial.number}",
            job_type="train",
            group=exp_token,
            mode=wandb_mode,
            reinit=True
        )

        explainer = explainer_model(config, _train_dataset, _rec_data, model, dist=config['cf_dist'], **kwargs)
        exp, model_preds = explainer.explain(user_data, _test_data, epochs, topk=topk)
        best_exp = utils.get_best_exp_early_stopping(exp, config)

        exp_metric = best_exp[utils.exp_col_index('exp_metric')]

        with run:
            run.log({'trial_exp_metric': exp_metric})

        return exp_metric

    study_name = exp_token + '_' + str([k for k in config['explainer_policies'] if config['explainer_policies'][k]])
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_name, load_if_exists=True)

    n_trials = 100
    if study.trials:
        n_trials -= study.trials[-1].number  # it is not done automatically by optuna
        if n_trials <= 0:
            raise ValueError(f"Optuna study with storage name {study_name}.db is already completed")

    study.optimize(objective, n_trials=n_trials)

    summary = utils.wandb_init(
        config,
        **wandb_env_data,
        name="summary",
        job_type="logging",
        group=exp_token,
        mode=wandb_mode
    )

    trials = study.trials

    print("Number of finished trials: ", len(trials))

    # WandB summary.
    for step, trial in enumerate(trials):
        # Logging the loss.
        summary.log({"trial_exp_metric": trial.value}, step=step)

        # Logging the parameters.
        for k, v in trial.params.items():
            summary.log({k: v}, step=step)

    with open(os.path.join(base_exps_file, 'best_params.json'), 'w') as param_file:
        json.dump(dict(trial.params.items()), param_file, indent=4)


def execute_explanation(model_file,
                        base_explainer_config_file,
                        explainer_config_file=os.path.join("config", "explainer.yaml"),
                        config_id=-1,
                        verbose=False,
                        wandb_mode="disabled",
                        cmd_config_args=None,
                        hyperoptimization=False,
                        overwrite=False):
    explainer_config = utils.update_base_explainer(base_explainer_config_file, explainer_config_file)

    # load trained model, config, dataset
    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(
        model_file,
        explainer_config,
        cmd_config_args=cmd_config_args
    )

    # force these evaluation metrics to be ready to be computed
    config['metrics'] = ['ndcg', 'recall', 'hit', 'mrr', 'precision']

    if config['exp_rec_data'] is not None:
        if config['exp_rec_data'] != 'train+valid':
            if config['exp_rec_data'] == 'train':
                rec_data = FullSortEvalDataLoader(config, train_data.dataset, train_data.sampler)
            elif config['exp_rec_data'] == 'rec':  # model recommendations are used as target
                rec_data = valid_data
            else:
                rec_data = locals()[f"{config['exp_rec_data']}_data"]
        else:
            valid_train_dataset = train_data.dataset.copy(
                pd.concat([train_data.dataset.inter_feat, valid_data.dataset.inter_feat], ignore_index=True)
            )
            rec_data = FullSortEvalDataLoader(config, valid_train_dataset, valid_data.sampler)
    else:
        rec_data = valid_data

    base_exps_filepath = get_base_exps_filepath(
        config,
        config_id=config_id,
        model_name=model.__class__.__name__,
        model_file=model_file,
        hyper=hyperoptimization
    )

    if not os.path.exists(base_exps_filepath):
        os.makedirs(base_exps_filepath)

    global wandb_env_data
    wandb_env_data = {}
    if os.path.exists("wandb_init.json"):
        with open("wandb_init.json", 'r') as wandb_file:
            wandb_env_data = json.load(wandb_file)

    kwargs = dict(
        verbose=verbose,
        wandb_mode=wandb_mode,
        overwrite=overwrite
    )

    if not hyperoptimization:
        explain(
            config,
            model,
            rec_data,
            dataset,
            train_data,
            valid_data,
            test_data,
            base_exps_filepath,
            **kwargs
        )
    else:
        optimize_explain(
            config,
            model,
            train_data.dataset,
            rec_data,
            test_data,
            base_exps_filepath,
            **kwargs
        )
