import os
import copy
import argparse

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from fa4gcf.utils import load_data_and_model
from fa4gcf.trainer import BeyondAccuracyPerturbationTrainer, PerturbationSampler


user_policies = [
    'users_zero_constraint',
    'users_furthest_constraint',
    'interaction_recency_constraint'
]

item_policies = [
    'items_preference_constraint',
    'items_timeless_constraint',
    'items_pagerank_constraint'
]


current_path = os.path.dirname(os.path.realpath(__file__))


def perturbation_sampler_kwargs_gen(conf):
    conf_pol_key = 'perturbation_policies'
    for pol in (user_policies + item_policies):
        new_conf = copy.deepcopy(conf)
        for conf_pol in new_conf[conf_pol_key]:
            new_conf[conf_pol_key][conf_pol] = False
        new_conf[conf_pol_key]['gradient_deactivation_constraint'] = True
        new_conf[conf_pol_key]['group_deletion_constraint'] = True
        new_conf[conf_pol_key][pol] = True

        # group_deletion_constraint = new_conf['explainer_policies']['group_deletion_constraint']
        # users_zero_constraint = new_conf['explainer_policies']['users_zero_constraint']
        # users_zero_constraint_value = new_conf['users_zero_constraint_value']
        # items_preference_constraint = new_conf['explainer_policies']['items_preference_constraint']
        # items_preference_constraint_ratio = new_conf['items_preference_constraint_ratio']
        # users_furthest_constraint = new_conf['explainer_policies']['users_furthest_constraint']
        # users_furthest_constraint_ratio = new_conf['users_furthest_constraint_ratio']
        # interaction_recency_constraint = new_conf['explainer_policies']['users_interaction_recency_constraint']
        # interaction_recency_constraint_ratio = new_conf['users_interaction_recency_constraint_ratio']
        # items_timeless_constraint = new_conf['explainer_policies']['items_timeless_constraint']
        # items_timeless_constraint_ratio = new_conf['items_timeless_constraint_ratio']
        # items_pagerank_constraint = new_conf['explainer_policies']['items_pagerank_constraint']
        # items_pagerank_constraint_ratio = new_conf['items_pagerank_constraint_ratio']
        #
        # kwargs = dict(
        #     group_deletion=group_deletion_constraint,
        #     users_zero_th=users_zero_constraint_value if users_zero_constraint else None,
        #     users_sparse_ratio=0,
        #     users_furthest_ratio=users_furthest_constraint_ratio if users_furthest_constraint else 0,
        #     users_low_degree_ratio=0,
        #     items_preference_ratio=items_preference_constraint_ratio if items_preference_constraint else 0,
        #     items_niche_ratio=0,
        #     users_interaction_recency_ratio=interaction_recency_constraint_ratio if interaction_recency_constraint else 0,
        #     items_timeless_ratio=items_timeless_constraint_ratio if items_timeless_constraint else 0,
        #     items_pagerank_ratio=items_pagerank_constraint_ratio if items_pagerank_constraint else 0
        # )

        yield pol, new_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', dest="dataset_name", required=True)
    parser.add_argument('--model', '-m', dest="model_name", required=True)
    parser.add_argument('--sensitive_attribute', '-sa', required=True)
    args = parser.parse_args()

    import numpy as np
    np.float = float

    dataset_config_file = os.path.join(current_path, 'config', 'dataset', f"{args.dataset_name.lower()}.yaml")
    if not os.path.exists(dataset_config_file):
        raise ValueError(f"No config existing for dataset [{args.dataset_name.lower()}]")
    model_file = [
        f.path for f in os.scandir(os.path.join(current_path, 'saved')) if args.dataset_name.upper() in f.name and \
                                                                           args.model_name in f.name
    ][0]

    dataset_pert_config_path = os.path.join(
        current_path, 'config', 'perturbation', f"{args.dataset_name.lower()}_perturbation.yaml"
    )

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file, perturbation_config_file=dataset_pert_config_path
    )
    user_list = torch.arange(1, dataset.user_num)

    if 'DeltaNDCG' in config['metrics']:
        config['metrics'].remove('DeltaNDCG')
    perturbation_trainer = BeyondAccuracyPerturbationTrainer(
        config,
        train_data.dataset,
        valid_data,
        model
    )

    _, rec_model_topk = perturbation_trainer._get_model_score_data(user_list, valid_data)
    perturbation_trainer.determine_adv_group(user_list.numpy(), rec_model_topk)
    pref_data = perturbation_trainer._pref_data_sens_and_metric(user_list.numpy(), rec_model_topk)

    sampled_data = {'user': [], 'item': []}
    for curr_policy, policy_conf in perturbation_sampler_kwargs_gen(config):
        pert_sampler = PerturbationSampler(
            perturbation_trainer.dataset,
            perturbation_trainer,
            policy_conf
        )
        sampled_users, sampled_items = pert_sampler.apply_policies(user_list, pref_data)

        if curr_policy in user_policies:
            sampled_data['user'].append([curr_policy, sampled_users])
        else:
            sampled_data['item'].append([curr_policy, sampled_items])

    out_data = {'user': [], 'item': []}
    for pol_type in ['user','item']:
        for samp_data_1 in sampled_data[pol_type]:
            for samp_data_2 in sampled_data[pol_type]:
                pol1, samp_list_1 = samp_data_1
                pol2, samp_list_2 = samp_data_2

                unique, counts = torch.unique(torch.cat((samp_list_1, samp_list_2)), return_counts=True)

                jaccard_sim = unique[counts > 1].shape[0] / unique.shape[0]

                out_data[pol_type].append([pol1, pol2, jaccard_sim])

        df = pd.DataFrame(out_data[pol_type], columns=["Policy 1", "Policy 2", "Jaccard Similarity"])
        out_path = os.path.join(
            current_path, "policy_overlap_comparison", f"{args.dataset_name.lower()}_{args.model_name.lower()}"
        )
        os.makedirs(out_path, exist_ok=True)

        file_name = os.path.join(out_path, f"{pol_type}_policies_jaccard_similarity")
        with open(file_name + ".csv", 'w') as file:
            file.write(
                f"{perturbation_trainer.sensitive_attribute}     "
                f"{dataset.field2id_token[perturbation_trainer.sensitive_attribute]}      "
                f"adv_group: {perturbation_trainer.adv_group}\n"
            )
            df.to_csv(file, index=False)

        fig, ax = plt.subplots(1, 1)
        plot = sns.heatmap(df.pivot(index='Policy 1', columns='Policy 2'), ax=ax)
        fig.savefig(file_name + ".png", dpi=200)
        plt.close(fig)
