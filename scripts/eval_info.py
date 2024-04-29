import os
import sys
import yaml
import pickle
import argparse

import scipy
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from recbole.evaluator import Evaluator

current_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file, os.pardir))

import fa4gcf.utils as utils
import fa4gcf.evaluation as evaluation
from fa4gcf.config import Config
from fa4gcf.utils.case_study import (
    pref_data_from_checkpoint,
    extract_metrics_from_perturbed_edges
)


if __name__ == "__main__":
    """It works only when called from outside of the scripts folder as a script (not as a module)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', '--e', required=True)
    parser.add_argument('--base_plots_path', '--bpp', default=os.path.join('scripts', 'plots'))
    parser.add_argument('--gpu_id', default=1)
    parser.add_argument('--psi_impact', action="store_true")
    args = parser.parse_args()

    consumer_group_map = {
        'gender': {'M': 'M', 'F': 'F'},
        'age': {'M': 'Y', 'F': 'O'},
        'user_wide_zone': {'M': 'America', 'F': 'Other'}
    }

    group_name_map = {
        "M": "Males",
        "F": "Females",
        "Y": "Younger",
        "O": "Older",
        "America": "America",
        "Other": "Other"
    }

    if args.exp_path[-1] != os.sep:
        args.exp_path += os.sep

    path_split_key = 'dp_perturbations' if 'dp_perturbations' in args.exp_path else 'dp_explanations'
    _, dset, mod, _, _, s_attr, eps, cid, _ = args.exp_path.split(path_split_key)[1].split(os.sep)
    eps = eps.replace('epochs_', '')

    model_files = os.scandir(os.path.join(os.path.dirname(sys.path[0]), 'saved'))
    model_file = [f.path for f in model_files if mod in f.name and dset.upper() in f.name][0]

    config = Config(
        model=mod,
        dataset=dset,
        config_file_list=[os.path.join(current_file, '..', 'config', 'perturbation', 'base_perturbation.yaml')],
        config_dict={"gpu_id": args.gpu_id}
    )
    perturbation_config = config.update_base_perturb_data(os.path.join(current_file, '..', args.exp_path, 'config.pkl'))
    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(
        model_file,
        perturbation_config
    )

    # Users policies
    zerousers_pol = 'ZN'
    furthestusers_pol = 'FR'
    interrecency_pol = 'IR'

    # Items policies
    itemspref_pol = 'IP'
    timelessitems_pol = 'IT'
    pagerankitems_pol = 'PR'

    policy_order_base = [
        zerousers_pol,
        furthestusers_pol,
        interrecency_pol,
        itemspref_pol,
        timelessitems_pol,
        pagerankitems_pol
    ]

    palette = dict(zip(policy_order_base, sns.color_palette("colorblind")))
    pol_hatches = dict(zip(policy_order_base, ['X', '.', '/', 'O', '*']))

    policy_map = {
        'users_zero_constraint': zerousers_pol,
        'users_furthest_constraint': furthestusers_pol,
        'items_preference_constraint': itemspref_pol,
        'interaction_recency_constraint': interrecency_pol,
        'items_timeless_constraint': timelessitems_pol,
        'items_pagerank_constraint': pagerankitems_pol
    }

    pol_key = 'perturbation_policies' if 'perturbation_policies' in config else 'explainer_policies'
    raw_exp_policies = [k for k, v in config[pol_key].items() if v and k in policy_map]
    exp_policies = [policy_map[k] for k in raw_exp_policies]
    curr_policy = '+'.join(exp_policies)

    edge_additions = config['edge_additions']
    eval_metric = config['eval_metric'].upper()
    plots_path = os.path.join(args.base_plots_path, dset, mod, s_attr, f"{cid}_{curr_policy}")
    if args.psi_impact:
        exp_policies_ratios = [config[k + "_ratio"] for k in raw_exp_policies]
        ratios_str = f" ({'+'.join(map(str, exp_policies_ratios))})"
        plots_path += ratios_str
        curr_policy += ratios_str
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    checkpoint = torch.load(model_file)
    orig_test_pref_data = pref_data_from_checkpoint(config, checkpoint, train_data, test_data)
    orig_valid_pref_data = pref_data_from_checkpoint(config, checkpoint, train_data, valid_data)

    demo_group_map = dataset.field2id_token[s_attr]

    evaluator = Evaluator(config)
    for _pref_data, _eval_data in zip([orig_test_pref_data, orig_valid_pref_data], [test_data.dataset, valid_data.dataset]):
        _pref_data['Demo Group'] = [
            demo_group_map[dg] for dg in dataset.user_feat[s_attr][_pref_data['user_id']].numpy()
        ]
        _pref_data["Demo Group"] = _pref_data["Demo Group"].map(consumer_group_map[s_attr.lower()]).to_numpy()

        metric_result = evaluation.compute_metric(evaluator, _eval_data, _pref_data, 'cf_topk_pred', 'ndcg')
        _pref_data['Value'] = metric_result[:, -1]
        _pref_data['Quantile'] = _pref_data['Value'].map(lambda x: np.ceil(x * 10) / 10 if x > 0 else 0.1)

    batch_exp = config['user_batch_exp']
    exps, rec_model_preds, test_model_preds = utils.load_dp_perturbations_file(args.exp_path)
    best_exp = utils.get_best_pert_early_stopping(exps[0], config)

    pert_edges = best_exp[utils.pert_col_index('del_edges')]

    _, valid_pert_df, test_pert_df = extract_metrics_from_perturbed_edges(
        {(dset, s_attr): pert_edges},
        models=[mod],
        metrics=["NDCG", "Recall"],
        models_path=os.path.join(current_file, os.pardir, 'saved'),
        on_bad_models='ignore',
        remap=False
    )

    test_pert_df = test_pert_df[test_pert_df['Metric'].str.upper() == eval_metric]
    valid_pert_df = valid_pert_df[valid_pert_df['Metric'].str.upper() == eval_metric]
    for _pert_df in [test_pert_df, valid_pert_df]:
        _pert_df['Quantile'] = _pert_df['Value'].map(lambda x: np.ceil(x * 10) / 10 if x > 0 else 0.1)
        _pert_df["Demo Group"] = _pert_df["Demo Group"].map(consumer_group_map[s_attr.lower()]).to_numpy()

    # print(f'{"*" * 15} Test {"*" * 15}')
    # print(f'{"*" * 15} {s_attr.title()} {"*" * 15}')
    # for dg, sa_dg_df in test_pert_df.groupby('Demo Group'):
    #     print(f'\n{"*" * 15} {dg.title()} {"*" * 15}')
    #     print(sa_dg_df.describe())

    dgs = list(consumer_group_map[s_attr.lower()].values())
    orig_pert_pval_dict = {'Valid': {}, 'Test': {}}
    plot_df_data = []
    for orig_dp_df, pert_dp_df, split in zip(
        [orig_test_pref_data, orig_valid_pref_data],
        [test_pert_df, valid_pert_df],
        ['Test', 'Valid']
    ):
        orig_pert_pval_dict[split][eval_metric] = scipy.stats.mannwhitneyu(
            orig_dp_df['Value'], pert_dp_df['Value']
        ).pvalue

        total = orig_dp_df['Value'].mean()
        metr_dg1 = orig_dp_df.loc[orig_dp_df['Demo Group'] == dgs[0], 'Value'].to_numpy()
        metr_dg2 = orig_dp_df.loc[orig_dp_df['Demo Group'] == dgs[1], 'Value'].to_numpy()
        _dp = evaluation.compute_DP(metr_dg1.mean(), metr_dg2.mean())
        pval = scipy.stats.mannwhitneyu(metr_dg1, metr_dg2).pvalue
        plot_df_data.append([_dp, split, 'Orig', metr_dg1.mean(), metr_dg2.mean(), total, pval])

        total = pert_dp_df['Value'].mean()
        metr_dg1 = pert_dp_df.loc[pert_dp_df['Demo Group'] == dgs[0], 'Value'].to_numpy()
        metr_dg2 = pert_dp_df.loc[pert_dp_df['Demo Group'] == dgs[1], 'Value'].to_numpy()
        _dp = evaluation.compute_DP(metr_dg1.mean(), metr_dg2.mean())

        pval = scipy.stats.mannwhitneyu(metr_dg1, metr_dg2).pvalue
        plot_df_data.append([_dp, split, curr_policy, metr_dg1.mean(), metr_dg2.mean(), total, pval])

        try:
            orig_pert_pval_dict[split]['DP'] = scipy.stats.wilcoxon(
                orig_dp_df.sort_values('user_id')['Value'].to_numpy(),
                pert_dp_df.sort_values('user_id')['Value'].to_numpy()
            ).pvalue
        except ValueError:  # zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
            # highest pvalue because the distributions are equal
            orig_pert_pval_dict[split]['DP'] = 1.0

    dp_plot_df = pd.DataFrame(plot_df_data, columns=['$\Delta$' + eval_metric, 'Split', 'Policy', *dgs, eval_metric, 'pvalue'])
    dp_plot_df.to_markdown(os.path.join(plots_path, 'DP_barplot.md'), index=False)
    dp_plot_df.to_latex(os.path.join(plots_path, 'DP_barplot.tex'), index=False)
    dp_plot_df.to_csv(os.path.join(plots_path, 'DP_barplot.csv'), index=False)
    with open(os.path.join(plots_path, 'orig_pert_pval_dict.pkl'), 'wb') as f:
        pickle.dump(orig_pert_pval_dict, f)
    print(dp_plot_df)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.barplot(x='Split', y='$\Delta$' + eval_metric, data=dp_plot_df, hue='Policy', ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_path, 'DP_barplot.png'), bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close()
