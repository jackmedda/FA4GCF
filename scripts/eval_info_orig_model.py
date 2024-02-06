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

import gnnuers

current_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file, os.pardir))

import fa4gcf.utils as utils
from fa4gcf.config import Config
from fa4gcf.evaluation import (
    Evaluator,
    pref_data_from_checkpoint,
    extract_metrics_from_perturbed_edges,
)


if __name__ == "__main__":
    """It works only when called from outside of the scripts folder as a script (not as a module)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', '-mf', required=True)
    parser.add_argument('--sensitive_attribute', '-sa', required=True)
    parser.add_argument('--gpu_id', default=1)
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

    checkpoint = torch.load(args.model_file)

    dset, mod = checkpoint["config"]["dataset"], checkpoint["config"]["model"]
    s_attr = args.sensitive_attribute

    config = Config(
        model=mod,
        dataset=dset,
        config_file_list=[
            os.path.join(current_file, '..', 'config', 'base_config.yaml'),
            os.path.join(current_file, '..', 'config', 'dataset', f'{dset}.yaml')
        ],
        config_dict={"gpu_id": args.gpu_id}
    )
    config, _, dataset, train_data, valid_data, test_data = utils.load_data_and_model(
        args.model_file,
        config.final_config_dict  # it could contain updated information, e.g., it should load the user_feat
    )

    orig_test_pref_data = pref_data_from_checkpoint(config, checkpoint, train_data, test_data)
    orig_valid_pref_data = pref_data_from_checkpoint(config, checkpoint, train_data, valid_data)

    demo_group_map = dataset.field2id_token[s_attr]

    evaluator = Evaluator(config)
    for _pref_data, _eval_data in zip([orig_test_pref_data, orig_valid_pref_data], [test_data.dataset, valid_data.dataset]):
        _pref_data['Demo Group'] = [
            demo_group_map[dg] for dg in dataset.user_feat[s_attr][_pref_data['user_id']].numpy()
        ]
        _pref_data["Demo Group"] = _pref_data["Demo Group"].map(consumer_group_map[s_attr.lower()]).to_numpy()

        metric_result = gnnuers.evaluation.compute_metric(evaluator, _eval_data, _pref_data, 'cf_topk_pred', 'ndcg')
        _pref_data['Value'] = metric_result[:, -1]
        _pref_data['Quantile'] = _pref_data['Value'].map(lambda x: np.ceil(x * 10) / 10 if x > 0 else 0.1)

    dgs = list(consumer_group_map[s_attr.lower()].values())
    plot_df_data = []
    for orig_dp_df, split in zip(
        [orig_test_pref_data, orig_valid_pref_data],
        ['Test', 'Valid']
    ):
        total = orig_dp_df['Value'].mean()
        metr_dg1 = orig_dp_df.loc[orig_dp_df['Demo Group'] == dgs[0], 'Value'].to_numpy()
        metr_dg2 = orig_dp_df.loc[orig_dp_df['Demo Group'] == dgs[1], 'Value'].to_numpy()
        _dp = gnnuers.evaluation.compute_DP(metr_dg1.mean(), metr_dg2.mean())
        pval = scipy.stats.mannwhitneyu(metr_dg1, metr_dg2).pvalue
        plot_df_data.append([_dp, split, 'Orig', metr_dg1.mean(), metr_dg2.mean(), total, pval])

    dp_plot_df = pd.DataFrame(plot_df_data, columns=['$\Delta$NDCG', 'Split', 'Policy', *dgs, 'NDCG', 'pvalue'])
    print(dp_plot_df)
