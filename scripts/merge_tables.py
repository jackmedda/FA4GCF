import os
import pdb
import re
import math
import pickle
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mpl_lines
import matplotlib.colors as mpl_colors
import matplotlib.ticker as mpl_tickers
import matplotlib.patches as mpl_patches
import matplotlib.transforms as mpl_trans
import matplotlib.legend_handler as mpl_legend_handlers


def update_plt_rc():
    SMALL_SIZE = 16
    MEDIUM_SIZE = 26
    BIGGER_SIZE = 30

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)


class HandlerEllipse(mpl_legend_handlers.HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = orig_handle.get_center()
        radius = orig_handle.get_radius()
        p = mpl_patches.Ellipse(
            xy=center, width=radius, height=radius
        )
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


if __name__ == "__main__":
    current_file = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '--d', required=True)
    parser.add_argument('--plots_path_to_merge', '--pptm', default=os.path.join(current_file, 'plots'))
    parser.add_argument('--base_plots_path', '--bpp', default=os.path.join(current_file, 'merged_plots'))
    parser.add_argument('--exclude', '--ex', nargs='+', help='Exclude certain config ids', default=None)
    parser.add_argument('--psi_impact', action="store_true")

    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    args.exclude = args.exclude or []
    print(args)

    sns.set_context("paper")
    update_plt_rc()
    out_path = os.path.join(args.base_plots_path, args.dataset)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    PVAL_001 = '^'
    PVAL_005 = '*'


    def pval_symbol(pval):
        if pval < 0.01:
            return PVAL_001
        elif pval < 0.05:
            return PVAL_005

        return ''

    def remove_str_pval(str_pval):
        return str_pval.replace('^', '').replace('*', '')

    def hl(val):
        return "\hl{" + val + "}"

    user_policies = ['ZN', 'FR', 'IR']
    item_policies = ['IP', 'IT', 'PR']
    user_item_policies = [f"{up}+{ip}" for up in user_policies for ip in item_policies]

    dataset_order = ['RENT', 'LF1K', 'FNYC', 'FTKY', 'ML1M']
    models_order = ['AutoCF', 'DirectAU', 'HMLET', 'LightGCN', 'NCL', 'NGCF', 'SGL', 'SVD-GCN', 'XSimGCL']
    policies_order = ['Orig'] + user_policies + item_policies + user_item_policies
    group_attr_order = ['Age', 'Gender']
    pert_type_order = ['Orig', 'Pert']

    loaded_dfs = []
    group_dfs = []
    dfs_across_exps = {'consumer': [], 'provider': []}
    edge_perturbation_impact = {}
    plots_path = os.path.join(args.plots_path_to_merge, args.dataset)
    for dirpath, dirnames, filenames in os.walk(plots_path):
        if filenames:
            for x in filenames:
                if x == 'DP_barplot.csv':
                    metadata = dirpath.split(args.dataset + os.sep)[1]
                    mod, s_attr, conf_pol = metadata.split(os.sep)
                    conf_id, policy = conf_pol.split('_')

                    metadata_map = {
                        'Dataset': args.dataset,
                        'Model': mod,
                        'GroupAttribute': s_attr
                    }
                    df = pd.read_csv(os.path.join(dirpath, x))

                    delta_col = df.columns[df.columns.str.contains('Delta')][0]
                    metric = delta_col.replace('$\Delta$', '')

                    df.rename(columns={delta_col: 'DP'}, inplace=True)
                    for key, val in metadata_map.items():
                        df[key] = val

                    rel_cols = ['Policy'] + list(metadata_map.keys())
                    loaded_dfs.append(df[rel_cols + ['Split', metric.upper(), 'DP', 'pvalue']])

                    g_df = df[df.columns[~df.columns.isin([delta_col, 'Split', 'DP', 'pvalue'])]].melt(rel_cols).rename(columns={
                        'variable': 'Metric', 'value': 'Value'
                    })
                    group_dfs.append(g_df)

    orig_pert_pval_data = []
    orig_pert_pval_cols = ['Dataset', 'Model', 'GroupAttribute', 'Policy', 'Split', 'Metric', 'P_value']
    all_dsets_path = os.path.dirname(plots_path)
    for dirpath, dirnames, filenames in os.walk(all_dsets_path):
        if filenames:
            for x in filenames:
                if x == 'orig_pert_pval_dict.pkl':
                    metadata = dirpath.split(all_dsets_path + os.sep)[1]
                    dset, mod, s_attr, conf_pol = metadata.split(os.sep)
                    conf_id, policy = conf_pol.split('_')

                    with open(os.path.join(dirpath, 'orig_pert_pval_dict.pkl'), 'rb') as f:
                        pval = pickle.load(f)
                        for spl, spl_pval in pval.items():
                            for metr, pval_value in spl_pval.items():
                                orig_pert_pval_data.append(
                                    [dset, mod, s_attr, policy, spl, metr, pval_value]
                                )

    dataset_map = {
        'rent_the_runway': 'RENT',
        'lastfm-1k': 'LF1K',
        'foursquare_nyc': 'FNYC',
        'foursquare_tky': 'FTKY',
        'ml-1m': 'ML1M'
    }

    # model_map = {
    #     'GCMC': 'GCMC',
    #     'LightGCN': 'LGCN',
    #     'NGCF': 'NGCF'
    # }

    setting_map = {
        'consumer | ndcg': 'CP',
        'consumer | softmax': 'CS',
        'provider | exposure': 'PE',
        'provider | visibility': 'PV'
    }

    pert_type_map = {
        'Orig | deletion': 'Orig',
        'Orig | addition': 'Orig',
        'Perturbed | deletion': '$\dotplus$ Del',
        'Perturbed | addition': '$\dotplus$ Add'
    }

    group_attr_map = {
        'gender': 'Gender',
        'age': 'Age'
    }

    orig_pert_pval_df = pd.DataFrame(orig_pert_pval_data, columns=orig_pert_pval_cols)

    cols_order = ["Dataset", "Model", "GroupAttribute", "Policy", "Split", "Metric", "Value", "pvalue"]
    first_df = pd.concat(loaded_dfs, ignore_index=True)
    first_df = first_df.melt(first_df.columns[~first_df.columns.isin([metric, 'DP'])], var_name='Metric', value_name='Value')
    first_df = first_df.drop_duplicates(subset=cols_order[:6])
    first_df = first_df[cols_order]

    first_df['Dataset'] = first_df['Dataset'].map(dataset_map)
    first_df['GroupAttribute'] = first_df['GroupAttribute'].map(group_attr_map)
    first_df.sort_values(cols_order[:5])

    orig_pert_pval_df['Dataset'] = orig_pert_pval_df['Dataset'].map(dataset_map)
    orig_pert_pval_df['GroupAttribute'] = orig_pert_pval_df['GroupAttribute'].map(group_attr_map)

    first_df['Value'] *= 100  # transform it to percentage
    first_df.to_csv(os.path.join(out_path, 'best_exp_raw_perc_values_table.csv'), index=False)

    first_merged_dfs_to_merge = []
    for dirpath, _, merged_csvs in os.walk(os.path.dirname(out_path)):
        if merged_csvs and '.ipynb_checkpoints' not in dirpath:
            for mc in merged_csvs:
                if mc == 'best_exp_raw_perc_values_table.csv':
                    first_merged_dfs_to_merge.append(pd.read_csv(os.path.join(dirpath, mc)))

    first_total_df = pd.concat(first_merged_dfs_to_merge, axis=0, ignore_index=True)
    first_total_df.to_csv(os.path.join(os.path.dirname(out_path), 'total_raw_perc_table.csv'), index=False)

    if args.psi_impact:
        if first_total_df.Policy.str.contains('IP+IR', regex=False).any():
            def remap_ipir_psi_values(val):
                psi_values = re.search('(?<=\().*(?=\))', val)[0]
                return f"IR+IP ({'+'.join(psi_values.split('+')[::-1])})"
            first_total_df['Policy'] = first_total_df['Policy'].map(
                lambda x: remap_ipir_psi_values(x) if 'IP+IR' in x else x
            )
        base_exp_ratios = '(0.35+0.2)'
        # base_exp_mask = first_df.Policy.str.contains('(0.35+0.2)', regex=False)
        # base_exp_df = first_df[base_exp_mask].reset_index(drop=True)
        # psi_impact_df = first_df[~base_exp_mask].reset_index(drop=True)

        first_total_df = first_total_df[first_total_df['Split'] == 'Test'].reset_index(drop=True)
        dp_key = '$\Delta$'
        first_total_df = first_total_df.replace('DP', dp_key)
        fixed_user_psi = first_total_df[first_total_df['Policy'].str.contains('0.35+', regex=False)]
        fixed_item_psi = first_total_df[first_total_df['Policy'].str.contains('+0.2', regex=False)]

        for i, (fixed_psi_df, varying_psi_type) in enumerate(
                zip([fixed_item_psi, fixed_user_psi], ['$\Psi_{\mathcal{U}}$', '$\Psi_{\mathcal{I}}$'])
        ):
            fig, axs = plt.subplots(len(fixed_psi_df['Dataset'].unique()), 1, figsize=(10, 8))  # , sharex='col')
            fixed_psi_df[varying_psi_type] = fixed_psi_df['Policy'].map(
                lambda p: p.split('(')[1].replace(')', '').split('+')[i]
            ).astype(float)
            fixed_psi_df['Policy'] = fixed_psi_df['Policy'].map(lambda p: p.split()[0])
            fixed_psi_df['Setting'] = fixed_psi_df[['Policy', 'Dataset', 'Model', 'GroupAttribute']].apply(
                lambda x: f'({",".join(x)})', axis=1
            )

            style_kws = dict(
                style='Metric',
                markers={dp_key: 'X', 'NDCG': 'P'},
                dashes={dp_key: (), 'NDCG': (2, 1)},
                errorbar=None,
                lw=5,
                markersize=30
            )

            psi_dsets_order = ['LF1K', 'ML1M']

            fixed_psi_df_gby = fixed_psi_df.groupby('Dataset')
            for dset_i, psi_dset in enumerate(psi_dsets_order):
                dset_psi_df = fixed_psi_df_gby.get_group(psi_dset)
                colors = sns.color_palette('cividis', n_colors=4)
                ax_color, axx_color = colors[0], colors[-2]
                axs[dset_i].margins(y=0.1)
                sns.lineplot(
                    x=varying_psi_type, y='Value', data=dset_psi_df[dset_psi_df['Metric'] == dp_key],
                    color=ax_color, ax=axs[dset_i], **style_kws
                )
                axs[dset_i].set_xlabel('')
                # axs[dset_i].set_ylabel(f'{dset_psi_df["Setting"].iloc[0]}\n{dp_key}', color=ax_color)
                axs[dset_i].set_ylabel(dp_key, color=ax_color)
                axs[dset_i].tick_params(axis='y', labelcolor=ax_color)

                axx = axs[dset_i].twinx()
                axx.margins(y=0.1)
                sns.lineplot(
                    x=varying_psi_type, y='Value', data=dset_psi_df[dset_psi_df['Metric'] == 'NDCG'],
                    color=axx_color, ax=axx, **style_kws
                )
                axx.set_xlabel('')
                axx.set_ylabel('NDCG', color=axx_color)
                axx.tick_params(axis='y', labelcolor=axx_color)

                axs[dset_i].grid(axis='both', which='major', ls=':', color='k')
                axs[dset_i].set_xticks(fixed_psi_df[varying_psi_type].unique())
                axs[dset_i].xaxis.set_major_formatter(mpl_tickers.FuncFormatter(lambda x, pos: f"{int(x * 100)}%"))
                axs[dset_i].yaxis.set_major_formatter(mpl_tickers.StrMethodFormatter("{x:.2f}"))
                axx.yaxis.set_major_formatter(mpl_tickers.StrMethodFormatter("{x:.2f}"))
                axs[dset_i].yaxis.set_major_locator(mpl_tickers.LinearLocator(6))
                axx.yaxis.set_major_locator(mpl_tickers.LinearLocator(6))

                ax_handles, ax_labels = axs[dset_i].get_legend_handles_labels()
                axx_handles, axx_labels = axx.get_legend_handles_labels()
                handles, labels = ax_handles + axx_handles, ax_labels + axx_labels
                axs[dset_i].get_legend().remove()
                axx.get_legend().remove()

            fig.subplots_adjust(hspace=0.5)
            for dset_i, dset in enumerate(psi_dsets_order):
                extent = axs[dset_i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())

                # bbox_args = [1.7, 1.1] if i == 0 else [1.1, 1.1]
                width = extent.width
                height = extent.height
                deltaw = (1.45 * width - width) / 2.0
                deltah = (1.36 * height - height) / 2.0  # (1.115 * height - height) / 2.0 if dset_i == 0 else (1.315 * height - height) / 2.0
                offsetw = -0.05 if dset_i == 0 or i == 0 else -0.14
                offseth = -0.660  # -0.101 if dset_i == 0 else -0.660
                a = np.array([[-deltaw - deltaw * offsetw, -deltah], [deltaw, deltah + deltah * offseth]])
                new_bbox = extent._points + a
                fig.savefig(
                    os.path.join(
                        os.path.dirname(out_path),
                        dset + ('_user_' if i == 0 else '_item_') + 'varying_psi_lineplot.png'
                    ),
                    bbox_inches=mpl_trans.Bbox(new_bbox), dpi=300
                )
            plt.close(fig)

        figlegend = plt.figure(figsize=(len(labels),  1))
        figlegend.legend(
            handles, labels, loc='center', frameon=False, fontsize=12, ncol=len(labels),
            markerscale=0.7, handlelength=5, # handletextpad=4, columnspacing=2, borderpad=0.1
        )
        figlegend.savefig(
            os.path.join(os.path.dirname(out_path), 'legend_psi_impact.png'),
            dpi=300, bbox_inches="tight", pad_inches=0
        )

        exit()
    else:
        first_total_df['Policy'] = first_total_df['Policy'].str.replace('IP+IR', 'IR+IP')
        orig_pert_pval_df['Policy'] = orig_pert_pval_df['Policy'].str.replace('IP+IR', 'IR+IP')

    # Raw best settings utility and fairness
    first_best_settings = []
    first_best_settings_cols = ["Dataset", "Model", "GroupAttribute", "Policy"]
    first_total_gby = first_total_df.groupby(["Dataset", "Model", "Split", "Metric"])
    for sa in ["Gender", "Age"]:
        for (dset, mod, spl, metr), first_total_group_df in first_total_gby:
            if spl == "Valid" and metr == "DP" and first_total_group_df["GroupAttribute"].str.contains(sa).any():
                first_total_sa_df = first_total_group_df[first_total_group_df["GroupAttribute"] == sa]
                first_total_sa_df = first_total_sa_df[first_total_sa_df["Policy"] != "Orig"]
                curr_best_setting = first_total_sa_df.iloc[first_total_sa_df.Value.argmin()]
                first_best_settings.append(curr_best_setting)

    first_best_settings_df = pd.concat(first_best_settings, axis=1).T
    first_bs_idx = first_best_settings_df.set_index(first_best_settings_cols)
    best_first_total_df = first_total_df.set_index(first_best_settings_cols).loc[first_bs_idx.index].reset_index()
    test_best_first_total_df = best_first_total_df[best_first_total_df["Split"] == "Test"]
    # test_best_pol_first_total_df = best_first_total_df.groupby(["Dataset", "Model", "GroupAttribute", "Metric"]).apply(
    #     lambda x: x.sort_values("Value", ascending=False).iloc[0]
    # ).reset_index(drop=True)

    test_bp_idx = test_best_first_total_df.set_index(first_best_settings_cols[:-1] + ["Split", "Metric"]).index
    test_orig_first_total_df = first_total_df[first_total_df["Policy"] == "Orig"].set_index(
        first_best_settings_cols[:-1] + ["Split", "Metric"]
    ).loc[test_bp_idx].reset_index()

    before_key, after_key = "Base", "Aug"
    first_best_pol_orig_df = pd.concat([test_best_first_total_df, test_orig_first_total_df], axis=0, ignore_index=True)
    first_best_pol_orig_df["Model"] = first_best_pol_orig_df["Model"].replace('SVD_GCN', 'SVD-GCN')
    first_best_pol_orig_df["Metric"] = first_best_pol_orig_df["Metric"].replace('DP', '$\Delta$')
    first_best_pol_orig_df["Status"] = first_best_pol_orig_df["Policy"].map(lambda p: before_key if p == "Orig" else after_key)
    first_best_pol_orig_df["GroupAttribute"] = first_best_pol_orig_df["GroupAttribute"].map({"Gender": "G", "Age": "A"})

    test_orig_pert_pval_df = orig_pert_pval_df[orig_pert_pval_df["Split"] == "Test"]
    test_orig_pert_pval_df["Model"] = test_orig_pert_pval_df["Model"].replace('SVD_GCN', 'SVD-GCN')
    test_orig_pert_pval_df["Metric"] = test_orig_pert_pval_df["Metric"].replace('DP', '$\Delta$')
    test_orig_pert_pval_df["GroupAttribute"] = test_orig_pert_pval_df["GroupAttribute"].map({"Gender": "G", "Age": "A"})

    pol_as_a_metric_df = first_best_pol_orig_df[(first_best_pol_orig_df["Status"] == "Aug") & (first_best_pol_orig_df["Metric"] == "NDCG")]
    pol_as_a_metric_df["Value"] = pol_as_a_metric_df["Policy"].map(lambda p: "{\\small \\emph{" + p + "}}")
    pol_as_a_metric_df["Metric"] = "PolAsMetric"

    # first_best_pol_orig_df = pd.concat([first_best_pol_orig_df, pol_as_a_metric_df], axis=0, ignore_index=True)
    pol_as_a_metric_df_pivot = pol_as_a_metric_df.pivot(
        index=["Dataset", "GroupAttribute", "Status"],
        columns=["Model", "Metric"],
        values="Value"
    )

    first_best_pol_orig_df_pivot = first_best_pol_orig_df.pivot(
        index=["Dataset", "GroupAttribute", "Status"],
        columns=["Model", "Metric"],
        values="Value"
    ).reindex(
        [before_key, after_key], axis=0, level=2
    ).reindex(
        ["NDCG", "$\Delta$"], axis=1, level=1
    )
    first_best_pol_orig_df_pivot.to_csv(os.path.join(os.path.dirname(out_path), "best_policy_compare_orig_dp_utility.csv"), sep='\t')


    def bold_row(row):
        metr_index = row.index.get_level_values(level=1)
        best_ndcg_idx = row[metr_index == "NDCG"].argmax() * 2
        best_dp_idx = row[metr_index == r"$\Delta$"].argmin() * 2 + 1
        row = row.apply(lambda x: f"{x:.2f}")
        row[best_ndcg_idx] = r"\textbf{" + row[best_ndcg_idx] + "}"
        row[best_dp_idx] = r"\textbf{" + row[best_dp_idx] + "}"
        return row


    def underline_improvements(col):
        float_col = col.str.replace(r"\textbf{", "").str.replace("}", "").astype(float)
        status_index = float_col.index.get_level_values(level=2)
        after_idx = float_col[status_index == after_key].values
        before_idx = float_col[status_index == before_key].values
        fn = "__gt__" if col.name[1] == "NDCG" else "__lt__"
        hl_mask = getattr(after_idx - before_idx, fn)(0)
        hl_idx = hl_mask.nonzero()[0] * 2 + 1
        col[hl_idx] = r"\underline{" + col[hl_idx] + "}"
        return col

    first_best_pol_orig_df_pivot = first_best_pol_orig_df_pivot.apply(bold_row, axis=1)
    first_best_pol_orig_df_pivot = first_best_pol_orig_df_pivot.apply(underline_improvements, axis=0)
    first_best_pol_orig_df_pivot = first_best_pol_orig_df_pivot.join(pol_as_a_metric_df_pivot).reindex(
        models_order, axis=1, level=0
    ).reindex(
        ["PolAsMetric", "NDCG", "$\Delta$"], axis=1, level=1
    )
    first_best_pol_orig_df_pivot = first_best_pol_orig_df_pivot.fillna('').rename(columns={'PolAsMetric': ''})

    first_best_pol_orig_df_pivot.index.names = [""] * len(first_best_pol_orig_df_pivot.index.names)
    first_best_pol_orig_df_pivot.columns.names = [""] * len(first_best_pol_orig_df_pivot.columns.names)

    pval_idx_columns = ["Dataset", "GroupAttribute", "Model", "Metric", "Policy"]
    test_orig_pert_pval_df_idx = test_orig_pert_pval_df.set_index(pval_idx_columns)
    for _, row in first_best_pol_orig_df_pivot.iterrows():
        row_idx = row.name
        for col_idx, _ in row.items():
            row_col_policy = first_best_pol_orig_df_pivot.loc[row_idx, (*col_idx[:1], '')].split('{')[-1].split('}')[0]
            if "Aug" in row_idx and '' not in col_idx:
                pval_str = ""
                if test_orig_pert_pval_df_idx.loc[(*row_idx[:2], *col_idx, row_col_policy), 'P_value'] < 0.05:
                    pval_str = f"$^{PVAL_005}$"
                row_col_val = first_best_pol_orig_df_pivot.loc[row_idx, col_idx]
                first_best_pol_orig_df_pivot.loc[row_idx, col_idx] = pval_str + row_col_val

    with open(os.path.join(os.path.dirname(out_path), "best_policy_compare_orig_dp_utility.tex"), "w") as tex_file:
        tex_file.write(
            first_best_pol_orig_df_pivot.to_latex(
                column_format=">{\\raggedright}p{7.5mm}>{\\raggedright}p{1mm}l*{9}{|>{\\raggedright}p{2.5mm}rr}",
                multicolumn=True,
                multicolumn_format="c|",
                multirow=True,
                escape=False
            ).replace(
                "NDCG", r"NDCG $\uparrow$"
            ).replace(
                "$\Delta$", "$\Delta$ $\downarrow_0$"
            ).replace(
                "\multirow[t]", "\multirow[c]"
            )
        )

    # Mini Heatmaps
    heatmaps_path = os.path.join(os.path.dirname(out_path), 'mini_heatmaps')
    os.makedirs(heatmaps_path, exist_ok=True)

    MAX_VMAX = 7
    for spl in ["Valid", "Test"]:
        cmap = sns.color_palette("cividis", as_cmap=True)
        first_total_heat_df = first_total_df[(first_total_df["Split"] == spl) & (first_total_df["Metric"] == "DP")]
        first_t_hm_gby = first_total_heat_df.groupby(["Model", "Dataset", "GroupAttribute"])
        vmin, vmax = 0, min(MAX_VMAX, first_total_heat_df["Value"].max())
        norm = mpl_colors.Normalize(vmin, vmax)
        for mod in models_order:
            mod_hm_df = first_total_heat_df[first_total_heat_df["Model"] == mod]
            for dset in dataset_order:
                for s_attr in ['Age', 'Gender']:
                    if (mod, dset, s_attr) not in first_t_hm_gby.groups:
                        continue

                    mini_heat_df = first_t_hm_gby.get_group((mod, dset, s_attr))

                    mini_heatmap_path = os.path.join(heatmaps_path, spl, mod)
                    os.makedirs(mini_heatmap_path, exist_ok=True)

                    orig_policy_dp = mini_heat_df.loc[mini_heat_df["Policy"] == "Orig", "Value"].item()
                    user_mh_df = mini_heat_df[mini_heat_df["Policy"].isin(user_policies)].set_index("Policy").reindex(
                        user_policies
                    )
                    item_mh_df = mini_heat_df[mini_heat_df["Policy"].isin(item_policies)].set_index("Policy").reindex(
                        item_policies
                    )
                    user_item_mh_df = mini_heat_df[mini_heat_df["Policy"].isin(user_item_policies)]

                    user_item_mh_df[['U', 'I']] = user_item_mh_df['Policy'].str.split('+', expand=True).values
                    user_item_hmap_data = user_item_mh_df.pivot(columns='I', index='U', values='Value').reindex(
                        user_policies, axis=0
                    ).reindex(
                        item_policies, axis=1
                    ).values
                    user_hmap_data = user_mh_df.loc[:, "Value"].values
                    item_hmap_data = item_mh_df.loc[:, "Value"].values

                    data = np.hstack([user_hmap_data.reshape([-1, 1]), user_item_hmap_data])
                    data = np.vstack([np.concatenate([[orig_policy_dp], item_hmap_data]).reshape([1, -1]), data])
                    data = pd.DataFrame(data, index=[''] + user_policies, columns=[''] + item_policies)

                    # annot = np.full_like(data, fill_value='', dtype='<U4')
                    greater_than_vmax_mask = data > MAX_VMAX
                    annot = data.applymap("{:.2f}".format).values
                    annot[0, 0] = "Base\n" + annot[0, 0]
                    annot[greater_than_vmax_mask] = f"> {MAX_VMAX}"

                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    sns.heatmap(
                        data, cmap=cmap, norm=norm, annot=annot, fmt="", square=True,
                        cbar=False, linewidths=.5, linecolor='white', ax=ax
                    )
                    ax.tick_params(length=0)
                    ax.xaxis.tick_top()
                    fig.savefig(
                        os.path.join(mini_heatmap_path, f"{dset}_{s_attr}_mini_heatmap.png"),
                        dpi=300, bbox_inches="tight", pad_inches=0
                    )
                    plt.close(fig)

        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar_fig, cbar_ax = plt.subplots(1, 1, figsize=(0.5, 10.8))
        colorbar = cbar_fig.colorbar(mappable, cax=cbar_ax, orientation="vertical")
        colorbar.ax.set_yticklabels([float(t.get_text()) for t in colorbar.ax.get_yticklabels()])
        cbar_fig.savefig(
            os.path.join(heatmaps_path, spl, "heatmaps_colorbar.png"), dpi=300, bbox_inches="tight", pad_inches=0
        )
        plt.close(cbar_fig)
