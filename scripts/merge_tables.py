import os
import re
import math
import pickle
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mpl_lines
import matplotlib.colors as mpl_colors
import matplotlib.ticker as mpl_tickers
import matplotlib.patches as mpl_patches
import matplotlib.legend_handler as mpl_legend_handlers


def update_plt_rc():
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
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

    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    args.exclude = args.exclude or []
    print(args)

    sns.set_context("paper")
    update_plt_rc()
    out_path = os.path.join(args.base_plots_path, args.dataset)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    def pval_symbol(pval):
        if pval < 0.01:
            return '^'
        elif pval < 0.05:
            return '*'

        return ''

    def remove_str_pval(str_pval):
        return str_pval.replace('^', '').replace('*', '')

    def hl(val):
        return "\hl{" + val + "}"

    user_policies = ['ZN', 'FR', 'IR']
    item_policies = ['IP', 'IT', 'PR']
    user_item_policies = [f"{up}+{ip}" for up in user_policies for ip in item_policies]

    dataset_order = ['RENT', 'LF1K', 'FNYC', 'FTKY', 'ML1M']
    models_order = ['AutoCF', 'DirectAU', 'HMLET', 'LightGCN', 'NCL', 'NGCF', 'SGL', 'SVD_GCN']
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

    cols_order = ["Dataset", "Model", "GroupAttribute", "Policy", "Split", "Metric", "Value", "pvalue"]
    first_df = pd.concat(loaded_dfs, ignore_index=True)
    first_df["Policy"] = first_df["Policy"].str.replace("IP+IR", "IR+IP")
    first_df = first_df.melt(first_df.columns[~first_df.columns.isin([metric, 'DP'])], var_name='Metric', value_name='Value')
    first_df = first_df.drop_duplicates(subset=cols_order[:6])
    first_df = first_df[cols_order]

    first_df['Dataset'] = first_df['Dataset'].map(dataset_map)
    first_df['GroupAttribute'] = first_df['GroupAttribute'].map(group_attr_map)
    first_df.sort_values(cols_order[:5])

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

    first_best_pol_orig_df = pd.concat([test_best_first_total_df, test_orig_first_total_df], axis=0, ignore_index=True)
    first_best_pol_orig_df["Model"] = first_best_pol_orig_df["Model"].replace('SVD_GCN', 'SVD-GCN')
    first_best_pol_orig_df["Metric"] = first_best_pol_orig_df["Metric"].replace('DP', '$\Delta$')
    first_best_pol_orig_df["Status"] = first_best_pol_orig_df["Policy"].map(lambda p: "Before" if p == "Orig" else "After")
    first_best_pol_orig_df_pivot = first_best_pol_orig_df.pivot(
        index=["Dataset", "GroupAttribute", "Status"],
        columns=["Model", "Metric"],
        values="Value"
    ).reindex(
        ["NDCG", "$\Delta$"], axis=1, level=1
    )


    def bold_row(row):
        metr_index = row.index.get_level_values(level=1)
        best_ndcg_idx = row[metr_index == "NDCG"].argmax() * 2
        best_dp_idx = row[metr_index == r"$\Delta$"].argmin() * 2 + 1
        row = row.apply(lambda x: f"{x:.2f}")
        row[best_ndcg_idx] = r"\textbf{" + row[best_ndcg_idx] + "}"
        row[best_dp_idx] = r"\textbf{" + row[best_dp_idx] + "}"
        return row

    first_best_pol_orig_df_pivot = first_best_pol_orig_df_pivot.apply(bold_row, axis=1)
    first_best_pol_orig_df_pivot.index.names = [""] * len(first_best_pol_orig_df_pivot.index.names)
    first_best_pol_orig_df_pivot.columns.names = [""] * len(first_best_pol_orig_df_pivot.columns.names)

    with open(os.path.join(os.path.dirname(out_path), "best_policy_compare_orig_dp_utility.tex"), "w") as tex_file:
        tex_file.write(
            first_best_pol_orig_df_pivot.to_latex(
                column_format="lll|rr|rr|rr|rr|rr|rr|rr|rr|rr",
                multicolumn=True,
                multicolumn_format="c|",
                multirow=True,
                escape=False
            ).replace(
                "NDCG", r"NDCG $\uparrow$"
            ).replace(
                "$\Delta$", "$\Delta$ $\downarrow$"
            )
        )

    first_pval_df = first_df.copy(deep=True)
    first_pval_df['Value'] = first_pval_df[['Value', 'pvalue']].apply(lambda row: f"{row['Value']:.2f}{pval_symbol(row['pvalue'])}", axis=1)
    del first_pval_df['pvalue']

    rel_delta_col = 'Rel. $Delta$ (%)'
    change_idx = ['Dataset', 'Model', 'GroupAttribute', 'Split', 'Metric', 'Policy']
    final_idx_df = first_pval_df.set_index(change_idx)
    for set_pert, gby_df in first_pval_df.groupby(change_idx[:-1]):
        gby_pol_df = gby_df.set_index(['Policy'])
        orig_val = float(remove_str_pval(gby_pol_df.loc['Orig', 'Value']))
        for pol in gby_pol_df.index:
            if pol != 'Orig':
                pol_val = float(remove_str_pval(gby_pol_df.loc[pol, 'Value']))
                pol_impact = (pol_val - orig_val) / orig_val * 100
                if pol_impact > 100:
                    pol_impact_bound = " ($>$+100%)"
                elif pol_impact < -100:
                    pol_impact_bound = " ($<$-100%)"
                else:
                    pol_impact_bound = f" ({pol_impact:+05.1f}%)"

                final_idx_df.loc[tuple([*set_pert, pol]), 'Value'] += pol_impact_bound
                final_idx_df.loc[tuple([*set_pert, pol]), rel_delta_col] = f"{pol_impact:.2f}"

    final_df = final_idx_df.reset_index()

    pivot_df = final_df.pivot(
        columns=['Dataset', 'Model', 'Metric'],
        index=['Split', 'Policy', 'GroupAttribute'],
        values='Value'
    )

    pivot_df = pivot_df.reindex(
        dataset_order, axis=1, level=0
    ).reindex(
        models_order, axis=1, level=1
    ).reindex(
        [metric, 'DP'], axis=1, level=2
    ).reindex(
        ['Valid', 'Test'], axis=0, level=0
    ).reindex(
        policies_order, axis=0, level=1
    ).reindex(
        group_attr_order, axis=0, level=2
    )

    pivot_df.to_csv(os.path.join(out_path, 'best_exp_change_table.csv'))

    merged_dfs_to_merge = []
    for dirpath, _, merged_csvs in os.walk(os.path.dirname(out_path)):
        if merged_csvs and '.ipynb_checkpoints' not in dirpath:
            for mc in merged_csvs:
                if mc == 'best_exp_change_table.csv':
                    merged_dfs_to_merge.append(pd.read_csv(os.path.join(dirpath, mc), index_col=[0,1,2], header=[0,1]))

    total_df = pd.concat(merged_dfs_to_merge, axis=1)

    # set_gr_idx = list(set([x[:2] for x in total_df.index]))
    # for col in total_df.columns:
    #     for sg_idx in set_gr_idx:
    #         col_setting_df = total_df.loc[sg_idx, col]
    #         to_hl = []
    #         for cs_df_idx in col_setting_df.index:
    #             if cs_df_idx != 'Orig':
    #                 change = float(re.search("([+-]\d+[.]\d+)|([+-]\d+)", col_setting_df.loc[cs_df_idx]).group(0))
    #                 if abs(change) < args.hl_threshold:
    #                     to_hl.append(True)
    #                 else:
    #                     to_hl.append(False)
    #         if all(to_hl):
    #             for cs_df_idx in col_setting_df.index:
    #                 if cs_df_idx != 'Orig':
    #                     col_setting_df.loc[cs_df_idx] = "\cellcolor{lightgray} " + col_setting_df.loc[cs_df_idx]
    #
    #         for cs_df_idx in col_setting_df.index:
    #             if cs_df_idx != 'Orig':
    #                 val, ch = col_setting_df.loc[cs_df_idx].split('(')
    #                 col_setting_df.loc[cs_df_idx] = val + '(\textit{' + ch[:-1] + '})'
    #
    #         # format cells with makecell
    #         # for cs_df_idx in col_setting_df.index:
    #         #     if cs_df_idx != 'Orig':
    #         #         left_val, right_val = col_setting_df.loc[cs_df_idx].split(' (')
    #         #         col_setting_df.loc[cs_df_idx] = "\makecell{" + left_val + r" \\ " + f"({right_val}" + "}"

    total_df.columns.names = ['' for x in total_df.columns.names]
    total_df.index.names = ['' for x in total_df.index.names]
    total_df.index = pd.MultiIndex.from_tuples([(f"{x[0]} - {x[1]}" if 'C' in x[0] else x[0], x[2]) for x in total_df.index])
    total_df.index = pd.MultiIndex.from_tuples([("\rotatebox[origin=c]{90}{" + x[0] + "}", x[1]) for x in total_df.index])

    # re.sub('(?<=\d)\*', '{\\\\scriptsize *}', aa)
    col_sep = "1.6cm"
    n_cols = total_df.columns.shape[0]
    with open(os.path.join(os.path.dirname(out_path), 'total_best_exp_change_table.tex'), 'w') as f:
        f.write(
            total_df.to_latex(
                column_format='cM{1.2cm}|' + ''.join(['M{' + col_sep + '}' + ('' if (i + 1) % 3 != 0 or (i + 1) == n_cols else '|') for i in range(n_cols)]),
                multicolumn_format='c',
                multirow=True,
                escape=False
            ).replace(
                '^', '\^{}'
            ).replace(
                '%', '\%'
            ).replace(
                '{*}', '{*}[-10pt]'
            ).replace(
                '{*}[-10pt]{\\rotatebox[origin=c]{90}{\\textbf{CS - Gender}', '{*}[-5pt]{\\rotatebox[origin=c]{90}{\\textbf{CS - Gender}'
            ).replace(
                '&       & \\multicolumn{3}{c}{INS}', '\\multicolumn{2}{c}{}       & \\multicolumn{3}{c}{INS}'
            ).replace(
                '\\cline{1-11}', '\\cline{2-11}\n\\cline{2-11}'
            ).replace(
                '\\bottomrule', '\\midrule\n\\bottomrule'
            ).replace(
                '\\toprule', '\\toprule\n\\midrule'
            )
        )

    # NEW RQ1
    P_VALUE_RQ1 = 0.05

    delta_pivot_df = final_df.pivot(
        columns=['Dataset', 'Model', 'Metric'],
        index=['Split', 'Policy', 'GroupAttribute'],
        values=rel_delta_col
    ).dropna()

    delta_pivot_df = delta_pivot_df.reindex(
        dataset_order, axis=1, level=0
    ).reindex(
        models_order, axis=1, level=1
    ).reindex(
        [metric, 'DP'], axis=1, level=2
    ).reindex(
        ['Valid', 'Test'], axis=0, level=0
    ).reindex(
        policies_order, axis=0, level=1
    ).reindex(
        group_attr_order, axis=0, level=2
    )

    delta_pivot_df.to_csv(os.path.join(out_path, 'best_exp_rel_delta_change_table.csv'))

    merged_delta_dfs_to_merge = []
    for dirpath, _, merged_csvs in os.walk(os.path.dirname(out_path)):
        if merged_csvs and '.ipynb_checkpoints' not in dirpath:
            for mc in merged_csvs:
                if mc == 'best_exp_rel_delta_change_table.csv':
                    merged_delta_dfs_to_merge.append(pd.read_csv(os.path.join(dirpath, mc), index_col=[0,1,2], header=[0,1,2]))

    delta_total_df = pd.concat(merged_delta_dfs_to_merge, axis=1)

    delta_total_df = delta_total_df.reset_index().melt(["Split", "Policy", "GroupAttribute"], value_name=rel_delta_col)

    orig_pert_pval_df = pd.DataFrame(orig_pert_pval_data, columns=orig_pert_pval_cols)
    orig_pert_pval_df["Policy"] = orig_pert_pval_df["Policy"].str.replace("IP+IR", "IR+IP")
    orig_pert_pval_df['Dataset'] = orig_pert_pval_df['Dataset'].map(dataset_map)
    # orig_pert_pval_df['Model'] = orig_pert_pval_df['Model'].map(model_map)
    orig_pert_pval_df['GroupAttribute'] = orig_pert_pval_df['GroupAttribute'].map(group_attr_map)

    pval_join_cols = ["Dataset", "Model", "GroupAttribute", "Policy", "Split", "Metric"]
    delta_total_df = delta_total_df.join(orig_pert_pval_df[pval_join_cols + ["P_value"]].set_index(pval_join_cols), on=pval_join_cols)
    print(delta_total_df)
    delta_total_df.to_csv(os.path.join(os.path.dirname(out_path), 'delta_total_change_table.csv'))

    palette = dict(zip([pert_type_map['Perturbed | addition'], pert_type_map['Perturbed | deletion']], sns.color_palette("colorblind")))
    markers = dict(zip([pert_type_map['Perturbed | addition'], pert_type_map['Perturbed | deletion']], ["s", "o"]))
    p_val_hatch = '+'
    markersize=20

    sns.set_context("paper")
    update_plt_rc()
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=16)

    xlims = {
        "GCMC": (3 * -10 ** 2, 10 ** 5 - 1),
        "LGCN": (3 * -10 ** 2, 10 ** 5 - 1),
        "NGCF": (3 * -10 ** 2, 10 ** 5 - 1)
    }
    delta_total_df_gby = delta_total_df.groupby("Dataset")
    for dset in dataset_order:
        if dset not in delta_total_df_gby.groups:
            continue
        delta_total_dset_df = delta_total_df_gby.get_group(dset)
        delta_total_dset_df_gby = delta_total_dset_df.groupby("Stakeholder")

        handles, labels = None, None
        fig = plt.figure(figsize=(6 * len(models_order), 2.8 if dset == 'LF1K' else 3), constrained_layout=True)
        height_ratios = (2, 1)
        gs = fig.add_gridspec(
            2, len(models_order),  height_ratios=height_ratios,
            # left=0.1, right=0.9, bottom=0.1, top=0.9,
            wspace=0.01
        )

        for plot_i, sk in enumerate(["Consumer", "Provider"]):
            delta_total_sk_df = delta_total_dset_df_gby.get_group(sk)
            delta_total_sk_gby = delta_total_sk_df.groupby("Model")
            for plot_j, mod in enumerate(models_order):
                delta_total_mod_df = delta_total_sk_gby.get_group(mod)

                kws = {}
                if plot_i > 0:
                    kws['sharex'] = fig.axes[plot_j]
                if plot_j > 0:
                    kws['sharey'] = fig.axes[gs.ncols * plot_i]
                ax = fig.add_subplot(gs[plot_i, plot_j], **kws)

                for pt_idx, (pt, delta_pt_df) in enumerate(delta_total_mod_df.groupby("Perturbation Type")):
                    sns.stripplot(
                        data=delta_pt_df, x="Rel. $Delta$ (%)", y="FullSetting", hue="Perturbation Type", palette={pt: palette[pt]},
                        marker=markers[pt], size=markersize, orient="h", jitter=False, linewidth=1, edgecolor="w", ax=ax
                    )

                    pt_collections = ax.collections[(pt_idx * len(delta_pt_df) + 1 * pt_idx):-1]
                    for collection, pval in zip(pt_collections, delta_pt_df['P_value'].values):
                        if pval < P_VALUE_RQ1:
                            collection.set_hatch(p_val_hatch)

                if handles is None and labels is None:
                    handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()

        for row in range(gs.nrows):
            for col in range(gs.ncols):
                ax = fig.axes[row * gs.ncols + col]

                if row < (gs.nrows - 1):
                    ax.tick_params(bottom=False, labelbottom=False)
                if col > 0:
                    ax.tick_params(left=False, labelleft=False)

                if dset == dataset_order[0] and row == 0:
                    ax.set_title(models_order[col])
                ax.set(xlabel='', ylabel='', xscale='symlog')
                ax.xaxis.grid(True)
                ax.yaxis.grid(True)
                ax.autoscale(False)
                # print(dset, sk, models_order[col], ax.get_xlim())
                ax.set_xlim(xlims[models_order[col]])
                if dset != "ML1M":
                    ax.tick_params(bottom=False, labelbottom=False)
                # xlim = ax.get_xlim()
                # ax.set_xlim((xlim[0] * (0.2 if dset == "INS" else 1), xlim[1] * 1.5))
                # if dset == "ML1M":
                #     ax.set_xlim((xlim[0] * 0.9, xlim[1] * 1.48))
                ax.plot([0, 0], list(ax.get_ylim()), "k", ls='-', lw=3, clip_on=False)

        handles, labels = map(list, zip(*[(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]))
        handles = [mpl_lines.Line2D([], [], color=hand.get_facecolor(), marker=markers[lab], linestyle='None', markersize=markersize) for hand, lab in zip(handles, labels)]
        handles.extend(
            [
                mpl_lines.Line2D([], [], alpha=0),
                mpl_patches.Circle((20, 4), radius=20, facecolor="w", edgecolor="k", hatch=p_val_hatch),
                mpl_patches.Circle((20, 4), radius=20, facecolor="w", edgecolor="k")
            ]
        )
        labels.extend(["Wilcoxon signed-rank\n test on $\Delta$ with 5% CI", "Significant", "Not Significant"])
        handles = [mpl_lines.Line2D([], [], alpha=0), *handles]
        labels = ['Perturbation\n      Type', *labels]

        figlegend = plt.figure(figsize=(len(labels), 1))
        figlegend.legend(
            handles, labels, loc='center', frameon=False, fontsize=15, ncol=len(labels),
            handler_map={mpl_patches.Circle: HandlerEllipse()}
            # markerscale=1.8, handletextpad=0.1, columnspacing=1, borderpad=0.1
        )
        figlegend.savefig(os.path.join(args.base_plots_path, 'legend_delta_dotplot.png'), dpi=300, bbox_inches="tight", pad_inches=0)

        fig.savefig(os.path.join(args.base_plots_path, f'{dset}_delta_dotplot.png'), dpi=300, bbox_inches="tight", pad_inches=0)
