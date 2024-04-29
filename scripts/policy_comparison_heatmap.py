import os

import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


policies_map = {
    'users_zero_constraint': 'ZN',
    'users_furthest_constraint': 'FR',
    'interaction_recency_constraint': 'IR',
    'items_preference_constraint': 'IP',
    'items_timeless_constraint': 'IT',
    'items_pagerank_constraint': 'PR'
}

user_policy_order = ['ZN', 'FR', 'IR']
item_policy_order = ['IP', 'IT', 'PR']

datasets = [
    "rent_the_runway",
    "foursquare_nyc",
    "lastfm-1m",
    "foursquare_tky",
    "ml-1m",
]

models = [
    "AutoCF",
    "NCL",
    "SGL",
    "XSimGCL",
    "NGCF",
]

dataset_map = {
        'rent_the_runway': 'RENT',
        'lastfm-1m': 'LF1M',
        'foursquare_nyc': 'FNYC',
        'foursquare_tky': 'FTKY',
        'ml-1m': 'ML1M'
    }


def update_plt_rc():
    SMALL_SIZE = 16
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 26

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)


current_path = os.path.dirname(os.path.realpath(__file__))
plots_path = os.path.join(current_path, 'policy_comparison_plots')
pol_compare_path = os.path.join(current_path, os.pardir, 'policy_overlap_comparison')

if not os.path.exists(plots_path):
    os.makedirs(plots_path, exist_ok=True)


if __name__ == "__main__":
    pol_compare_folders = os.listdir(pol_compare_path)

    update_plt_rc()
    vmin, vmax = 0, 1
    norm = mpl.colors.Normalize(vmin, vmax)
    cmap = sns.color_palette("cividis", as_cmap=True)
    for j, (dset, mod) in enumerate(zip(datasets, models)):
        for pol_type in ['user', 'item']:
            foldername = f"{dset.lower()}_{mod.lower()}"
            df = pd.read_csv(
                os.path.join(pol_compare_path, foldername, f"{pol_type}_policies_jaccard_similarity.csv"),
                skiprows=1
            )
            df["Policy 1"] = df["Policy 1"].map(policies_map)
            df["Policy 2"] = df["Policy 2"].map(policies_map)

            policy_order = user_policy_order if pol_type == "user" else item_policy_order

            df = df.pivot(
                columns="Policy 1", index="Policy 2", values="Jaccard Similarity"
            ).reindex(
                policy_order, axis=1
            ).reindex(
                policy_order, axis=0
            )

            fig, ax = plt.subplots(1, 1)
            sns.heatmap(
                df, cbar=False, norm=norm, cmap=cmap,
                linewidths=.5, linecolor='k', annot=True, fmt=".2f", square=True, ax=ax
            )
            ax.set_xlabel("")
            if dset == "foursquare_tky":
                ax.set_ylabel(pol_type.upper())
            else:
                ax.set_ylabel("")
            if pol_type == "user":
                ax.set_title(f"{dataset_map[dset]} ({mod})")
            else:
                ax.set_title("")
            ax.tick_params("both", length=0)
            fig.savefig(
                os.path.join(plots_path, f"{pol_type}_{foldername}.png"), dpi=300, bbox_inches="tight", pad_inches=0
            )
            plt.close(fig)

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_fig, cbar_ax = plt.subplots(1, 1, figsize=(0.4, 10.8))
    colorbar = cbar_fig.colorbar(mappable, cax=cbar_ax, orientation="vertical")
    cbar_fig.savefig(os.path.join(plots_path, "colorbar.png"), dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(cbar_fig)
