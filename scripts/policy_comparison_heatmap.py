import os

import pandas as pd
import seaborn as sns
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
    # "rent_the_runway",
    "foursquare_nyc",
    "lastfm-1k",
    "foursquare_tky",
    "ml-1m",
]

models = [
    # "rent_the_runway",
    "XSimGCL",
    "SGL",
    "UltraGCN",
    "SVD_GCN",
]


current_path = os.path.dirname(os.path.realpath(__file__))
plots_path = os.path.join(current_path, 'policy_comparison_plots')
pol_compare_path = os.path.join(current_path, os.pardir, 'policy_overlap_comparison')

if not os.path.exists(plots_path):
    os.makedirs(plots_path, exist_ok=True)


if __name__ == "__main__":
    pol_compare_folders = os.listdir(pol_compare_path)
    for dset, mod in zip(datasets, models):
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
            plot = sns.heatmap(df, ax=ax)
            fig.savefig(os.path.join(plots_path, foldername + ".png"), dpi=200)
            plt.close(fig)
