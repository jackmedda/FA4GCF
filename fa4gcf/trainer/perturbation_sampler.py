import copy

import igraph
import torch
import numpy as np
import pandas as pd

import fa4gcf.data.utils as data_utils
import fa4gcf.evaluation.graph_evaluation as graph_eval


class PerturbationSampler:

    DEFAULT_RATIOS = {
        # User Policies
        'USERS_ZERO': 0,
        'USERS_LOW_DEGREE': 0.35,
        'USERS_FURTHEST': 0.35,
        'USERS_SPARSE': 0.35,
        'USERS_INTERACTION_RECENCY': 0.35,
        # Item Policies
        'ITEMS_PREFERENCE': 0.2,
        'ITEMS_NICHE': 0.2,
        'ITEMS_TIMELESS': 0.2,
        'ITEMS_PAGERANK': 0.2
    }

    POLICIES_OFF_RATIOS = {
        'USERS_ZERO': None,
        'USERS_LOW_DEGREE': 0,
        'USERS_FURTHEST': 0,
        'USERS_SPARSE': 0,
        'USERS_INTERACTION_RECENCY': 0,
        'ITEMS_PREFERENCE': 0,
        'ITEMS_NICHE': 0,
        'ITEMS_TIMELESS': 0,
        'ITEMS_PAGERANK': 0
    }

    def __init__(self,
                 dataset,
                 perturbation_trainer,
                 config):

        self.dataset = dataset
        self.perturbation_trainer = perturbation_trainer
        policies = config['perturbation_policies']

        # User-/item policies
        self.policies_ratios = self._get_policies_ratios(config)

        # Other User-sampling Policies
        self.group_deletion = policies['group_deletion_constraint']

        # Other sampling Policies (it can be chained with other policies)
        self.random_sampling_size = config['random_sampling_policy_data']  # (user_size: float, item_size: float)

    def _get_policies_ratios(self, config):
        policies = config['perturbation_policies']
        policies_ratios = copy.copy(self.POLICIES_OFF_RATIOS)

        for policy in policies_ratios:
            constraint_key = f'{policy.lower()}_constraint'
            ratio_key = f'{policy.lower()}_constraint'

            if policies[constraint_key]:
                policies_ratios[policy] = (config[ratio_key] or self.DEFAULT_RATIOS[policy])

        return policies_ratios

    def _apply_group_deletion_policy(self, users_list):
        if self.group_deletion:
            user_feat = self.dataset.user_feat
            adv_group = self.perturbation_trainer.adv_group
            sensitive_attribute = self.perturbation_trainer.sensitive_attribute

            advantaged_mask = user_feat[sensitive_attribute][users_list] == adv_group
            users_list = users_list[advantaged_mask]

        return users_list

    def _apply_users_zero_policy(self, users_list, pref_data):
        policy_key = 'USERS_ZERO'
        policy_ratio = self.policies_ratios[policy_key]
        if policy_ratio != self.POLICIES_OFF_RATIOS[policy_key]:
            eval_metric = self.perturbation_trainer.eval_metric

            zero_th_users = torch.from_numpy(
                pref_data.loc[(pref_data[eval_metric] <= policy_ratio), 'user_id'].to_numpy()
            )

            users_with_zero_th_list, counts = torch.cat((users_list, zero_th_users)).unique(return_counts=True)
            users_list = users_with_zero_th_list[counts > 1]

        return users_list

    def _apply_users_low_degree_policy(self, users_list):
        policy_key = 'USERS_LOW_DEGREE'
        policy_ratio = self.policies_ratios[policy_key]
        if policy_ratio != self.POLICIES_OFF_RATIOS[policy_key]:
            _, _, history_len = self.dataset.history_item_matrix()
            history_len = history_len[users_list]

            n_low_degree_users = int(policy_ratio * history_len.shape[0])
            lowest_degree = torch.argsort(history_len)[:n_low_degree_users]
            users_list = users_list[lowest_degree]

        return users_list

    def _apply_users_furthest_policy(self, users_list, full_users_list):
        policy_key = 'USERS_FURTHEST'
        policy_ratio = self.policies_ratios[policy_key]
        if policy_ratio != self.POLICIES_OFF_RATIOS[policy_key]:
            user_feat = self.dataset.user_feat
            disadv_group = self.perturbation_trainer.disadv_group
            sensitive_attribute = self.perturbation_trainer.sensitive_attribute

            disadvantaged_mask = user_feat[sensitive_attribute][full_users_list] == disadv_group
            disadvantaged_users = full_users_list[disadvantaged_mask].numpy()

            # due to the removal of user/item padding in the igraph graph,
            # the ids are first shifted back by 1 and then forward by 1 for the argsort on the distances
            igg = data_utils.get_bipartite_igraph(self.dataset, remove_first_row_col=True)
            mean_dist = np.array(igg.distances(source=users_list - 1, target=disadvantaged_users - 1)).mean(axis=1)
            furthest_users = np.argsort(mean_dist)

            n_furthest = int(policy_ratio * furthest_users.shape[0])
            users_list = users_list[furthest_users[-n_furthest:]]

        return users_list

    def _apply_users_sparse_policy(self, users_list):
        """ Sparse users are connected to low-degree items """
        policy_key = 'USERS_SPARSE'
        policy_ratio = self.policies_ratios[policy_key]
        if policy_ratio != self.POLICIES_OFF_RATIOS[policy_key]:
            sparsity_df = graph_eval.extract_graph_metrics_per_node(
                self.dataset, remove_first_row_col=True, metrics=["Sparsity"]
            )

            users_sparsity = sparsity_df.set_index('Node').loc[users_list.numpy(), 'Sparsity']
            users_sparsity = torch.from_numpy(users_sparsity.to_numpy())

            n_most_sparse_users = int(policy_ratio * users_sparsity.shape[0])
            most_sparse = torch.argsort(users_sparsity)[-n_most_sparse_users:]
            users_list = users_list[most_sparse]

        return users_list

    def _apply_users_interaction_recency_policy(self, users_list):
        policy_key = 'USERS_INTERACTION_RECENCY'
        policy_ratio = self.policies_ratios[policy_key]
        if policy_ratio != self.POLICIES_OFF_RATIOS[policy_key]:
            uid_field, time_field = self.dataset.uid_field, self.dataset.time_field

            users_list_feat_mask = torch.isin(self.dataset.inter_feat[uid_field], users_list)
            users_list_feat = self.dataset.inter_feat[users_list_feat_mask]  # makes a copy

            df = pd.DataFrame(users_list_feat.numpy())
            latest_inter_df = df.groupby(uid_field).max().reset_index().sort_values(time_field, ascending=False)
            users_interaction_recency = torch.from_numpy(latest_inter_df[uid_field].to_numpy())

            n_most_recent_users = int(policy_ratio * users_list.shape[0])
            most_recent = users_interaction_recency[:n_most_recent_users]
            users_list = most_recent

        return users_list

    def _apply_items_preference_policy(self, items_list):
        """
        advantaged_preference_ratio > 1 means the advantaged group prefers those items w.r.t. to their representation.
        self.perturbation_trainer.adv_group always stores the group to be perturbed. So, if the config param
        `perturb_adv_group` is False, then adv_group will actually be the disadvantaged group
        """
        policy_key = 'ITEMS_PREFERENCE'
        policy_ratio = self.policies_ratios[policy_key]
        if policy_ratio != self.POLICIES_OFF_RATIOS[policy_key]:
            item_history, _, item_history_len = self.dataset.history_user_matrix()
            adv_group = self.perturbation_trainer.adv_group
            sensitive_attribute = self.perturbation_trainer.sensitive_attribute

            sensitive_attribute_map = self.dataset.user_feat[sensitive_attribute]
            n_advantaged_users = (sensitive_attribute_map == adv_group).sum()
            advantaged_ratio = n_advantaged_users / (sensitive_attribute_map.shape[0] - 1)

            sensitive_item_history = sensitive_attribute_map[item_history]
            advantaged_item_history_ratio = (sensitive_item_history == adv_group).sum(dim=1) / item_history_len
            advantaged_item_history_ratio = torch.nan_to_num(advantaged_item_history_ratio, nan=0)

            advantaged_preference_ratio = advantaged_item_history_ratio / advantaged_ratio

            n_advantaged_most_preferred_items = int(policy_ratio * advantaged_preference_ratio.shape[0])
            most_preferred = torch.argsort(advantaged_preference_ratio)[-n_advantaged_most_preferred_items:]

            items_with_most_preferred_list, counts = torch.cat((items_list, most_preferred)).unique(return_counts=True)
            items_list = items_with_most_preferred_list[counts > 1]

        return items_list

    def _apply_items_niche_policy(self, items_list):
        policy_key = 'ITEMS_NICHE'
        policy_ratio = self.policies_ratios[policy_key]
        if policy_ratio != self.POLICIES_OFF_RATIOS[policy_key]:
            _, _, item_history_len = self.dataset.history_user_matrix()
            items_list_history_len = item_history_len[items_list]

            most_niche = int(policy_ratio * items_list_history_len.shape[0])
            items_list = items_list[torch.argsort(items_list_history_len)[:most_niche]]

        return items_list

    def _apply_items_timeless_policy(self, items_list):
        policy_key = 'ITEMS_TIMELESS'
        policy_ratio = self.policies_ratios[policy_key]
        if policy_ratio != self.POLICIES_OFF_RATIOS[policy_key]:
            iid_field, time_field = self.dataset.iid_field, self.dataset.time_field

            items_list_feat_mask = torch.isin(self.dataset.inter_feat[iid_field], items_list)
            items_list_feat = self.dataset.inter_feat[items_list_feat_mask]  # makes a copy

            df = pd.DataFrame(items_list_feat.numpy())
            df_by_item = df.groupby(iid_field)
            latest_inter_df = df_by_item[time_field].max()
            oldest_inter_df = df_by_item[time_field].min()

            items_timeless = (latest_inter_df - oldest_inter_df).sort_values(ascending=False).index
            items_timeless = torch.from_numpy(items_timeless.to_numpy())

            n_most_timeless_items = int(policy_ratio * items_timeless.shape[0])
            most_timeless = items_timeless[:n_most_timeless_items]
            items_list = most_timeless

        return items_list

    def _apply_items_pagerank_policy(self, items_list):
        policy_key = 'ITEMS_PAGERANK'
        policy_ratio = self.policies_ratios[policy_key]
        if policy_ratio != self.POLICIES_OFF_RATIOS[policy_key]:
            igg: igraph.Graph = data_utils.get_bipartite_igraph(self.dataset, remove_first_row_col=True)

            # due to the removal of user/item padding in the igraph graph,
            # the ids are first shifted back by 1 and then forward by 1 for the argsort on the distances
            items_pagerank = torch.Tensor(igg.pagerank(items_list - 1, directed=False))
            highest_pagerank = torch.argsort(items_pagerank)

            n_highest_pagerank = int(policy_ratio * items_pagerank.shape[0])
            items_list = items_list[highest_pagerank[-n_highest_pagerank:]]

        return items_list

    def _apply_random_policy(self, users_list, items_list):
        """Randomly samples a subset of users or items"""
        if self.random_sampling_size is not None:
            user_size, item_size = self.random_sampling_size
            user_random_kwargs, item_random_kwargs = {"size": user_size or 0}, {"size": item_size or 0}

            if user_random_kwargs["size"] > 0:
                users_list = np.random.choice(users_list, **user_random_kwargs)
            if item_random_kwargs["size"] > 0:
                items_list = np.random.choice(items_list, **item_random_kwargs)

        return users_list, items_list

    def apply_policies(self, batched_data, pref_data):
        """ No modification should be applied internally by each policy. batched_data should never be modified. """
        sampled_users = batched_data
        sampled_items = torch.arange(1, self.dataset.item_num)

        sampled_users = self._apply_group_deletion_policy(sampled_users)
        sampled_users = self._apply_users_zero_policy(sampled_users, pref_data)
        sampled_users = self._apply_users_low_degree_policy(sampled_users)
        sampled_users = self._apply_users_furthest_policy(sampled_users, batched_data)
        sampled_users = self._apply_users_sparse_policy(sampled_users)
        sampled_users = self._apply_users_interaction_recency_policy(sampled_users)

        sampled_items = self._apply_items_preference_policy(sampled_items)
        sampled_items = self._apply_items_niche_policy(sampled_items)
        sampled_items = self._apply_items_timeless_policy(sampled_items)
        sampled_items = self._apply_items_pagerank_policy(sampled_items)

        sampled_users, sampled_items = self._apply_random_policy(sampled_users, sampled_items)

        return sampled_users, sampled_items
