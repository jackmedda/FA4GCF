import time

import torch
import torch.cuda.amp as amp
from torch_geometric.typing import SparseTensor
from gnnuers.explainers import DPBG

from fa4gcf.model import PygPerturbedModel
from fa4gcf.trainer import PerturbationSampler


class PerturbationTrainer(DPBG):

    def __init__(self, config, dataset, rec_data, model, dist="damerau_levenshtein", **kwargs):
        super(PerturbationTrainer, self).__init__(
            config, dataset, rec_data, model, dist=dist, **kwargs
        )

        self.enable_amp = config["enable_amp"]

        self.interaction_recency_constraint = config['explainer_policies']['interaction_recency_constraint']
        self.interaction_recency_constraint_ratio = config['interaction_recency_constraint_ratio'] or 0.35
        self.items_timeless_constraint = config['explainer_policies']['items_timeless_constraint']
        self.items_timeless_constraint_ratio = config['items_timeless_constraint_ratio'] or 0.2
        self.items_pagerank_constraint = config['explainer_policies']['items_pagerank_constraint']
        self.items_pagerank_constraint_ratio = config['items_pagerank_constraint_ratio'] or 0.2

        self.random_sampling_policy_data = config['random_sampling_policy_data']
        self.pert_sampler = PerturbationSampler(
            self.dataset,
            self,
            group_deletion=self.group_deletion_constraint,
            users_zero_th=self.users_zero_constraint_value if self.users_zero_constraint else None,
            users_sparse_ratio=self.sparse_users_constraint_ratio if self.sparse_users_constraint else 0,
            users_furthest_ratio=self.users_furthest_constraint_ratio if self.users_furthest_constraint else 0,
            users_low_degree_ratio=self.users_low_degree_ratio if self.users_low_degree else 0,
            items_preference_ratio=self.items_preference_constraint_ratio if self.items_preference_constraint else 0,
            items_niche_ratio=self.niche_items_constraint_ratio if self.niche_items_constraint else 0,
            users_interaction_recency_ratio=self.interaction_recency_constraint_ratio if self.interaction_recency_constraint else 0,
            items_timeless_ratio=self.items_timeless_constraint_ratio if self.items_timeless_constraint else 0,
            items_pagerank_ratio=self.items_pagerank_constraint_ratio if self.items_pagerank_constraint else 0,
            random_sampling_size=self.random_sampling_policy_data
        )

    def _check_policies(self, batched_data, rec_model_topk):
        test_model_topk, test_scores_args, rec_scores_args = [None] * 3
        if self.increase_disparity:
            batched_data, test_model_topk, test_scores_args,\
                rec_model_topk, rec_scores_args = self.increase_dataset_unfairness(
                    batched_data,
                    test_data,
                    rec_model_topk,
                    topk=topk
                )

        self.determine_adv_group(batched_data.detach().numpy(), rec_model_topk)
        pref_data = self._pref_data_sens_and_metric(batched_data.detach().numpy(), rec_model_topk)
        filtered_users, filtered_items = self.pert_sampler.apply_policies(batched_data, pref_data)

        return batched_data, filtered_users, filtered_items, (test_model_topk, test_scores_args, rec_model_topk, rec_scores_args)

    def initialize_cf_model(self, **kwargs):
        kwargs["random_perturbation"] = self.random_perturbation

        # Instantiate CF model class, load weights from original model
        self.cf_model = PygPerturbedModel(self.config, self.dataset, self.model, **kwargs).to(self.model.device)
        # self.parallel_cf_model = torch_parallel.DistributedDataParallel(self.cf_model)
        # self.parallel_cf_model = torch_parallel.DataParallel(self.cf_model)
        # for attr in ['device', 'full_sort_predict']:
        #     setattr(self.parallel_cf_model, attr, getattr(self.cf_model, attr))
        #
        # self.cf_model = self.parallel_cf_model
        self.logger.info(self.cf_model)

        self.initialize_optimizer()

    def initialize_optimizer(self):
        lr = self.config['cf_learning_rate']
        momentum = self.config["momentum"] or 0.0
        sgd_kwargs = {'momentum': momentum, 'nesterov': True if momentum > 0 else False}
        if self.config["cf_optimizer"] == "SGD":
            self.cf_optimizer = torch.optim.SGD(self.cf_model.parameters(), lr=lr, **sgd_kwargs)
        elif self.config["cf_optimizer"] == "Adadelta":
            self.cf_optimizer = torch.optim.Adadelta(self.cf_model.parameters(), lr=lr)
        elif self.config["cf_optimizer"] == "Adagrad":
            self.cf_optimizer = torch.optim.Adagrad(self.cf_model.parameters(), lr=lr)
        elif self.config["cf_optimizer"] == "AdamW":
            self.cf_optimizer = torch.optim.AdamW(self.cf_model.parameters(), lr=lr)
        elif self.config["cf_optimizer"] == "Adam":
            self.cf_optimizer = torch.optim.Adam(self.cf_model.parameters(), lr=lr)
        elif self.config["cf_optimizer"] == "RMSprop":
            self.cf_optimizer = torch.optim.RMSprop(self.cf_model.parameters(), lr=lr)
        else:
            raise NotImplementedError("CF Optimizer not implemented")

    def train(self, epoch, scores_args, users_ids, **kwargs):
        """
        Training procedure of explanation
        :param epoch:
        :param topk:
        :return:
        """
        train_start = time.time()
        torch.cuda.empty_cache()

        # Only the 500 itmes with the highest predicted relevance will be used to measure the approx NDCG
        # This prevents the usage of a tremendous amount of memory, due to the pairwise preference function
        MEMORY_PERFORMANCE_MAX_LOSS_TOPK_ITEMS = 500
        if self._exp_loss.ranking_loss_function.__MAX_TOPK_ITEMS__ != MEMORY_PERFORMANCE_MAX_LOSS_TOPK_ITEMS:
            self._exp_loss.ranking_loss_function.__MAX_TOPK_ITEMS__ = MEMORY_PERFORMANCE_MAX_LOSS_TOPK_ITEMS

        if self.mini_batch_descent:
            self.cf_optimizer.zero_grad()
        self.cf_model.train()

        user_feat = self.get_batch_user_feat(users_ids)
        target = self.get_target(user_feat)

        if self._exp_loss.is_data_feat_needed():
            data_feat = self.get_loss_data_feat(users_ids, user_feat)
            self._exp_loss.update_data_feat(data_feat)

        with amp.autocast(enabled=self.enable_amp):
            loss_total, orig_loss_graph_dist, loss_graph_dist, exp_loss, adj_sub_cf_adj = self.cf_model.loss(
                scores_args,
                self._exp_loss,
                target
            )

        torch.cuda.empty_cache()
        # import torchviz
        # dot = torchviz.make_dot(loss_total, params=dict(self.cf_model.named_parameters()))
        # dot.graph_attr.update(size='600,600')
        # dot.render("comp_graph", format="png")
        loss_total.backward()

        # for name, param in self.cf_model.named_parameters():
        #     print(param.grad)
        # import pdb; pdb.set_trace()

        exp_loss = exp_loss.mean().item() if exp_loss is not None else torch.nan
        self.log_epoch(train_start, epoch, loss_total, exp_loss, loss_graph_dist, orig_loss_graph_dist)

        cf_stats = None
        if orig_loss_graph_dist.item() > 0:
            cf_stats = self.get_batch_cf_stats(
                adj_sub_cf_adj, loss_total, loss_graph_dist, exp_loss, epoch
            )

        return cf_stats, loss_total.item(), exp_loss

    def get_target(self, user_feat):
        target_shape = (user_feat[self.dataset.uid_field].shape[0], self.dataset.item_num)
        target = torch.zeros(target_shape, dtype=torch.float, device=self.cf_model.device)

        if not self._pred_as_rec:
            hist_matrix, _, _ = self.rec_data.dataset.history_item_matrix()
            rec_data_interactions = hist_matrix[user_feat[self.dataset.uid_field]]
        else:
            rec_data_interactions = self._test_history_matrix[user_feat[self.dataset.uid_field]]
        target[torch.arange(target.shape[0])[:, None], rec_data_interactions] = 1
        target[:, 0] = 0  # item 0 is a padding item

        return target

    def get_batch_cf_stats(self, adj_sub_cf_adj, loss_total, loss_graph_dist, exp_loss, epoch):
        adj_pert_edges = adj_sub_cf_adj.detach().cpu()
        if isinstance(adj_pert_edges, SparseTensor):
            row, col, vals = adj_pert_edges.coo()
            pert_edges = torch.stack((row, col), dim=0)[:, vals.nonzero().squeeze()]
        else:
            pert_edges = adj_pert_edges.indices()[:, adj_pert_edges.values().nonzero().squeeze()]

        # remove duplicated edges
        pert_edges = pert_edges[:, (pert_edges[0, :] < self.dataset.user_num) & (pert_edges[0, :] > 0)].numpy()

        cf_stats = [loss_total.item(), loss_graph_dist.item(), exp_loss, None, pert_edges, epoch + 1]

        if self.neighborhood_perturbation:
            self.cf_model.update_neighborhood(torch.Tensor(pert_edges))

        return cf_stats
