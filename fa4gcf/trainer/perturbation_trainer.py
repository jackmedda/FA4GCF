import torch
from gnnuers.explainers import DPBG

from fa4gcf.model import PygPerturbedModel


class PerturbationTrainer(DPBG):

    def initialize_cf_model(self, **kwargs):
        # Freeze weights from original model in cf_model
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # Instantiate CF model class, load weights from original model
        self.cf_model = PygPerturbedModel(self.config, self.model, **kwargs).to(self.model.device)
        # self.parallel_cf_model = torch_parallel.DistributedDataParallel(self.cf_model)
        # self.parallel_cf_model = torch_parallel.DataParallel(self.cf_model)
        # for attr in ['device', 'full_sort_predict']:
        #     setattr(self.parallel_cf_model, attr, getattr(self.cf_model, attr))
        #
        # self.cf_model = self.parallel_cf_model

    def get_batch_cf_stats(self, adj_sub_cf_adj, loss_total, loss_graph_dist, exp_loss, epoch):
        adj_pert_edges = adj_sub_cf_adj.detach().cpu()
        if self.cf_model.inner_pyg_model.edge_weight is None:
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
