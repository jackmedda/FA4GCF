import torch
from gnnuers.explainer import DPBG


class FA4GCF(gnnuers_DPGB):

    def get_batch_cf_stats(self, adj_sub_cf_adj, loss_total, loss_graph_dist, exp_loss, epoch):
        # Compute distance between original and perturbed list. Explanation maintained only if dist > 0
        # cf_dist = [self.dist(_pred, _topk_idx) for _pred, _topk_idx in zip(cf_topk_pred_idx, self.model_topk_idx)]
        cf_dist = None

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
