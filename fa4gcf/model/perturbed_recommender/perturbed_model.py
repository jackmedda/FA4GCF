import torch
import numpy as np
from torch_geometric.typing import SparseTensor
from recbole.model.abstract_recommender import GeneralRecommender as RecboleModel  # to inherit model attributes

import fa4gcf.data.utils as data_utils
import fa4gcf.evaluation as eval_utils
from fa4gcf.model.perturbed_recommender.customized_perturbation import PerturbationApplier
from fa4gcf.model.general_recommender import (
    LightGCL,
    SVD_GCN
)


class PygPerturbedModel(RecboleModel):
    def __init__(self,
                 config,
                 dataset,
                 model,
                 filtered_users=None,
                 filtered_items=None,
                 random_perturbation=False):
        super(PygPerturbedModel, self).__init__(config, dataset)

        self.device = model.device
        self.inner_pyg_model = model
        self.num_all = model.n_users + model.n_items

        # Freeze weights of inner model
        for name, param in self.inner_pyg_model.named_parameters():
            param.requires_grad = False

        self.beta = config['cf_beta']

        if isinstance(self.inner_pyg_model, LightGCL):
            raise NotImplementedError("Current implementation of LightGCL cannot be perturbed.")

        if hasattr(self.inner_pyg_model, "edge_weight") and hasattr(self.inner_pyg_model, "edge_index"):
            edge_index, edge_weight = self.inner_pyg_model.edge_index, self.inner_pyg_model.edge_weight
        else:  # necessary for SVD_GCN that deletes edge_index and edge_weight to reduce memory usage
            edge_index, edge_weight = dataset.get_norm_adj_mat(enable_sparse=config["enable_sparse"])

        if edge_weight is None:
            self.adj = edge_index.fill_value(1.).to(self.inner_pyg_model.device)
        else:
            self.adj = torch.sparse_coo_tensor(
                edge_index.clone(),
                torch.ones_like(edge_weight),
                (self.num_all, self.num_all),
                device=self.inner_pyg_model.device
            )

        gcn_norm_kwargs = {}
        if isinstance(self.inner_pyg_model, SVD_GCN):
            gcn_norm_kwargs['alpha'] = self.inner_pyg_model.alpha
            if not hasattr(self.inner_pyg_model, "q"):
                self.inner_pyg_model.q = self.inner_pyg_model._update_q(config)

        self.graph_perturbation_layer = GraphPerturbation(
            config,
            dataset,
            edge_index,
            edge_weight,
            filtered_users=filtered_users,
            filtered_items=filtered_items,
            random_perturbation=random_perturbation,
            add_self_loops=dataset.add_self_loops,
            gcn_norm_kwargs=gcn_norm_kwargs
        )
        self.perturbation_applier = PerturbationApplier()

    def cf_state_dict(self):
        return self.graph_perturbation_layer.cf_state_dict()

    def load_cf_state_dict(self, ckpt):
        self.graph_perturbation_layer.load_cf_state_dict(ckpt)

    def loss(self, scores_args, main_loss_f, main_loss_target):
        """

        :param scores_args: arguments to compute cf_model scores
        :param main_loss_f: fair loss function
        :param main_loss_target: fair loss target

        :return:
        """
        # generate discrete P_loss (non-differentiable adj matrix) to compute the graph dist loss
        self.eval()
        with torch.no_grad():
            cf_adj = self.graph_perturbation_layer(pred=True, return_only_P_loss=True)

        self.train()
        # Need to change this otherwise loss_graph_dist has no gradient
        if isinstance(cf_adj, SparseTensor):
            cf_adj.requires_grad_(True)
        else:
            cf_adj.requires_grad = True

        cf_scores = eval_utils.get_scores(self, *scores_args, pred=False)
        cf_scores = torch.nan_to_num(  # remove neginf from output
            cf_scores, neginf=(torch.min(cf_scores[~torch.isinf(cf_scores)]) - 1).item()
        )

        ########### Main Loss ###########
        fair_loss = main_loss_f(cf_scores, main_loss_target)

        ########### Dist Loss ###########
        if isinstance(cf_adj, SparseTensor):
            orig_dist = cf_adj.add(self.adj.mul_nnz(torch.tensor(-1), layout='coo'))  # cf_adj - self.adj
        else:
            orig_dist = (cf_adj - self.adj)
        if not orig_dist.is_coalesced():
            orig_dist = orig_dist.coalesce()
        if isinstance(cf_adj, SparseTensor):
            _, _, vals = orig_dist.coo()
        else:
            vals = orig_dist.values()

        # compute normalized graph dist loss (logistic sigmoid is not used because reaches too fast 1)
        orig_loss_graph_dist = torch.sum(vals.abs()) / 2  # Number of edges changed (symmetrical)
        loss_graph_dist = orig_loss_graph_dist / (1 + abs(orig_loss_graph_dist))  # sigmoid dist

        loss_total = fair_loss + self.beta * loss_graph_dist.to(fair_loss.device)

        return loss_total, orig_loss_graph_dist, loss_graph_dist, fair_loss, orig_dist

    def forward(self, pred):
        """
        Perturbs the adjacency matrix in a differentiable way. Then, it re-creates the normalized adjacency matrix,
        and updates it for trained GNN model.

        :param pred: if True, the perturbation is discrete, i.e., \hat{p} \in \{0, 1\}. Used for inference.
        :return:
        """
        pert_edge_index, pert_edge_weight = self.graph_perturbation_layer(pred=pred)
        self.perturbation_applier.apply(self.inner_pyg_model, pert_edge_index, pert_edge_weight)

    def predict(self, interaction, pred=False):
        self.forward(pred=pred)
        return self.inner_pyg_model.predict(interaction)

    def full_sort_predict(self, interaction, pred=False):
        self.forward(pred=pred)
        return self.inner_pyg_model.full_sort_predict(interaction)


class GraphPerturbation(torch.nn.Module):

    def __init__(self,
                 config,
                 dataset,
                 edge_index,
                 edge_weight,
                 filtered_users=None,
                 filtered_items=None,
                 random_perturbation=False,
                 add_self_loops=False,
                 gcn_norm_kwargs=None):
        super(GraphPerturbation, self).__init__()

        self.edge_additions = config['edge_additions']
        self.random_perturb_p = (config['random_perturbation_p'] or 0.05) if random_perturbation else None
        self.random_perturb_rs = np.random.RandomState(config['seed'] or 0)
        self.initialization = config['perturbation_initialization']
        self.add_self_loops = add_self_loops
        self.gcn_norm_kwargs = gcn_norm_kwargs or {}

        self.num_all = dataset.user_num + dataset.item_num

        if edge_weight is None:
            self.edge_index = edge_index.fill_value_(1.).to('cpu')
            self.edge_weight = edge_weight
        else:
            self.edge_index = edge_index.to('cpu')
            self.edge_weight = torch.ones_like(edge_weight, device='cpu')

        self.mask_sub_adj = None
        self.mask_filter = None
        self.mask_neighborhood = None
        self.build_edge_masks(dataset, filtered_users, filtered_items)

        self.P_symm = None
        self._initialize_P_symm()

        self.force_removed_edges = None
        if not self.edge_additions:
            if config['explainer_policies']['force_removed_edges']:
                self.force_removed_edges = torch.ones(self.P_symm_size, dtype=torch.float).to('cpu')
            if config['explainer_policies']['neighborhood_perturbation']:
                self.mask_neighborhood = self.mask_filter.clone().detach()

    def _initialize_P_symm(self):
        if self.edge_additions:
            if self.initialization != 'random':
                self.P_symm_init = -5  # to get sigmoid closer to 0
                self.P_symm_func = "zeros"
            else:
                self.P_symm_init = -6
                self.P_symm_func = "rand"
            self.P_symm_size = self.mask_sub_adj.shape[1]
        else:
            if self.initialization != 'random':
                self.P_symm_init = 1
                self.P_symm_func = "ones"
            else:
                self.P_symm_init = 2
                self.P_symm_func = "rand"
            self.P_symm_size = self.mask_filter.nonzero().shape[0] // 2

        temp_param = getattr(torch, self.P_symm_func)(self.P_symm_size, dtype=torch.float)
        self.P_symm = torch.nn.Parameter(temp_param + self.P_symm_init)

    def reset_param(self):
        with torch.no_grad():
            self._parameters['P_symm'].copy_(
                torch.FloatTensor(getattr(torch, self.P_symm_func)(self.P_symm_size)) + self.P_symm_init
            )
        if self.force_removed_edges is not None:
            self.force_removed_edges = torch.FloatTensor(torch.ones(self.P_symm_size)).to('cpu')
        if self.mask_neighborhood is not None:
            self.mask_neighborhood = self.mask_filter.clone().detach()

    def cf_state_dict(self):
        return {
            **self.state_dict(),
            'mask_sub_adj': self.mask_sub_adj if self.edge_additions else None,
            'mask_filter': self.mask_filter.detach() if self.mask_filter is not None else None,
            'force_removed_edges': self.force_removed_edges.detach() if self.force_removed_edges is not None else None,
            'mask_neighborhood': self.mask_neighborhood.detach() if self.mask_neighborhood is not None else None
        }

    def load_cf_state_dict(self, ckpt):
        state_dict = self.state_dict()
        state_dict.update({'P_symm': ckpt['P_symm']})
        self.mask_sub_adj = ckpt['mask_sub_adj']
        self.mask_filter = ckpt['mask_filter']
        self.force_removed_edges = ckpt['force_removed_edges']
        self.mask_neighborhood = ckpt['mask_neighborhood']

    def build_edge_masks(self, dataset, filtered_users=None, filtered_items=None):
        """
        the perturbation vector is always P_symm, no matter if edges should be added or deleted
        edge_additions = True:
            we need to store also the indices to add, such that they are concatenated to existing edge_index
        edge_additions = False:
            we just need a boolean mask, filter, that swaps the values inside edge_weight with the perturbed ones

        :param dataset:
        :param filtered_users: subset of users that will receive the perturbation
        :param filtered_items: subset of items that will receive the perturbation
        :return:
        """
        n_users = dataset.user_num
        if self.edge_additions:
            # OTTIMIZZARE => BISOGNA SEMPLICEMENTE PRENDERE GLI EDGE DOVE LA MATRICE DI ADIACENZA È ZERO,
            # CORRISPONDE AL PRODOTTO CARTESIANO FRA USER E ITEM SENZA GLI EDGE GIÀ NELLA MATRICE. SAREBBE MEGLIO?
            # tipo product(torch.arange(1, user_num), torch.arange(1, item_num) + user_num) - self.edge_index
            # sulla base di filtered users and items si può preventivamente generare ridotta la mask_sub_adj
            self.mask_sub_adj = np.stack((dataset.inter_matrix(form="coo") == 0).nonzero())

            self.mask_sub_adj = self.mask_sub_adj[:, (self.mask_sub_adj[0] != 0) & (self.mask_sub_adj[1] != 0)]
            self.mask_sub_adj[1] += n_users
            self.mask_sub_adj = torch.tensor(self.mask_sub_adj, dtype=torch.int, device='cpu')

            if filtered_users is not None:
                filtered_users = filtered_users.to(self.mask_sub_adj.device)
                filtered_nodes_mask = data_utils.edges_filter_nodes(self.mask_sub_adj[[0]], filtered_users)
                if filtered_items is not None:
                    filtered_items = filtered_items.to(self.mask_sub_adj.device)
                    filtered_nodes_mask &= data_utils.edges_filter_nodes(
                        self.mask_sub_adj[[1]], filtered_items + n_users
                    )

                self.mask_sub_adj = self.mask_sub_adj[:, filtered_nodes_mask]
        else:
            # POICHÉ MASK_SUB_ADJ COINCIDE ESATTAMENTE CON EDGE_INDEX, MASK_SUB_ADJ È SUPERFLUO
            # SERVE SOLO MASK_FILTER (booleano) PER CAPIRE QUALI EDGE VENGONO EFFETTIVAMENTE PERTURBATI
            # if isinstance(self.edge_index, SparseTensor):
            #     row, col, _ = self.edge_index.coo()
            #     self.mask_sub_adj = torch.stack((row, col), dim=0).to('cpu')
            # else:
            #     self.mask_sub_adj: torch.Tensor = self.edge_index.to('cpu')
            if isinstance(self.edge_index, SparseTensor):
                row, col, _ = self.edge_index.coo()
                mask_sub_adj = torch.stack((row, col), dim=0).to('cpu')
            else:
                mask_sub_adj: torch.Tensor = self.edge_index.to('cpu')

            self.mask_filter = torch.ones(mask_sub_adj.size(1), dtype=torch.bool, device='cpu')

            if filtered_users is not None and self.random_perturb_p is None:
                filtered_users = filtered_users.to(self.mask_filter.device)
                self.mask_filter &= data_utils.edges_filter_nodes(mask_sub_adj, filtered_users)

            if filtered_items is not None:
                filtered_items = filtered_items.to(self.mask_filter.device)
                self.mask_filter &= data_utils.edges_filter_nodes(mask_sub_adj, filtered_items + n_users)

    def update_neighborhood(self, nodes: torch.Tensor):
        if self.mask_neighborhood is None:
            raise NotImplementedError(
                "neighborhood can be updated only on edge deletion and if the config parameter is set"
            )
        nodes = nodes.flatten().to(self.mask_sub_adj.device)
        nodes_filter = data_utils.edges_filter_nodes(self.mask_sub_adj[:, :self.mask_neighborhood.shape[0]], nodes)
        self._update_neighborhood(nodes_filter)

    def _update_neighborhood(self, nhood: torch.Tensor):
        if not nhood.dtype == torch.bool:
            raise TypeError(f"neighborhood update except a bool Tensor, got {nhood.dtype}")
        self.mask_neighborhood &= nhood

    def _update_P_symm_on_neighborhood(self, P_symm):
        if (self.mask_filter != self.mask_neighborhood).any():
            filtered_idxs_asymm = self.mask_filter.nonzero().T.squeeze()[:P_symm.shape[0]]
            P_symm_nhood_mask = self.mask_neighborhood[filtered_idxs_asymm].to(P_symm.device)
            return torch.where(P_symm_nhood_mask, P_symm, torch.ones_like(P_symm, device=P_symm.device))

    def forward(self, pred=False, return_only_P_loss=False):
        P_symm = self.P_symm
        # import pdb; pdb.set_trace()
        if not self.edge_additions:
            if self.force_removed_edges is not None:
                if self.random_perturb_p is not None:  # RANDOM_POLICY is active
                    if not pred:
                        p = self.random_perturb_p
                        random_perb = torch.FloatTensor(
                            (self.random_perturb_rs.rand(self.force_removed_edges.size(0)) > p).astype(int)
                        ).to(self.force_removed_edges.device)
                        self.force_removed_edges = self.force_removed_edges * random_perb
                    # the minus 1 assigns (0 - 1) = -1 to the already removed edges, such that the sigmoid is < 0.5
                    P_symm = self.force_removed_edges.to(self.P_symm.device) - 1
                else:
                    if self.mask_neighborhood is not None:
                        P_symm = self._update_P_symm_on_neighborhood(P_symm)

                    force_removed_edges = self.force_removed_edges.to(self.P_symm.device)
                    force_removed_edges = (torch.sigmoid(P_symm.detach()) >= 0.5).float() * force_removed_edges
                    # the minus 1 assigns (0 - 1) = -1 to the already removed edges, such that the sigmoid is < 0.5
                    P_symm = torch.where(force_removed_edges == 0, force_removed_edges - 1, P_symm)
                    self.force_removed_edges = force_removed_edges.to('cpu')
                    del force_removed_edges

            elif self.mask_neighborhood is not None:
                P_symm = self._update_P_symm_on_neighborhood(P_symm)

        if pred:
            P_hat_symm = (torch.sigmoid(P_symm) >= 0.5).float()
        else:
            P_hat_symm = torch.sigmoid(P_symm)

        # pert_edge_index, pert_edge_weight = data_utils.create_sparse_symm_matrix_from_vec(
        #     P_hat_symm, self.mask_sub_adj.to(self.P_symm.device), self.edge_index, self.edge_weight,
        #     edge_deletions=not self.edge_additions,
        #     mask_filter=self.mask_filter.to(self.P_symm.device) if self.mask_filter is not None else None
        # )
        pert_edge_index = self.mask_sub_adj if self.edge_additions else self.mask_filter
        pert_edge_index, pert_edge_weight = data_utils.create_sparse_symm_matrix_from_vec(
            P_hat_symm,
            pert_edge_index.to(P_hat_symm.device),
            self.edge_index.to(P_hat_symm.device),
            self.edge_weight.to(P_hat_symm.device) if self.edge_weight is not None else None,
            edge_deletions=not self.edge_additions,
        )
        _, _, val = pert_edge_index.coo()

        # if pred is True the perturbed matrix without normalization is saved to check distance with original matrix
        if pred:
            if pert_edge_weight is None:
                P_loss = pert_edge_index
            else:
                P_loss = torch.sparse_coo_tensor(
                    pert_edge_index, pert_edge_weight, (self.num_all, self.num_all)
                )

            if return_only_P_loss:
                return P_loss

        # Graph normalization
        if pert_edge_weight is None:
            pert_edge_index = differentiable_gcn_norm(
                pert_edge_index, None, self.num_all,
                add_self_loops=self.add_self_loops, **self.gcn_norm_kwargs
            )
        else:
            pert_edge_index, pert_edge_weight = differentiable_gcn_norm(
                pert_edge_index, pert_edge_weight, self.num_all,
                add_self_loops=self.add_self_loops, **self.gcn_norm_kwargs
            )

        return pert_edge_index, pert_edge_weight


def differentiable_gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False, alpha=0,
                            add_self_loops=True, flow="source_to_target", dtype=None):
    from torch_geometric.typing import torch_sparse
    from torch_geometric.utils import add_remaining_self_loops
    from torch_geometric.utils import add_self_loops as add_self_loops_fn
    from torch_geometric.utils import (
        is_torch_sparse_tensor,
        scatter,
        to_edge_index,
    )
    from torch_geometric.utils.num_nodes import maybe_num_nodes
    from torch_geometric.utils.sparse import set_sparse_value

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1) + 1e-7  # avoids NaN in backpropagation
        deg += alpha  # specific of some models (e.g., SVD_GCN)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum') + 1e-7  # avoids NaN in backpropagation
        deg += alpha  # specific of some models (e.g., SVD_GCN)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum') + 1e-7  # avoids NaN in backpropagation
    deg += alpha  # specific of some models (e.g., SVD_GCN)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight
