r"""
GFCF
################################################
Reference:
    Yifei Shen et al. "How Powerful is Graph Convolution for Recommendation?" in CIKM 2021.

Reference code:
    https://github.com/yshenaw/GF_CF
    https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/external/models/gfcf/GFCF.py
"""

import torch
import scipy
import numpy as np

from recbole.utils import ModelType, InputType

from fa4gcf.model.abstract_recommender import GeneralGraphRecommender


class GFCF(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(GFCF, self).__init__(config, dataset)

        # load parameters info
        self.svd_factors = config['svd_factors']
        self.alpha = config['gfcf_alpha']

        # generate adj_matrix to be trained with scipy and sparsesvd
        self.adj_mat = dataset.inter_matrix(form='coo').tolil()
        self.restore_user_rating = None

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

        self._run_svd()

        # we convert the resulting vectors to pytorch tensors to allow the perturbation vector to be optimized
        # it does not affect the final computation. The prediction is just performed with pytorch
        # self._gfcf_data_to_tensor()

    def _run_svd(self):
        from sparsesvd import sparsesvd

        self._generate_gfcf_data()

        _, _, self.vt = sparsesvd(self.norm_adj, self.svd_factors)

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def _generate_gfcf_data(self):
        # if self.training:
        adj_mat = self.adj_mat
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = scipy.sparse.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = scipy.sparse.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = scipy.sparse.diags(1 / d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        # else:
        #     if not isinstance(self.adj_mat, (torch.Tensor, torch.sparse.Tensor)):
        #         raise ValueError("The perturbation process requires the adjacency matrix to be a pytorch tensor")
        #
        #     adj_mat = self.adj_mat
        #     rowsum = adj_mat.sum(dim=1)
        #     d_inv = torch.pow(rowsum, -0.5)
        #     d_inv[torch.isinf(d_inv)] = 0.
        #     d_mat = torch.sparse_coo_tensor(
        #         torch.tile(torch.arange(d_inv.shape[0]), (2, 1)),
        #         d_inv,
        #         (d_inv.shape[0], d_inv.shape[0]),
        #         device=d_inv.device
        #     )
        #     norm_adj = torch.sparse.mm(d_mat, self.adj_mat)
        #
        #     colsum = adj_mat.sum(dim=0)
        #     d_inv = torch.pow(colsum, -0.5)
        #     d_inv[torch.isinf(d_inv)] = 0.
        #     d_mat = torch.sparse_coo_tensor(
        #         torch.tile(torch.arange(d_inv.shape[0]), (2, 1)),
        #         d_inv,
        #         (d_inv.shape[0], d_inv.shape[0]),
        #         device=d_inv.device
        #     )
        #     self.d_mat_i = d_mat
        #     self.d_mat_i_inv = torch.sparse_coo_tensor(
        #         torch.tile(torch.arange(d_inv.shape[0]), (2, 1)),
        #         1 / d_inv,
        #         (d_inv.shape[0], d_inv.shape[0]),
        #         device=d_inv.device
        #     )
        #     self.norm_adj = torch.sparse.mm(norm_adj, d_mat)

    def _generate_user_rating(self):
        U_2 = self.adj_mat @ self.norm_adj.T @ self.norm_adj
        U_1 = self.adj_mat @ self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv

        self.restore_user_rating = torch.tensor(U_2 + self.alpha * U_1).to(self.device)

        # U_2 = torch.sparse.mm(torch.sparse.mm(self.adj_mat, self.norm_adj.T), self.norm_adj)
        # U_1 = torch.sparse.mm(
        #     torch.sparse.mm(
        #         torch.sparse.mm(
        #             torch.sparse.mm(
        #                 torch.sparse.mm(self.adj_mat, self.d_mat_i)
        #             ),
        #             self.vt.T
        #         ),
        #         self.vt
        #     ),
        #     self.d_mat_i_inv
        # )
        #
        # self.restore_user_rating = U_2 + self.alpha * U_1

    def _gfcf_data_to_tensor(self):
        self.vt = torch.from_numpy(self.vt).to_sparse_coo().to(self.device)
        self.d_mat_i = torch.sparse_coo_tensor(
            torch.tile(torch.arange(self.d_mat_i.shape[0]), (2, 1)),
            torch.from_numpy(self.d_mat_i.data.squeeze()),
            self.d_mat_i.shape,
            device=self.device
        )
        self.d_mat_i_inv = torch.sparse_coo_tensor(
            torch.tile(torch.arange(self.d_mat_i_inv.shape[0]), (2, 1)),
            torch.from_numpy(self.d_mat_i_inv.data.squeeze()),
            self.d_mat_i_inv.shape,
            device=self.device
        )
        coo_norm_adj = self.norm_adj.tocoo()
        self.norm_adj = torch.sparse_coo_tensor(
            torch.stack((torch.from_numpy(coo_norm_adj.row), torch.from_numpy(coo_norm_adj.col))),
            torch.from_numpy(coo_norm_adj.data),
            coo_norm_adj.shape,
            device=self.device
        )
        coo_adj_mat = self.adj_mat.tocoo()

        self.adj_mat = torch.sparse_coo_tensor(
            torch.stack((torch.from_numpy(coo_adj_mat.row), torch.from_numpy(coo_adj_mat.col))),
            torch.from_numpy(coo_adj_mat.data),
            coo_adj_mat.shape,
            device=self.device
        )

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if self.restore_user_rating is None:
            self._generate_user_rating()

        user_rating = self.restore_user_rating

        return user_rating[user, item]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_rating is None:
            self._generate_user_rating()

        user_rating = self.restore_user_rating

        return user_rating[user]
