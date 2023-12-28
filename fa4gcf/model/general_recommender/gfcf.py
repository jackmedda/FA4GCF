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
from sparsesvd import sparsesvd

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
        self.adj_mat = dataset.inter_matrix().tolil()
        self.user_rating = None

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

        self._run_svd()

    def _run_svd(self):
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

        _, _, self.vt = sparsesvd(self.norm_adj, self.svd_factors)

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def _generate_user_rating(self):
        U_2 = self.adj_mat @ self.norm_adj.T @ self.norm_adj
        U_1 = self.adj_mat @ self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv

        self.user_rating = torch.tensor(U_2 + self.alpha * U_1).to(self.device)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if self.user_rating is None:
            self._generate_user_rating()

        return self.user_rating[user, item]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.user_rating is None:
            self._generate_user_rating()

        return self.user_rating[user]
