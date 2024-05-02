r"""
GFCF
################################################
Reference:
    Shaowen Peng et al. "SVD-GCN: A Simplified Graph Convolution Paradigm for Recommendation" in CIKM 2022.

Reference code:
    https://github.com/tanatosuu/svd_gcn
    https://github.com/sisinflab/Topology-Graph-Collaborative-Filtering/blob/master/external/models/svd_gcn/SVDGCN.py
"""

import torch
import numpy as np

from recbole.utils import InputType, ModelType
from recbole.model.loss import BPRLoss

from fa4gcf.model.abstract_recommender import GeneralGraphRecommender


def svd_gcn_norm(edge_index, alpha=1.0):
    adj_t = edge_index

    degree_u = adj_t.sum(dim=1).to_dense() + alpha
    degree_i = adj_t.sum(dim=0).to_dense() + alpha

    degree_u.pow_(-0.5)
    degree_i.pow_(-0.5)

    degree_u.masked_fill_(degree_u == float('inf'), 0)
    degree_i.masked_fill_(degree_i == float('inf'), 0)

    return degree_u.unsqueeze(1) * adj_t * degree_i


class SVD_GCN(GeneralGraphRecommender):
    model_type = ModelType.TRADITIONAL
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SVD_GCN, self).__init__(config, dataset)
        # not used, reduce memory usage
        del self.edge_index
        del self.edge_weight

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.k_singular_values = config['k_singular_values']  # req_vec
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.parametric = config['parametric']
        self.user_coefficient = config['user_coefficient'] or 0.0
        self.item_coefficient = config['item_coefficient'] or 0.0
        self.reg_weight = config['reg_weight']
        self.rate_matrix_weight = config['rate_matrix_weight'] or 1000

        # load input data keys info
        self.NEIGHBOR_USER_ID = config["NEIGHBOR_USER_ID"]
        self.NEG_NEIGHBOR_USER_ID = config["NEG_NEIGHBOR_USER_ID"]
        self.NEIGHBOR_ITEM_ID = config["NEIGHBOR_ITEM_ID"]
        self.NEG_NEIGHBOR_ITEM_ID = config["NEG_NEIGHBOR_ITEM_ID"]

        self.q = self._update_q(config)

        # generate approximated SVD
        self.rate_matrix = self.get_rate_matrix(dataset)
        U, value, V = self.get_svd(self.q)

        weighted_value = self._apply_value_weight(value)

        self.user_vector = U * weighted_value
        self.item_vector = V * weighted_value

        # define layers and loss
        self.mf_loss = BPRLoss()
        self.W = self._init_params()

        # torch.svd_lowrank is not deterministic, so user and item vectors are saved to improve replicability
        self.other_parameter_name = ['user_vector', 'item_vector']

    def _init_params(self):
        if self.parametric:
            W = torch.nn.Parameter(torch.randn(self.k_singular_values, self.embedding_size, device=self.device))

            # parameters initialization
            torch.nn.init.uniform_(
                W,
                a=-np.sqrt(6. / (self.k_singular_values + self.embedding_size)),
                b=np.sqrt(6. / (self.k_singular_values + self.embedding_size))
            )
        else:
            W = torch.nn.Parameter(torch.zeros(1))  # fake_loss

        return W

    def get_rate_matrix(self, dataset):
        inter_matrix = dataset.inter_matrix(form="coo")
        rate_matrix = torch.sparse_coo_tensor(
            torch.stack((torch.from_numpy(inter_matrix.row), torch.from_numpy(inter_matrix.col)), dim=0),
            torch.from_numpy(inter_matrix.data).float(),  # to float32
            size=inter_matrix.shape,
            device=self.device
        )
        del inter_matrix
        return svd_gcn_norm(rate_matrix, alpha=self.alpha)

    def get_svd(self, q):
        U, value, V = torch.svd_lowrank(self.rate_matrix, q=q, niter=30)

        return U[:, :self.k_singular_values], value[:self.k_singular_values], V[:, :self.k_singular_values]

    def _apply_value_weight(self, value):
        weighted_value = value
        while True and self.beta != 0:
            # the exp of the singular values could lead to inf with torch.float32, so we take half of beta
            weighted_value = torch.exp(self.beta * value)  # weight_func/kernel
            if not torch.isinf(weighted_value).any():
                break
            else:
                self.logger.warning(f"SVD_GCN beta = {self.beta} leads to inf values. Reduced by half")
                self.beta /= 2

        return weighted_value

    def _update_q(self, config):
        """
        To make a more accurate SVD, q should be (slightly) larger than k_singular_values. q=400 as set by authors
        However, a value of q that is too close to the number of users and items could cause inf values and nan loss
        """
        q = config['q'] or 400

        # svd_lowrank does not work if q is greater than the number of users or items
        if q > min(self.n_users, self.n_items):
            q = min(self.n_users, self.n_items)
            q = q // 2  # makes q lower than the number of users or items
            self.logger.warning(f"q for SVD_GCN is larger than the number of users or items. q set to {q}")
            if q < self.k_singular_values:
                self.logger.warning(
                    "To make a more accurate SVD, q should be (slightly) larger than k_singular_values. "
                    "k_singular_values will now be set equal to q - 1 to correctly perform the model training"
                )
                self.k_singular_values = q - 1
        return q

    def forward(self):
        if self.parametric:
            return self.user_vector.mm(self.W), self.item_vector.mm(self.W)
        else:
            return self.user_vector, self.item_vector

    def calculate_loss(self, interaction):
        if self.parametric:
            user = interaction[self.USER_ID]
            pos_item = interaction[self.ITEM_ID]
            neg_item = interaction[self.NEG_ITEM_ID]

            user_all_embeddings, item_all_embeddings = self.forward()
            u_embs = user_all_embeddings[user]
            pos_embs = item_all_embeddings[pos_item]
            neg_embs = item_all_embeddings[neg_item]

            # calculate BPR-like Loss
            pos_scores = torch.mul(u_embs, pos_embs).sum(dim=1)
            neg_scores = torch.mul(u_embs, neg_embs).sum(dim=1)
            mf_loss = self.mf_loss(pos_scores, neg_scores)

            if self.user_coefficient > 0:
                pos_user_neighbor = interaction[self.NEIGHBOR_USER_ID]
                neg_user_neighbor = interaction[self.NEG_NEIGHBOR_USER_ID]

                pos_user_neighbor_embs = user_all_embeddings[pos_user_neighbor]
                neg_user_neighbor_embs = user_all_embeddings[neg_user_neighbor]

                mf_loss += self.user_coefficient * self.mf_loss(pos_user_neighbor_embs, neg_user_neighbor_embs)
            else:
                pos_user_neighbor_embs, neg_user_neighbor_embs = 0., 0.

            if self.item_coefficient > 0:
                pos_item_neighbor = interaction[self.NEIGHBOR_ITEM_ID]
                neg_item_neighbor = interaction[self.NEG_NEIGHBOR_ITEM_ID]

                pos_item_neighbor_embs = item_all_embeddings[pos_item_neighbor]
                neg_item_neighbor_embs = item_all_embeddings[neg_item_neighbor]

                mf_loss += self.user_coefficient * self.mf_loss(pos_item_neighbor_embs, neg_item_neighbor_embs)
            else:
                pos_item_neighbor_embs, neg_item_neighbor_embs = 0., 0.

            reg_loss = (
                u_embs ** 2 +
                pos_embs ** 2 +
                neg_embs ** 2 +
                pos_user_neighbor_embs ** 2 +
                neg_user_neighbor_embs ** 2 +
                pos_item_neighbor_embs ** 2 +
                neg_item_neighbor_embs ** 2
            )
            reg_loss = (reg_loss / u_embs.shape[0]).sum()  # not explicitly using mean can help to avoid an inf loss

            loss = mf_loss + self.reg_weight * reg_loss
            if torch.isinf(loss).any():
                raise ValueError("SVD_GCN loss is inf. Try lowering `beta` and/or `q`")
            if torch.isnan(loss).any():
                raise ValueError("SVD_GCN loss is nan. Try lowering `beta` and/or `q`")

            return loss
        else:
            return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]

        A = self.rate_matrix.index_select(0, user).index_select(1, item) * self.rate_matrix_weight
        scores = u_embeddings.mm(i_embeddings.t()).sigmoid() - A

        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]

        A = self.rate_matrix.index_select(0, user) * self.rate_matrix_weight
        scores = u_embeddings.mm(item_all_embeddings.t()).sigmoid() - A

        return scores
