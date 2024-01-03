r"""
UltraGCN
################################################
Reference:
    Kelong Mao et al. "UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation" in CIKM 2021.

Reference code:
    https://github.com/reczoo/RecZoo
    https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/external/models/ultragcn
"""

import torch
import numpy as np

from recbole.utils import ModelType, InputType

from fa4gcf.model.abstract_recommender import GeneralGraphRecommender
from fa4gcf.model.loss import LLoss, ILoss, NormLoss


class UltraGCN(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE
    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(UltraGCN, self).__init__(config, dataset)

        self.w1 = config["w1"]
        self.w2 = config["w2"]
        self.w3 = config["w3"]
        self.w4 = config["w4"]

        # load parameters info
        self.embedding_size = config['embedding_size']  # int type:the embedding size of lightGCN
        self.ii_neighbor_num = config["ii_neighbor_num"]  # int type: neighbors in the item-item co-occurrence graph
        self.initial_weight = config['initial_weight']  # float type: standard deviation of normal initialization
        self.gamma = config['gamma']  # float type: weight parameter of norm_loss
        self.lambda_ = config['ILoss_lambda']  # float type: weight parameter of ILoss
        self.negative_num = config['train_neg_sample_args']['sample_num']

        # constraint matrix generation, item-item constraint matrix generation
        user_degree = dataset.history_item_matrix()[-1]
        item_degree = dataset.history_user_matrix()[-1]
        self.beta_uD = (torch.sqrt(user_degree + 1) / user_degree).to(self.device)
        self.beta_iD = (1 / torch.sqrt(item_degree + 1)).to(self.device)
        self.ii_neighbor_matrix, self.ii_constraint_matrix = self._get_ii_constrain_mat(
            dataset.inter_matrix(form='coo')
        )

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embedding_size)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embedding_size)
        self.l_loss = LLoss(negative_weight=config['negative_weight'])
        self.i_loss = ILoss()
        self.norm_loss = NormLoss()

        # parameters initialization
        torch.nn.init.normal_(self.user_embedding.weight, std=self.initial_weight)
        torch.nn.init.normal_(self.item_embedding.weight, std=self.initial_weight)

    def _get_ii_constrain_mat(self, adj_mat):
        try:
            A = adj_mat.T.dot(adj_mat).tocoo()
            A = torch.sparse_coo_tensor(
                torch.stack((torch.from_numpy(A.row), torch.from_numpy(A.col))),
                torch.from_numpy(A.data),
                A.shape,
                device=self.device
            )

            item_col_degree = A.sum(dim=1).to_dense()  # users_D
            item_row_degree = A.sum(dim=0).to_dense()  # items_D

            beta_item_colD = (torch.sqrt(item_col_degree + 1) / item_col_degree).reshape(-1, 1)  # beta_uD
            beta_item_rowD = (1 / torch.sqrt(item_row_degree + 1)).reshape(1, -1)  # beta_iD
            all_ii_constrain_mat = torch.mm(beta_item_colD, beta_item_rowD)

            ii_neighbors_matrix = (all_ii_constrain_mat * A).to_dense()
            ii_neighbors_sim, ii_neighbors_idxs = torch.topk(ii_neighbors_matrix, k=self.ii_neighbor_num, dim=1)
        except RuntimeError:  # or torch.cuda.OutOfMemoryError
            A = adj_mat.T.dot(adj_mat).tocoo()

            item_col_degree = np.sum(A, axis=1).reshape(-1)  # users_D
            item_row_degree = np.sum(A, axis=0).reshape(-1)  # items_D

            beta_item_colD = (np.sqrt(item_col_degree + 1) / item_col_degree).reshape(-1, 1)  # beta_uD
            beta_item_rowD = (1 / np.sqrt(item_row_degree + 1)).reshape(1, -1)  # beta_iD

            all_ii_constrain_mat = beta_item_colD.dot(beta_item_rowD)

            ii_neighbors_matrix = torch.from_numpy(A.multiply(all_ii_constrain_mat).todense()).to(self.device)
            ii_neighbors_sim, ii_neighbors_idxs = torch.topk(ii_neighbors_matrix, k=self.ii_neighbor_num, dim=1)

        return ii_neighbors_idxs, ii_neighbors_sim

    def get_omegas(self, users, pos_items, neg_items):
        if self.w2 > 0:
            pos_weight = torch.mul(self.beta_uD[users], self.beta_iD[pos_items])
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items))

        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(self.beta_uD[users], self.beta_iD[neg_items.flatten()])
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.shape[0])

        omega_weight = torch.cat((pos_weight, neg_weight))

        return omega_weight

    # def cal_loss_l(self, users, pos_items, neg_items, omega_weight):
    #     user_all_embeddings, item_all_embeddings = self.forward()
    #     u_embeddings = user_all_embeddings[user]
    #     pos_embeddings = item_all_embeddings[pos_item]
    #     neg_embeddings = item_all_embeddings[neg_item]
    #
    #     pos_scores = (u_embeddings * pos_embeddings).sum(dim=-1)  # batch_size
    #     u_embeddings = u_embeddings.unsqueeze(1)
    #     neg_scores = (u_embeddings * neg_embeddings).sum(dim=-1)  # batch_size * negative_num
    #
    #     neg_labels = torch.zeros(neg_scores.size()).to(self.device)
    #     neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
    #         neg_scores, neg_labels,
    #         weight=omega_weight[len(pos_scores):].view(neg_scores.size()),
    #         reduction='none'
    #     ).mean(dim=-1)
    #
    #     pos_labels = torch.ones(pos_scores.size()).to(self.device)
    #     pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
    #         pos_scores, pos_labels,
    #         weight=omega_weight[:len(pos_scores)],
    #         reduction='none'
    #     )
    #
    #     loss = pos_loss + neg_loss * self.negative_weight
    #
    #     return loss.sum()

    # def cal_loss_i(self, users, pos_items):
    #     neighbor_embeds = self.item_embedding(self.ii_neighbor_mat[pos_items])
    #     sim_scores = self.ii_constraint_mat[pos_items]
    #     user_embeds = self.user_embedding(users).unsqueeze(1)
    #
    #     loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
    #
    #     # loss = loss.sum(-1)
    #     return loss.sum()

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        omega_weight = self.get_omegas(user, pos_item, neg_item)

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = (u_embeddings * pos_embeddings).sum(dim=-1)
        neg_scores = (u_embeddings * neg_embeddings).sum(dim=-1)
        loss = self.l_loss(pos_scores, neg_scores, omega_weight)

        loss += self.gamma * self.norm_loss(self.parameters())

        neighbor_scores = u_embeddings.unsqueeze(1) * item_all_embeddings[self.ii_neighbor_matrix[pos_item]]
        sim_scores = self.ii_constraint_matrix[pos_item]
        loss += self.lambda_ * self.i_loss(sim_scores, neighbor_scores)

        return loss

    def forward(self):
        return self.user_embedding.weight, self.item_embedding.weight

    # def forward(self, users, pos_items, neg_items):
    #     omega_weight = self.get_omegas(users, pos_items, neg_items)
    #
    #     loss = self.cal_loss_l(users, pos_items, neg_items, omega_weight)
    #     loss += self.gamma * self.norm_loss()
    #     loss += self.lambda_ * self.cal_loss_i(users, pos_items)
    #     return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]

        scores = torch.matmul(u_embeddings, i_embeddings.t())

        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]

        scores = torch.matmul(u_embeddings, item_all_embeddings.t())

        return scores.view(-1)
