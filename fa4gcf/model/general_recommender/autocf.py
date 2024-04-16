r"""
AutoCF
################################################
Reference:
    Lianghao Xia et al. "Automated Self-Supervised Learning for Recommendation." in WWW 2023.

Reference code:
    https://github.com/HKUDS/AutoCF
"""

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import RegLoss
from recbole.utils import InputType

from fa4gcf.model.abstract_recommender import GeneralGraphRecommender
from fa4gcf.model.loss import ContrastiveLoss
from fa4gcf.model.layers import LightGCNConv


class AutoCF(GeneralGraphRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(AutoCF, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']   # int type:the embedding size of AutoCF
        self.n_gcn_layers = config['n_gcn_layers']       # int type:the gcn layer num of AutoCF
        self.n_gt_layers = config['n_gt_layers']         # int type:the graph transformer layer num of AutoCF
        self.reg_weight = config['reg_weight']           # float type: weight decay regularizer
        self.ssl_reg_weight = config['ssl_reg_weight']   # contrastive\self-supervised learning regularizer

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embedding_size)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embedding_size)
        self.gcn_conv = LightGCNConv(dim=self.embedding_size)
        self.GTlayers = torch.nn.ModuleList(
            [GTLayer(self.embedding_size, config['head']) for _ in range(self.n_gt_layers)]
        )
        self.reg_loss = RegLoss()
        self.contrast_loss = ContrastiveLoss()

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, encoder_edge_index, encoder_edge_weight, decoder_edge_index=None):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for gcn_layer_idx in range(self.n_gcn_layers):
            all_embeddings = self.gcn_conv(embeddings_list[-1], encoder_edge_index, encoder_edge_weight)
            embeddings_list.append(all_embeddings)

        if decoder_edge_index is not None:
            for gt_layer in self.GTlayers:
                all_embeddings = gt_layer(embeddings_list[-1], decoder_edge_index)
                embeddings_list.append(all_embeddings)

        auto_cf_all_embeddings = torch.stack(embeddings_list, dim=1)
        auto_cf_all_embeddings = torch.sum(auto_cf_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(auto_cf_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction, encoder_edge_index, encoder_edge_weight, decoder_edge_index):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        _ = interaction[self.NEG_ITEM_ID]  # not used

        user_all_embeddings, item_all_embeddings = self.forward(
            encoder_edge_index, encoder_edge_weight, decoder_edge_index=decoder_edge_index
        )
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]

        # calculate BPR loss
        bpr_loss = (-torch.mul(u_embeddings, pos_embeddings).sum(dim=-1)).mean()

        # calculate regularization Loss
        reg_loss = self.reg_loss(self.parameters())

        # calculate contrastive loss
        contr_loss = self.contrast_loss(user, user_all_embeddings) + self.contrast_loss(pos_item, item_all_embeddings)
        contr_loss *= self.ssl_reg_weight
        contr_loss += self.contrast_loss(user, user_all_embeddings, item_all_embeddings)

        loss = bpr_loss + self.reg_weight * reg_loss + contr_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward(
            self.edge_index, self.edge_weight, decoder_edge_index=self.edge_index
        )

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        user_all_embeddings, item_all_embeddings = self.forward(
            self.edge_index, self.edge_weight, decoder_edge_index=self.edge_index
        )

        u_embeddings = user_all_embeddings[user]

        scores = torch.matmul(u_embeddings, item_all_embeddings.transpose(0, 1))
        return scores.view(-1)


class GTLayer(torch.nn.Module):

    def __init__(self, embedding_size, head):
        super(GTLayer, self).__init__()

        self.embedding_size = embedding_size
        self.head = head
        # ensure embedding can be divided into <head> heads
        assert self.embedding_size % self.head == 0, "Embeddings size must be a multiple of transformer heads"

        self.q = torch.nn.Parameter(torch.empty(self.embedding_size, self.embedding_size))
        self.k = torch.nn.Parameter(torch.empty(self.embedding_size, self.embedding_size))
        self.v = torch.nn.Parameter(torch.empty(self.embedding_size, self.embedding_size))

    def apply(self, fn):
        torch.nn.init.xavier_uniform_(self.q)
        torch.nn.init.xavier_uniform_(self.k)
        torch.nn.init.xavier_uniform_(self.v)

    def forward(self, all_embeddings, edge_index):
        torch.cuda.empty_cache()
        if isinstance(edge_index, torch.Tensor):
            rows, cols = edge_index
        else:  # it should be a pytorch_sparse SparseTensor
            rows, cols, _ = edge_index.coo()

        row_embeddings = all_embeddings[rows]
        col_embeddings = all_embeddings[cols]

        # if torch.cuda.device_count() > 1:
        q_embeddings = (row_embeddings @ self.q).view([-1, self.head, self.embedding_size // self.head]).to('cuda:1')
        k_embeddings = (col_embeddings @ self.k).view([-1, self.head, self.embedding_size // self.head]).to('cuda:1')
        v_embeddings = (col_embeddings @ self.v).view([-1, self.head, self.embedding_size // self.head]).to('cuda:1')
        # q_embeddings = (row_embeddings @ self.q).view([-1, self.head, self.embedding_size // self.head])
        # k_embeddings = (col_embeddings @ self.k).view([-1, self.head, self.embedding_size // self.head])
        # v_embeddings = (col_embeddings @ self.v).view([-1, self.head, self.embedding_size // self.head])

        att = torch.einsum('ehd, ehd -> eh', q_embeddings, k_embeddings)
        att = torch.clamp(att, -10.0, 10.0)

        exp_att = torch.exp(att).to(rows.device)
        tem = torch.zeros([all_embeddings.shape[0], self.head], device=exp_att.device)
        att_norm = (tem.index_add_(0, rows, exp_att))[rows]
        att = exp_att / (att_norm + 1e-8)  # eh

        # if torch.cuda.device_count() > 1:
        out_embeddings = torch.einsum('eh, ehd -> ehd', att.to('cuda:1'), v_embeddings).view([-1, self.embedding_size])
        # out_embeddings = torch.einsum('eh, ehd -> ehd', att, v_embeddings).view([-1, self.embedding_size])

        tem = torch.zeros([all_embeddings.shape[0], self.embedding_size], dtype=out_embeddings.dtype, device=out_embeddings.device)

        # if torch.cuda.device_count() > 1:
        out_embeddings = tem.index_add_(0, rows.to('cuda:1'), out_embeddings)  # nd
        # out_embeddings = tem.index_add_(0, rows, out_embeddings)  # nd

        return out_embeddings.to(all_embeddings.device)


class LocalGraphSampler(torch.nn.Module):

    def __init__(self, n_seeds):
        super(LocalGraphSampler, self).__init__()

        self.n_seeds = n_seeds

    @staticmethod
    def add_noise(scores):
        noise = torch.rand(scores.shape, device=scores.device)
        noise = -torch.log(-torch.log(noise))
        return torch.log(scores) + noise

    def forward(self, all_embeddings, edge_index, edge_weight):
        """

        :param edge_index:
        :param edge_weight:
        :param all_embeddings: should be zero-order embeddings
        :return:
        """
        # all_ones_adj should be with self-loops
        # if edge_weight is None:
        #     all_ones_adj = edge_index.fill_value(1).add(
        #         edge_index.eye(*edge_index.sparse_sizes(), device=edge_index.device())
        #     )
        # else:
        #     adj = torch.sparse_coo_tensor(
        #         indices=edge_index,
        #         values=torch.ones_like(edge_weight, device=edge_weight.device),
        #         size=(all_embeddings.shape[0], all_embeddings.shape[0])
        #     )
        #     all_ones_adj = adj + torch.eye(all_embeddings.shape[0], dtype=adj.dtype, device=adj.device)
        if edge_weight is None:
            all_ones_adj = edge_index.fill_value(1)
        else:
            all_ones_adj = torch.sparse_coo_tensor(
                indices=edge_index,
                values=torch.ones_like(edge_weight, device=edge_weight.device),
                size=(all_embeddings.shape[0], all_embeddings.shape[0])
            )

        order = all_ones_adj.sum(dim=-1).to_dense().view([-1, 1])
        first_embeddings = all_ones_adj.matmul(all_embeddings) - all_embeddings
        first_num = order

        second_embeddings = (all_ones_adj.matmul(first_embeddings) - first_embeddings) - first_num * all_embeddings
        second_num = (all_ones_adj.matmul(first_num) - first_num) - first_num
        subgraph_embeddings = (first_embeddings + second_embeddings) / (first_num + second_num + 1e-8)
        subgraph_embeddings = torch.nn.functional.normalize(subgraph_embeddings, p=2)

        all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2)
        scores = torch.sigmoid(torch.sum(subgraph_embeddings * all_embeddings, dim=-1))

        scores = self.add_noise(scores)
        _, seeds = torch.topk(scores, k=self.n_seeds)

        return scores, seeds


class SubgraphRandomMasker(torch.nn.Module):

    def __init__(self, mask_depth, keep_rate, num_all):
        super(SubgraphRandomMasker, self).__init__()

        self.mask_depth = mask_depth
        self.keep_rate = keep_rate
        self.num_all = num_all

    def forward(self, edge_index, edge_weight, seeds):
        if edge_weight is None:
            rows, cols, values = edge_index.coo()
        else:
            rows, cols = edge_index
            values = edge_weight

        mask_nodes = [seeds]

        for i in range(self.mask_depth):
            next_seeds = []
            cur_seeds = seeds if i == 0 else next_seeds
            for seed in cur_seeds:
                row_idct = (rows == seed)
                col_idct = (cols == seed)
                idct = torch.logical_or(row_idct, col_idct)

                if i != self.mask_depth - 1:
                    mask_rows = rows[idct]
                    mask_cols = cols[idct]
                    next_seeds.append(mask_rows)
                    next_seeds.append(mask_cols)

                rows = rows[torch.logical_not(idct)]
                cols = cols[torch.logical_not(idct)]

            if len(next_seeds) > 0:
                next_seeds = torch.unique(torch.concat(next_seeds))
                mask_nodes.append(next_seeds)

        sample_num = int(self.num_all * self.keep_rate)
        sampled_nodes = torch.randint(self.num_all, size=[sample_num]).to(values.device)

        mask_nodes.append(sampled_nodes)
        mask_nodes = torch.unique(torch.concat(mask_nodes))

        if edge_weight is None:
            import torch_sparse
            encoder_edge_index = torch_sparse.SparseTensor(
                row=rows,
                col=cols,
                value=torch.ones_like(rows, dtype=torch.float, device=rows.device),
                sparse_sizes=(self.num_all, self.num_all)
            )
            encoder_edge_index = gcn_norm(encoder_edge_index, None, rows.shape[0], add_self_loops=False)
            encoder_edge_weight = None
        else:
            encoder_edge_index, encoder_edge_weight = gcn_norm(
                torch.stack((rows, cols), dim=0),
                torch.ones_like(rows, dtype=torch.float, device=rows.device),
                rows.shape[0],
                add_self_loops=False
            )

        tem_num = mask_nodes.shape[0]
        tem_rows = mask_nodes[torch.randint(tem_num, size=[values.shape[0]])]
        tem_cols = mask_nodes[torch.randint(tem_num, size=[values.shape[0]])]

        new_rows = torch.concat([tem_rows, tem_cols, torch.arange(self.num_all, device=tem_rows.device), rows])
        new_cols = torch.concat([tem_cols, tem_rows, torch.arange(self.num_all, device=tem_rows.device), cols])

        # filter duplicated
        hash_val = new_rows * self.num_all + new_cols
        hash_val = torch.unique(hash_val)
        new_cols = hash_val % self.num_all
        new_rows = ((hash_val - new_cols) / self.num_all).long()

        decoder_edge_index = torch.stack((new_rows, new_cols), dim=0)

        return encoder_edge_index, encoder_edge_weight, decoder_edge_index
