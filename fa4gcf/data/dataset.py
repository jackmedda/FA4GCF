import torch
from torch_geometric.typing import torch_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from gnnuers.data import Dataset as GNNUERS_Dataset

import fa4gcf.data.utils as utils
from fa4gcf.model.general_recommender import AutoCF


class Dataset(GNNUERS_Dataset):

    @staticmethod
    def edge_index_to_adj_t(edge_index, edge_weight, m_num_nodes, n_num_nodes):
        return utils.edge_index_to_adj_t(edge_index, edge_weight, m_num_nodes, n_num_nodes)

    def get_norm_adj_mat(self, enable_sparse=False):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """
        self.is_sparse = torch_sparse is not object

        row = self.inter_feat[self.uid_field]
        col = self.inter_feat[self.iid_field] + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        edge_weight = torch.ones(edge_index.size(1))
        num_nodes = self.user_num + self.item_num

        self.add_self_loops = self.config.model in [AutoCF.__name__]

        if enable_sparse:
            if not self.is_sparse:
                self.logger.warning(
                    "Import `torch_sparse` error, please install corresponding version of `torch_sparse`."
                    "Dense edge_index will be used instead of SparseTensor in dataset."
                )
            else:
                adj_t = self.edge_index_to_adj_t(edge_index, edge_weight, num_nodes, num_nodes)
                adj_t = gcn_norm(adj_t, None, num_nodes, add_self_loops=self.add_self_loops)
                return adj_t, None

        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops=self.add_self_loops)

        return edge_index, edge_weight
