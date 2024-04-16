from typing import Union

import torch
import numpy as np
import igraph as ig
import pandas as pd
from torch_geometric.typing import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from recbole.data.utils import get_dataloader, create_samplers
from recbole.utils import FeatureSource, FeatureType

from fa4gcf.data.interaction import Interaction


def get_dataset_with_perturbed_edges(pert_edges: np.ndarray, dataset):
    user_num = dataset.user_num
    uid_field, iid_field = dataset.uid_field, dataset.iid_field
    pert_edges = pert_edges.copy()

    pert_edges = torch.tensor(pert_edges)
    pert_edges[1] -= user_num  # remap items in range [0, item_num)

    orig_inter_feat = dataset.inter_feat
    pert_inter_feat = {}
    for i, col in enumerate([uid_field, iid_field]):
        pert_inter_feat[col] = torch.cat((orig_inter_feat[col], pert_edges[i]))

    unique, counts = torch.stack(
        (pert_inter_feat[uid_field], pert_inter_feat[iid_field]),
    ).unique(dim=1, return_counts=True)
    pert_inter_feat[uid_field], pert_inter_feat[iid_field] = unique[:, counts == 1]

    return dataset.copy(Interaction(pert_inter_feat))


def get_dataloader_with_perturbed_edges(pert_edges: np.ndarray, config, dataset, train_data, valid_data, test_data):
    pert_edges = pert_edges.copy()

    train_dataset = get_dataset_with_perturbed_edges(pert_edges, train_data.dataset)
    valid_dataset = get_dataset_with_perturbed_edges(pert_edges, valid_data.dataset)
    test_dataset = get_dataset_with_perturbed_edges(pert_edges, test_data.dataset)

    built_datasets = [train_dataset, valid_dataset, test_dataset]
    train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

    train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=False)
    valid_data = get_dataloader(config, 'valid')(config, valid_dataset, valid_sampler, shuffle=False)
    test_data = get_dataloader(config, 'test')(config, test_dataset, test_sampler, shuffle=False)

    return train_data, valid_data, test_data


def symmetrically_sort(idxs: torch.Tensor, value: torch.Tensor = None):
    assert idxs.shape[0] == 2
    symm_offset = idxs.shape[1] // 2
    left_idx, right_idx = idxs[:, :symm_offset], idxs[[1, 0], symm_offset:]
    left_sorter, right_sorter = torch.argsort(left_idx[0]), torch.argsort(right_idx[0])
    sorter = torch.cat((left_sorter, right_sorter + symm_offset))

    idxs = idxs[:, sorter]
    if value is not None:
        value = value[sorter]
    assert is_symmetrically_sorted(idxs)

    return (idxs, value) if value is not None else idxs


def is_symmetrically_sorted(idxs: torch.Tensor):
    assert idxs.shape[0] == 2
    symm_offset = idxs.shape[1] // 2
    return (idxs[:, :symm_offset] == idxs[[1, 0], symm_offset:]).all()


def get_sorter_indices(base_idxs, to_sort_idxs):
    unique, inverse = torch.cat((base_idxs, to_sort_idxs), dim=1).unique(dim=1, return_inverse=True)
    inv_base, inv_to_sort = torch.split(inverse, to_sort_idxs.shape[1])
    sorter = torch.arange(to_sort_idxs.shape[1], device=inv_to_sort.device)[torch.argsort(inv_to_sort)]

    return sorter, inv_base


def create_sparse_symm_matrix_from_vec(pert_vector,
                                       pert_index_filter,
                                       edge_index: Union[torch.Tensor, SparseTensor],
                                       edge_weight,
                                       num_nodes=None,
                                       edge_deletions=False):
    is_sparse = False
    if edge_weight is None:
        is_sparse = True
        if not edge_index.is_coalesced():
            edge_index = edge_index.coalesce()
        num_nodes = edge_index.sparse_size(dim=0) if num_nodes is None else num_nodes
        row, col, edge_weight = edge_index.coo()
        edge_index = torch.stack([row, col], dim=0)

    if not edge_deletions:  # if pass is edge additions
        pert_vector_mask = pert_vector != 0  # reduces memory footprint

        pert_index_filter = pert_index_filter[:, pert_vector_mask]
        pert_vector = pert_vector[pert_vector_mask]
        del pert_vector_mask
        torch.cuda.empty_cache()

        edge_index = edge_index.to(pert_vector.device)
        pert_index_filter = pert_index_filter.to(pert_vector.device)
        edge_index = torch.cat((edge_index, pert_index_filter, pert_index_filter[[1, 0]]), dim=1)
        edge_weight = torch.cat((edge_weight, pert_vector, pert_vector))
    else:
        pert_vector = torch.cat((pert_vector, pert_vector))
        if pert_index_filter is not None:
            # sorter, pert_index_inverse = get_sorter_indices(pert_index, edge_index)
            # edge_weight = edge_weight[sorter][pert_index_inverse]
            # assert is_symmetrically_sorted(edge_index[:, sorter][:, pert_index_inverse])
            edge_weight[pert_index_filter] = pert_vector

    torch.cuda.empty_cache()

    if is_sparse and num_nodes is not None:
        edge_index = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=edge_weight,
            sparse_sizes=(num_nodes, num_nodes)
        ).t()
        edge_weight = None

    return edge_index, edge_weight


def edge_index_to_adj_t(edge_index, edge_weight, m_num_nodes, n_num_nodes):
    adj = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=edge_weight,
        sparse_sizes=(m_num_nodes, n_num_nodes)
    )
    return adj.t()


def get_norm_adj_mat(edge_index, edge_weight, num_nodes, add_self_loops=False, enable_sparse=False, is_sparse=False):
    if enable_sparse:
        if is_sparse:
            adj_t = edge_index_to_adj_t(edge_index, edge_weight, num_nodes, num_nodes)
            adj_t = gcn_norm(adj_t, None, num_nodes, add_self_loops=add_self_loops)
            return adj_t, None

    edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops=add_self_loops)

    return edge_index, edge_weight


def edges_filter_nodes(edges: torch.LongTensor, nodes: torch.LongTensor):
    try:
        adj_filter = torch.isin(edges, nodes).any(dim=0)
    except AttributeError:
        if edges.shape[0] == 1:
            adj_filter = torch.from_numpy(isin_backcomp(edges, nodes))
        else:
            adj_filter = torch.from_numpy(isin_backcomp(edges[0], nodes) | isin_backcomp(edges[1], nodes))
            # adj_filter = (edges[0][:, None] == nodes).any(-1) | (edges[1][:, None] == nodes).any(-1)
    return adj_filter


def isin_backcomp(ar1: torch.Tensor, ar2: torch.Tensor):
    ar1 = ar1.detach().numpy()
    ar2 = ar2.detach().numpy()
    return np.in1d(ar1, ar2)


def get_bipartite_igraph(dataset, remove_first_row_col=False):
    inter_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
    if remove_first_row_col:
        inter_matrix = inter_matrix[1:, 1:]

    incid_adj = ig.Graph.Incidence(inter_matrix.todense().tolist())
    bip_info = np.concatenate([np.zeros(inter_matrix.shape[0], dtype=int), np.ones(inter_matrix.shape[1], dtype=int)])

    return ig.Graph.Bipartite(bip_info, incid_adj.get_edgelist())


def update_item_feat_discriminative_attribute(dataset,
                                              item_discriminative_attribute,
                                              item_discriminative_ratio,
                                              item_discriminative_map):
    if dataset.item_feat is None:
        dataset.item_feat = dataset.get_item_feature()
    elif item_discriminative_attribute in dataset.item_feat:
        return

    if isinstance(dataset.inter_feat, pd.DataFrame):
        # inter_feat does not contain the padding user and item, so + 1 re-indexes the indices by 1
        pop = torch.tensor(dataset.inter_feat[dataset.iid_field].value_counts(sort=False).argsort()[::-1].values + 1)
    else:
        pop = torch.argsort(dataset.history_user_matrix()[2], descending=True)
    sh_size = round(item_discriminative_ratio * pop.shape[0])

    # field2seqlen[dest_field]

    exposure_group = torch.zeros_like(dataset.item_feat[dataset.iid_field])
    exposure_group[pop[:sh_size]] = 1
    exposure_group[pop[sh_size:]] = 2
    exposure_group[0] = 0

    dataset.item_feat[item_discriminative_attribute] = exposure_group
    dataset.field2type[item_discriminative_attribute] = FeatureType('token')
    dataset.field2source[item_discriminative_attribute] = FeatureSource('item')
    dataset.field2seqlen[item_discriminative_attribute] = 1
    dataset.field2id_token[item_discriminative_attribute] = np.asarray(item_discriminative_map)
