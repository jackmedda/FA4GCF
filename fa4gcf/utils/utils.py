import copy
from typing import Union

import torch
try:
    import torch_sparse
except ImportError:
    pass


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
                                       pert_index,
                                       edge_index: Union[torch.Tensor, "torch_sparse.SparseTensor"],
                                       edge_weight,
                                       num_nodes=None,
                                       edge_deletions=False,
                                       mask_filter=None):
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

        pert_index = pert_index[:, pert_vector_mask]
        pert_vector = pert_vector[pert_vector_mask]
        del vector_zeros_mask
        torch.cuda.empty_cache()

        edge_index = edge_index.to(pert_vector.device)
        pert_index = pert_index.to(pert_vector.device)
        edge_index = torch.cat((edge_index, pert_index, pert_index[[1, 0]]), dim=1)
        edge_weight = torch.cat((edge_weight, pert_vector, pert_vector))
    else:
        pert_vector = torch.cat((pert_vector, pert_vector))
        if mask_filter is not None:
            sorter, pert_index_inv = get_sorter_indices(pert_index, edge_index)
            edge_weight = edge_weight[sorter][pert_index_inv]
            assert is_symmetrically_sorted(edge_index[:, sorter][:, pert_index_inv])
            edge_weight[mask_filter] = pert_vector

    torch.cuda.empty_cache()

    if is_sparse and num_nodes is not None:
        edge_index = torch_sparse.SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=edge_weight,
            sparse_sizes=(num_nodes, num_nodes)
        ).t()
        edge_weight = None

    return edge_index, edge_weight
