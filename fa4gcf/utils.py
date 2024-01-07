import copy
import yaml
import warnings
import importlib
from logging import getLogger
from typing import Union, Literal

import torch
from torch_geometric.typing import SparseTensor
from recbole.sampler import KGSampler
from recbole.data.dataloader import *
from recbole.utils import (
    ModelType,
    set_color,
    get_model as get_recbole_model,
    get_trainer as get_recbole_trainer
)
from recbole.data.utils import (
    load_split_dataloaders,
    create_samplers,
    save_split_dataloaders,
    get_dataloader as get_recbole_dataloader
)

from fa4gcf.data.custom_dataloader import SVD_GCNDataLoader


def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['[valid|test]_eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
        dataset._change_feat_format()
    else:
        model_type = config["MODEL_TYPE"]
        built_datasets = dataset.build()

        train_dataset, valid_dataset, test_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler = create_samplers(
            config, dataset, built_datasets
        )

        if model_type != ModelType.KNOWLEDGE:
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, shuffle=config["shuffle"]
            )
        else:
            kg_sampler = KGSampler(
                dataset,
                config["train_neg_sample_args"]["distribution"],
                config["train_neg_sample_args"]["alpha"],
            )
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, kg_sampler, shuffle=True
            )

        valid_data = get_dataloader(config, "valid")(
            config, valid_dataset, valid_sampler, shuffle=False
        )
        test_data = get_dataloader(config, "test")(
            config, test_dataset, test_sampler, shuffle=False
        )
        if config["save_dataloaders"]:
            save_split_dataloaders(
                config, dataloaders=(train_data, valid_data, test_data)
            )

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["train_batch_size"]}]', "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f'[{config["train_neg_sample_args"]}]', "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["eval_batch_size"]}]', "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f'[{config["eval_args"]}]', "yellow")
    )
    return train_data, valid_data, test_data


def get_dataloader(config, phase: Literal["train", "valid", "test", "evaluation"]):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take 4 values: 'train', 'valid', 'test' or 'evaluation'.
            Notes: 'evaluation' has been deprecated, please use 'valid' or 'test' instead.
    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase not in ["train", "valid", "test", "evaluation"]:
        raise ValueError(
            "`phase` can only be 'train', 'valid', 'test' or 'evaluation'."
        )
    if phase == "evaluation":
        phase = "test"
        warnings.warn(
            "'evaluation' has been deprecated, please use 'valid' or 'test' instead.",
            DeprecationWarning,
        )

    register_table = {
        "SVD_GCN": _get_SVD_GCN_dataloader
    }

    if config["model"] in register_table:
        return register_table[config["model"]](config, phase)
    else:
        return get_recbole_dataloader(config, phase)


def _get_SVD_GCN_dataloader(config, phase: Literal["train", "valid", "test", "evaluation"]):
    """Customized function for SVD_GCN to get correct dataloader class.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take 4 values: 'train', 'valid', 'test' or 'evaluation'.
            Notes: 'evaluation' has been deprecated, please use 'valid' or 'test' instead.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase not in ["train", "valid", "test", "evaluation"]:
        raise ValueError(
            "`phase` can only be 'train', 'valid', 'test' or 'evaluation'."
        )
    if phase == "evaluation":
        phase = "test"
        warnings.warn(
            "'evaluation' has been deprecated, please use 'valid' or 'test' instead.",
            DeprecationWarning,
        )

    if phase == "train":
        return SVD_GCNDataLoader
    else:
        eval_mode = config["eval_args"]["mode"][phase]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_submodule = [
        'general_recommender'
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['fa4gcf.model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        model_class = get_recbole_model(model_name)
    else:
        model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name
    Args:
        model_type (ModelType): model type
        model_name (str): model name
    Returns:
        Trainer: trainer class
    """
    try:
        return getattr(importlib.import_module('fa4gcf.trainer.trainer'), model_name + 'Trainer')
    except AttributeError:
        return get_recbole_trainer(model_type, model_name)


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
                                       edge_index: Union[torch.Tensor, SparseTensor],
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
