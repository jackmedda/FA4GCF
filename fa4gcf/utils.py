import os
import copy
import yaml
import warnings
import importlib
from logging import getLogger
from typing import Literal

import torch
import numpy as np
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

from gnnuers.data import Interaction

from fa4gcf.data import Dataset
from fa4gcf.data.custom_dataloader import SVD_GCNDataLoader


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
    valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
    test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)

    return train_data, valid_data, test_data


def load_data_and_model(model_file, explainer_config=None, cmd_config_args=None, perturbed_dataset=None):
    r"""Load filtered dataset, split dataloaders and saved model.
    Args:
        model_file (str): The path of saved model file.
    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']

    if explainer_config is not None:
        if isinstance(explainer_config, str):
            with open(explainer_config, 'r', encoding='utf-8') as f:
                explain_config_dict = yaml.load(f.read(), Loader=config.yaml_loader)
        elif isinstance(explainer_config, dict):
            explain_config_dict = explainer_config
        else:
            raise ValueError(f'explainer_config cannot be `{type(explainer_config)}`. Only `str` and `dict` are supported')

        config.final_config_dict.update(explain_config_dict)

    if cmd_config_args is not None:
        for arg, val in cmd_config_args.items():
            conf = config
            if '.' in arg:
                subargs = arg.split('.')
                for subarg in subargs[:-1]:
                    conf = conf[subarg]
                arg = subargs[-1]

            if conf[arg] is None:
                try:
                    new_val = float(val)
                    new_val = int(new_val) if new_val.is_integer() else new_val
                except ValueError:
                    new_val = int(val) if val.isdigit() else val
                conf[arg] = new_val
            else:
                try:
                    arg_type = type(conf[arg])
                    if arg_type == bool:
                        new_val = val.title() == 'True'
                    else:
                        new_val = arg_type(val)  # cast to same type in config
                    conf[arg] = new_val
                except (ValueError, TypeError):
                    warnings.warn(f"arg [{arg}] taken from cmd not valid for explainer config file")

    config['data_path'] = config['data_path'].replace('\\', os.sep)
    # config['device'] = torch.device('cuda')

    logger = getLogger()
    logger.info(config)

    if perturbed_dataset is not None:
        dataset = perturbed_dataset
    else:
        dataset = Dataset(config)
        default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-{dataset.__class__.__name__}.pth')
        file = config['dataset_save_path'] or default_file
        if os.path.exists(file):
            with open(file, 'rb') as f:
                dataset = pickle.load(f)
            dataset_args_unchanged = True
            for arg in dataset_arguments + ['seed', 'repeatable']:
                if config[arg] != dataset.config[arg]:
                    dataset_args_unchanged = False
                    break
            if dataset_args_unchanged:
                logger.info(set_color('Load filtered dataset from', 'pink') + f': [{file}]')

        if config['save_dataset']:
            dataset.save()

    logger.info(dataset)

    config["train_neg_sample_args"]['sample_num'] = 1  # deactivate negative sampling
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data


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
        return getattr(importlib.import_module("fa4gcf.trainer"), "Trainer")
