import os
import pickle
import yaml
import importlib
import functools
from typing import Literal

import wandb
import numba
from recbole.sampler import KGSampler
from recbole.data.dataloader import *
from recbole.trainer import (
    Trainer as RecboleTrainer,
    TraditionalTrainer as RecboleTraditionalTrainer
)
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
from recbole.utils.argument_list import dataset_arguments

from fa4gcf.data import Dataset
from fa4gcf.data.custom_dataloader import *


def load_data_and_model(model_file, perturbation_config_file=None, cmd_config_args=None, perturbed_dataset=None):
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

    if perturbation_config_file is not None:
        if isinstance(perturbation_config_file, str):
            perturbation_config_dict = config.update_base_perturb_data(perturbation_config_file)
        elif isinstance(perturbation_config_file, dict):
            perturbation_config_dict = perturbation_config_file
        else:
            raise ValueError(
                f'perturbation_config cannot be `{type(perturbation_config_file)}`. Only `str` and `dict` are supported'
            )

        config.final_config_dict.update(perturbation_config_dict)

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
                    warnings.warn(f"arg [{arg}] taken from cmd not valid for perturbation config file")

    config['data_path'] = config['data_path'].replace('\\', os.sep)

    logger = getLogger('FA4GCF')
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

    logger = getLogger('FA4GCF')
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
        "SVD_GCN": functools.partial(_get_custom_train_dataloader, train_dataloader=SVD_GCNDataLoader),
        "AutoCF": functools.partial(_get_custom_train_dataloader, train_dataloader=AutoCFDataLoader)
    }

    if config["model"] in register_table:
        return register_table[config["model"]](config, phase)
    else:
        return get_recbole_dataloader(config, phase)


def _get_custom_train_dataloader(config,
                                 phase: Literal["train", "valid", "test", "evaluation"],
                                 train_dataloader: TrainDataLoader = None):
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
        return train_dataloader
    else:
        eval_mode = config["eval_args"]["mode"][phase]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def _get_model_module(model_file_name):
    model_submodule = [
        'general_recommender'
    ]

    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['fa4gcf.model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    return model_module


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_file_name = model_name.lower()
    model_module = _get_model_module(model_file_name)

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
        return getattr(importlib.import_module('fa4gcf.trainer'), model_name + 'Trainer')
    except AttributeError:
        if model_type == ModelType.TRADITIONAL:
            return getattr(
                importlib.import_module("fa4gcf.trainer"), "TraditionalTrainer"
            )
        else:
            loaded_recbole_trainer = get_recbole_trainer(model_type, model_name)
            if loaded_recbole_trainer is RecboleTrainer:
                return getattr(importlib.import_module("fa4gcf.trainer"), "Trainer")
            elif loaded_recbole_trainer is RecboleTraditionalTrainer:
                return getattr(importlib.import_module("fa4gcf.trainer"), "TraditionalTrainer")
            elif loaded_recbole_trainer.__name__.lower().startswith(model_name.lower()):
                # Dynamically replaces FA4GCF Trainer as superclass instead of Recbole Trainer
                loaded_recbole_trainer.__bases__ = (getattr(importlib.import_module("fa4gcf.trainer"), "Trainer"),)
                return loaded_recbole_trainer
            else:
                # Other Trainer types (KG, context-aware) are not supported in FA4GCF
                warnings.warn(
                    f"[{model_name}] uses a Trainer not supported in FA4GCF. Successful execution not guaranteed.",
                    RuntimeWarning
                )
                return loaded_recbole_trainer


def wandb_init(config, policies=None, **kwargs):
    config = config.final_config_dict if not isinstance(config, dict) else config

    tags = None
    policies = config.get("perturbation_policies", policies) or config.get("explainer_policies", policies)
    if policies is not None:
        tags = [k for k in policies if policies[k]]
    config['wandb_tags'] = tags

    return wandb.init(
        **kwargs,
        tags=tags,
        config=config
    )


def read_recbole_config_skip_errors(filepath, config):
    with open(filepath, 'r') as yaml_path:
        def construct_undefined(self, node):
            if isinstance(node, yaml.nodes.ScalarNode):
                value = self.construct_scalar(node)
            elif isinstance(node, yaml.nodes.SequenceNode):
                value = self.construct_sequence(node)
            elif isinstance(node, yaml.nodes.MappingNode):
                value = self.construct_mapping(node)
            else:
                assert False, f"unexpected node: {node!r}"
            return {node.__str__(): value}

        config.yaml_loader.add_constructor(None, construct_undefined)
        return yaml.load(yaml_path.read(), Loader=config.yaml_loader)


def damerau_levenshtein_distance(s1, s2):
    import numpy as np

    s1 = [s1] if np.ndim(s1) == 1 else s1
    s2 = [s2] if np.ndim(s2) == 1 else s2

    out = np.zeros((len(s1, )), dtype=int)
    for i, (_s1, _s2) in enumerate(zip(s1, s2)):
        try:
            numb_s1, numb_s2 = numba.typed.List(_s1), numba.typed.List(_s2)
        except TypeError:  # python < 3.8
            numb_s1, numb_s2 = numba.typed.List(), numba.typed.List()
            for el in _s1:
                numb_s1.append(el)
            for el in _s2:
                numb_s2.append(el)

        out[i] = _damerau_levenshtein_distance(numb_s1, numb_s2)

    return out.item() if out.shape == (1,) else out


@numba.jit(nopython=True)
def _damerau_levenshtein_distance(s1, s2):
    """
    Copyright (c) 2015, James Turk
    https://github.com/jamesturk/jellyfish/blob/main/jellyfish/_jellyfish.py
    """

    # _check_type(s1)
    # _check_type(s2)

    len1 = len(s1)
    len2 = len(s2)
    infinite = len1 + len2

    # character array
    da = {}

    # distance matrix
    score = [[0] * (len2 + 2) for _ in range(len1 + 2)]

    score[0][0] = infinite
    for i in range(0, len1 + 1):
        score[i + 1][0] = infinite
        score[i + 1][1] = i
    for i in range(0, len2 + 1):
        score[0][i + 1] = infinite
        score[1][i + 1] = i

    for i in range(1, len1 + 1):
        db = 0
        for j in range(1, len2 + 1):
            i1 = da[s2[j - 1]] if s2[j - 1] in da else 0
            j1 = db
            cost = 1
            if s1[i - 1] == s2[j - 1]:
                cost = 0
                db = j

            score[i + 1][j + 1] = min(
                score[i][j] + cost,
                score[i + 1][j] + 1,
                score[i][j + 1] + 1,
                score[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
            )
        da[s1[i - 1]] = i

    return score[len1 + 1][len2 + 1]
