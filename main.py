import os
import yaml
import shutil
import pickle
import inspect
import argparse
import logging

import numpy as np
import pandas as pd
from recbole.config import Config
from recbole.trainer import HyperTuning
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color, get_local_time

from gnnuers.models.ngcf import NGCF
from gnnuers.data import PerturbedDataset
from gnnuers.explain import execute_explanation
import gnnuers.utils as utils


def training(_config, saved=True, model_file=None, hyper=False, perturbed_dataset=None):
    logger = logging.getLogger() if not hyper else None

    if model_file is not None and perturbed_dataset is None:
        _config, _model, _dataset, train_data, valid_data, test_data = utils.load_data_and_model(
            model_file,
            perturbed_dataset=perturbed_dataset
        )
    else:
        # dataset filtering
        _dataset = perturbed_dataset if perturbed_dataset is not None else create_dataset(_config)

        # dataset splitting
        train_data, valid_data, test_data = data_preparation(_config, _dataset)

        # model loading and initialization
        if _config['model'].upper() == 'ngcf':
            _model = NGCF()(_config, train_data.dataset).to(_config['device'])
        else:
            _model = get_model(_config['model'])(_config, train_data.dataset).to(_config['device'])

        if not hyper:
            logger.info(_config)
            logger.info(_dataset)
            logger.info(_model)

    # trainer loading and initialization
    trainer = get_trainer(_config['MODEL_TYPE'], _config['model'])(_config, _model)
    if perturbed_dataset is not None:
        explanations_path = perturbed_dataset.explanations_path
        perturbed_suffix = "_PERTURBED"
        split_saved_file = os.path.basename(trainer.saved_model_file).split('-')
        perturbed_model_path = os.path.join(
            explanations_path,
            '-'.join(
                split_saved_file[:1] + [_dataset.dataset_name.upper()] + split_saved_file[1:]
            ).replace('.pth', '') + perturbed_suffix + '.pth'
        )

        resume_perturbed_training = False
        for f in os.scandir(explanations_path):
            if _config['model'].lower() in f.name.lower() and \
               _config['dataset'].lower() in f.name.lower() and \
               perturbed_suffix in f.name:
                perturbed_model_path = f.path
                resume_perturbed_training = True
                break

        trainer.saved_model_file = perturbed_model_path
        if resume_perturbed_training:
            trainer.resume_checkpoint(perturbed_model_path)
    elif model_file is not None:
        trainer.resume_checkpoint(model_file)
    else:
        split_saved_file = os.path.basename(trainer.saved_model_file).split('-')
        trainer.saved_model_file = os.path.join(
            os.path.dirname(trainer.saved_model_file),
            '-'.join(split_saved_file[:1] + [_dataset.dataset_name.upper()] + split_saved_file[1:])
        )

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=saved,
        show_progress=_config['show_progress'] and not hyper,
        verbose=not hyper
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data,
        load_best_model=saved,
        show_progress=_config['show_progress'] and not hyper
    )

    if not hyper:
        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': _config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def recbole_hyper(base_config, params_file, config_file_list, saved=True):
    def objective_function(c_dict, c_file_list):
        config = Config(
            model=base_config['model'],
            dataset=base_config['dataset'],
            config_file_list=c_file_list,
            config_dict=c_dict
        )
        config['data_path'] = os.path.join(base_config.file_config_dict['data_path'], base_config.dataset)
        init_seed(base_config['seed'], config['reproducibility'])
        logging.basicConfig(level=logging.ERROR)

        return training(config, saved=False, hyper=True)

    hp = HyperTuning(
        objective_function, algo='exhaustive', params_file=params_file, fixed_config_file_list=config_file_list
    )
    hp.run()

    output_path = os.path.join(base_config['checkpoint_dir'], 'hyper', base_config['dataset'], base_config['model'])
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{get_local_time()}.txt")

    hp.export_result(output_file=output_file)
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])

    with open(output_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(
            'Best Params and Results\n' +
            str(hp.best_params).rstrip('\r\n') + '\n' +
            str(hp.params2result[hp.params2str(hp.best_params)]) + '\n\n' + content
        )


def main(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True, seed=None, hyper_params_file=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset
    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    config['data_path'] = os.path.join(config.file_config_dict['data_path'], config.dataset)
    seed = seed or config['seed']
    init_seed(seed, config['reproducibility'])
    import torch; torch.use_deterministic_algorithms(True)

    # logger initialization
    init_logger(config)
    logger = logging.getLogger()

    # if args.run == 'evaluate_perturbed' or args.run == 'graph_stats':
    #     orig_config, orig_model, orig_dataset, orig_train_data, orig_valid_data, orig_test_data = \
    #         utils.load_data_and_model(args.original_model_file, args.explainer_config_file)

    if args.use_perturbed_graph:
        perturbed_dataset = PerturbedDataset(config, args.explanations_path, args.best_exp)
        if args.run == 'train':
            training(config, saved=saved, model_file=args.model_file, perturbed_dataset=perturbed_dataset)
            # elif args.run == 'explain':
            #     runner(*explain_args)
            # elif args.run == 'evaluate_perturbed':
            #     logger.info("EVALUATE PERTURBED MODEL")
            #     _, pert_model, pert_dataset, _, _, _ = utils.load_data_and_model(args.model_file,
            #                                                                      args.explainer_config_file)
            #     runner(
            #         orig_config,
            #         orig_model,
            #         pert_model,
            #         orig_dataset,
            #         pert_dataset,
            #         orig_train_data,
            #         orig_test_data,
            #         topk=args.topk,
            #         perturbed_model_file=os.path.splitext(os.path.basename(args.model_file))[0]
            #     )
            # elif args.run == 'graph_stats':
            #     pert_config, _, _, pert_train_data, _, _ = utils.load_data_and_model(args.model_file,
            #                                                                          args.explainer_config_file)
            #     runner(
            #         pert_config,
            #         orig_train_data,
            #         orig_valid_data,
            #         orig_test_data,
            #         pert_train_data,
            #         args.original_model_file,
            #         sens_attr,
            #         c_id,
            #         *args.best_exp
            #     )
    else:
        dataset = create_dataset(config)
        logger.info(dataset)

        if args.run == 'train':
            training(config, saved=saved, model_file=args.model_file)
        elif args.run == 'explain':
            execute_explanation(args.model_file, *explain_args)
        elif args.run == 'recbole_hyper':
            config['seed'] = seed
            recbole_hyper(config, hyper_params_file, config_file_list, saved=saved)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    perturbed_train_group = parser.add_argument_group(
        "perturbed_train",
        "All the arguments related to training with augmented data"
    )
    explain_group = parser.add_argument_group(
        "explain",
        "All the arguments related to create explanations"
    )
    recbole_hyper_group = parser.add_argument_group(
        "recole_hyper",
        "All the arguments related to run the hyperparameter optimization on the recbole models for training"
    )

    parser.add_argument('--run', default='train', choices=['train', 'explain', 'recbole_hyper'], required=True)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--model', default='GCMC')
    parser.add_argument('--dataset', default='ml-100k')
    parser.add_argument('--config_file_list', nargs='+', default=None)
    parser.add_argument('--config_dict', default=None)
    parser.add_argument('--model_file', default=None)
    explain_group.add_argument('--base_explainer_config_file', default=os.path.join("config", "base_explainer.yaml"))
    explain_group.add_argument('--explainer_config_file', default=os.path.join("config", "explainer.yaml"))
    # explain_group.add_argument('--load', action='store_true')
    explain_group.add_argument('--explain_config_id', default=-1)
    explain_group.add_argument('--verbose', action='store_true')
    explain_group.add_argument('--wandb_online', action='store_true')
    explain_group.add_argument('--hyper_optimize', action='store_true')
    explain_group.add_argument('--overwrite', action='store_true')
    perturbed_train_group.add_argument('--use_perturbed_graph', action='store_true')
    perturbed_train_group.add_argument('--best_exp', nargs="*",
                                       help="one of ['fairest', 'fairest_before_exp', 'fixed_exp'] with "
                                            "the chosen exp number for the last two types")
    perturbed_train_group.add_argument('--explanations_path', default=None)
    recbole_hyper_group.add_argument('--params_file', default=None)

    args, unk_args = parser.parse_known_args()
    print(args)

    unk_args[::2] = map(lambda s: s.replace('-', ''), unk_args[::2])
    unk_args = dict(zip(unk_args[::2], unk_args[1::2]))
    print("Unknown args", unk_args)

    if args.hyper_optimize and not args.verbose:
        from tqdm import tqdm
        from functools import partialmethod

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    if args.use_perturbed_graph:
        if args.explainer_config_file:
            # TODO: check that the path has the experiments folder setup
            pass

    args.wandb_online = {False: "offline", True: "online"}[args.wandb_online]
    explain_args = [
        args.base_explainer_config_file,
        args.explainer_config_file,
        args.explain_config_id,
        args.verbose,
        args.wandb_online,
        unk_args,
        args.hyper_optimize,
        args.overwrite
    ]

    if args.run not in ['train', 'explain', 'recbole_hyper']:
        raise NotImplementedError(f"The run `{args.run}` is not supported.")

    main(
        args.model,
        args.dataset,
        args.config_file_list,
        args.config_dict,
        seed=args.seed,
        hyper_params_file=args.params_file
    )
