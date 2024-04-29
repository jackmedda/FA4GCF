import os
import re
import yaml
import argparse
import logging

import torch
from recbole.utils import init_logger, init_seed, set_color, get_local_time

import fa4gcf.utils as utils
from fa4gcf.config import Config
from fa4gcf.model.utils import is_model_saveable
from fa4gcf.data import Dataset, PerturbedDataset
from fa4gcf.trainer import HyperTuning
from perturb import execute_perturbation


def training(model,
             dataset,
             config_file_list,
             config_dict,
             saved=True,
             model_file=None,
             hyper=False,
             **kwargs):

    # configurations initialization
    _config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(_config['seed'], _config['reproducibility'])

    # logger initialization
    init_logger(_config)
    if hyper:
        logging.basicConfig(level=logging.ERROR)
        _config['data_path'] = os.path.join(_config.file_config_dict['data_path'], dataset)
        _config['train_neg_sample_args']['sample_num'] = 1  # hyper-optimization is performed with 1 negative sample
    logger = logging.getLogger('FA4GCF')

    use_perturbed_graph = kwargs.get('use_perturbed_graph', False)
    if model_file is not None and not use_perturbed_graph:
        loaded_config, _model, _dataset, train_data, valid_data, test_data = utils.load_data_and_model(
            model_file,
            perturbation_config=_config.final_config_dict,  # not used for perturbation, but to update params
            perturbed_dataset=False
        )
        loaded_config.final_config_dict.update(_config.final_config_dict)
    else:
        # dataset filtering
        if use_perturbed_graph:
            perturbations_path = kwargs.get('perturbations_path')
            best_perturbation = kwargs.get('best_perturbation')

            logger.info(
                f"Training with perturbed graph with {best_perturbation} pert "
                f"from perturbations_path: {perturbations_path}"
            )
            _dataset = PerturbedDataset(_config, perturbations_path, best_perturbation)
        else:
            _dataset = Dataset(_config)

        # dataset splitting
        train_data, valid_data, test_data = utils.data_preparation(_config, _dataset)

        # model loading and initialization
        _model = utils.get_model(_config['model'])(_config, train_data.dataset).to(_config['device'])

        logger.info(_config)
        logger.info(_dataset)
        logger.info(_model)

    # trainer loading and initialization
    trainer = utils.get_trainer(_config['MODEL_TYPE'], _config['model'])(_config, _model)
    if use_perturbed_graph:
        perturbations_path = _dataset.perturbations_path
        perturbed_suffix = "_PERTURBED"
        split_saved_file = os.path.basename(trainer.saved_model_file).split('-')
        perturbed_model_path = os.path.join(
            perturbations_path,
            '-'.join(
                split_saved_file[:1] + [_dataset.dataset_name.upper()] + split_saved_file[1:]
            ).replace('.pth', '') + perturbed_suffix + '.pth'
        )

        resume_perturbed_training = False
        for f in os.scandir(perturbations_path):
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
        saved=saved and is_model_saveable(_config, trainer),
        show_progress=_config['show_progress'] and not hyper,
        verbose=not hyper
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data,
        load_best_model=saved and is_model_saveable(_config, trainer),
        show_progress=_config['show_progress'] and not hyper
    )

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    result = {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': _config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

    if not _config["single_spec"]:
        torch.distributed.destroy_process_group()

    queue = kwargs.get('queue')
    if _config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result


def recbole_hyper(model, dataset, config_file_list, config_dict, params_file):
    # configurations initialization
    base_config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(base_config['seed'], base_config['reproducibility'])

    # logger initialization
    init_logger(base_config)

    model_name = base_config['model']
    if model_name.lower() == 'svd_gcn':
        parametric = base_config['parametric'] if base_config['parametric'] is not None else True
        if not parametric:
            model_name += '_S'

    def objective_function(c_dict, c_file_list):
        if model_name.startswith('svd_gcn'):
            c_dict['parametric'] = base_config['parametric'] if base_config['parametric'] is not None else True

        import traceback
        try:
            train_result = training(
                base_config['model'], dataset, c_file_list, c_dict, saved=False, hyper=True
            )
        except Exception as e:
            print(e)
            print(traceback.format_exc())
        train_result['model'] = model_name
        return train_result

    hp = HyperTuning(
        objective_function,
        model_name,
        algo='exhaustive',
        params_file=params_file,
        fixed_config_file_list=config_file_list,
        early_stop=10,
        ignore_errors=True
    )
    hp.run()

    output_path = os.path.join(base_config['checkpoint_dir'], 'hyper', dataset, model_name)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{get_local_time()}.txt")

    hp.export_result(output_file=output_file)
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])

    with open(output_file, 'r+') as f:
        from pprint import pprint
        content = f.read()
        f.seek(0, 0)
        f.write(
            'Best Params and Results\n' +
            str(hp.best_params).rstrip('\r\n') + '\n'
        )
        pprint(hp.params2result[hp.params2str(hp.best_params)], stream=f)
        f.write('\n\n' + content)

    return hp.params2result[hp.params2str(hp.best_params)]


def main(run,
         model=None,
         dataset=None,
         config_file_list=None,
         config_dict=None,
         hyper_params_file=None,
         queue=None,
         **kwargs):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset
    Args:
        run (str): Choices ['train', 'perturb', 'recbole_hyper']
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        hyper_params_file:
        queue: used internally for torch multiprocessing
    """

    if run == 'train':
        return training(model, dataset, config_file_list, config_dict, queue=queue)
    elif run == 'perturb':
        torch.use_deterministic_algorithms(True)
        model_file = kwargs.get('model_file')
        return execute_perturbation(model_file, *perturb_args)
    elif run == 'recbole_hyper':
        return recbole_hyper(model, dataset, config_file_list, config_dict, hyper_params_file)


def mp_main(rank, *run_args):
    kwargs = run_args[-1]
    config_dict = run_args[-2]
    config_dict['local_rank'] = rank

    main(
        *run_args[:-1],
        **kwargs,
    )


def run_process(*run_args, **kwargs):
    if (args.nproc == 1 and args.world_size <= 0) or args.run not in ['train', 'perturb']:
        res = main(args.run, *run_args, **kwargs)
    else:
        if args.world_size == -1:
            args.world_size = args.nproc
        import torch.multiprocessing as mp

        # Refer to https://discuss.pytorch.org/t/problems-with-torch-multiprocess-spawn-and-simplequeue/69674/2
        # https://discuss.pytorch.org/t/return-from-mp-spawn/94302/2
        queue = mp.get_context("spawn").SimpleQueue()

        config_dict = run_args[-1]
        config_dict.update(
            {
                "world_size": args.world_size,
                "ip": args.ip,
                "port": args.port,
                "nproc": args.nproc,
                "offset": args.group_offset,
            }
        )
        kwargs["queue"] = queue

        mp.spawn(
            mp_main,
            args=(args.run, *run_args, kwargs),
            nprocs=args.nproc,
            join=True,
        )

        # Normally, there should be only one item in the queue
        res = None if queue.empty() else queue.get()
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    perturbed_train_group = parser.add_argument_group(
        "perturbed_train",
        "All the arguments related to training with augmented data"
    )
    perturb_group = parser.add_argument_group(
        "perturb",
        "All the arguments related to create perturbations"
    )
    recbole_hyper_group = parser.add_argument_group(
        "recole_hyper",
        "All the arguments related to run the hyperparameter optimization on the recbole models for training"
    )

    parser.add_argument('--run', default='train', choices=['train', 'perturb', 'recbole_hyper'], required=True)
    parser.add_argument('--model', default='GCMC')
    parser.add_argument('--dataset', default='ml-100k')
    parser.add_argument('--config_file_list', nargs='+', default=None)
    parser.add_argument('--model_file', default=None)
    parser.add_argument('--use_best_params', action='store_true')
    parser.add_argument("--nproc", type=int, default=1, help="the number of process in this group")
    parser.add_argument("--ip", type=str, default="localhost", help="the ip of master node")
    parser.add_argument("--port", type=str, default="5678", help="the port of master node")
    parser.add_argument("--world_size", type=int, default=-1, help="total number of jobs")
    parser.add_argument("--group_offset", type=int, default=0, help="the global rank offset of this group")
    perturb_group.add_argument('--perturbation_file', default=None)
    perturb_group.add_argument('--perturbation_id', default=-1)
    perturb_group.add_argument('--verbose', action='store_true')
    perturb_group.add_argument('--wandb_online', action='store_true')
    perturb_group.add_argument('--hyper_optimize', action='store_true')
    perturb_group.add_argument('--overwrite', action='store_true')
    perturbed_train_group.add_argument('--use_perturbed_graph', action='store_true')
    perturbed_train_group.add_argument('--best_pert', nargs="*",
                                       help="one of ['fairest', 'fairest_before_pert', 'fixed_pert'] with "
                                            "the chosen pert number for the last two types")
    perturbed_train_group.add_argument('--perturbations_path', default=None)
    recbole_hyper_group.add_argument('--params_file', default=None)

    args, parsed_unk_args = parser.parse_known_args()
    print(args)
    conf_dict = {}

    # workaround for NumPy deprecation error in Recbole code
    import numpy as np
    np.float = float

    unk_args = parsed_unk_args[:]
    unk_args[::2] = map(lambda s: s.replace('-', ''), unk_args[::2])
    unk_args = dict(zip(unk_args[::2], unk_args[1::2]))
    print("Unknown args", unk_args)

    logging.getLogger('FA4GCF').setLevel('DEBUG')

    if args.hyper_optimize and not args.verbose:
        from tqdm import tqdm
        from functools import partialmethod

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    current_file = os.path.dirname(os.path.realpath(__file__))

    base_overall_config = os.path.join(current_file, "config", "base_config.yaml")
    if os.path.isfile(base_overall_config):
        if args.config_file_list is None:
            args.config_file_list = [base_overall_config]
        else:
            args.config_file_list = [base_overall_config] + args.config_file_list

    # it handles the convention of the min_interactions at the end of the dataset name when a dataset is pre-processed
    if re.search(r'_\d+$', args.dataset) is not None:
        dataset_name = re.sub(r'_\d+$', '', args.dataset)
    else:
        dataset_name = args.dataset

    all_dataset_configs = os.path.join(current_file, "config", "dataset")
    dataset_config = os.path.join(all_dataset_configs, f"{dataset_name.lower()}.yaml")
    if os.path.isfile(dataset_config):
        if args.config_file_list is None:
            args.config_file_list = [dataset_config]
        else:
            args.config_file_list.append(dataset_config)

    all_model_configs = os.path.join(current_file, "config", "model")
    model_config = os.path.join(all_model_configs, f"{args.model}.yaml")
    if os.path.isfile(model_config):
        args.config_file_list.append(model_config)

    if args.run == "perturb":
        if args.perturbation_file is None:
            all_perturbation_configs = os.path.join(current_file, "config", "perturbation")
            perturbation_config = os.path.join(all_perturbation_configs, f"{dataset_name.lower()}_perturbation.yaml")

            if os.path.isfile(perturbation_config):
                args.perturbation_file = perturbation_config

        if args.model_file is None:
            saved_models_path = os.path.join(current_file, "saved")
            maybe_model_file = [
                f for f in os.listdir(saved_models_path)
                if dataset_name.lower() in f.lower()
                and args.model.lower() in f.lower()
                and f.endswith('.pth')
            ]
            if len(maybe_model_file) == 1:
                args.model_file = os.path.join(saved_models_path, maybe_model_file[0])
            else:
                raise FileNotFoundError(
                    f'`model_file` is None and no unique {args.model} trained on {dataset_name} found'
                )

    if args.run == "train":
        if args.use_best_params:
            model_best_config = os.path.join(all_model_configs, f"{args.model}_best.yaml")
            if os.path.isfile(model_best_config):
                with open(model_best_config, 'r') as best_conf_file:
                    best_conf_dict = yaml.load(best_conf_file, Loader=Config._build_yaml_loader())
                if args.model.lower() == "svd_gcn":
                    # handles best_params for SVD_GCN or SVD_GCN_S
                    parametric = None
                    for conf_arg in args.config_file_list:
                        with open(conf_arg, 'r') as conf_arg_file:
                            conf_arg_dict = yaml.load(conf_arg_file, Loader=Config._build_yaml_loader())
                        if 'parametric' in conf_arg_dict:
                            parametric = conf_arg_dict['parametric']

                    cmd_line_parametric = [cmd for cmd in parsed_unk_args if '--parametric=' in cmd]
                    if cmd_line_parametric:
                        parametric = cmd_line_parametric[0].split('=')[1] == 'True'
                    if parametric is None:
                        raise ValueError("`parametric` was not set for SVD_GCN using best_params")

                    best_conf_dict = best_conf_dict[args.dataset.lower()][parametric]
                else:
                    best_conf_dict = best_conf_dict[args.dataset.lower()]
                conf_dict.update(best_conf_dict)

        if args.use_perturbed_graph:
            if args.perturbation_file:
                # TODO: check that the path has the experiments folder setup
                pass

    args.wandb_online = {False: "offline", True: "online"}[args.wandb_online]
    perturb_args = [
        args.perturbation_file,
        args.perturbation_id,
        args.verbose,
        args.wandb_online,
        unk_args,
        args.hyper_optimize,
        args.overwrite
    ]

    run_process(
        args.model,
        args.dataset,
        args.config_file_list,
        conf_dict,
        hyper_params_file=args.params_file,
        model_file=args.model_file,
        perturbations_path=args.perturbations_path,
        best_perturbation=args.best_pert
    )
