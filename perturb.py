import os
import json
import yaml
import pickle
import logging
import inspect

import torch
import optuna
import pandas as pd

from recbole.data.dataloader import FullSortEvalDataLoader

import fa4gcf.utils as utils
from fa4gcf.trainer import BeyondAccuracyPerturbationTrainer


script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))


def get_base_perts_filepath(config,
                            config_id=-1,
                            model_name=None,
                            model_file="",
                            hyper=False):
    """
    return the filepath where perturbations are saved
    :param config:
    :param config_id:
    :param model_name:
    :param model_file:
    :return:
    """
    epochs = config["cf_epochs"]
    model_name = model_name or config["model"]
    perturber = BeyondAccuracyPerturbationTrainer.__name__
    pert_type = "dp_perturbations" if not hyper else "hyperoptimization"
    base_perts_file = os.path.join(script_path, "experiments", pert_type, config["dataset"], model_name, perturber)

    pert_metadata = config["sensitive_attribute"].lower() if 'consumer' in config["pert_metric"].lower() else ''
    pert_loss = config["pert_metric"] + "_loss"
    base_perts_file = os.path.join(base_perts_file, pert_loss, pert_metadata, f"epochs_{epochs}")

    if os.path.exists(base_perts_file):
        if config_id == -1:
            paths_c_ids = sorted(filter(str.isdigit, os.listdir(base_perts_file)), key=int)
            if len(paths_c_ids) == 0:
                config_id = 1
            else:
                int_paths_c_ids = list(map(int, paths_c_ids))
                candidates = set(range(1, max(int_paths_c_ids) + 1)) - set(int_paths_c_ids)
                config_id = str(min(candidates, default=max(int_paths_c_ids) + 1))

            for path_c in paths_c_ids:
                config_path = os.path.join(base_perts_file, path_c, "config.pkl")
                if os.path.exists(config_path):
                    with open(config_path, 'rb') as f:
                        _c = pickle.load(f)

                    if config.final_config_dict == _c.final_config_dict:
                        if model_file != "" and "perturbed" in model_file:
                            check_perturb = input("The perturbations of the perturbed graph could overwrite the "
                                                  "perturbations from which the perturbed graph was generated. Type "
                                                  "y/yes to confirm this outcome. Other inputs will assign a new id: ")
                            if check_perturb.lower() != "y" and check_perturb.lower() != "yes":
                                continue
                        config_id = os.path.join(base_perts_file, str(path_c))
                        break
                elif hyper:
                    config_id = os.path.join(base_perts_file, str(path_c))
                    break

        base_perts_file = os.path.join(base_perts_file, str(config_id))
    else:
        base_perts_file = os.path.join(base_perts_file, "1")

    if not os.path.exists(base_perts_file):
        os.makedirs(base_perts_file)

    with open(os.path.join(base_perts_file, "config.yaml"), 'w') as pert_file:
        pert_file.write(yaml.dump(config.final_config_dict, default_flow_style=False))

    return base_perts_file


def perturb(config, model, _rec_data, _full_dataset, _train_data, _valid_data, _test_data, base_perts_file, **kwargs):
    """
    Function that starts the perturbation process, that is generates perturbed graphs.
    :param config:
    :param model:
    :param _train_dataset:
    :param _rec_data:
    :param _test_data:
    :param base_perts_file:
    :param kwargs:
    :return:
    """
    epochs = config['cf_epochs']
    wandb_mode = kwargs.get("wandb_mode", "disabled")
    overwrite = kwargs.get("overwrite", False)

    perts_filename = os.path.join(base_perts_file, f"cf_data.pkl")
    users_order_file = os.path.join(base_perts_file, f"users_order.pkl")
    model_preds_file = os.path.join(base_perts_file, f"model_rec_test_preds.pkl")
    checkpoint_path = os.path.join(base_perts_file, "checkpoint.pth")

    sh = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(base_perts_file, "perturbation_trainer.log"), encoding='utf-8')
    sh.setLevel(logging.DEBUG)
    fh.setLevel(logging.DEBUG)
    logger = logging.getLogger('FA4GCF')
    logger.addHandler(sh)
    logger.addHandler(fh)

    if overwrite and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    user_source = _rec_data if _rec_data is not None else _test_data
    user_data = user_source.user_df[user_source.uid_field][torch.randperm(user_source.user_df.length)]

    with open(os.path.join(base_perts_file, "config.pkl"), 'wb') as config_file:
        pickle.dump(config, config_file)

    utils.wandb_init(
        config,
        **wandb_env_data,
        name="Perturbation",
        job_type="train",
        group=f"{model.__class__.__name__}_{config['dataset']}_{config['sensitive_attribute'].title()}_epochs{config['cf_epochs']}_pert{os.path.basename(base_perts_file)}",
        mode=wandb_mode
    )
    # wandb.config.update({"pert": os.path.basename(base_perts_file)})

    logger.info(config)
    perturbation_trainer = BeyondAccuracyPerturbationTrainer(
        config,
        _train_data.dataset,
        _rec_data,
        model,
        dist=config['cf_dist'],
        **kwargs
    )
    perturbation_trainer.set_checkpoint_path(checkpoint_path)
    logger.info(f"Rec Evaluation data for optimization of {BeyondAccuracyPerturbationTrainer.__name__}")
    logger.info(_rec_data.dataset)

    pert, users_order, model_preds = perturbation_trainer.perturb(
        user_data,
        _full_dataset,
        _train_data,
        _valid_data,
        _test_data,
        epochs
    )

    with open(perts_filename, 'wb') as f:
        pickle.dump(pert, f)
    with open(users_order_file, 'wb') as f:
        pickle.dump(users_order, f)
    with open(model_preds_file, 'wb') as f:
        pickle.dump(model_preds, f)

    logger.info(f"Saved perturbations at path {base_perts_file}")


def optimize_perturbation(config, model, _train_dataset, _rec_data, _test_data, base_perts_file, **kwargs):
    """
    Function that perturbs, that is generates perturbed graphs.
    :param config:
    :param model:
    :param _train_dataset:
    :param _rec_data:
    :param _test_data:
    :param base_perts_file:
    :param kwargs:
    :return:
    """
    epochs = config['cf_epochs']
    topk = config['cf_topk']
    wandb_mode = kwargs.get("wandb_mode", "disabled")

    user_source = _rec_data if _rec_data is not None else _test_data
    user_data = user_source.user_df[user_source.uid_field][torch.randperm(user_source.user_df.length)]

    pert_token = f"{model.__class__.__name__}_" + \
                 f"{config['dataset']}_" + \
                 f"{config['sensitive_attribute'].title()}_" + \
                 f"epochs{config['cf_epochs']}_" + \
                 f"pert{os.path.basename(base_perts_file)}"

    def objective(trial):
        wandb_config_keys = [
            'cf_learning_rate',
            'user_batch_pert',
            'cf_beta',
            'dropout_prob'
        ]

        config['cf_learning_rate'] = trial.suggest_int('cf_learning_rate', 1000, 10000)

        config['user_batch_pert'] = trial.suggest_int(
            'user_batch_pert',
            min(int(_test_data.dataset.user_num * 0.1), 32),
            min(int(_test_data.dataset.user_num * 0.33), 220)
        )

        config['cf_beta'] = trial.suggest_float('cf_beta', 0.01, 10.0)

        # config['dropout_prob'] = trial.suggest_float('dropout_prob', 0, 0.3)

        wandb_config = {k: config[k] for k in wandb_config_keys}

        run = utils.wandb_init(
            wandb_config,
            **wandb_env_data,
            policies=config['perturbation_policies'],
            name=f"Perturbation_trial{trial.number}",
            job_type="train",
            group=pert_token,
            mode=wandb_mode,
            reinit=True
        )

        perturbation_trainer = BeyondAccuracyPerturbationTrainer(
            config,
            _train_dataset,
            _rec_data,
            model,
            dist=config['cf_dist'],
            **kwargs
        )
        pert, model_preds = perturbation_trainer.perturb(
            user_data,
            _test_data,
            epochs
        )
        best_pert = utils.get_best_pert_early_stopping(pert, config)

        pert_metric = best_pert[utils.pert_col_index('pert_metric')]

        with run:
            run.log({'trial_pert_metric': pert_metric})

        return pert_metric

    study_name = pert_token + '_' + str([k for k in config['perturbation_policies'] if config['perturbation_policies'][k]])
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_name, load_if_exists=True)

    n_trials = 100
    if study.trials:
        n_trials -= study.trials[-1].number  # it is not done automatically by optuna
        if n_trials <= 0:
            raise ValueError(f"Optuna study with storage name {study_name}.db is already completed")

    study.optimize(objective, n_trials=n_trials)

    summary = utils.wandb_init(
        config,
        **wandb_env_data,
        name="summary",
        job_type="logging",
        group=pert_token,
        mode=wandb_mode
    )

    trials = study.trials

    print("Number of finished trials: ", len(trials))

    # WandB summary.
    for step, trial in enumerate(trials):
        # Logging the loss.
        summary.log({"trial_pert_metric": trial.value}, step=step)

        # Logging the parameters.
        for k, v in trial.params.items():
            summary.log({k: v}, step=step)

    with open(os.path.join(base_perts_file, 'best_params.json'), 'w') as param_file:
        json.dump(dict(trial.params.items()), param_file, indent=4)


def execute_perturbation(model_file,
                         perturb_config_file=os.path.join("config", "perturbation", "base_perturbation.yaml"),
                         config_id=-1,
                         verbose=False,
                         wandb_mode="disabled",
                         cmd_config_args=None,
                         hyperoptimization=False,
                         overwrite=False):
    # load trained model, config, dataset
    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(
        model_file,
        perturb_config_file,
        cmd_config_args=cmd_config_args
    )

    # force these evaluation metrics to be ready to be computed
    config['metrics'] = ['ndcg', 'recall', 'hit', 'mrr', 'precision']

    if config['pert_rec_data'] is not None:
        if config['pert_rec_data'] != 'train+valid':
            if config['pert_rec_data'] == 'train':
                rec_data = FullSortEvalDataLoader(config, train_data.dataset, train_data.sampler)
            elif config['pert_rec_data'] == 'rec':  # model recommendations are used as target
                rec_data = valid_data
            else:
                rec_data = locals()[f"{config['pert_rec_data']}_data"]
        else:
            valid_train_dataset = train_data.dataset.copy(
                pd.concat([train_data.dataset.inter_feat, valid_data.dataset.inter_feat], ignore_index=True)
            )
            rec_data = FullSortEvalDataLoader(config, valid_train_dataset, valid_data.sampler)
    else:
        rec_data = valid_data

    base_perts_filepath = get_base_perts_filepath(
        config,
        config_id=config_id,
        model_name=model.__class__.__name__,
        model_file=model_file,
        hyper=hyperoptimization
    )

    if not os.path.exists(base_perts_filepath):
        os.makedirs(base_perts_filepath)

    global wandb_env_data
    wandb_env_data = {}
    if os.path.exists("wandb_init.json"):
        with open("wandb_init.json", 'r') as wandb_file:
            wandb_env_data = json.load(wandb_file)

    kwargs = dict(
        verbose=verbose,
        wandb_mode=wandb_mode,
        overwrite=overwrite
    )

    if not hyperoptimization:
        perturb(
            config,
            model,
            rec_data,
            dataset,
            train_data,
            valid_data,
            test_data,
            base_perts_filepath,
            **kwargs
        )
    else:
        optimize_perturbation(
            config,
            model,
            train_data.dataset,
            rec_data,
            test_data,
            base_perts_filepath,
            **kwargs
        )
