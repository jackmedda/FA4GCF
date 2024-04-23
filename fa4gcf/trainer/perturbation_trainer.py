import os
import time
import logging

import wandb
import torch
import numpy as np
import pandas as pd
import torch.cuda.amp as amp
from tqdm import tqdm
from recbole.utils import set_color
from torch_geometric.typing import SparseTensor

import fa4gcf.utils as utils
import fa4gcf.data.utils as data_utils
import fa4gcf.trainer.utils as trainer_utils
from fa4gcf.data import Interaction
from fa4gcf.evaluation import (
    Evaluator,
    compute_metric,
    compute_beyondaccuracy_metric,
    get_scores,
    get_top_k
)
from fa4gcf.model import (
    PygPerturbedModel,
    get_ranking_loss,
    get_loss_from_beyondacc_metric
)
from fa4gcf.trainer.perturbation_sampler import PerturbationSampler
from fa4gcf.trainer.early_stopping import EarlyStopping


class PerturbationTrainer:

    def __init__(self, config, dataset, rec_data, model, **kwargs):
        self.config = config

        # self.cf_model = None
        self.model = model
        self.model.eval()
        self._cf_model = None

        self.dataset = dataset
        self.rec_data = rec_data
        self._pred_as_rec = config['pert_rec_data'] == 'rec'
        self._test_history_matrix = None

        self.cf_optimizer = None
        self.mini_batch_descent = config['mini_batch_descent']

        self.beta = config['cf_beta']
        self.device = config['device']
        self.cf_topk = config['cf_topk']
        self.user_batch_pert = config['user_batch_pert']
        self.unique_graph_dist_loss = config['save_unique_graph_dist_loss']

        self.logger = logging.getLogger()
        self.enable_amp = config["enable_amp"]
        self.verbose = kwargs.get("verbose", False)

        self.model_scores, self.model_scores_topk, self.model_topk_idx = None, None, None

        self.eval_metric = config['eval_metric'] or 'ndcg'
        self.evaluator = Evaluator(config)
        self.pert_metric = config['pert_metric'] or 'consumer_DP_across_random_samples'

        self._metric_loss = get_ranking_loss(config['metric_loss'] or 'ndcg')
        self._pert_loss = get_loss_from_beyondacc_metric(self.pert_metric)

        self.earlys = EarlyStopping(
            config['early_stopping']['patience'],
            config['early_stopping']['ignore'],
            method=config['early_stopping']['method'],
            fn=config['early_stopping']['mode'],
            delta=config['early_stopping']['delta']
        )
        self.earlys_check_criterion = config['early_stopping']['check_criterion']

        self.ckpt_loading_path = None

        pert_policies = config['perturbation_policies']
        self.random_perturbation = pert_policies['random_perturbation']
        self.neighborhood_perturbation = pert_policies['neighborhood_perturbation']

        torch.autograd.set_detect_anomaly(True)

    @property
    def cf_model(self):
        if self._cf_model is None:
            print("Counterfactual Model Explainer is not initialized yet. Execute 'explain' to initialize it.")
        else:
            return self._cf_model

    @cf_model.setter
    def cf_model(self, value):
        self._cf_model = value

    def initialize_cf_model(self, **kwargs):
        kwargs["random_perturbation"] = self.random_perturbation

        # Instantiate CF model class, load weights from original model
        self.cf_model = PygPerturbedModel(self.config, self.dataset, self.model, **kwargs).to(self.model.device)
        # self.parallel_cf_model = torch_parallel.DistributedDataParallel(self.cf_model)
        # self.parallel_cf_model = torch_parallel.DataParallel(self.cf_model)
        # for attr in ['device', 'full_sort_predict']:
        #     setattr(self.parallel_cf_model, attr, getattr(self.cf_model, attr))
        #
        # self.cf_model = self.parallel_cf_model
        self.logger.info(self.cf_model)

        self.initialize_optimizer()

    def initialize_optimizer(self):
        lr = self.config['cf_learning_rate']
        momentum = self.config["momentum"] or 0.0
        sgd_kwargs = {'momentum': momentum, 'nesterov': True if momentum > 0 else False}
        if self.config["cf_optimizer"] == "SGD":
            self.cf_optimizer = torch.optim.SGD(self.cf_model.parameters(), lr=lr, **sgd_kwargs)
        elif self.config["cf_optimizer"] == "Adadelta":
            self.cf_optimizer = torch.optim.Adadelta(self.cf_model.parameters(), lr=lr)
        elif self.config["cf_optimizer"] == "Adagrad":
            self.cf_optimizer = torch.optim.Adagrad(self.cf_model.parameters(), lr=lr)
        elif self.config["cf_optimizer"] == "AdamW":
            self.cf_optimizer = torch.optim.AdamW(self.cf_model.parameters(), lr=lr)
        elif self.config["cf_optimizer"] == "Adam":
            self.cf_optimizer = torch.optim.Adam(self.cf_model.parameters(), lr=lr)
        elif self.config["cf_optimizer"] == "RMSprop":
            self.cf_optimizer = torch.optim.RMSprop(self.cf_model.parameters(), lr=lr)
        else:
            raise NotImplementedError("CF Optimizer not implemented")

    def _initialize_pert_loss(self):
        raise NotImplementedError()

    def set_checkpoint_path(self, path):
        self.ckpt_loading_path = path

    def _resume_or_start_checkpoint(self):
        if self.ckpt_loading_path is not None and os.path.exists(self.ckpt_loading_path):
            pert_losses, starting_epoch, best_cf_example = self._resume_checkpoint()
        else:
            pert_losses, starting_epoch, best_cf_example = [], 0, []

        return pert_losses, starting_epoch, best_cf_example

    def _resume_checkpoint(self):
        ckpt = torch.load(self.ckpt_loading_path)
        epoch = ckpt['starting_epoch']
        loss_ckpt_keys = ['pert_losses', 'exp_losses', 'fair_losses']
        for loss_key in loss_ckpt_keys:  # retro-compatibility
            if loss_key in ckpt:
                pert_losses = ckpt[loss_key]
                break
        else:
            raise AttributeError(f'checkpoint loss key not in [{loss_ckpt_keys}]')

        self.earlys = ckpt['early_stopping']
        last_earlys_check_value = self.earlys.history.pop()
        if self.earlys.check(last_earlys_check_value):
            raise AttributeError("A checkpoint of a completed run cannot be resumed")

        best_cf_example = ckpt['best_cf_example']
        self.cf_model.load_cf_state_dict(ckpt)

        return pert_losses, epoch, best_cf_example

    def _save_checkpoint(self, epoch, pert_losses, best_cf_example):
        cf_state_dict = self.cf_model.cf_state_dict()
        ckpt = {
            'starting_epoch': epoch,
            'pert_losses': pert_losses,
            'early_stopping': self.earlys,
            'best_cf_example': best_cf_example,
            **cf_state_dict
        }
        torch.save(ckpt, self.ckpt_loading_path)

    def _check_early_stopping(self, check_value, epoch, *update_best_example_args):
        if self.earlys.check(check_value):
            self.logger.info(self.earlys)
            best_epoch = epoch + 1 - self.earlys.patience
            self.logger.info(f"Early Stopping: best epoch {best_epoch}")

            # stub example added to find again the best epoch when explanations are loaded
            self.update_best_cf_example(*update_best_example_args, force_update=True)

            return True
        return False

    def _early_stopping_step(self, pert_losses, epoch_pert_metric, epoch, best_cf_example, *update_best_example_args):
        earlys_check_value = {
            'pert_loss': pert_losses[-1],
            'pert_metric': epoch_pert_metric
        }[self.earlys_check_criterion]
        if self._pred_as_rec and earlys_check_value == epoch_pert_metric:
            raise ValueError(f"`pert_rec_data` = `rec` stores test data to log explanation metric. "
                             f"Cannot be used as value for early stopping check")

        earlys_check = self._check_early_stopping(earlys_check_value, epoch, *update_best_example_args)
        print("*" * 7 + " Early Stopping History " + "*" * 7)
        print(self.earlys.history)
        print("*" * 7 + "************************" + "*" * 7)

        if self.ckpt_loading_path is not None:
            if not earlys_check:
                # epoch + 1 because the current one is already finished
                self._save_checkpoint(epoch + 1, pert_losses, best_cf_example)

        return earlys_check

    def compute_model_predictions(self, scores_args):
        """
        Compute the predictions of the original model without perturbation
        :param scores_args: arguments needed by recbole to compute scores
        :return:
        """
        self.model_scores = get_scores(self.model, *scores_args, pred=None)

        # topk_idx contains the ids of the topk items
        self.model_scores_topk, self.model_topk_idx = get_top_k(self.model_scores, topk=self.cf_topk)

    def _get_scores_args(self, batched_data, dataset):
        dset_batch_data = self.prepare_batched_data(batched_data, dataset)
        return [
            dset_batch_data,
            self.dataset.item_num,
            self.dataset.get_item_feature().to(self.model.device)
        ]

    def _get_model_score_data(self, batched_data, dataset):
        dset_scores_args = self._get_scores_args(batched_data, dataset)
        self.compute_model_predictions(dset_scores_args)
        dset_model_topk = self.model_topk_idx.detach().cpu().numpy()

        return dset_scores_args, dset_model_topk

    def _get_no_grad_pred_model_score_data(self, scores_args):
        self.cf_model.eval()
        # When recommendations are generated passing test set the items in train and validation are considered watched
        with torch.no_grad():
            cf_scores_pred = get_scores(self.cf_model, *scores_args, pred=True)
            _, cf_topk_pred_idx = get_top_k(cf_scores_pred, topk=self.cf_topk)
        cf_topk_pred_idx = cf_topk_pred_idx.detach().cpu().numpy()

        return cf_topk_pred_idx

    def _pref_data_and_metric(self, pref_users, model_topk, eval_data=None):
        pref_data = pd.DataFrame(zip(pref_users, model_topk), columns=['user_id', 'topk_pred'])
        pref_data[self.eval_metric] = self.compute_eval_metric(
            eval_data or self.rec_data.dataset, pref_data, 'topk_pred'
        )[:, -1]

        return pref_data

    def compute_eval_metric(self, dataset, pref_data, col):
        return compute_metric(self.evaluator, dataset, pref_data, col, self.eval_metric)

    def compute_eval_result(self, pref_users: np.ndarray, model_topk: np.ndarray, eval_data=None):
        return self._pref_data_and_metric(pref_users, model_topk, eval_data=eval_data)[self.eval_metric].to_numpy()

    def compute_pert_metric(self, pref_users, model_topk, dataset):
        raise NotImplementedError()

    @staticmethod
    def prepare_batched_data(batched_data, data, item_data=None):
        return trainer_utils.prepare_batched_data(batched_data, data, item_data=item_data)

    def prepare_iter_batched_data(self, batched_data):
        return batched_data[torch.randperm(batched_data.shape[0])].split(self.user_batch_pert)

    def get_iter_data(self, user_data):
        user_data = user_data.split(self.user_batch_pert)

        return (
            tqdm(
                user_data,
                total=len(user_data),
                ncols=100,
                desc=set_color(f"Explaining   ", 'pink'),
            )
        )

    def _prepare_test_history_matrix(self, test_data):
        uids = test_data.dataset.user_feat.interaction[test_data.dataset.uid_field][1:]

        dset_scores_args, dset_model_topk = self._get_model_score_data(uids, test_data)

        # add -1 as item ids for the padding user
        dset_model_topk = np.vstack((np.array([-1] * self.cf_topk, dtype=dset_model_topk.dtype), dset_model_topk))
        self._test_history_matrix = dset_model_topk

        return dset_scores_args, dset_model_topk

    def update_best_cf_example(self,
                               best_cf_example,
                               new_example,
                               loss_total,
                               best_loss,
                               model_topk=None,
                               force_update=False):
        """
        Updates the perturbations with new perturbation (if not None) depending on new loss value
        :param best_cf_example:
        :param new_example:
        :param loss_total:
        :param best_loss:
        :param model_topk:
        :param force_update:
        :return:
        """
        update_check = new_example is not None and (abs(loss_total) < best_loss or self.unique_graph_dist_loss)
        if force_update or update_check:
            if self.unique_graph_dist_loss and len(best_cf_example) > 0:
                self.old_graph_dist = best_cf_example[-1][1]
                new_graph_dist = new_example[1]
                if not force_update and not (self.old_graph_dist != new_graph_dist):
                    return best_loss

            best_cf_example.append(new_example)
            self.old_graph_dist = new_example[1]

            if not self.unique_graph_dist_loss:
                return abs(loss_total)

        return best_loss

    def update_new_example(self,
                           new_example,
                           detached_batched_data,
                           full_dataset,
                           train_data,
                           valid_data,
                           test_data,
                           test_model_topk,
                           rec_model_topk,
                           epoch_pert_loss):
        perturbed_edges = new_example[-2]

        try:
            if self.config['pert_rec_data'] == 'rec':
                raise NotImplementedError(
                    'pert metric evaluation with dataloaders with perturbed edges '
                    'not implemented for `pert_rec_data` == `rec`'
                )

            try:
                pert_sets = data_utils.get_dataloader_with_perturbed_edges(
                    perturbed_edges, self.config, full_dataset, train_data, valid_data, test_data
                )
            except ValueError as e:
                if repr(e) == "ValueError('Some users have interacted with all items, " \
                              "which we can not sample negative items for them. " \
                              "Please set `user_inter_num_interval` to filter those users.')":
                    return False, None
                raise e

            pert_sets_dict = dict(zip(['train', 'valid', 'test'], pert_sets))

            test_scores_args = self._get_scores_args(
                detached_batched_data, pert_sets_dict['test']
            )
            rec_scores_args = self._get_scores_args(
                detached_batched_data, pert_sets_dict[self.config['pert_rec_data']]
            )

            test_cf_topk_pred_idx = self._get_no_grad_pred_model_score_data(test_scores_args)
            rec_cf_topk_pred_idx = self._get_no_grad_pred_model_score_data(rec_scores_args)

            epoch_rec_pert_metric = self.compute_pert_metric(
                detached_batched_data,
                rec_cf_topk_pred_idx,
                pert_sets_dict[self.config['pert_rec_data']].dataset
            )

            epoch_test_pert_metric = self.compute_pert_metric(
                detached_batched_data,
                test_cf_topk_pred_idx,
                pert_sets_dict['test'].dataset
            )

        except TypeError as e:
            if self.earlys_check_criterion != 'pert_loss':
                raise NotImplementedError(
                    'must check how to solve the problem when certain users have'
                    'no items in the history, e.g., after removing/adding edges'
                ) from e
            else:
                epoch_rec_pert_metric = None
                epoch_test_pert_metric = None

        new_example[utils.pert_col_index('pert_loss')] = epoch_pert_loss
        new_example[utils.pert_col_index('pert_metric')] = epoch_rec_pert_metric

        wandb.log({
            'loss': epoch_pert_loss,
            'rec_pert_metric': epoch_rec_pert_metric,
            'test_pert_metric': epoch_test_pert_metric,
            '# Pert Edges': perturbed_edges.shape[1],
            'epoch': new_example[-1]
        })

        self.logger.info(str({
            'loss': epoch_pert_loss,
            'rec_pert_metric': epoch_rec_pert_metric,
            'test_pert_metric': epoch_test_pert_metric,
            '# Pert Edges': perturbed_edges.shape[1],
            'epoch': new_example[-1]
        }))

        return new_example, epoch_rec_pert_metric

    def run_perturb(self, iter_epochs, batched_data, rec_model_topk, pert_losses, best_cf_example, *example_args):
        best_loss = np.inf

        for epoch in iter_epochs:
            new_example, loss_total = self.run_epoch(epoch, batched_data, pert_losses)

            if self.verbose:
                self._verbose_plot(pert_losses, epoch)

            if new_example is not None:
                new_example, epoch_pert_metric = self.update_new_example(
                    new_example,
                    *example_args,
                    rec_model_topk,
                    pert_losses[-1]
                )

                if new_example is False:
                    self.logger.warning('Interrupting because one of the users interacted with all the items')
                    break

                update_best_example_args = [best_cf_example, new_example, loss_total, best_loss]

                earlys_check = self._early_stopping_step(
                    pert_losses, epoch_pert_metric, epoch, best_cf_example, *update_best_example_args
                )

                if earlys_check:
                    break
                elif epoch == (iter_epochs.iterable - 1):
                    # stub perturbation that means the code stopped because of the
                    # number of epochs limit, not because of the early stopping
                    update_best_example_args[1] = utils.PERT_END_EPOCHS_STUB
                    self.update_best_cf_example(*update_best_example_args, model_topk=rec_model_topk)
                    break

                best_loss = self.update_best_cf_example(*update_best_example_args, model_topk=rec_model_topk)

            # the optimizer step of the last epoch_step is done here to prevent computations
            # done to update new example to be related to the new state of the model
            torch.nn.utils.clip_grad_norm_(self.cf_model.parameters(), 2.0)
            if self.mini_batch_descent:
                self.cf_optimizer.step()

            self.logger.info("{} CF examples".format(len(best_cf_example)))

    def run_epoch(self, epoch, batched_data, pert_losses):
        iter_data = self.prepare_iter_batched_data(batched_data)
        iter_data = [iter_data[i] for i in np.random.permutation(len(iter_data))]

        if not self.mini_batch_descent:
            self.cf_optimizer.zero_grad()

        epoch_pert_loss = []
        new_example, loss_total = None, None
        for batch_idx, batch_user in enumerate(iter_data):
            batch_scores_args = self._get_scores_args(batch_user, self.rec_data)

            torch.cuda.empty_cache()
            new_example, loss_total, pert_loss = self.train(epoch, batch_scores_args)
            epoch_pert_loss.append(pert_loss)

            if batch_idx != len(iter_data) - 1:
                torch.nn.utils.clip_grad_norm_(self.cf_model.parameters(), 2.0)
                if self.mini_batch_descent:
                    self.cf_optimizer.step()

        epoch_pert_loss = np.mean(epoch_pert_loss)
        pert_losses.append(epoch_pert_loss)

        return new_example, loss_total

    def perturb(self, batched_data, full_dataset, train_data, valid_data, test_data, epochs):
        """
        The method from which starts the perturbation of the graph by optimization of `pert_loss`
        :param batched_data:
        :param test_data:
        :param epochs:
        :return:
        """
        self._check_loss_trend_epoch_images()

        test_scores_args, test_model_topk = self._get_model_score_data(batched_data, test_data)

        # recommendations generated by the model are considered the ground truth
        if self._pred_as_rec:
            rec_scores_args, rec_model_topk = self._prepare_test_history_matrix(test_data)
        else:
            rec_scores_args, rec_model_topk = self._get_model_score_data(batched_data, self.rec_data)

        self._initialize_pert_loss()

        # logs of explanation consider validation as seen when model recommendations are used as ground truth
        if self._pred_as_rec:
            self.rec_data = test_data

        detached_batched_data = batched_data.detach().numpy()
        self.initialize_cf_model()
        self.initialize_optimizer()

        pert_losses, starting_epoch, best_cf_example = self._resume_or_start_checkpoint()

        iter_epochs = tqdm(
            range(starting_epoch, epochs),
            total=epochs,
            ncols=100,
            initial=starting_epoch,
            desc=set_color(f"Epochs   ", 'blue'),
        )

        example_args = [detached_batched_data, full_dataset, train_data, valid_data, test_data, test_model_topk]
        self.run_perturb(
            iter_epochs,
            batched_data,
            rec_model_topk,
            pert_losses,
            best_cf_example,
            *example_args
        )

        return best_cf_example, detached_batched_data, (rec_model_topk, test_model_topk)

    def train(self, epoch, batch_scores_args):
        raise NotImplementedError()

    def log_epoch(self, initial_time, epoch, *losses, **verbose_kws):
        loss_total, pert_loss, loss_graph_dist, orig_loss_graph_dist = losses
        elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - initial_time))
        self.logger.info(f"{self.cf_model.__class__.__name__} " +
                         f"Explain duration: {elapsed_time}, " +
                         # 'User id: {}, '.format(str(users_ids)) +
                         'Epoch: {}, '.format(epoch + 1) +
                         'loss: {:.4f}, '.format(loss_total.item()) +
                         'exp loss: {:.4f}, '.format(pert_loss) +
                         'graph loss: {:.4f}, '.format(loss_graph_dist.item()) +
                         'perturbed edges: {:.4f}, '.format(int(orig_loss_graph_dist.item())))
        if self.verbose:
            self.logger.info('Orig output: {}\n'.format(self.model_scores) +
                             # 'Output: {}\n'.format(verbose_kws.get('cf_scores', None)) +
                             # 'Output nondiff: {}\n'.format(verbose_kws.get('cf_scores_pred', None)) +
                             '{:20}: {},\n {:20}: {},\n {:20}: {}\n'.format(
                                 'orig pred', self.model_topk_idx,
                                 'new pred', verbose_kws.get('cf_topk_idx', None),
                                 'new pred nondiff', verbose_kws.get('cf_topk_pred_idx', None))
                             )
        self.logger.info(" ")

    @staticmethod
    def _verbose_plot(pert_losses, epoch):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import scipy.signal as sp_signal

        if os.path.isfile(f'loss_trend_epoch{epoch}.png'):
            os.remove(f'loss_trend_epoch{epoch}.png')
        ax = sns.lineplot(
            x='epoch',
            y='pert loss',
            data=pd.DataFrame(zip(np.arange(1, epoch + 2), pert_losses), columns=['epoch', 'pert loss'])
        )
        if len(pert_losses) > 20:
            sns.lineplot(
                x=np.arange(1, epoch + 2),
                y=sp_signal.savgol_filter(pert_losses, window_length=len(pert_losses) // 2, polyorder=2),
                ax=ax
            )
        plt.savefig(f'loss_trend_epoch{epoch + 1}.png')
        plt.close()

    def _check_loss_trend_epoch_images(self):
        cwd_files = [f for f in os.listdir() if f.startswith('loss_trend_epoch')]
        if len(cwd_files) > 0 and os.path.isfile(cwd_files[0]) and cwd_files[0][-3:] == 'png':
            os.remove(cwd_files[0])


class BeyondAccuracyPerturbationTrainer(PerturbationTrainer):

    def __init__(self, config, dataset, rec_data, model, dist="damerau_levenshtein", **kwargs):
        super(BeyondAccuracyPerturbationTrainer, self).__init__(config, dataset, rec_data, model, **kwargs)

        if dist == "set":
            self.dist = lambda topk_idx, cf_topk_idx: len(topk_idx) - (len(set(topk_idx) & set(cf_topk_idx)))
        elif dist == "damerau_levenshtein":
            self.dist = utils.damerau_levenshtein_distance

        # Init policies
        pert_policies = config['perturbation_policies']
        self.gradient_deactivation_constraint = pert_policies['gradient_deactivation_constraint'] or False
        self.increase_disparity = pert_policies['increase_disparity']
        self.group_deletion_constraint = pert_policies['group_deletion_constraint']

        self.results = None
        self.adv_group, self.disadv_group = None, None
        self.only_adv_group = config['only_adv_group']

        # User/Consumer Fairness
        self.sensitive_attribute = config['sensitive_attribute']
        attr_map = dataset.field2id_token[self.sensitive_attribute]
        self.f_idx, self.m_idx = (attr_map == 'F').nonzero()[0][0], (attr_map == 'M').nonzero()[0][0]

        # Item/Provider Fairness
        self.item_discriminative_attribute = config['item_discriminative_attribute'] or 'exposure'
        # SH / LT Explainable Fairness in Recommendation
        self.item_discriminative_ratio = config['short_head_item_discriminative_ratio'] or 0.2
        self.item_discriminative_groups_distrib = [1, 1 / self.item_discriminative_ratio - 1]
        self.item_discriminative_map = ['[PAD]', 'SH', 'LT']  # SH: Short Head, LT: Long Tail

        # User Coverage
        self.coverage_min_relevant_items = config['coverage_min_relevant_items'] or 0
        self.coverage_loss_only_relevant = config['coverage_loss_only_relevant'] or True

        self._pert_loss_args = {
            'consumer_dp': [self.sensitive_attribute],
            'consumer_dp_across_random_samples': [self.sensitive_attribute],
            'provider_dp': [self.item_discriminative_attribute],
            'uc': [self.coverage_min_relevant_items]
        }

        self.pert_sampler = PerturbationSampler(
            self.dataset,
            self,
            config
        )

    def _check_policies(self, batched_data, rec_model_topk, test_data=None):
        test_model_topk, test_scores_args, rec_scores_args = [None] * 3
        if self.increase_disparity:
            batched_data, test_model_topk, test_scores_args,\
                rec_model_topk, rec_scores_args = self.increase_dataset_unfairness(
                    batched_data,
                    test_data,
                    rec_model_topk
                )

        self.determine_adv_group(batched_data.detach().numpy(), rec_model_topk)
        pref_data = self._pref_data_sens_and_metric(batched_data.detach().numpy(), rec_model_topk)
        filtered_users, filtered_items = self.pert_sampler.apply_policies(batched_data, pref_data)

        return batched_data, filtered_users, filtered_items, (test_model_topk, test_scores_args, rec_model_topk, rec_scores_args)

    def initialize_cf_model(self, **kwargs):
        kwargs["random_perturbation"] = self.random_perturbation

        # Instantiate CF model class, load weights from original model
        self.cf_model = PygPerturbedModel(self.config, self.dataset, self.model, **kwargs).to(self.model.device)
        # self.parallel_cf_model = torch_parallel.DistributedDataParallel(self.cf_model)
        # self.parallel_cf_model = torch_parallel.DataParallel(self.cf_model)
        # for attr in ['device', 'full_sort_predict']:
        #     setattr(self.parallel_cf_model, attr, getattr(self.cf_model, attr))
        #
        # self.cf_model = self.parallel_cf_model
        self.logger.info(self.cf_model)

        self.initialize_optimizer()

    def _initialize_pert_loss(self):
        self._pert_loss = self._pert_loss(
            *self._pert_loss_args[self.pert_metric.lower()],
            topk=self.cf_topk,
            loss=self._metric_loss,
            adv_group_data=(self.only_adv_group, self.disadv_group, self.results[self.disadv_group]),
            deactivate_gradient=self.gradient_deactivation_constraint,
            only_relevant=self.coverage_loss_only_relevant,
            groups_distrib=self.item_discriminative_groups_distrib
        )

        if self._pert_loss.loss_type() == 'Provider':
            self._check_item_feat_integrity()

    def logging_pert_per_group(self, new_example, model_topk):
        em_str = self.eval_metric.upper()

        pref_data = pd.DataFrame(
            zip(*new_example[:2], model_topk),
            columns=['user_id', 'cf_topk_pred', 'topk_pred']
        )
        orig_res = self.compute_eval_metric(self.rec_data.dataset, pref_data, 'topk_pred')
        cf_res = self.compute_eval_metric(self.rec_data.dataset, pref_data, 'cf_topk_pred')

        user_feat = Interaction(
            {k: v[torch.tensor(new_example[0])] for k, v in self.dataset.user_feat.interaction.items()}
        )

        m_users = user_feat[self.sensitive_attribute] == self.m_idx
        f_users = user_feat[self.sensitive_attribute] == self.f_idx

        orig_f, orig_m = np.mean(orig_res[f_users, -1]), np.mean(orig_res[m_users, -1])
        cf_f, cf_m = np.mean(cf_res[f_users, -1]), np.mean(cf_res[m_users, -1])
        self.logger.info(
            f"Original => {em_str} F: {orig_f}, {em_str} M: {orig_m}, Diff: {np.abs(orig_f - orig_m)}\n"
            f"CF       => {em_str} F: {cf_f}, {em_str} M: {cf_m}, Diff: {np.abs(cf_f - cf_m)}"
        )

    def _before_perturbation_log(self, full_dataset, test_data, batched_data, rec_model_topk, test_model_topk):
        # orig_rec_dp = eval_utils.compute_DP(
        #     self.results[self.adv_group], self.results[self.disadv_group]
        # )
        # orig_test_dp = eval_utils.compute_DP(
        #     *self.compute_f_m_result(detached_batched_data, test_model_topk, eval_data=test_data.dataset)
        # )

        orig_rec_beyondacc_metric_rec = self.compute_pert_metric(
            batched_data, rec_model_topk, self.rec_data.dataset
        )
        orig_rec_beyondacc_metric_test = self.compute_pert_metric(
            batched_data, test_model_topk, test_data.dataset
        )

        self.logger.info("*********** Rec Data ***********")
        if self._pert_loss.loss_type() == 'Consumer':
            self.logger.info(self.results)
            self.logger.info(f"M idx: {self.m_idx}")
        self.logger.info(f"Original Rec Explanation Metric Value: {orig_rec_beyondacc_metric_rec}")
        self.logger.info("*********** Test Data ***********")
        self.logger.info(f"Original Test Explanation Metric Value: {orig_rec_beyondacc_metric_test}")

        full_users_list = full_dataset.user_feat[full_dataset.uid_field][1:].numpy()
        _, full_users_test_model_topk = self._get_model_score_data(full_users_list, test_data)
        pref_data = self._pref_data_sens_and_metric(
            full_users_list, full_users_test_model_topk, eval_data=test_data.dataset
        )
        self.logger.info(f"Original Test Overall {self.eval_metric.upper()}: {pref_data[self.eval_metric].mean()}")

    def prepare_iter_batched_data(self, batched_data):
        if self._pert_loss.loss_type() == 'Consumer':
            if self.only_adv_group != "global":
                iter_data = trainer_utils.randperm2groups(
                    batched_data, self.sensitive_attribute, self.dataset.user_feat, self.user_batch_pert
                )
                # check if each batch has at least 2 groups
                while any(self.dataset.user_feat[self.sensitive_attribute][d].unique().shape[0] < 2 for d in iter_data):
                    iter_data = trainer_utils.randperm2groups(
                        batched_data, self.sensitive_attribute, self.dataset.user_feat, self.user_batch_pert
                    )
            else:
                batched_attr_data = self.dataset.user_feat[self.sensitive_attribute][batched_data]
                iter_data = batched_data[batched_attr_data == self.adv_group].split(self.user_batch_pert)
        else:
            iter_data = super(BeyondAccuracyPerturbationTrainer, self).prepare_iter_batched_data(batched_data)

        return iter_data

    def _check_item_feat_integrity(self):
        if self.dataset.item_feat is None or self.item_discriminative_attribute not in self.dataset.item_feat:
            data_utils.update_item_feat_discriminative_attribute(
                self.dataset, self.item_discriminative_attribute, self.item_discriminative_ratio, self.item_discriminative_map
            )

    def _pref_data_sens_and_metric(self, pref_users, model_topk, eval_data=None):
        pref_data = pd.DataFrame(
            zip(pref_users, model_topk, self.dataset.user_feat[self.sensitive_attribute][pref_users].numpy()),
            columns=['user_id', 'topk_pred', 'Demo Group']
        )
        pref_data[self.eval_metric] = self.compute_eval_metric(
            eval_data or self.rec_data.dataset, pref_data, 'topk_pred'
        )[:, -1]

        return pref_data

    def compute_eval_result(self, pref_users: np.ndarray, model_topk: np.ndarray, eval_data=None):
        return self._pref_data_sens_and_metric(pref_users, model_topk, eval_data=eval_data)[self.eval_metric].to_numpy()

    def compute_f_m_result(self, batched_data: np.ndarray, model_topk, eval_data=None):
        pref_data = self._pref_data_sens_and_metric(batched_data, model_topk, eval_data=eval_data)

        f_result = pref_data.loc[pref_data['Demo Group'] == self.f_idx, self.eval_metric].mean()
        m_result = pref_data.loc[pref_data['Demo Group'] == self.m_idx, self.eval_metric].mean()

        return f_result, m_result

    def compute_pert_metric(self, pref_users, model_topk, dataset, iterations=100):
        pref_data = self._pref_data_sens_and_metric(pref_users, model_topk, eval_data=dataset)

        return compute_beyondaccuracy_metric(
            self.pert_metric,
            pref_data,
            self.eval_metric,
            dataset.dataset_name,
            self.sensitive_attribute,
            self.user_batch_pert,
            dataset=self.dataset,
            discriminative_attribute=self.item_discriminative_attribute,
            groups_distrib=self.item_discriminative_groups_distrib,
            iterations=iterations,
            coverage_min_relevant_items=self.coverage_min_relevant_items
        )

    def determine_adv_group(self, batched_data: np.ndarray, rec_model_topk):
        f_result, m_result = self.compute_f_m_result(batched_data, rec_model_topk)

        check_func = "__ge__" if self.config['perturb_adv_group'] else "__lt__"

        self.adv_group = self.m_idx if getattr(m_result, check_func)(f_result) else self.f_idx
        self.disadv_group = self.f_idx if self.adv_group == self.m_idx else self.m_idx
        self.results = dict(zip([self.m_idx, self.f_idx], [m_result, f_result]))

    def increase_dataset_unfairness(self, batched_data, test_data, rec_model_topk):
        pref_data = pd.DataFrame(
            zip(batched_data.numpy(), rec_model_topk.tolist()),
            columns=['user_id', 'topk_pred']
        )
        pref_data['result'] = self.compute_eval_metric(self.rec_data.dataset, pref_data, 'topk_pred')[:, -1]

        batched_data = trainer_utils.increase_user_unfairness(pref_data, self.sensitive_attribute, self.dataset)

        # recompute recommendations of original model
        test_scores_args, test_model_topk = self._get_model_score_data(batched_data, test_data)
        rec_scores_args, rec_model_topk = self._get_model_score_data(batched_data, self.rec_data)

        return batched_data, test_model_topk, test_scores_args, rec_model_topk, rec_scores_args

    def update_best_cf_example(self,
                               best_cf_example,
                               new_example,
                               loss_total,
                               best_loss,
                               model_topk=None,
                               force_update=False):
        """
        Updates the perturbations with new perturbation (if not None) depending on new loss value
        :param best_cf_example:
        :param new_example:
        :param loss_total:
        :param best_loss:
        :param model_topk:
        :param force_update:
        :return:
        """
        update_check = new_example is not None and (abs(loss_total) < best_loss or self.unique_graph_dist_loss)
        if force_update or update_check:
            if self.unique_graph_dist_loss and len(best_cf_example) > 0:
                self.old_graph_dist = best_cf_example[-1][1]
                new_graph_dist = new_example[1]
                if not force_update and not (self.old_graph_dist != new_graph_dist):
                    return best_loss

            best_cf_example.append(new_example)
            self.old_graph_dist = new_example[1]

            if self.verbose and model_topk is not None:
                self.logging_pert_per_group(new_example, model_topk)

            if not self.unique_graph_dist_loss:
                return abs(loss_total)

        return best_loss

    def perturb(self, batched_data, full_dataset, train_data, valid_data, test_data, epochs):
        """
        The method from which starts the perturbation of the graph by optimization of `pert_loss`
        :param batched_data:
        :param test_data:
        :param epochs:
        :return:
        """
        self._check_loss_trend_epoch_images()

        test_scores_args, test_model_topk = self._get_model_score_data(batched_data, test_data)

        # recommendations generated by the model are considered the ground truth
        if self._pred_as_rec:
            rec_scores_args, rec_model_topk = self._prepare_test_history_matrix(test_data)
        else:
            rec_scores_args, rec_model_topk = self._get_model_score_data(batched_data, self.rec_data)

        batched_data, filtered_users, filtered_items, inc_disp_model_data = self._check_policies(
            batched_data, rec_model_topk, test_data=test_data
        )
        if self.increase_disparity:
            test_model_topk, test_scores_args, rec_model_topk, rec_scores_args = inc_disp_model_data

        self._initialize_pert_loss()

        # logs of explanation consider validation as seen when model recommendations are used as ground truth
        if self._pred_as_rec:
            self.rec_data = test_data

        detached_batched_data = batched_data.detach().numpy()
        self.initialize_cf_model(filtered_users=filtered_users, filtered_items=filtered_items)
        self.initialize_optimizer()

        pert_losses, starting_epoch, best_cf_example = self._resume_or_start_checkpoint()

        self._before_perturbation_log(
            full_dataset, test_data, detached_batched_data, rec_model_topk, test_model_topk
        )

        iter_epochs = tqdm(
            range(starting_epoch, epochs),
            total=epochs,
            ncols=100,
            initial=starting_epoch,
            desc=set_color(f"Epochs   ", 'blue'),
        )

        example_args = [detached_batched_data, full_dataset, train_data, valid_data, test_data, test_model_topk]
        self.run_perturb(
            iter_epochs,
            batched_data,
            rec_model_topk,
            pert_losses,
            best_cf_example,
            *example_args
        )

        return best_cf_example, detached_batched_data, (rec_model_topk, test_model_topk)

    def train(self, epoch, scores_args):
        """
        Training procedure of perturbation
        :param epoch:
        :return:
        """
        train_start = time.time()
        torch.cuda.empty_cache()

        # Only the 500 items with the highest predicted relevance will be used to measure the approx NDCG
        # This prevents the usage of a tremendous amount of memory, due to the pairwise preference function
        MEMORY_PERFORMANCE_MAX_LOSS_TOPK_ITEMS = 500
        if self._pert_loss.ranking_loss_function.__MAX_TOPK_ITEMS__ != MEMORY_PERFORMANCE_MAX_LOSS_TOPK_ITEMS:
            self._pert_loss.ranking_loss_function.__MAX_TOPK_ITEMS__ = MEMORY_PERFORMANCE_MAX_LOSS_TOPK_ITEMS

        if self.mini_batch_descent:
            self.cf_optimizer.zero_grad()
        self.cf_model.train()

        user_feat = self.get_batch_user_feat(scores_args[0][0][self.dataset.uid_field])
        target = self.get_target(user_feat)

        if self._pert_loss.is_data_feat_needed():
            data_feat = self.get_loss_data_feat(user_feat)
            self._pert_loss.update_data_feat(data_feat)

        with amp.autocast(enabled=self.enable_amp):
            loss_total, orig_loss_graph_dist, loss_graph_dist, pert_loss, adj_sub_cf_adj = self.cf_model.loss(
                scores_args,
                self._pert_loss,
                target
            )

        torch.cuda.empty_cache()
        # import torchviz
        # dot = torchviz.make_dot(loss_total, params=dict(self.cf_model.named_parameters()))
        # dot.graph_attr.update(size='600,600')
        # dot.render("comp_graph", format="png")
        loss_total.backward()

        # for name, param in self.cf_model.named_parameters():
        #     print(param.grad)
        # import pdb; pdb.set_trace()

        pert_loss = pert_loss.mean().item() if pert_loss is not None else torch.nan
        self.log_epoch(train_start, epoch, loss_total, pert_loss, loss_graph_dist, orig_loss_graph_dist)

        cf_stats = None
        if orig_loss_graph_dist.item() > 0:
            cf_stats = self.get_batch_cf_stats(
                adj_sub_cf_adj, loss_total, loss_graph_dist, pert_loss, epoch
            )

        return cf_stats, loss_total.item(), pert_loss

    def get_target(self, user_feat):
        target_shape = (user_feat[self.dataset.uid_field].shape[0], self.dataset.item_num)
        target = torch.zeros(target_shape, dtype=torch.float, device=self.cf_model.device)

        if not self._pred_as_rec:
            hist_matrix, _, _ = self.rec_data.dataset.history_item_matrix()
            rec_data_interactions = hist_matrix[user_feat[self.dataset.uid_field]]
        else:
            rec_data_interactions = self._test_history_matrix[user_feat[self.dataset.uid_field]]
        target[torch.arange(target.shape[0])[:, None], rec_data_interactions] = 1
        target[:, 0] = 0  # item 0 is a padding item

        return target

    def get_batch_cf_stats(self, adj_sub_cf_adj, loss_total, loss_graph_dist, pert_loss, epoch):
        adj_pert_edges = adj_sub_cf_adj.detach().cpu()
        if isinstance(adj_pert_edges, SparseTensor):
            row, col, vals = adj_pert_edges.coo()
            pert_edges = torch.stack((row, col), dim=0)[:, vals.nonzero().squeeze()]
        else:
            pert_edges = adj_pert_edges.indices()[:, adj_pert_edges.values().nonzero().squeeze()]

        # remove duplicated edges
        pert_edges = pert_edges[:, (pert_edges[0, :] < self.dataset.user_num) & (pert_edges[0, :] > 0)].numpy()

        cf_stats = [loss_total.item(), loss_graph_dist.item(), pert_loss, None, pert_edges, epoch + 1]

        if self.neighborhood_perturbation:
            self.cf_model.update_neighborhood(torch.Tensor(pert_edges))

        return cf_stats

    def get_batch_user_feat(self, users_ids):
        user_feat = self.dataset.user_feat
        user_id_mask = users_ids.unsqueeze(-1) if users_ids.dim() == 0 else users_ids
        return {k: feat[user_id_mask] for k, feat in user_feat.interaction.items()}

    def get_loss_data_feat(self, user_feat):
        if self._pert_loss.loss_type() == 'Consumer':
            return user_feat
        elif self._pert_loss.loss_type() == 'Provider':
            return self.dataset.item_feat[1:]
        else:
            raise ValueError('A `FairLoss` subclass should contain "Consumer" or "Provider" in its name')
