import math
from time import time

import torch
import torch.cuda.amp as amp
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from recbole.trainer import Trainer as RecboleTrainer
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.utils import (
    early_stopping,
    dict2str,
    set_color,
    get_gpu_usage,
    EvaluatorType
)

from fa4gcf.model.general_recommender.autocf import LocalGraphSampler, SubgraphRandomMasker


class Trainer(RecboleTrainer):

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        # Add user sensitive attribute information to the evaluation collector
        if eval_data:
            self.eval_collector.eval_data_collect(eval_data)
        return super(Trainer, self).evaluate(
            eval_data,
            load_best_model=load_best_model,
            model_file=model_file,
            show_progress=show_progress
        )

    @torch.no_grad()
    def evaluate_from_scores(
            self, eval_data, scores, show_progress=False
    ):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            scores (torch.Tensor): scores predicted by model.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return
        else:
            self.eval_collector.eval_data_collect(eval_data)

        if isinstance(eval_data, FullSortEvalDataLoader):
            if self.item_tensor is None:
                self.item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        else:
            raise NotImplementedError("Evaluation from scores not supported with negative sampling")
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, history_index, positive_u, positive_i = batched_data
            batch_scores = scores.clone()[interaction[eval_data.dataset.uid_field]]
            batch_scores[:, 0] = -torch.inf
            batch_scores[history_index] = -torch.inf
            self.eval_collector.eval_batch_collect(
                batch_scores, interaction, positive_u, positive_i
            )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result


class TraditionalTrainer(Trainer):

    def __init__(self, config, model):
        super(TraditionalTrainer, self).__init__(config, model)
        self.epochs = 1
        self.eval_step = 1  # the model is also evaluated on the validation set


class SVD_GCNTrainer(TraditionalTrainer):

    def __init__(self, config, model):
        super(SVD_GCNTrainer, self).__init__(config, model)
        if config['parametric']:
            self.epochs = config['epochs']  # overwrites the single epoch with the value in config
        else:
            self.saved_model_file = self.saved_model_file.replace('SVD_GCN', 'SVD_GCN_S')


class GFCFTrainer(TraditionalTrainer):

    def __init__(self, config, model):
        super(GFCFTrainer, self).__init__(config, model)
        self.eval_step = 1  # the model is also evaluated on the validation set


class NCLTrainer(Trainer):
    def __init__(self, config, model):
        super(NCLTrainer, self).__init__(config, model)

        self.num_m_step = config['m_step']
        assert self.num_m_step is not None

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.
        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.
        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        self.eval_collector.data_collect(train_data)

        for epoch_idx in range(self.start_epoch, self.epochs):

            # only differences from the original trainer
            if epoch_idx % self.num_m_step == 0:
                self.logger.info("Running E-step ! ")
                self.model.e_step()
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color('Saving current', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch
        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.
        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                if epoch_idx < self.config['warm_up_step']:
                    losses = losses[:-1]
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss


class HMLETTrainer(Trainer):
    def __init__(self, config, model):
        super(HMLETTrainer, self).__init__(config, model)

        self.warm_up_epochs = config['warm_up_epochs']
        self.ori_temp = config['ori_temp']
        self.min_temp = config['min_temp']
        self.gum_temp_decay = config['gum_temp_decay']
        self.epoch_temp_decay = config['epoch_temp_decay']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if epoch_idx > self.warm_up_epochs:
            # Temp decay
            gum_temp = self.ori_temp * math.exp(-self.gum_temp_decay*(epoch_idx - self.warm_up_epochs))
            self.model.gum_temp = max(gum_temp, self.min_temp)
            self.logger.info(f'Current gumbel softmax temperature: {self.model.gum_temp}')

            for gating in self.model.gating_nets:
                self.model._gating_freeze(gating, True)
        return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)


class AutoCFTrainer(Trainer):
    def __init__(self, config, model):
        super(AutoCFTrainer, self).__init__(config, model)

        self.sampled_graph_steps = config['sampled_graph_steps']

        self.local_graph_sampler = LocalGraphSampler(config['n_seeds'])
        self.subgraph_random_masker = SubgraphRandomMasker(
            config['mask_depth'],
            config['keep_rate'],
            model.n_users + model.n_items
        )

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        sample_scores = None
        encoder_edge_index, encoder_edge_weight, decoder_edge_index = None, None, None

        scaler = amp.GradScaler(enabled=self.enable_scaler)
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                ################## AutoCF ####################
                if batch_idx % self.sampled_graph_steps == 0:
                    sample_scores, seeds = self.local_graph_sampler(
                        self.model.get_ego_embeddings(), self.model.edge_index, self.model.edge_weight
                    )
                    encoder_edge_index, encoder_edge_weight, decoder_edge_index = self.subgraph_random_masker(
                        self.model.edge_index, self.model.edge_weight, seeds
                    )
                ##############################################

                losses = loss_func(interaction, encoder_edge_index, encoder_edge_weight, decoder_edge_index)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )

            ################## AutoCF ####################
            if batch_idx % self.sampled_graph_steps == 0:
                local_global_loss = -sample_scores.mean()
                loss += local_global_loss
            ##############################################

            self._check_nan(loss)
            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
        return total_loss
