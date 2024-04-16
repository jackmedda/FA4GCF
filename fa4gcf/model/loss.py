import abc
import math
from typing import Tuple, Dict, Type

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """L_Loss used by UltraGCN"""

    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, nodes, nodes_embeddings, other_nodes_embeddings=None):
        if other_nodes_embeddings is None:
            nodes = torch.unique(nodes)
            other_nodes_embeddings = nodes_embeddings

        nodes_embeddings = nodes_embeddings[nodes]
        scores = torch.logsumexp(nodes_embeddings.mm(other_nodes_embeddings.T), dim=-1).mean()

        return scores


class LLoss(nn.Module):
    """LLoss used by UltraGCN"""

    def __init__(self, negative_weight=200):
        super(LLoss, self).__init__()
        self.negative_weight = negative_weight

    def forward(self, pos_scores, neg_scores, omega_weight):
        neg_labels = torch.zeros(neg_scores.size()).to(neg_scores.device)
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            neg_scores, neg_labels,
            weight=omega_weight[len(pos_scores):],
            reduction='none'
        ).mean(dim=-1)

        pos_labels = torch.ones(pos_scores.size()).to(pos_scores.device)
        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pos_scores, pos_labels,
            weight=omega_weight[:len(pos_scores)],
            reduction='none'
        )

        loss = pos_loss + neg_loss * self.negative_weight

        return loss.sum()


class ILoss(nn.Module):
    """ILoss used by UltraGCN"""

    def __init__(self):
        super(ILoss, self).__init__()

    def forward(self, sim_scores, neighbor_scores):
        loss = neighbor_scores.sum(dim=-1).sigmoid().log()
        loss = -sim_scores * loss

        return loss.sum()


class NormLoss(nn.Module):
    """NormLoss, based on UltraGCN Normalization Loss of parameters"""

    def __init__(self):
        super(NormLoss, self).__init__()

    def forward(self, parameters):
        loss = 0.0
        for param in parameters:
            loss += torch.sum(param ** 2)
        return loss / 2


# Ranking Loss
class RankingLoss(torch.nn.modules.loss._Loss, metaclass=abc.ABCMeta):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', temperature=0.01, **kwargs) -> None:
        super(RankingLoss, self).__init__(size_average, reduce, reduction)
        self.temperature = temperature

    @abc.abstractmethod
    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses must implement this method")


class TopKLoss(RankingLoss):
    def __init__(self, size_average=None, reduce=None, topk=None, reduction: str = 'mean', temperature=0.01) -> None:
        super(TopKLoss, self).__init__(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature
        )
        self.topk = topk

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses must implement this method")


class NDCGApproxLoss(TopKLoss):
    __MAX_TOPK_ITEMS__ = 10000

    def __init__(self, size_average=None, reduce=None, topk=None, reduction: str = 'mean', temperature=0.01) -> None:
        """
        Lower values of `temperature` makes the loss more accurate in approximating NDCG
        """

        super(NDCGApproxLoss, self).__init__(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            topk=topk,
            temperature=temperature
        )

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _input_temp = torch.nn.ReLU()(_input) / self.temperature

        if _input_temp.shape[1] > self.__MAX_TOPK_ITEMS__:
            topk = self.__MAX_TOPK_ITEMS__ or target.shape[1]
            _, _input_topk = torch.topk(_input_temp, dim=1, k=topk)
            _input_temp = _input_temp[torch.arange(_input_temp.shape[0])[:, None], _input_topk]
            target = target[torch.arange(target.shape[0])[:, None], _input_topk]

        def approx_ranks(inp):
            shape = inp.shape[1]

            a = torch.tile(torch.unsqueeze(inp, 2), [1, 1, shape])
            a = torch.transpose(a, 1, 2) - a
            return torch.sum(torch.sigmoid(a), dim=-1) + .5

        def inverse_max_dcg(_target,
                            gain_fn=lambda _target: torch.pow(2.0, _target) - 1.,
                            rank_discount_fn=lambda rank: 1. / rank.log1p()):
            topk = self.topk or _target.shape[1]
            ideal_sorted_target = torch.topk(_target, topk).values
            rank = (torch.arange(ideal_sorted_target.shape[1]) + 1).to(_target.device)
            discounted_gain = gain_fn(ideal_sorted_target).to(_target.device) * rank_discount_fn(rank)
            discounted_gain = torch.sum(discounted_gain, dim=1, keepdim=True)
            return torch.where(discounted_gain > 0., 1. / discounted_gain, torch.zeros_like(discounted_gain))

        def ndcg(_target, _ranks):
            topk = self.topk or _target.shape[1]
            sorted_target, sorted_idxs = torch.topk(_target, topk)
            discounts = 1. / _ranks[torch.arange(_ranks.shape[0])[:, None], sorted_idxs].log1p()
            gains = torch.pow(2., sorted_target).to(_target.device) - 1.
            dcg = torch.sum(gains * discounts, dim=-1, keepdim=True)
            return dcg * inverse_max_dcg(_target)

        ranks = approx_ranks(_input_temp)

        return -ndcg(target, ranks)


class SigmoidBCELoss(RankingLoss):
    """"
    Based on TensorFlow Ranking SigmoidBCELoss
    """

    def __init__(self, size_average=None, reduce=None, topk=None, reduction: str = 'mean', temperature=0.1) -> None:
        super(SigmoidBCELoss, self).__init__(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            topk=topk,
            temperature=temperature
        )

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _input_temp = _input / self.temperature

        target_sum = target.sum(dim=1, keepdim=True)
        padded_target = torch.where(target_sum > 0, target, torch.ones_like(target) * 1e-7)
        padded_target_sum = padded_target.sum(dim=1, keepdim=True)
        target = torch.nan_to_num(padded_target / padded_target_sum, nan=0)

        return torch.nn.BCEWithLogitsLoss(reduction='none')(_input_temp, target).mean(dim=1, keepdim=True)


# Beyond-accuracy Loss
class BeyondAccLoss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.01, topk=10, **kwargs) -> None:
        super(BeyondAccLoss, self).__init__(size_average, reduce, reduction)
        self.temperature = temperature
        self.topk = topk

    def loss_type(self):
        return self.__class__.__name__

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses must implement this method")

    def is_data_feat_needed(self):
        raise NotImplementedError("subclasses must implement this method")


class FairLoss(BeyondAccLoss):
    def __init__(self,
                 discriminative_attribute: str,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.01, **kwargs) -> None:
        super(FairLoss, self).__init__(size_average, reduce, reduction, temperature, **kwargs)

        self.discriminative_attribute = discriminative_attribute
        self.data_feat = None

    def loss_type(self):
        if 'consumer' in self.__class__.__name__.lower():
            return 'Consumer'
        elif 'provider' in self.__class__.__name__.lower():
            return 'Provider'
        else:
            raise ValueError('A `FairLoss` subclass should contain "Consumer" or "Provider" in its name')

    def update_data_feat(self, data_feat):
        self.data_feat = data_feat

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses must implement this method")

    def is_data_feat_needed(self):
        return True


class ConsumerDPLoss(FairLoss):
    def __init__(self,
                 sensitive_attribute: str,
                 loss: Type[RankingLoss] = NDCGApproxLoss,
                 adv_group_data: Tuple[str, int, float] = None,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.01, **kwargs) -> None:
        super(ConsumerDPLoss, self).__init__(
            sensitive_attribute,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature,
            **kwargs
        )

        self.ranking_loss_function: RankingLoss = loss(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature,
            topk=self.topk
        )

        self.adv_group_data = adv_group_data
        self.deactivate_gradient = kwargs.get("deactivate_gradient", True)

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.data_feat is None:
            raise AttributeError('each forward call should be preceded by a call of `update_data_feat`')

        groups = self.data_feat[self.discriminative_attribute].unique()
        masks = []
        for gr in groups:
            masks.append((self.data_feat[self.discriminative_attribute] == gr))
        masks = torch.stack(masks)

        loss_values = self.ranking_loss_function(_input, target)

        masked_loss = []
        for gr, mask in zip(groups, masks):
            masked_loss.append(loss_values[mask].mean(dim=0))
        masked_loss = torch.stack(masked_loss)

        fair_loss = None
        total_loss = None
        for gr_i_idx in range(len(groups)):
            if self.adv_group_data[0] == "global":
                if groups[gr_i_idx] != self.adv_group_data[1]:
                    # the loss optimizes towards -1, but the global loss is positive
                    fair_loss = (masked_loss[gr_i_idx] - (-self.adv_group_data[2])).abs()
                    total_loss = fair_loss if total_loss is None else total_loss + fair_loss
            else:
                gr_i = groups[gr_i_idx]
                for gr_j_idx in range(gr_i_idx + 1, len(groups)):
                    l_val = masked_loss[gr_i_idx]
                    r_val = masked_loss[gr_j_idx]

                    if self.adv_group_data[0] == "local" and self.deactivate_gradient:
                        if self.adv_group_data[1] == gr_i:
                            l_val = l_val.detach()
                        else:
                            r_val = r_val.detach()

                    fair_loss = (l_val - r_val).abs()
                    total_loss = fair_loss if total_loss is None else total_loss + fair_loss

        self.update_data_feat(None)

        return fair_loss / max(int(math.comb(len(groups), 2)), 1)


class ProviderDPLoss(FairLoss):
    __TOPK_OFFSET__ = 0.1

    def __init__(self,
                 discriminative_attribute: str,
                 adv_group_data: Tuple[str, int, float] = None,
                 groups_distrib: Dict[int, float] = None,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.01, **kwargs) -> None:
        super(ProviderDPLoss, self).__init__(
            discriminative_attribute,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature,
            **kwargs
        )

        self.adv_group_data = adv_group_data
        self.groups_distrib = groups_distrib

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Instead of estimating the visibility on the top-k, we take the top 10% of items, because the top-k could not
        include items of both groups (short-head, long-tail) due to recommendation of only short-head items or because
        the top-k during training include also interactions in the training set.
        """
        if self.data_feat is None:
            raise AttributeError('each forward call should be preceded by a call of `update_data_feat`')

        _input = _input[:, 1:]  # data_feat does not have padding item 0

        input_topk_vals, input_topk_idxs = torch.topk(_input, k=round(_input.shape[1] * self.__TOPK_OFFSET__), dim=1)

        groups = self.data_feat[self.discriminative_attribute].unique().numpy()
        groups_recs_distrib = []
        for i, gr in enumerate(groups):
            mask = (self.data_feat[self.discriminative_attribute].to(_input.device)[input_topk_idxs] == gr).to(_input.device)

            if self.discriminative_attribute.lower() == 'visibility':
                input_topk_vals = torch.where(mask, input_topk_vals, 0)  # reduce the relevance of items not in the group

                mask = mask.float()
                mask_sum = mask.sum(dim=1, keepdim=True)
                padded_mask = torch.where(mask_sum > 0, mask, torch.ones_like(mask) * 1e-7)
                padded_mask_sum = padded_mask.sum(dim=1, keepdim=True)
                mask = torch.nan_to_num(padded_mask / padded_mask_sum, nan=0)

                visibility = torch.nn.BCEWithLogitsLoss(reduction='mean')(input_topk_vals, mask)

                groups_recs_distrib.append(visibility)
            elif self.discriminative_attribute.lower() == 'exposure':
                exposure = torch.where(mask, input_topk_vals, 0).sum(dim=1)

                groups_recs_distrib.append(exposure)
            else:
                raise NotImplementedError(f'Provider fairness loss with discriminative attribute `{self.discriminative_attribute}` is not supported')

        disparity = groups_recs_distrib[0] / self.groups_distrib[0] - groups_recs_distrib[1] / self.groups_distrib[1]
        if self.discriminative_attribute.lower() == 'visibility':
            fair_loss = disparity.abs()
        elif self.discriminative_attribute.lower() == 'exposure':
            fair_loss = (disparity.sum() / input_topk_vals.sum()).abs()

        return fair_loss


class UserCoverageLoss(BeyondAccLoss):

    def __init__(self,
                 min_relevant_items=0,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.1, **kwargs) -> None:
        super(UserCoverageLoss, self).__init__(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature,
            **kwargs
        )
        self.min_relevant_items = min(min_relevant_items, round(0.5 * self.topk))
        self.only_relevant = kwargs.get('only_relevant', True)

    def is_data_feat_needed(self):
        return False

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _input_temp = _input / self.temperature

        _, _input_topk_idxs = torch.topk(_input, k=self.topk, dim=1)
        relevant_idxs = target.gather(dim=1, index=_input_topk_idxs)
        relevant_recs = relevant_idxs.sum(dim=1)

        # take only users with #relevant_items < min_relevant_items
        rel_mask = relevant_recs <= self.min_relevant_items

        # exclude negative relevance
        _input_temp = torch.relu(_input_temp)[rel_mask]
        target = target[rel_mask]

        # the 1st only increases the relevance of relevant items, the 2nd also decreases the relevance of non-relevant items
        if self.only_relevant:
            loss = torch.where(target == 1, _input_temp * -1 - 1e-7, torch.relu(_input_temp * -1)).mean()
        else:
            loss = torch.where(target == 1, _input_temp * -1 - 1e-7, _input_temp).mean()

        return torch.nan_to_num(loss, nan=0)  # if all users have enough relevant items `loss` is NaN
