import torch
import torch.nn as nn


class L_Loss(nn.Module):
    """L_Loss used by UltraGCN"""

    def __init__(self, negative_weight=200):
        super(L_Loss, self).__init__()
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


class I_Loss(nn.Module):
    """I_Loss used by UltraGCN"""

    def __init__(self):
        super(I_Loss, self).__init__()

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
