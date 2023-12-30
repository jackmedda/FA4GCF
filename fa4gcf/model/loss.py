import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """L_Loss used by UltraGCN"""

    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, nodes, nodes_embeddings, other_nodes_embeddings=None):
        if other_nodes_embeddings is not None:
            nodes_embeddings = nodes_embeddings[nodes]
            scores = torch.log(
                torch.exp(nodes_embeddings.mm(other_nodes_embeddings.T)).sum(-1)
            ).mean()
        else:
            unique_nodes = torch.unique(nodes)
            nodes_embeddings = nodes_embeddings[unique_nodes]
            scores = torch.log(
                torch.exp(nodes_embeddings.mm(nodes_embeddings.T)).sum(-1)
            ).mean()

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
