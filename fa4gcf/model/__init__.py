from fa4gcf.model.perturbed_recommender.perturbed_model import PygPerturbedModel
from fa4gcf.model.loss import *


def get_ranking_loss(loss="ndcg"):
    return {
        "ndcg": NDCGApproxLoss,
        "sigmoid_bce": SigmoidBCELoss
    }[loss.lower()]


def get_beyondacc_loss(loss="consumer_dp"):
    return {
        "consumer_dp": ConsumerDPLoss,
        "provider_dp": ProviderDPLoss,
        "uc": UserCoverageLoss
    }[loss.lower()]


__beyondacc_metrics_to_losses_map__ = {
    "consumer_DP": ConsumerDPLoss,
    "consumer_DP_across_random_samples": ConsumerDPLoss,
    "UC": UserCoverageLoss,
    "provider_DP": ProviderDPLoss
}


def get_loss_from_beyondacc_metric(beyond_metric):
    if beyond_metric in __beyondacc_metrics_to_losses_map__:
        return __beyondacc_metrics_to_losses_map__[beyond_metric]
    else:
        raise NotImplementedError(f'The beyond-accuracy metric `{beyond_metric}` is not supported')
