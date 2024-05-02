from recbole.trainer import TraditionalTrainer as RecboleTraditionalTrainer

from fa4gcf.trainer import TraditionalTrainer


def is_model_saveable(config, trainer):
    if config["model"] == "SVD_GCN":
        return True  # config["parametric"]

    return not isinstance(trainer, (RecboleTraditionalTrainer, TraditionalTrainer))
