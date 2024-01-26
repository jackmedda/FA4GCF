from recbole.evaluator import Evaluator as RecboleEvaluator
from recbole.evaluator.register import metrics_dict as recbole_metrics_dict

from fa4gcf.evaluation.register import metrics_dict as fa4gcf_metrics_dict


class Evaluator(RecboleEvaluator):

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self.metric_class = {}

        metrics_dict = {**recbole_metrics_dict, **fa4gcf_metrics_dict}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)
