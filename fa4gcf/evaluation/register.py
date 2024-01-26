from recbole.evaluator.register import cluster_info
from recbole.evaluator import Register as RecboleRegister
from recbole.evaluator.register import metric_information as recbole_metric_information


metric_module_name = "fa4gcf.evaluation.metrics"
smaller_metrics, fa4gcf_metric_information, metric_types, metrics_dict = cluster_info(
    metric_module_name
)


class Register(RecboleRegister):

    def _build_register(self):
        metric_information = {**recbole_metric_information, **fa4gcf_metric_information}
        for metric in self.metrics:
            metric_needs = metric_information[metric]
            for info in metric_needs:
                setattr(self, info, True)
