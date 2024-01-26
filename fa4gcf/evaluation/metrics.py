from recbole.evaluator.base_metric import AbstractMetric
from recbole.evaluator.metrics import NDCG
from recbole.utils import EvaluatorType

import gnnuers


class DeltaNDCG(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ["eval_data.user_feat", "rec.users"]
    smaller = True

    def __init__(self, config):
        super(DeltaNDCG, self).__init__(config)
        self.ndcg_metric = NDCG(config)
        self.sensitive_attribute = config["sensitive_attribute"]

    def used_info(self, dataobject):
        user_feat = dataobject.get("eval_data.user_feat")
        interaction_users = dataobject.get("rec.users")

        # 0 is the padding sensitive attribute
        group_mask1 = user_feat[self.sensitive_attribute][interaction_users] == 1
        group_mask2 = user_feat[self.sensitive_attribute][interaction_users] == 2

        return group_mask1, group_mask2

    def calculate_metric(self, dataobject):
        group_mask1, group_mask2 = self.used_info(dataobject)
        pos_index, pos_len = self.ndcg_metric.used_info(dataobject)
        result = self.ndcg_metric.metric_info(pos_index, pos_len)

        group1_result = result[group_mask1, :].mean(axis=0)
        group2_result = result[group_mask2, :].mean(axis=0)

        metric_dict = {}
        avg_result = gnnuers.evaluation.compute_DP(group1_result, group2_result)
        for k in self.ndcg_metric.topk:
            key = "{}@{}".format("deltandcg", k)
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict
