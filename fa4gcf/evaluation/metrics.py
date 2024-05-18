import numpy as np
import torch
from recbole.evaluator.base_metric import AbstractMetric, TopkMetric
from recbole.evaluator.metrics import NDCG, Precision
from recbole.utils import EvaluatorType

from fa4gcf.evaluation.recommender_evaluation import compute_DP, compute_raw_exposure


class ConsumerMetric(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ["eval_data.user_feat", "rec.users"]
    smaller = True

    def __init__(self, config):
        super(ConsumerMetric, self).__init__(config)
        self.ranking_metric = None
        self.sensitive_attribute = config["sensitive_attribute"]

    def used_info(self, dataobject):
        user_feat = dataobject.get("eval_data.user_feat")
        interaction_users = dataobject.get("rec.users")

        # 0 is the padding sensitive attribute
        group_mask1 = user_feat[self.sensitive_attribute][interaction_users] == 1
        group_mask2 = user_feat[self.sensitive_attribute][interaction_users] == 2

        return group_mask1, group_mask2

    def ranking_metric_info(self, pos_index, pos_len):
        raise NotImplementedError("Use a subclass of ConsumerMetric to calculate a specific ranking metric")

    def calculate_metric(self, dataobject):
        group_mask1, group_mask2 = self.used_info(dataobject)
        pos_index, pos_len = self.ranking_metric.used_info(dataobject)
        result = self.ranking_metric_info(pos_index, pos_len)

        group1_result = result[group_mask1, :].mean(axis=0)
        group2_result = result[group_mask2, :].mean(axis=0)

        if group1_result.shape != group2_result.shape:
            avg_result = torch.full_like(torch.from_numpy(result), torch.inf)
        else:
            avg_result = compute_DP(group1_result, group2_result)

        return self.ranking_metric.topk_result(self.__class__.__name__.lower(), avg_result[None, :])


class DeltaNDCG(ConsumerMetric):

    def __init__(self, config):
        super(DeltaNDCG, self).__init__(config)
        self.ranking_metric = NDCG(config)

    def ranking_metric_info(self, pos_index, pos_len):
        return self.ranking_metric.metric_info(pos_index, pos_len)


class DeltaPrecision(ConsumerMetric):

    def __init__(self, config):
        super(DeltaPrecision, self).__init__(config)
        self.ranking_metric = Precision(config)

    def ranking_metric_info(self, pos_index, pos_len):
        return self.ranking_metric.metric_info(pos_index)


class ProviderMetric(TopkMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ["eval_data.item_discriminative_feat", "rec.items"]
    smaller = True

    def __init__(self, config):
        super(ProviderMetric, self).__init__(config)

        self.SH_KEY = 'SH'
        self.LT_KEY = 'LT'

        self.item_disciminative_attribute = config["item_discriminative_attribute"]
        self.short_head_item_discriminative_ratio = config["short_head_item_discriminative_ratio"]
        self.item_discriminative_map = np.array(config["item_discriminative_map"])
        self.group_distribution = {
            "SH": 1,
            "LT": 1 / self.short_head_item_discriminative_ratio - 1
        }

    def used_info(self, dataobject):
        item_feat = dataobject.get("eval_data.item_discriminative_feat")
        interaction_items = dataobject.get("rec.items")

        sh_id = (self.item_discriminative_map == self.SH_KEY).nonzero()[0].item()
        lt_id = (self.item_discriminative_map == self.LT_KEY).nonzero()[0].item()

        # 0 is the padding
        sh_group = item_feat[self.item_disciminative_attribute] == sh_id
        lt_group = item_feat[self.item_disciminative_attribute] == lt_id

        return sh_group, lt_group, interaction_items

    def get_provider_topk_result(self, sh_result, lt_result):
        sh_result = (sh_result.sum() / self.group_distribution["SH"]).item()
        lt_result = (lt_result.sum() / self.group_distribution["LT"]).item()

        result = compute_DP(sh_result, lt_result)

        return round(result, self.decimal_place)

    def calculate_metric(self, dataobject):
        raise NotImplementedError("Use a subclass of ProviderMetric to calculate a specific metric")


class DeltaExposure(ProviderMetric):

    def calculate_metric(self, dataobject):
        sh_group_mask, lt_group_mask, sorted_recs = self.used_info(dataobject)

        metric_dict = {}
        for k in self.topk:
            topk_recs = sorted_recs[:, :k].numpy()
            exposure_discount = np.log2(np.arange(1, k + 1) + 1)

            sh_exposure = compute_raw_exposure(topk_recs, sh_group_mask.numpy(), exposure_discount)
            lt_exposure = compute_raw_exposure(topk_recs, lt_group_mask.numpy(), exposure_discount)

            key = "{}@{}".format("deltaexposure", k)
            metric_dict[key] = self.get_provider_topk_result(sh_exposure, lt_exposure)
        return metric_dict


class DeltaVisibility(ProviderMetric):

    def calculate_metric(self, dataobject):
        sh_group_mask, lt_group_mask, sorted_recs = self.used_info(dataobject)

        metric_dict = {}
        n_items = sh_group_mask.shape[0]
        for k in self.topk:
            topk_recs = sorted_recs[:, :k]
            raw_visibility = torch.bincount(topk_recs.flatten(), minlength=n_items)
            visibility_prob = raw_visibility / (topk_recs.shape[0] * n_items)  # items exposure / (n_users * n_items)

            sh_visibility_prob = visibility_prob[sh_group_mask]
            lt_visibility_prob = visibility_prob[lt_group_mask]

            key = "{}@{}".format("deltavisibility", k)
            metric_dict[key] = self.get_provider_topk_result(sh_visibility_prob, lt_visibility_prob)
        return metric_dict
