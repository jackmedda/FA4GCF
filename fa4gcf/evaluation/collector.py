import copy

import torch
from recbole.evaluator import Collector as RecboleCollector

from fa4gcf.data.utils import update_item_feat_discriminative_attribute


class Collector(RecboleCollector):

    def __init__(self, config):
        super(Collector, self).__init__(config)

    def eval_data_collect(self, eval_data):
        if self.register.need("eval_data.user_feat"):
            if not hasattr(eval_data.dataset, "user_feat"):
                raise AttributeError("Evaluation data does not include user features.")
            self.data_struct.set("eval_data.user_feat", eval_data.dataset.user_feat)

        if self.register.need("eval_data.item_discriminative_feat"):
            temp_dataset = copy.deepcopy(eval_data.dataset)
            update_item_feat_discriminative_attribute(
                temp_dataset,
                self.config['item_discriminative_attribute'],
                self.config['short_head_item_discriminative_ratio'],
                self.config['item_discriminative_map']
            )
            self.data_struct.set("eval_data.item_discriminative_feat", copy.deepcopy(temp_dataset.item_feat))
            del temp_dataset

    def eval_batch_collect(
        self,
        scores_tensor: torch.Tensor,
        interaction,
        positive_u: torch.Tensor,
        positive_i: torch.Tensor,
    ):
        if self.register.need("rec.users"):
            uid_field = self.config["USER_ID_FIELD"]
            self.data_struct.update_tensor("rec.users", interaction[uid_field].to(self.device))

        super(Collector, self).eval_batch_collect(
            scores_tensor,
            interaction,
            positive_u,
            positive_i
        )

    def get_data_struct(self):
        """Get all the evaluation resource that been collected.
        And reset some of outdated resource.
        """
        returned_struct = copy.deepcopy(self.data_struct)
        register_keys = [
            "rec.users",
            "rec.topk",
            "rec.meanrank",
            "rec.score",
            "rec.items",
            "data.label"
        ]
        for key in register_keys:
            if key in self.data_struct:
                del self.data_struct[key]
        return returned_struct
