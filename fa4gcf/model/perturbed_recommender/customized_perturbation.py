import torch
from torch_geometric.typing import SparseTensor

from fa4gcf.model.general_recommender import (
    DirectAU,
    SVD_GCN
)


class PerturbationApplier:

    GENERIC_APPLIER_KEY = "GENERIC_APPLIER"

    def __init__(self):
        self._model2applier = {
            self.GENERIC_APPLIER_KEY: self._apply_generic,
            "DirectAU": self._apply_direct_au,
            "SVD_GCN": self._apply_svd_gcn
        }
        self._model2class = {
            "DirectAU": DirectAU,
            "SVD_GCN": SVD_GCN
        }

    def get_applier(self, model):
        applier_key = model.__class__.__name__
        if applier_key not in self._model2applier:
            # No specific applier support for model [{model.__class__.__name__}]", using the generic one
            applier_key = self.GENERIC_APPLIER_KEY
        else:
            model_class = self._model2class[applier_key]
            if not isinstance(model, model_class):
                raise ValueError(f"The model with name [{applier_key}] is not an instance of [{model_class}]")

        return self._model2applier[applier_key]

    def apply(self, model, edge_index, edge_weight):
        applier = self.get_applier(model)
        applier(model, edge_index, edge_weight)

        if hasattr(model, "restore_user_e") and hasattr(model, "restore_item_e"):
            model.restore_user_e, model.restore_item_e = None, None

        if hasattr(model, "restore_user_rating"):
            model.restore_user_rating = None

    @staticmethod
    def _apply_generic(model, edge_index, edge_weight):
        model.edge_index = edge_index
        model.edge_weight = edge_weight

    @staticmethod
    def _apply_svd_gcn(model, edge_index, edge_weight):
        if isinstance(edge_index, SparseTensor):
            rate_matrix = edge_index
        else:
            rate_matrix = torch.sparse_coo_tensor(
                edge_index,
                edge_weight,
                size=(model.n_users, model.n_items),
                device=model.device
            )
        model.rate_matrix = rate_matrix
        U, value, V = model.get_svd(q)

        weighted_value = self._apply_value_weight(value)

        model.user_vector = U * weighted_value
        model.item_vector = V * weighted_value

    @staticmethod
    def _apply_direct_au(model, edge_index, edge_weight):
        model.encoder.edge_index = edge_index
        model.encoder.edge_weight = edge_weight
