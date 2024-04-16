import warnings

import torch
import numpy as np
from recbole.utils.enum_type import InputType
from recbole.data.dataloader.general_dataloader import TrainDataLoader

from fa4gcf.data import Interaction


class NegSampleUserItemNeighborDataLoader(TrainDataLoader):
    """:class:`NegSampleUserItemNeighborDataLoader` is a dataloader for sampling pos/neg user item neighbors.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffled after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        self.neighbor_prefix = config["NEIGHBOR_PREFIX"] or "neighbor"

        self.NEIGHBOR_USER_ID = f"{self.neighbor_prefix}_{self.uid_field}"
        self.NEG_NEIGHBOR_USER_ID = f"{self.neg_prefix}_{self.NEIGHBOR_USER_ID}"
        self.NEIGHBOR_ITEM_ID = f"{self.neighbor_prefix}_{self.iid_field}"
        self.NEG_NEIGHBOR_ITEM_ID = f"{self.neg_prefix}_{self.NEIGHBOR_ITEM_ID}"
        config["NEIGHBOR_USER_ID"] = self.NEIGHBOR_USER_ID
        config["NEG_NEIGHBOR_USER_ID"] = self.NEG_NEIGHBOR_USER_ID
        config["NEIGHBOR_ITEM_ID"] = self.NEIGHBOR_ITEM_ID
        config["NEG_NEIGHBOR_ITEM_ID"] = self.NEG_NEIGHBOR_ITEM_ID

        self._init_user_item_neighbors_matrix()

    def update_config(self, config):
        self._set_neg_sample_args(
            config,
            self._dataset,
            InputType.PAIRWISE,
            config["train_neg_sample_args"],
        )
        super().update_config(config)

    def _init_user_item_neighbors_matrix(self):
        inter_matrix = self._dataset.inter_matrix()

        self.user_neighbor_matrix = torch.from_numpy(
            (inter_matrix.dot(inter_matrix.T) != 0).astype(float).toarray()
        )
        self.item_neighbor_matrix = torch.from_numpy(
            (inter_matrix.T.dot(inter_matrix) != 0).astype(float).toarray()
        )

        self.neg_user_neighbor_matrix = 1 - self.user_neighbor_matrix
        self.neg_item_neighbor_matrix = 1 - self.item_neighbor_matrix

    def _one_sample_from_distribution(self, inter_data, data_matrix, distribution='multinomial'):
        if distribution == 'multinomial':
            # neg neighbor sampling always works thanks to the padding user/item
            sampled_data = torch.multinomial(data_matrix[inter_data], 1, True).squeeze()
        else:
            raise NotImplementedError(
                f'the distribution function [{distribution}] is not supported for NegSampleUserItemNeighbors transform'
            )

        return sampled_data

    def _times_sample_from_distribution(self, inter_data, data_matrix, distribution='multinomial'):
        """
        Returns the multinomial distribution based on the negative sampling.
        The sampling is performed only once on data_matrix of the size `self.times`, then flattened accordingly.
        More efficient than _one_sample_from_distribution
        """
        if distribution == 'multinomial':
            # neg neighbor sampling always works thanks to the padding user/item
            sampled_data = torch.multinomial(data_matrix[inter_data], self.times, True).squeeze()
            if sampled_data.dim() > 2:
                print("MULTINOMIAL DISTRIBUTION OUTCOME HAS MORE THAN 2 DIMS")
                import pdb; pdb.set_trace()
            sampled_data = sampled_data.T.flatten()
        else:
            raise NotImplementedError(
                f'the distribution function [{distribution}] is not supported for NegSampleUserItemNeighbors transform'
            )

        return sampled_data

    def _user_item_neighbor_sampling(self,
                                     transformed_feat,
                                     neg_sampled_feat,
                                     neighbor_matrix,
                                     neg_neighbor_matrix,
                                     field,
                                     neighbor_field,
                                     neg_neighbor_field,
                                     sample_fn="times"):
        if sample_fn == "times":
            pos_neighbor = self._times_sample_from_distribution(transformed_feat[field], neighbor_matrix)
            neg_neighbor = self._times_sample_from_distribution(transformed_feat[field], neg_neighbor_matrix)
        elif sample_fn == "one":
            pos_neighbor = self._times_sample_from_distribution(neg_sampled_feat[field], neighbor_matrix)
            neg_neighbor = self._times_sample_from_distribution(neg_sampled_feat[field], neg_neighbor_matrix)
        else:
            raise NotImplementedError(f"`_user_item_neighbor_sampling` inner sample_fn cannot be [{sample_fn}]")

        neighbor_feat = Interaction({
            neighbor_field: pos_neighbor,
            neg_neighbor_field: neg_neighbor
        })
        neighbor_feat = self._dataset.join(neighbor_feat)
        neg_sampled_feat.update(neighbor_feat)
        return neg_sampled_feat

    def _user_neighbor_sampling(self, transformed_feat, neg_sampled_feat):
        return self._user_item_neighbor_sampling(
            transformed_feat,
            neg_sampled_feat,
            self.user_neighbor_matrix,
            self.neg_user_neighbor_matrix,
            self.uid_field,
            self.NEIGHBOR_USER_ID,
            self.NEG_NEIGHBOR_USER_ID
        )

    def _item_neighbor_sampling(self, transformed_feat, neg_sampled_feat):
        return self._user_item_neighbor_sampling(
            transformed_feat,
            neg_sampled_feat,
            self.item_neighbor_matrix,
            self.neg_item_neighbor_matrix,
            self.iid_field,
            self.NEIGHBOR_ITEM_ID,
            self.NEG_NEIGHBOR_ITEM_ID
        )

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        transformed_data = self.transform(self._dataset, data)
        output_data = self._neg_sampling(transformed_data)
        output_data = self._user_neighbor_sampling(transformed_data, output_data)
        output_data = self._item_neighbor_sampling(transformed_data, output_data)

        return output_data


class SVD_GCNDataLoader(NegSampleUserItemNeighborDataLoader):
    """:class:`SVD_GCNDataLoader` is a dataloader for training SVD_GCN.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffled after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.use_user_sampling = (
                config['user_coefficient'] is not None and
                config['user_coefficient'] > 0 and
                config['parametric'] is True
        )
        self.use_item_sampling = (
                config['item_coefficient'] is not None and
                config['item_coefficient'] > 0 and
                config['parametric'] is True
        )

        super().__init__(config, dataset, sampler, shuffle=shuffle)

        if not config['parametric']:
            self.neg_sample_num = 1
            self.times = 1

    def _init_user_item_neighbors_matrix(self):
        if self.use_user_sampling or self.use_item_sampling:
            inter_matrix = self._dataset.inter_matrix()

            if self.use_user_sampling:
                self.user_neighbor_matrix = torch.from_numpy(
                    (inter_matrix.dot(inter_matrix.T) != 0).astype(float).toarray()
                )
                self.neg_user_neighbor_matrix = 1 - self.user_neighbor_matrix
            if self.use_item_sampling:
                self.item_neighbor_matrix = torch.from_numpy(
                    (inter_matrix.T.dot(inter_matrix) != 0).astype(float).toarray()
                )
                self.neg_item_neighbor_matrix = 1 - self.item_neighbor_matrix

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        transformed_data = self.transform(self._dataset, data)
        output_data = self._neg_sampling(transformed_data)

        if self.use_user_sampling:
            output_data = self._user_neighbor_sampling(transformed_data, output_data)
        if self.use_item_sampling:
            output_data = self._item_neighbor_sampling(transformed_data, output_data)

        return output_data


class AutoCFDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        neg_sample_num = config["train_neg_sample_args"]["sample_num"]
        if neg_sample_num > 1:
            warnings.warn("AutoCF does not use negative sampling. Setting [sample_num] to 1.")

        config["train_neg_sample_args"]["sample_num"] = 1
        super().__init__(config, dataset, sampler, shuffle=shuffle)
