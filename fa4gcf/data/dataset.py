import os
import pickle

import torch
import numpy as np
from torch_geometric.typing import torch_sparse
from recbole.data.dataset import Dataset as RecboleDataset

import fa4gcf.data.utils as utils
from fa4gcf.data.interaction import Interaction


class Dataset(RecboleDataset):
    SPLITS = ['train', 'validation', 'test']

    def __init__(self, config):
        if 'LRS' not in config['eval_args']['split']:
            raise ValueError("Perturbed graph can be used only when splits are loaded.")

        super(Dataset, self).__init__(config)

    def _discretization(self):
        if self.config['numerical_features'] is None:
            self.config['numerical_features'] = []
        super(Dataset, self)._discretization()

    def _load_data_split(self, split):
        filename = os.path.join(self.dataset_path, f'{self.dataset_name}.{split}')
        if not os.path.isfile(filename):
            if split in ['train', 'test']:
                raise NotImplementedError(f'The splitting method "LRS" needs at least train and test.')
            return None

        with open(filename, 'rb') as split_file:
            split_data: dict = pickle.load(split_file)

        return split_data

    def split_by_loaded_splits(self):
        next_df = []
        for split in self.SPLITS:
            split_data = self._load_data_split(split)

            if split_data is None:
                next_df.append(self.inter_feat[[]])
            else:
                split_keys = list(split_data.keys())
                if self.uid_field not in split_keys or self.iid_field not in split_keys or len(split_keys) != 2:
                    raise ValueError(f'The splitting grouping method "LRS" should contain only the fields '
                                     f'`{self.uid_field}` and `{self.iid_field}`.')

                for field in [self.uid_field, self.iid_field]:
                    data = split_data[field]
                    data = data.numpy() if isinstance(data, torch.Tensor) else data
                    split_data[field] = torch.LongTensor([self.field2token_id[field][val] for val in data.astype(str)])

                split_df = self.inter_feat.setintersection_by_user_item_ids(
                    Interaction(split_data),
                    uid_field=self.uid_field,
                    iid_field=self.iid_field,
                    return_unique_counts=False
                )

                next_df.append(split_df)

        next_ds = [self.copy(_) for _ in next_df]
        for ds in next_ds:
            ds._drop_unused_col()
        return next_ds

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        self._change_feat_format()

        if self.benchmark_filename_list is not None:
            self._drop_unused_col()
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [
                self.copy(self.inter_feat[start:end])
                for start, end in zip([0] + cumsum[:-1], cumsum)
            ]
            return datasets

        # ordering
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args == "RO":
            self.shuffle()
        elif ordering_args == "TO":
            self.sort(by=self.time_field)
        else:
            raise NotImplementedError(
                f"The ordering_method [{ordering_args}] has not been implemented."
            )

        # splitting & grouping
        split_args = self.config["eval_args"]["split"]
        if split_args is None:
            raise ValueError("The split_args in eval_args should not be None.")
        if not isinstance(split_args, dict):
            raise ValueError(f"The split_args [{split_args}] should be a dict.")

        split_mode = list(split_args.keys())[0]
        assert len(split_args.keys()) == 1
        group_by = self.config["eval_args"]["group_by"]
        if split_mode == "RS":
            if not isinstance(split_args["RS"], list):
                raise ValueError(f'The value of "RS" [{split_args}] should be a list.')
            if group_by is None or group_by.lower() == "none":
                datasets = self.split_by_ratio(split_args["RS"], group_by=None)
            elif group_by == "user":
                datasets = self.split_by_ratio(
                    split_args["RS"], group_by=self.uid_field
                )
            else:
                raise NotImplementedError(
                    f"The grouping method [{group_by}] has not been implemented."
                )
        elif split_mode == "LS":
            datasets = self.leave_one_out(
                group_by=self.uid_field, leave_one_mode=split_args["LS"]
            )
        elif split_mode == "LRS":  # Load Ready Splits
            datasets = self.split_by_loaded_splits()
        else:
            raise NotImplementedError(
                f"The splitting_method [{split_mode}] has not been implemented."
            )

        return datasets

    @staticmethod
    def edge_index_to_adj_t(edge_index, edge_weight, m_num_nodes, n_num_nodes):
        return utils.edge_index_to_adj_t(edge_index, edge_weight, m_num_nodes, n_num_nodes)

    def get_norm_adj_mat(self, enable_sparse=False):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """
        self.is_sparse = torch_sparse is not object

        row = self.inter_feat[self.uid_field]
        col = self.inter_feat[self.iid_field] + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        edge_weight = torch.ones(edge_index.size(1))
        num_nodes = self.user_num + self.item_num

        self.add_self_loops = self.config.model in ["AutoCF"]

        if enable_sparse:
            if not self.is_sparse:
                self.logger.warning(
                    "Import `torch_sparse` error, please install corresponding version of `torch_sparse`."
                    "Dense edge_index will be used instead of SparseTensor in dataset."
                )

        return utils.get_norm_adj_mat(
            edge_index,
            edge_weight,
            num_nodes,
            add_self_loops=self.add_self_loops,
            enable_sparse=enable_sparse,
            is_sparse=self.is_sparse
        )
