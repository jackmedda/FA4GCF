import os
import pickle
import logging

import torch
import numpy as np
import pandas as pd
from torch_geometric.typing import torch_sparse
from recbole.utils import FeatureType, FeatureSource, set_color

import fa4gcf.data.utils as data_utils
import fa4gcf.utils as utils
from fa4gcf.data.dataset import Dataset
from fa4gcf.data.interaction import np_unique_cat_recbole_interaction


class PerturbedDataset(Dataset):
    FEATS = [FeatureSource.INTERACTION, FeatureSource.ITEM]  # the perturbed dataset never deletes users

    def __init__(self, config, perturbations_path, best_pert):
        self.perturbations_path = perturbations_path
        self.__base_dataset = Dataset(config)  # must be instantiated to get the recbole mappings of user and item ids

        self.best_pert = best_pert
        self.best_perturbation = None
        self.pert_config = None
        self.perturbed_edges = self.load_perturbed_edges()
        self.mapped_perturbed_edges = utils.remap_edges_recbole_ids(
            self.__base_dataset, self.perturbed_edges, field2id_token=True
        )

        super(PerturbedDataset, self).__init__(config)

    def get_best_perturbation(self, perts, pert_config):
        if self.best_pert[0] == "fairest":
            best_pert = utils.get_best_pert_early_stopping(perts, pert_config)
        elif self.best_pert[0] == "fixed_pert":
            best_pert = utils.get_pert_by_epoch(perts, self.best_pert[1])
        else:
            raise ValueError("`best_pert` must be set to select the perturbation of a specific epoch")

        return best_pert

    def load_perturbed_edges(self):
        logger = logging.getLogger('FA4GCF')

        with open(os.path.join(self.perturbations_path, 'cf_data.pkl'), 'rb') as perts_file:
            perts = pickle.load(perts_file)
        logger.info(f"Original Fair Loss: {perts[0][-1]}")

        self.pert_config = utils.read_recbole_config_skip_errors(
            os.path.join(self.perturbations_path, 'config.yaml'),
            self.__base_dataset.config
        )
        logger.info(self.pert_config)

        pert_rec_data_key = 'pert_rec_data' if 'pert_rec_data' in self.pert_config else 'exp_rec_data'
        if self.pert_config[pert_rec_data_key] != 'valid':
            logger.warning('Performing Graph Augmentation on Perturbation NOT produced on Validation Data.')

        self.best_perturbation = self.get_best_perturbation(perts, self.pert_config)

        return self.best_perturbation[utils.pert_col_index('del_edges')]

    def perturb_split(self, split):
        if split in self.SPLITS:
            spl_file = os.path.join(self.config['data_path'], f"{self.config['dataset']}.{split}")
            if not os.path.isfile(spl_file):
                return None

            with open(spl_file, 'rb') as split_data_file:
                split_data = pickle.load(split_data_file)

            if isinstance(split_data[self.uid_field], torch.Tensor):
                split_data = {k: v.numpy() for k, v in split_data.items()}

            # Recbole tokens are always treated as strings (e.g., user\item ids)
            split_data = {k: v.astype(str) for k, v in split_data.items()}

            mapped_split_data = utils.remap_edges_recbole_ids(
                self.__base_dataset,
                np.stack((split_data[self.uid_field], split_data[self.iid_field])),
                field2id_token=False
            )

            new_split_data, unique, counts = np_unique_cat_recbole_interaction(
                mapped_split_data, self.perturbed_edges,
                uid_field=self.uid_field, iid_field=self.iid_field, return_unique_counts=True
            )

            if split != "train":
                # validation and test set should not be affected by any deleted edge in the training set
                # but if an edge is added to the training set, it must be removed from the validation or test set
                common_interactions = unique[:, counts > 1]
                new_split_data = np_unique_cat_recbole_interaction(
                    mapped_split_data, common_interactions, uid_field=self.uid_field, iid_field=self.iid_field
                )

            new_split_data = utils.remap_edges_recbole_ids(
                self.__base_dataset,
                np.stack((new_split_data[self.uid_field], new_split_data[self.iid_field])),
                field2id_token=True
            )

            return dict(zip([self.uid_field, self.iid_field], new_split_data))
        else:
            raise ValueError(f"split `{split}` is not supported for LRS configuration")

    def perturb_feat(self, source=FeatureSource.INTERACTION):
        if source in self.FEATS:
            # new dataset is the merge of splits => it takes care of particular cases. For instance, an edge in the
            # validation set that was added to the training set is removed from the validation set and kept in training
            pert_splits = [self.perturb_split(split) for split in self.SPLITS]
            merged_splits = pert_splits[0]
            for p_split in pert_splits[1:]:
                for key in merged_splits:
                    merged_splits[key] = np.concatenate([merged_splits[key], p_split[key]])

            # Recbole tokens are always treated as strings (e.g., user\item ids)
            pert_df = pd.DataFrame(merged_splits).astype(str)

            feat_df_cols = [self.uid_field, self.iid_field]
            if source == FeatureSource.INTERACTION:
                feat_df = self.__base_dataset.inter_feat
            elif source == FeatureSource.ITEM and self.__base_dataset.item_feat is not None:
                feat_df = self.__base_dataset.item_feat
                pert_df = pert_df[[self.iid_field]]
            else:
                raise ValueError(f"feat {source} not supported for modification when re-training with perturbed data")

            for field, token_map in self.__base_dataset.field2id_token.items():
                if field in feat_df.columns:
                    # Recbole tokens are always treated as strings (e.g., user\item ids)
                    feat_df[field] = feat_df[field].map(token_map.__getitem__).astype(str)
            new_feat_df = pert_df.join(feat_df.set_index(feat_df_cols), on=feat_df_cols, how="left")

            return new_feat_df
        else:
            raise ValueError(f"feat {source} not supported for modification when re-training with perturbed data")

    def _remap(self, remap_list):
        tokens, split_point = self._concat_remaped_tokens(remap_list)

        common_field = remap_list[0][1]
        new_ids_list = [self.__base_dataset.field2token_id[common_field][orig_id] for orig_id in tokens]

        new_ids_list = np.split(new_ids_list, split_point)

        mp = self.__base_dataset.field2id_token[common_field]
        token_id = self.__base_dataset.field2token_id[common_field]

        for (feat, field, ftype), new_ids in zip(remap_list, new_ids_list):
            if field not in self.field2id_token:
                self.field2id_token[field] = mp
                self.field2token_id[field] = token_id
            if ftype == FeatureType.TOKEN:
                feat[field] = new_ids
            elif ftype == FeatureType.TOKEN_SEQ:
                split_point = np.cumsum(feat[field].agg(len))[:-1]
                feat[field] = np.split(new_ids, split_point)

    def _load_data_split(self, split):
        filename = os.path.join(self.dataset_path, f'{self.dataset_name}.{split}')
        if not os.path.isfile(filename):
            if split in ['train', 'test']:
                raise NotImplementedError(f'The splitting method "LRS" needs at least train and test.')

        split_data = self.perturb_split(split)

        return split_data

    def _load_feat(self, filepath, source):
        """Load features according to source into :class:`pandas.DataFrame`.

        Set features' properties, e.g. type, source and length.

        Args:
            filepath (str): path of input file.
            source (FeatureSource or str): source of input file.

        Returns:
            pandas.DataFrame: Loaded feature

        Note:
            For sequence features, ``seqlen`` will be loaded, but data in DataFrame will not be cut off.
            Their length is limited only after calling :meth:`~_dict_to_interaction` or
            :meth:`~_dataframe_to_interaction`
        """
        self.logger.debug(set_color(f'Loading feature from [{filepath}] (source: [{source}]).', 'green'))

        load_col, unload_col = self._get_load_and_unload_col(source)
        if load_col == set():
            return None

        field_separator = self.config['field_separator']
        columns = []
        usecols = []
        dtype = {}
        encoding = self.config['encoding']
        with open(filepath, 'r', encoding=encoding) as f:
            head = f.readline()[:-1]
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(':')
            try:
                ftype = FeatureType(ftype)
            except ValueError:
                raise ValueError(f'Type {ftype} from field {field} is not supported.')
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            if isinstance(source, FeatureSource) or source != 'link':
                self.field2source[field] = source
                self.field2type[field] = ftype
                if not ftype.value.endswith('seq'):
                    self.field2seqlen[field] = 1
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == FeatureType.FLOAT else str

        if len(columns) == 0:
            self.logger.warning(f'No columns has been loaded from [{source}]')
            return None

        if source in self.FEATS:
            df: pd.DataFrame = self.perturb_feat(source)
        else:
            df = pd.read_csv(
                filepath, delimiter=field_separator, usecols=usecols, dtype=dtype, encoding=encoding, engine='python'
            )
            df.columns = columns

        seq_separator = self.config['seq_separator']
        for field in columns:
            ftype = self.field2type[field]
            if not ftype.value.endswith('seq'):
                continue
            df[field].fillna(value='', inplace=True)
            if ftype == FeatureType.TOKEN_SEQ:
                df[field] = [np.array(list(filter(None, _.split(seq_separator)))) for _ in df[field].values]
            elif ftype == FeatureType.FLOAT_SEQ:
                df[field] = [np.array(list(map(float, filter(None, _.split(seq_separator))))) for _ in df[field].values]
            self.field2seqlen[field] = max(map(len, df[field].values))
        return df

    @staticmethod
    def edge_index_to_adj_t(edge_index, edge_weight, m_num_nodes, n_num_nodes):
        return data_utils.edge_index_to_adj_t(edge_index, edge_weight, m_num_nodes, n_num_nodes)

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

        return data_utils.get_norm_adj_mat(
            edge_index,
            edge_weight,
            num_nodes,
            add_self_loops=self.add_self_loops,
            enable_sparse=enable_sparse,
            is_sparse=self.is_sparse
        )
