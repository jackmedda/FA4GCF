import  os
import pickle

import numpy as np


PERT_END_EPOCHS_STUB = ['STUB', 'STUB']


_PERT_COLUMNS = [
    "loss_total",
    "loss_graph_dist",
    "pert_loss",
    "pert_metric",
    "del_edges",
    "epoch",
]


def pert_col_index(col):
    try:
        idx = _PERT_COLUMNS.index(col)
    except ValueError:
        idx = col
    return idx


def load_dp_perturbations_file(base_pert_file):
    cf_data_file = os.path.join(base_pert_file, 'cf_data.pkl')
    model_preds_file = os.path.join(base_pert_file, 'model_rec_test_preds.pkl')

    with open(cf_data_file, 'rb') as file:
        exps = [pickle.load(file)]
    with open(model_preds_file, 'rb') as file:
        model_preds = pickle.load(file)

    return exps, *model_preds


def get_best_pert_early_stopping(pert_data, config_dict):
    if pert_data[-1] == PERT_END_EPOCHS_STUB:
        return pert_data[-2]

    best_epoch = get_best_epoch_early_stopping(pert_data, config_dict)
    epoch_idx = pert_col_index('epoch')

    return [e for e in sorted(pert_data, key=lambda x: abs(x[epoch_idx] - best_epoch)) if e[epoch_idx] <= best_epoch][0]


def get_pert_by_epoch(pert_data, query_epoch):
    epoch_idx = pert_col_index('epoch')

    queried_pert = [e for e in pert_data if e[epoch_idx] == query_epoch]

    return queried_pert[0] if len(queried_pert) > 0 else None


def get_best_epoch_early_stopping(pert_data, config_dict):
    try:
        patience = config_dict['early_stopping']['patience']
    except TypeError:
        patience = config_dict['earlys_patience']

    if pert_data[-1] == PERT_END_EPOCHS_STUB:
        return pert_data[-2][pert_col_index('epoch')]

    max_epoch = max([e[pert_col_index('epoch')] for e in pert_data])
    # the training process stopped because of other condition
    if max_epoch <= patience:
        return max_epoch

    return max_epoch - patience


def remap_edges_recbole_ids(dataset, edges, field2id_token=True):
    mp_edges = []
    for i, _field in enumerate([dataset.uid_field, dataset.iid_field]):
        mp_edges.append([])
        for val in edges[i]:
            idx_val = val

            if field2id_token:
                if _field == dataset.iid_field:
                    idx_val = val - dataset.user_num

                mp_edges[-1].append(dataset.field2id_token[_field][idx_val])
            else:
                if _field == dataset.iid_field:
                    mp_val = dataset.field2token_id[_field][idx_val] + dataset.user_num
                else:
                    mp_val = dataset.field2token_id[_field][idx_val]

                mp_edges[-1].append(mp_val)
    return np.stack(mp_edges)
