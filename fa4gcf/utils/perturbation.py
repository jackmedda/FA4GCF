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


def get_best_pert_early_stopping(pert_data, config_dict):
    if pert_data[-1] == _PERT_END_EPOCHS_STUB:
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

    if pert_data[-1] == _PERT_END_EPOCHS_STUB:
        return pert_data[-2][pert_col_index('epoch')]

    max_epoch = max([e[pert_col_index('epoch')] for e in pert_data])
    # the training process stopped because of other condition
    if max_epoch <= patience:
        return max_epoch

    return max_epoch - patience
