import torch
import numba
import numpy as np
import pandas as pd

from recbole.evaluator.collector import DataStruct


def get_scores(model, batched_data, tot_item_num, item_tensor, **kwargs):
    interaction, history_index, _, _ = batched_data
    inter_data = interaction.to(model.device)
    if 'pred' in kwargs:
        if kwargs['pred'] is None:
            kwargs.pop('pred')
    full_sort_predict = False
    try:
        scores = model.full_sort_predict(inter_data, **kwargs)
        full_sort_predict = True
    except NotImplementedError:
        scores = []
        for user_i in range(len(inter_data)):
            new_inter = inter_data[[user_i]].repeat_interleave(tot_item_num)
            new_inter.update(item_tensor)
            scores.append(model.predict(new_inter, **kwargs))
        scores = torch.cat(scores, dim=0)

    scores = scores.view(-1, tot_item_num)
    scores[:, 0] = -np.inf
    if model.ITEM_ID in interaction and full_sort_predict:
        scores = scores[:, inter_data[model.ITEM_ID]]
    if history_index is not None:
        scores[history_index] = -np.inf

    return scores


def get_top_k(scores_tensor, topk=10):
    scores_top_k, topk_idx = torch.topk(scores_tensor, topk, dim=-1)  # n_users x k

    return scores_top_k, topk_idx


def compute_metric(evaluator, dataset, pref_data, pred_col, metric, hist_matrix=None):
    # useful to use a different history from the dataset one
    if hist_matrix is None:
        hist_matrix, _, _ = dataset.history_item_matrix()
    dataobject = DataStruct()
    uid_list = pref_data['user_id'].to_numpy()

    pos_matrix = np.zeros((dataset.user_num, dataset.item_num), dtype=int)
    pos_matrix[uid_list[:, None], hist_matrix[uid_list]] = 1
    pos_matrix[:, 0] = 0
    pos_len_list = torch.tensor(pos_matrix.sum(axis=1, keepdims=True))
    pos_idx = torch.tensor(pos_matrix[uid_list[:, None], np.stack(pref_data[pred_col].values)])
    pos_data = torch.cat((pos_idx, pos_len_list[uid_list]), dim=1)

    dataobject.set('rec.topk', pos_data)

    pos_index, pos_len = evaluator.metric_class[metric].used_info(dataobject)
    if metric in ['hit', 'mrr', 'precision']:
        result = evaluator.metric_class[metric].metric_info(pos_index)
    else:
        result = evaluator.metric_class[metric].metric_info(pos_index, pos_len)

    return result


def compute_DP_across_random_samples(df,
                                     sens_attr,
                                     demo_group_field,
                                     dataset_name,
                                     metric,
                                     iterations=100,
                                     batch_size=64,
                                     seed=124):
    np.random.seed(seed)

    if not hasattr(compute_DP_across_random_samples, "generated_groups"):
        compute_DP_across_random_samples.generated_groups = {}

    df = df.sort_values(demo_group_field)
    max_user = df['user_id'].max() + 1

    n_users = 0
    demo_groups_order = []
    size_perc = np.zeros((2,), dtype=float)
    groups = np.zeros((2, max_user), dtype=int)
    for i, (dg, gr_df) in enumerate(df.groupby(demo_group_field)):
        gr_users = gr_df['user_id'].unique()
        groups[i, gr_users] = 1
        n_users += gr_users.shape[0]
        size_perc[i] = gr_users.shape[0]
        demo_groups_order.append(dg)
    size_perc /= n_users

    gr_data = np.zeros(max_user)
    for gr_users in groups:
        pos = gr_users.nonzero()[0]
        gr_data[pos] = df.set_index('user_id').loc[pos, metric].to_numpy()

    if (dataset_name, sens_attr) not in compute_DP_across_random_samples.generated_groups:
        compute_DP_across_random_samples.generated_groups[(dataset_name, sens_attr)] = np.zeros(
            (iterations, 2, max_user), dtype=np.bool_
        )

    return _compute_DP_random_samples(
        gr_data,
        groups,
        size_perc,
        compute_DP_across_random_samples.generated_groups[(dataset_name, sens_attr)],
        batch_size=batch_size,
        iterations=iterations
    ), demo_groups_order


@numba.jit(nopython=True, parallel=True)
def _compute_DP_random_samples(group_data, groups, size_perc, out_samples, batch_size=64, iterations=100):
    out = np.empty((iterations, 3), dtype=np.float32)
    check = out_samples.nonzero()[0].shape[0] == 0
    for i in numba.prange(iterations):
        if check:
            samples = np.zeros_like(groups, dtype=np.bool_)
            for gr_i in range(len(groups)):
                sample_size = round(batch_size * size_perc[gr_i])
                samples[gr_i][np.random.choice(groups[gr_i].nonzero()[0], sample_size, replace=False)] = True
            out_samples[i] = samples

        gr1_mean = group_data[out_samples[i, 0]].mean()
        gr2_mean = group_data[out_samples[i, 1]].mean()

        dp = compute_DP_with_masks(group_data, out_samples[i, 0], out_samples[i, 1])
        out[i] = [gr1_mean, gr2_mean, dp]

    return out


@numba.jit(nopython=True)
def compute_DP_with_masks(eval_data, gr1_mask, gr2_mask):
    gr1_mean = eval_data[gr1_mask].mean()
    gr2_mean = eval_data[gr2_mask].mean()
    return compute_DP(gr1_mean, gr2_mean)


@numba.jit(nopython=True)
def compute_DP(gr1_result, gr2_result):
    return np.abs(gr1_result - gr2_result)


def compute_provider_DP(pref_data, dataset, discriminative_attribute, groups_distribution_ratio, topk_column='topk_pred'):
    prov_metr_per_gr = compute_provider_metric_per_group(
        pref_data, dataset, discriminative_attribute, groups_distribution_ratio, topk_column=topk_column
    )
    return compute_DP(*[prov_metr.numpy() for prov_metr in prov_metr_per_gr.values()])


def compute_provider_metric_per_group(pref_data,
                                      dataset,
                                      discriminative_attribute,
                                      groups_distribution_ratio,
                                      topk_column='topk_pred',
                                      raw=False):
    groups_distrib = [groups_distribution_ratio[0] / groups_distribution_ratio[1], 1 - groups_distribution_ratio[0] / groups_distribution_ratio[1]]

    item_discrim_map = np.asarray(dataset.field2id_token[discriminative_attribute])

    metric_sh = compute_provider_raw_metric(
        pref_data, dataset, discriminative_attribute, (item_discrim_map == 'SH').nonzero()[0].item(), topk_column=topk_column
    )
    metric_lt = compute_provider_raw_metric(
        pref_data, dataset, discriminative_attribute, (item_discrim_map == 'LT').nonzero()[0].item(), topk_column=topk_column
    )

    if raw:
        item_df = pd.DataFrame({k: v for k, v in dataset.item_feat.numpy().items() if k in [dataset.iid_field, discriminative_attribute]})
        item_df.rename(columns={discriminative_attribute: discriminative_attribute + '_group'}, inplace=True)

        # they are vectors with same dimension, but completely different indices > 0
        metric_norm_distrib = (metric_sh / groups_distrib[0]) + (metric_lt / groups_distrib[1])
        item_df[discriminative_attribute] = metric_norm_distrib

        result = item_df
    else:
        metric_sh = metric_sh.sum() / groups_distrib[0]
        metric_lt = metric_lt.sum() / groups_distrib[1]

        result = {'SH': metric_sh, 'LT': metric_lt}

    return result


def compute_provider_raw_metric(pref_data, dataset, provider_metric, group, topk_column='topk_pred'):
    topk_recs = torch.stack(tuple(pref_data[topk_column].map(torch.Tensor).values)).long()
    k = topk_recs.shape[1]

    mask = dataset.item_feat[provider_metric] == group

    if provider_metric == 'visibility':
        visibility = torch.bincount(topk_recs.flatten(), minlength=dataset.item_num)
        visibility = visibility[mask] / torch.multiply(*topk_recs.shape)

        metric = torch.zeros_like(mask, dtype=visibility.dtype)
        metric[mask] = visibility
    elif provider_metric == 'exposure':
        exposure_discount = np.log2(np.arange(1, k + 1) + 1)

        # metric = ((metric / exposure_discount).sum(dim=1) / (1 / exposure_discount).sum()).mean()
        metric = torch.from_numpy(compute_raw_exposure(topk_recs.numpy(), mask.numpy(), exposure_discount))
    else:
        raise NotImplementedError(f'The provider metric `{provider_metric}` is not supported')

    return metric


@numba.jit(nopython=True, parallel=True)
def compute_raw_exposure(topk_recs, mask, exposure_discount):
    exposure = np.zeros_like(mask, dtype=np.float32)
    items_ids = np.flatnonzero(mask)
    exp_disc_sum = (1 / exposure_discount).sum()

    for i in numba.prange(items_ids.shape[0]):
        item_id = items_ids[i]

        item_mask = np.zeros_like(mask, dtype=np.bool_)
        item_mask[item_id] = True

        item_presence = np.take(item_mask, topk_recs)
        item_exposure = item_presence / exposure_discount
        item_exposure = (item_exposure / exp_disc_sum).sum(axis=1).mean()
        exposure[item_id] = item_exposure

    return exposure


def compute_beyondaccuracy_metric(beyondacc_metric, *args, **kwargs):
    if beyondacc_metric == "consumer_DP":
        return compute_consumer_DP(*args, **kwargs)
    elif beyondacc_metric == "consumer_DP_across_random_samples":
        return compute_consumer_DP_across_random_samples(*args, **kwargs)
    elif beyondacc_metric == "provider_DP":
        dataset = kwargs.pop('dataset')
        discrim_attr = kwargs.pop('discriminative_attribute')
        groups_distrib = kwargs.pop('groups_distrib')

        return compute_provider_DP(
            args[0],
            dataset,
            discrim_attr,
            groups_distrib,
            topk_column='topk_pred'
        )
    elif beyondacc_metric == "UC":
        return compute_UC(*args, **kwargs)
    else:
        raise NotImplementedError(f'beyond-accuracy metric `{beyondacc_metric}` is not implemented.')


def compute_consumer_DP(pref_data,
                        eval_metric,
                        *args,
                        **kwargs):
    gr_results = []
    for gr_mask in pref_data.groupby('Demo Group').groups.values():
        gr_results.append(pref_data.loc[gr_mask, eval_metric].mean())

    return compute_DP(*gr_results)


def compute_consumer_DP_across_random_samples(pref_data,
                                              eval_metric,
                                              dset_name,
                                              sens_attr,
                                              batch_size,
                                              iterations=100):
    # minus 1 encodes the demographic groups to {0, 1} instead of {1, 2}
    pref_data['Demo Group'] -= 1

    # it prevents from using memoization
    if hasattr(compute_DP_across_random_samples, "generated_groups"):
        if (dset_name, sens_attr) in compute_DP_across_random_samples.generated_groups:
            del compute_DP_across_random_samples.generated_groups[(dset_name, sens_attr)]

    exp_metric_value, _ = compute_DP_across_random_samples(
        pref_data, sens_attr, 'Demo Group', dset_name, eval_metric, iterations=iterations, batch_size=batch_size
    )

    return exp_metric_value[:, -1].mean()


def compute_UC(pref_data,
               eval_metric,
               *args,
               **kwargs):
    min_rel_items = kwargs.get('coverage_min_relevant_items', 0)
    if min_rel_items > 0:
        raise NotImplementedError('`coverage_min_relevant_items` cannot be > 0. Not implemented')

    return (pref_data.loc[:, eval_metric] > min_rel_items).astype(int).sum() / pref_data.shape[0]


def compute_edge_perturbation_impact(dataset, pert_edges, attribute, consumer=False):
    pert_edges = pert_edges.copy()
    total_pert_edges = pert_edges.shape[1]
    attr_map = dataset.field2id_token[attribute]

    if (pert_edges[1] > dataset.user_num).all():
        pert_edges[1] -= dataset.user_num

    if consumer:
        data_feat = dataset.user_feat
        data_num = dataset.user_num
        pert_edges_idx = 0
    else:
        data_feat = dataset.item_feat
        data_num = dataset.item_num
        pert_edges_idx = 1

    gr1_mask = data_feat[attribute] == 1
    gr2_mask = data_feat[attribute] == 2

    pert_edges_per_gr = np.bincount(pert_edges[pert_edges_idx], minlength=data_num)
    gr1_pert_edges = pert_edges_per_gr[gr1_mask]
    gr2_pert_edges = pert_edges_per_gr[gr2_mask]

    gr1_ei_ratio = gr1_pert_edges.sum() / total_pert_edges
    gr2_ei_ratio = gr2_pert_edges.sum() / total_pert_edges

    gr1_repr_ratio = gr1_mask.sum() / data_num
    gr2_repr_ratio = gr2_mask.sum() / data_num

    return {
        attr_map[1]: (gr1_ei_ratio / gr1_repr_ratio).item(),
        attr_map[2]: (gr2_ei_ratio / gr2_repr_ratio).item()
    }
