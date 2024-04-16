import math

import torch
import numpy as np
import pandas as pd

from fa4gcf.data import Interaction


def prepare_batched_data(input_data, data, item_data=None):
    """
    Prepare the batched data according to the "recbole" pipeline
    :param input_data:
    :param data:
    :param item_data:
    :return:
    """
    data_df = Interaction({k: v[input_data] for k, v in data.dataset.user_feat.interaction.items()})

    if item_data is not None:
        data_df.update(Interaction({data.dataset.iid_field: item_data}))

    if hasattr(data, "uid2history_item"):
        history_item = data.uid2history_item[data_df[data.dataset.uid_field]]

        if len(input_data) > 1:
            history_item = [x if x is not None else torch.Tensor([]) for x in history_item]
            history_u = torch.cat([torch.full_like(hist_iid, i, dtype=int) for i, hist_iid in enumerate(history_item)])
            history_i = torch.cat(list(history_item)).long()
        else:
            history_u = torch.full_like(history_item, 0)
            history_i = history_item

        history_index = (history_u, history_i)
    else:
        history_index = None

    return data_df, history_index, None, None


def randperm2groups(batched_data, sensitive_attribute, feature_data, batch_size):
    """
    At least 2 groups are represented in the batch following the distribution in the dataset.
    eps is used to select as an offset with respect to the fixed distribution. If a group has a 70% of distribution
    and the batch size is 32, then 22 +- (22 * eps) items are allocated for that group and the current batch
    :param batched_data:
    :return:
    """
    iter_data = []
    n_samples = batched_data.shape[0]
    n_batch = math.ceil(n_samples / batch_size)

    attr = sensitive_attribute
    user_feat = feature_data[attr][batched_data]
    groups = user_feat.unique().numpy()

    masks = []
    for gr in groups:
        masks.append((user_feat == gr).numpy())
    masks = np.stack(masks)

    distrib = []
    for mask in masks:
        distrib.append(mask.nonzero()[0].shape[0] / batched_data.shape[0])

    for batch in range(n_batch):
        distrib = []
        for mask in masks:
            distrib.append(mask.nonzero()[0].shape[0] / n_samples)

        # n_samples is lower than batch_size only for last batch
        batch_len = min(n_samples, batch_size)
        batch_counter = batch_len
        batch_data = []
        for mask_i, mask_idx in enumerate(np.random.permutation(np.arange(masks.shape[0]))):
            if mask_i == (masks.shape[0] - 1):
                n_mask_samples = batch_counter
            else:
                if batch_counter < batch_len:
                    n_mask_samples = max(min(round(distrib[mask_idx] * batch_len), batch_counter), 1)
                else:
                    n_mask_samples = max(min(round(distrib[mask_idx] * batch_len), batch_counter - 1), 1)
            mask_samples = np.random.permutation(masks[mask_idx].nonzero()[0])
            if batch != (n_batch - 1):
                if mask_samples.shape[0] == n_mask_samples:
                    n_mask_samples = max(n_mask_samples - (n_batch - 1) - batch, 1)

                mask_samples = mask_samples[:n_mask_samples]
            batch_data.append(batched_data[mask_samples])
            # affects groups where these users belong (e.g. gender and age group)
            masks[mask_idx, mask_samples] = False
            batch_counter -= mask_samples.shape[0]
            n_samples -= mask_samples.shape[0]

            if batch_counter == 0:
                break
        iter_data.append(torch.cat(batch_data))

    return iter_data


def increase_user_unfairness(pref_data, sensitive_attribute, dataset):
    users = pref_data['user_id'].to_numpy()
    users_mask_1 = dataset.user_feature[sensitive_attribute][users] == 1
    users_mask_2 = dataset.user_feature[sensitive_attribute][users] == 2

    size_1, size_2 = users_mask_1.nonzero().shape[0], users_mask_2.nonzero().shape[0]
    if size_1 >= size_2:
        gr_to_reduce, gr_fixed = size_1, size_2
        steps = np.linspace(size_2, size_1, 10, dtype=int)[::-1]
    else:
        gr_to_reduce, gr_fixed = size_2, size_1
        steps = np.linspace(size_1, size_2, 10, dtype=int)[::-1]

    df_res = pd.DataFrame(zip(
        users,
        pref_data['result'],
        dataset.user_feature[sensitive_attribute][users].numpy()
    ), columns=['user_id', 'result', sensitive_attribute])

    if dataset.dataset_name == "lastfm-1k":
        ascending = True
        def check_func(gr_red_res, gr_fix_res): return gr_red_res <= gr_fix_res / 2
    else:
        ascending = False
        def check_func(gr_red_res, gr_fix_res): return gr_fix_res <= gr_red_res / 2

    for step in steps:
        step_df = df_res.groupby(sensitive_attribute).apply(
            lambda x: x.sort_values('result', ascending=ascending)[:step]
        ).reset_index(drop=True)
        mean_metric = step_df.groupby(sensitive_attribute).mean()

        users = torch.tensor(step_df['user_id'].to_numpy())

        if check_func(mean_metric.loc[gr_to_reduce, 'result'], mean_metric.loc[gr_fixed, 'result']):
            print(mean_metric)
            break

    return users
