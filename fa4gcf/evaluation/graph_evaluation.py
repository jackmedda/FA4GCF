from typing import Iterable

import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp

from fa4gcf.data.utils import get_bipartite_igraph


class GraphMetricsExtractor:

    BASE_METRICS = ["Degree", "Sparsity", "Reachability", "Sharing Potentiality", "UPI"]

    def __init__(self, dataset, graph_metrics_df=None, metrics="all", sp_kwargs=None, upi_kwargs=None):
        self.dataset = dataset
        self.graph_metrics_df = graph_metrics_df
        self.node_col = 'Node'
        self.metrics = self.BASE_METRICS if metrics == "all" else metrics

        self._item_hist = None
        self._item_hist_len = None

        self._user_hist = None
        self._user_hist_len = None

        self._igg = None

        self.sp_kwargs = sp_kwargs or dict(length=2, depth=2)
        self.upi_kwargs = upi_kwargs or {'sensitive_attribute': ['gender']}

    @property
    def item_hist(self):
        if self._item_hist is None:
            self._item_hist, _, self._item_hist_len = self.dataset.history_user_matrix()
        return self._item_hist

    @property
    def item_hist_len(self):
        if self._item_hist_len is None:
            self._item_hist, _, self._item_hist_len = self.dataset.history_user_matrix()
        return self._item_hist_len

    @property
    def user_hist(self):
        if self._user_hist is None:
            self._user_hist, _, self._user_hist_len = self.dataset.history_item_matrix()
        return self._user_hist

    @property
    def user_hist_len(self):
        if self._user_hist_len is None:
            self._user_hist, _, self._user_hist_len = self.dataset.history_item_matrix()
        return self._user_hist

    @property
    def igg(self):
        if self._igg is None:
            self._igg = get_bipartite_igraph(self.dataset, remove_first_row_col=True)
        return self._igg

    def extract_graph_metrics_per_node(self):
        G_df = self.graph_metrics_df.copy(deep=True)

        for metr in self.metrics:
            if G_df is not None and metr in G_df.columns:
                continue

            if metr == "Degree":
                df = pd.DataFrame(dict(zip(self.igg.vs.indices, self.igg.degree())).items(), columns=[self.node_col, metr])
            elif metr == "Reachability":
                last_user_id = self.dataset.user_num - 2
                user_reach = get_user_reachability(self.igg, last_user_id=last_user_id)

                first_item_id = last_user_id + 1
                item_reach = get_item_reachability(self.igg, first_item_id=first_item_id)

                df = pd.DataFrame({**user_reach, **item_reach}.items(), columns=[self.node_col, metr])
            elif metr == "Sparsity" or metr == "Density":
                _item_hist = self.item_hist[1:].where(self.item_hist[1:] == 0, self.item_hist[1:] - 1)
                item_pop = self.item_hist_len
                _user_hist = self.user_hist[1:].where(self.user_hist[1:] == 0, self.user_hist[1:] - 1)
                user_hist_len = self.user_hist_len

                user_density = np.nan_to_num(
                    ((item_pop[_user_hist] / _user_hist.shape[0]).sum(dim=1) / user_hist_len[1:]).numpy(),
                    nan=0
                )
                user_metric = 1 - user_density if metr == "Sparsity" else user_density

                # item density represents the activity of the users that interact with an item.
                # If only a user interact with item X and this user interacted with all the items in the catalog, then
                # the density of X is maximum. A low density then means a high sparsity, which means the users that interact
                # with that item interact with a few others
                item_density = np.nan_to_num(
                    ((user_hist_len[_item_hist] / _item_hist.shape[0]).sum(dim=1) / item_pop[1:]).numpy(),
                    nan=0
                )
                item_metric = 1 - item_density if metr == "Sparsity" else item_density

                df = pd.DataFrame(
                    zip(self.igg.vs.indices, np.concatenate([user_metric, item_metric])), columns=[self.node_col, metr]
                )
            elif metr == "Sharing Potentiality":
                n_users = self.user_hist[1:].shape[0]
                user_user = get_node_node_graph_data(self.user_hist[1:].numpy())
                user_user = np.asarray(sp.coo_matrix(
                    (user_user[:, -1], (user_user[:, 0], user_user[:, 1])), shape=(n_users, n_users)
                ).todense())
                user_sp = compute_sharing_potentiality(
                    user_user, self.user_hist_len[1:].numpy(), **self.sp_kwargs
                )

                n_items = self.item_hist[1:].shape[0]
                item_item = get_node_node_graph_data(self.item_hist[1:].numpy())
                item_item = np.asarray(sp.coo_matrix(
                    (item_item[:, -1], (item_item[:, 0], item_item[:, 1])), shape=(n_items, n_items)
                ).todense())
                item_sp = compute_sharing_potentiality(
                    item_item, self.item_hist_len[1:].numpy(), **self.sp_kwargs
                )

                sp_data = np.concatenate([user_sp, item_sp])
                df = pd.DataFrame(zip(range(len(sp_data)), sp_data), columns=[self.node_col, metr])
            elif metr == "Unprotected Preference Index" or metr.upper() == "UPI":
                if self.upi_kwargs is None:
                    raise('UPI needs the upi_kwargs to know the sensitive attribute and the disadvantaged group')
                else:
                    sens_attrs = self.upi_kwargs.pop('sensitive_attribute')
                    sens_attrs = [sens_attrs] if not isinstance(sens_attrs, Iterable) else sens_attrs

                sa_dfs = []
                for sa in sens_attrs:
                    if "UPI " + sa.title() not in G_df.columns:
                        _item_hist = self.item_hist[1:].where(self.item_hist[1:] == 0, self.item_hist[1:] - 1)
                        _user_hist = self.user_hist[1:].where(self.user_hist[1:] == 0, self.user_hist[1:] - 1)

                        sens_map = self.dataset.user_feat[sa]
                        grs_size = sens_map[1:].bincount()
                        minor_gr = ((grs_size > 0) & (grs_size < grs_size.max())).nonzero().item()
                        n_minor = (sens_map == minor_gr).sum() / (sens_map.shape[0] - 1)

                        sens_item_hist = sens_map[_item_hist]
                        items_upi = ((sens_item_hist == minor_gr).sum(dim=1) / self.item_hist_len[1:]).nan_to_num(nan=0)
                        # values higher than 1 means the advantaged group prefer those items w.r.t. to their representation
                        items_upi = items_upi / n_minor

                        users_upi = items_upi[_user_hist].mean(dim=1)

                        upi = np.concatenate((users_upi.numpy(), items_upi.numpy()))
                        sa_dfs.append(pd.DataFrame(zip(range(len(upi)), upi), columns=[self.node_col, "UPI " + sa.title()]).set_index('Node'))
                if not sa_dfs:
                    continue
                df = pd.concat(sa_dfs, axis=1).reset_index()
            if G_df is None:
                G_df = df
            else:
                G_df = G_df.join(df.set_index(self.node_col), on=self.node_col)
        return G_df


def get_user_reachability(graph, last_user_id):
    return get_reachability_per_node(graph, last=last_user_id)


def get_item_reachability(graph, first_item_id):
    return get_reachability_per_node(graph, first=first_item_id)


def get_reachability_per_node(graph, first=None, last=None, nodes=None):
    if nodes is not None:
        nodes = sorted(nodes)
    else:
        nodes = np.arange(graph.vcount())

#         dist = dist[first:][:, first:] if first is not None else dist
#         nodes = nodes[first:] if first is not None else nodes

#         dist = dist[:(last + 1)][:, :(last + 1)] if last is not None else dist
#         nodes = nodes[:(last + 1)] if last is not None else nodes
        if first is not None:
            nodes = nodes[first:]
        if last is not None:
            nodes = nodes[:(last + 1)]

    dist = np.array(graph.distances(source=nodes, target=nodes))

    reach = _get_reachability_per_node(dist)

    return dict(zip(nodes, reach))


def get_reachability_source_target(graph, source, target):
    dist = np.array(graph.distances(source=source, target=target))

    reach = _get_reachability_per_node(dist)

    return dict(zip(source, reach))


@numba.jit(nopython=True, parallel=True)
def _get_reachability_per_node(dist):
    n_nodes = dist.shape[0]
    reach = np.zeros((n_nodes,), dtype=np.float32)

    for i in numba.prange(n_nodes):
        n_dist = dist[i]
        mask = (~np.isinf(n_dist)) & (n_dist > 0)
        n_reach = np.bincount(n_dist[mask].astype(np.int32))
        n_reach = n_reach[n_reach > 0]
        reach[i] = compute_reachability(n_reach) / n_nodes

    return reach


@numba.jit(nopython=True, parallel=True)
def compute_reachability(reach):
    discount = np.arange(reach.shape[0]) + 1
    return (reach / discount).sum()


@numba.jit(nopython=True, parallel=True)
def compute_sharing_potentiality(common_data, hist_len, length=2, depth=2):
    n = common_data.shape[0]
    res = np.zeros((n,), dtype=np.float32)
    for i in numba.prange(n):
        most_sim = np.argsort(common_data[i])[::-1][:length]
        res[i] += _compute_sp_length(i, most_sim, common_data, hist_len)
        for d in range(depth - 1):
            most_d_sim = np.argsort(common_data[most_sim[d]])[::-1][:length]
            res[i] += _compute_sp_length(most_sim[d], most_d_sim, common_data, hist_len) / (2 + d)
    return res


@numba.jit(nopython=True)
def _compute_sp_length(data_i, most_sim, common_data, hist_len):
    sp_length = 0
    for i in numba.prange(most_sim.shape[0]):
        sim = common_data[data_i, most_sim[i]]
        if hist_len[data_i] == 0 and hist_len[most_sim[i]] > 0:
            sp_length += 1  # most_sim[i] can share every item
        elif hist_len[data_i] == 0 or hist_len[most_sim[i]] == 0:
            sp_length += 0
        else:
            sp_length += sim / hist_len[data_i] * (1 - sim / hist_len[most_sim[i]])
    return sp_length / most_sim.shape[0]


def get_user_user_data_sens_df(dataset, user_df, sens_attr, attr_map=None):
    user_history, _, _ = dataset.history_item_matrix()
    user_graph_df = get_node_node_data_feature_df(user_history.numpy(), user_df, dataset.uid_field, sens_attr, attr_map=attr_map)

    return user_graph_df


def get_item_item_data_pop_df(dataset, item_df, pop_attr):
    item_history, _, _ = dataset.history_user_matrix()
    item_graph_df = get_node_node_data_feature_df(item_history.numpy(), item_df, dataset.iid_field, pop_attr)

    return item_graph_df


def get_node_node_data_feature_df(history, node_df, id_label, feat_attr, attr_map=None):
    graph = get_node_node_graph_data(history)
    graph = graph[graph[:, -1] > 0]

    graph_df = pd.DataFrame(graph, columns=[f'{id_label}_1', f'{id_label}_2', 'n_common_edges'])
    graph_df = graph_df.join(
        node_df.set_index(id_label), on=f"{id_label}_1"
    ).reset_index(drop=True).rename(columns={feat_attr: f"{feat_attr}_1"})

    if attr_map is not None:
        graph_df[f"{feat_attr}_1"] = graph_df[f"{feat_attr}_1"].map(attr_map)

    graph_df = graph_df.join(
        node_df.set_index(id_label), on=f'{id_label}_2'
    ).reset_index(drop=True).rename(columns={feat_attr: f"{feat_attr}_2"})

    if attr_map is not None:
        graph_df[f"{feat_attr}_2"] = graph_df[f"{feat_attr}_2"].map(attr_map)

    return graph_df


def get_node_node_graph_data(history):
    return _get_node_node_graph_data(history)


@numba.jit(nopython=True, parallel=True)
def _get_node_node_graph_data(history):
    hist = [set(h) for h in history]
    n_nodes = history.shape[0] - 1  # removed padding 0
    node_node = np.zeros((n_nodes * (n_nodes - 1) // 2, 3), dtype=np.int32)  # number of combinations

    for n1 in numba.prange(1, n_nodes + 1):
        _inner_combinations(n1, n_nodes, hist, node_node)

    return node_node


@numba.jit(nopython=True, parallel=True)
def _inner_combinations(n1, n_nodes, hist, node_node):
    for n2 in numba.prange(n1 + 1, n_nodes + 1):
        node_node[sum(range(n_nodes - n1 + 1, n_nodes)) + (n2 - n1 - 1)] = [
            n1, n2, len(hist[n1] & hist[n2]) - 1
        ]


def compute_homophily(graph_df, group_sizes, feat_attr):
    homophily = {}
    for (gr1, gr2), gr_df in graph_df.groupby([f"{feat_attr}_1", f"{feat_attr}_2"]):
        if gr1 == gr2:
            all_gr_edges = graph_df.loc[
                (graph_df[f"{feat_attr}_1"] == gr1) |  # interactions of gr1 with any group (also gr1)
                ((graph_df[f"{feat_attr}_2"] == gr1) & (graph_df[f"{feat_attr}_1"] != gr1)),  # interactions from gr2 to gr1
                'n_common_edges'
            ].sum()
            homophily[gr1] = gr_df['n_common_edges'].sum() / all_gr_edges - group_sizes[gr1]

    return homophily
