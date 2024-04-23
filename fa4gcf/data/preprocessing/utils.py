import polars as pl
import polars.selectors as cs
from pandas.api.types import is_numeric_dtype


# def split_data_temporally(interactions,
#                           start_train_split=0.80,
#                           test_split=None,
#                           validation_split=None,
#                           user_field='user_id',
#                           time_field='timestamp'):
#     # TODO: each temporal split could be yielded instead of returning an entire structure with all the splits
#     # TODO: funzione che trova il set di giorni più vicini al N% (e.g. 10%, 20%), in modo che i giorni siano un numero
#     #       tondo, tipo 90 giorni (3 mesi) o 100
#     # TODO: trovare un modo efficiente per controllare se un timestamp sta in un intervallo di tempo
#     train_set = []
#     test_set = []
#     val_set = []
#     groups = interactions.groupby([user_field])
#     for i, (_, group) in enumerate(tqdm.tqdm(groups, desc=f"Splitting data per_user")):
#         if time_field:
#             sorted_group = group.sort_values(time_field)
#         else:
#             sorted_group = group
#
#         if isinstance(train_split, float) or isinstance(test_split, float):
#             n_rating_train = int(len(sorted_group.index) * train_split) if train_split is not None else 0
#             n_rating_test = int(len(sorted_group.index) * test_split) if test_split is not None else 0
#             n_rating_val = int(len(sorted_group.index) * validation_split) if validation_split is not None else 0
#
#             if len(sorted_group.index) > (n_rating_train + n_rating_test + n_rating_val):
#                 n_rating_train += len(sorted_group.index) - (n_rating_train + n_rating_test + n_rating_val)
#         else:
#             raise ValueError(f"split type not accepted")
#
#         if n_rating_train == 0:
#             start_index = len(sorted_group) - n_rating_test
#             start_index = start_index - n_rating_val if n_rating_val is not None else start_index
#             train_set.append(sorted_group.iloc[:start_index])
#         else:
#             train_set.append(sorted_group.iloc[:n_rating_train])
#             start_index = n_rating_train
#
#         if n_rating_val > 0:
#             val_set.append(sorted_group.iloc[start_index:(start_index + n_rating_val)])
#             start_index += n_rating_val
#
#         if n_rating_test > 0:
#             test_set.append(sorted_group.iloc[start_index:(start_index + n_rating_test)])
#         else:
#             test_set.append(sorted_group.iloc[start_index:])
#
#     train, test = pd.concat(train_set), pd.concat(test_set)
#     validation = pd.concat(val_set) if val_set else None
#
#     return train, validation, test


def split_data_per_user(interactions: pl.DataFrame,
                        test_split=0.2,
                        validation_split=None,
                        user_field='user_id',
                        time_field='timestamp'):
    splits = [validation_split, test_split]
    if time_field is not None:
        interactions = interactions.sort(time_field)

    interactions = interactions.group_by(user_field).agg(pl.all())

    exclude = [user_field]  # columns not aggregated
    train = interactions.select(
        pl.col(exclude), subset_slice(splits, 'train', exclude=exclude)
    ).explode(pl.exclude(exclude))
    valid = interactions.select(
        pl.col(exclude), subset_slice(splits, 'valid', exclude=exclude)
    ).explode(pl.exclude(exclude))
    test = interactions.select(
        pl.col(exclude), subset_slice(splits, 'test', exclude=exclude)
    ).explode(pl.exclude(exclude))

    return train, valid, test


def random_split(interactions: pl.DataFrame,
                 test_split=0.2,
                 validation_split=None,
                 seed=120):
    interactions = interactions.sample(fraction=1, seed=seed)

    n_train, n_val, n_test = compute_set_sizes(len(interactions), [validation_split, test_split])

    train = interactions.slice(0, n_train)
    valid = interactions.slice(n_train, n_val)
    test = interactions.slice(n_train + n_val, n_test)

    return train, valid, test


def subset_slice(splits, subset, exclude=None):
    exclude = exclude or []
    n_data = pl.exclude(exclude).list.len()
    valid_split, test_split = splits

    if isinstance(test_split, float):
        n_rating_val = (n_data * valid_split).floor().cast(int) if valid_split is not None else 0
        n_rating_test = (n_data * test_split).floor().cast(int) if test_split is not None else 0
        n_rating_train = n_data - n_rating_val - n_rating_test
    else:
        raise ValueError(f"split type not accepted")

    if subset == 'train':
        start, offset = 0, n_rating_train
    elif subset in ['validation', 'valid']:
        start, offset = n_rating_train, n_rating_val
    else:
        start, offset = n_rating_train + n_rating_val, n_rating_test

    return pl.exclude(exclude).list.slice(start, offset)


def compute_set_sizes(n_data, splits):
    validation_split, test_split = splits

    if isinstance(test_split, float):
        n_rating_val = int(n_data * validation_split) if validation_split is not None else 0
        n_rating_test = int(n_data * test_split) if test_split is not None else 0

        n_rating_train = n_data - (n_rating_test + n_rating_val)
    else:
        raise ValueError(f"split type not accepted")

    return n_rating_train, n_rating_val, n_rating_test


def filter_min_interactions(interactions, by='user_id', min_interactions=20):
    return interactions.filter(pl.all_horizontal(pl.exclude(by).len().over(by) >= min_interactions))


def k_core(interactions, user_field='user_id', item_field='item_id', min_interactions=20, iterations=10):
    i = 0
    while True:
        for filter_field, other_field in zip([user_field, item_field], [item_field, user_field]):
            interactions = filter_min_interactions(interactions, other_field, min_interactions=min_interactions)

        user_check = interactions.groupby(user_field).agg(
            pl.col(item_field).len() >= min_interactions
        ).select(
            pl.col(item_field).all()
        ).item()

        item_check = interactions.groupby(item_field).agg(
            pl.col(user_field).len() >= min_interactions
        ).select(
            pl.col(user_field).all()
        ).item()

        if (user_check & item_check) or i == iterations:
            break
        i += 1

    return interactions


def add_token(df, user_df, args):
    def rename_cols(_df):
        token_fields = [token for token in args.token_fields if token in _df]
        token_df = _df.select(pl.col(token_fields).suffix(':token'))
        numeric_df = _df.select(pl.all().exclude(token_fields).suffix(':float'))

        return pl.concat([token_df, numeric_df], how='horizontal')

    df = df.with_columns(pl.col(args.user_field).cast(pl.String))
    df = df.with_columns(pl.col(args.item_field).cast(pl.String))
    user_df = user_df.with_columns(pl.col(args.user_field).cast(pl.String))

    df = rename_cols(df)
    user_df = rename_cols(user_df)

    args.user_field += ':token'
    args.item_field += ':token'
    args.time_field += ':float'

    return df, user_df
