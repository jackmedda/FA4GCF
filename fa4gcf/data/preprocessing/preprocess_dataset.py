import os
import pickle
import pprint
import argparse

import polars as pl

import fa4gcf.data.preprocessing.utils as preproc_utils


if __name__ == "__main__":
    """
    python -m fa4gcf.data.preprocessing.preprocess_dataset \
        --train_split 0.7 \
        --test_split 0.2 \
        --validation_split 0.1 \
        --split_type per_user \
        --user_field user_id \
        --item_field adgroup_id \
        --time_field time_stamp \
        --in_filepath dataset/taobao_ad/original_data/click_data.csv \
        --out_folderpath dataset/taobao_ad/ \
        --user_filepath dataset/taobao_ad/original_data/user_profile.csv \
        --item_filepath dataset/taobao_ad/original_data/ad_feature.csv \
        --min_interactions 5 \
        --dataset_name taobao_ad \
        --add_token \
        -k_core \
        --sep , \
        --token_fields user_id adgroup_id pid cms_segid cms_group_id shopping_level pvalue_level new_user_class_level age_level gender cate_id campaign_id brand customer_id \
        --sensitive_attributes gender age_level
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_filepath', required=True)
    parser.add_argument('--user_filepath', required=True)
    parser.add_argument('--item_filepath', default=None)
    parser.add_argument('--out_folderpath', required=True)
    parser.add_argument('--sep', default='\t')
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--split_type', choices=['random', 'per_user'], default='per_user')
    parser.add_argument('--train_split', default=0.7, type=float)
    parser.add_argument('--train_to_test', default=0, type=int,
                        help="it moves the last N train interactions to test")
    parser.add_argument('--test_split', default=0.2, type=float)
    parser.add_argument('--validation_split', default=0.1, type=float)
    parser.add_argument('--user_field', default='user_id')
    parser.add_argument('--item_field', default='item_id')
    parser.add_argument('--time_field', default='timestamp')
    parser.add_argument('--sensitive_attributes', nargs='+', default=None)
    parser.add_argument('--random_seed', type=int, default=120)
    parser.add_argument('--min_interactions', type=int, default=0)
    parser.add_argument('--min_interactions_filtering_type',
                        choices=["user", "item", "both"], type=str, default="user")
    parser.add_argument('--k_core', action='store_true')
    parser.add_argument('--k_core_iters', type=int, default=10)
    parser.add_argument('--add_token', action='store_true', help='add `token` or `float` to header')
    parser.add_argument('--token_fields', nargs='+', help='fields to tokenize')

    args = parser.parse_args()
    args.sep = '\t' if args.sep == '\\t' else args.sep
    if args.train_to_test > 0:
        args.train_split = (args.train_split, args.train_to_test)

    df = pl.read_csv(args.in_filepath, separator=args.sep)
    user_df = pl.read_csv(args.user_filepath, separator=args.sep)
    item_df = pl.read_csv(args.item_filepath, separator=args.sep) if args.item_filepath is not None else None

    pprint.pprint(vars(args))

    os.makedirs(args.out_folderpath, exist_ok=True)

    # Remove users with no sensitive information
    if args.sensitive_attributes is not None:
        user_df = user_df.drop_nulls(subset=args.sensitive_attributes)
        df = df.join(user_df, on=args.user_field, how='semi').select(df.columns)

    # Remove duplicated interactions
    df = df.unique(subset=[args.user_field, args.item_field])

    if args.min_interactions > 0:
        if args.k_core:
            print(f"> K-core filtering by min_interactions {args.min_interactions} for {args.k_core_iters} iterations")
            df = preproc_utils.k_core(
                df,
                user_field=args.user_field,
                item_field=args.item_field,
                min_interactions=args.min_interactions,
                iterations=args.k_core_iters
            )
        else:
            print(f"> Filtering by min_interactions {args.min_interactions}")
            filtering_fields = {
                "user": [args.user_field],
                "item": [args.item_field],
                "both": [args.item_field, args.user_field]
            }[args.min_interactions_filtering_type]

            for filt_field in filtering_fields:
                df = preproc_utils.filter_min_interactions(df, by=filt_field, min_interactions=args.min_interactions)

        user_df = user_df.join(df, on=args.user_field, how='semi').select(user_df.columns)
        if item_df is not None:
            item_df = item_df.join(df, on=args.item_field, how='semi').select(item_df.columns)

        print(df)
        print(df.describe())
        print(df.select(pl.all().n_unique()))

    if args.add_token:
        if 'token' not in args.user_field:
            df, user_df = preproc_utils.add_token(df, user_df, args)
        else:
            print("`token` already present in headers")

    df.write_csv(os.path.join(args.out_folderpath, f"{args.dataset_name}.inter"), separator='\t')
    user_df.write_csv(os.path.join(args.out_folderpath, f"{args.dataset_name}.user"), separator='\t')
    if item_df is not None:
        item_df.write_csv(os.path.join(args.out_folderpath, f"{args.dataset_name}.item"), separator='\t')

    if args.split_type == "per_user":
        print("> Splitting per user")
        train, val, test = preproc_utils.split_data_per_user(df,
                                                             test_split=args.test_split,
                                                             validation_split=args.validation_split,
                                                             user_field=args.user_field,
                                                             time_field=args.time_field)
    elif args.split_type == "random":
        print("> Splitting randomly")
        train, val, test = preproc_utils.random_split(df,
                                                      test_split=args.test_split,
                                                      validation_split=args.validation_split,
                                                      seed=args.random_seed)
    else:
        raise NotImplementedError(f"`split_type` = `{args.split_type}` not supported")
    print(train)
    remove_token = 'token' in args.user_field

    for data, data_name in zip([train, val, test], ["train", "validation", "test"]):
        with open(os.path.join(args.out_folderpath, f'{args.dataset_name}.{data_name}'), 'wb') as f:
            cols = [args.user_field, args.item_field]
            out_data = dict(zip(
                map(lambda x: x.split(':')[0] if remove_token else x, cols),
                data.select(pl.col(cols)).to_numpy().T
            ))
            pickle.dump(out_data, f)
