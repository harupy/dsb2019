import re
from functools import partial
import numpy as np
import pandas as pd

from utils.common import remove_dir_ext, prefix_list
from utils.io import read_from_raw, save_features, save_feature_meta
from utils.dataframe import apply_funcs, concat_dfs
from utils.encoding import build_one_hot_encoder, get_categories
from features.funcs import (filter_assessment_attempt,
                            assign_attempt_result,
                            calc_attempt_stats)


def best_accuracy_before(df, encoder):
    columns = get_categories(encoder)
    columns = [re.sub(r'[^a-zA-Z\d]+', '_', c).strip('_') for c in columns]
    columns = prefix_list(columns, 'acg')

    onehot = encoder.transform(df[['title']]).toarray()
    expanded = pd.DataFrame(df[['accuracy_group']].values * np.where(onehot == 1.0, onehot, np.nan), columns=columns)
    expanded = concat_dfs([df, expanded], axis=1)

    def func(gdf):
        return concat_dfs([gdf.drop(columns, axis=1), gdf[columns].expanding().max().shift(1)], axis=1)

    expanded = expanded.groupby('installation_id', sort=False).apply(func)
    return expanded.assign(cum_best_accuracy_group=np.sum(np.nan_to_num(expanded[columns].values) * onehot, axis=1))


def main():
    train = read_from_raw('train.csv', reduce_mem=True)
    test = read_from_raw('test.csv', reduce_mem=True)

    funcs = [
        filter_assessment_attempt,
        assign_attempt_result,
        partial(calc_attempt_stats, keep_title=True),
    ]

    train = apply_funcs(train, funcs)
    test = apply_funcs(test, funcs)

    encoder = build_one_hot_encoder(train, test, ['title'])
    train = best_accuracy_before(train, encoder)
    test = best_accuracy_before(test, encoder)

    name = remove_dir_ext(__file__)
    save_features(train, name, 'train')
    save_features(test, name, 'test')

    use_cols = ['installation_id', 'game_session', 'cum_best_accuracy_group']
    meta = {'use_cols': use_cols}
    save_feature_meta(meta, name)


if __name__ == '__main__':
    main()
