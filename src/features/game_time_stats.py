import os
import numpy as np
import pandas as pd

from utils.common import remove_ext, suffix_list
from utils.io import read_from_clean, save_features
from utils.encoding import build_one_hot_encoder, get_categories
from utils.dataframe import suffix_columns


def game_time_stats(df, encoder):
    keys = ['installation_id', 'game_session']
    last_session = df.groupby(keys, sort=False).last().reset_index()
    one_hot = encoder.transform(last_session[['type']]).toarray()
    one_hot = one_hot.astype(np.float64)  # convert to float to accept NaN.
    one_hot[one_hot == 0] = np.nan  # fill zero with NaN
    feature_names = suffix_list(get_categories(encoder), 'duration')
    dur = pd.DataFrame(one_hot * last_session[['game_time']].values, columns=feature_names)
    dur = pd.concat((last_session[keys], dur), axis=1)

    def cum_shift(gdf):
        return pd.concat([
            gdf[keys],
            suffix_columns(gdf.drop(keys, axis=1).expanding().mean().shift(1), 'cum_mean'),
            suffix_columns(gdf.drop(keys, axis=1).expanding().std().shift(1), 'cum_std'),
            suffix_columns(gdf.drop(keys, axis=1).expanding().min().shift(1), 'cum_min'),
            suffix_columns(gdf.drop(keys, axis=1).expanding().max().shift(1), 'cum_max'),
        ], axis=1)

    return dur.groupby(['installation_id'], sort=False).apply(cum_shift).reset_index()


def main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    encoder = build_one_hot_encoder(train, test, ['type'])

    gt_stats_train = game_time_stats(train, encoder)
    gt_stats_test = game_time_stats(test, encoder)

    name = remove_ext(os.path.basename(__file__))
    save_features(gt_stats_train, name, 'train')
    save_features(gt_stats_test, name, 'test')


if __name__ == '__main__':
    main()
