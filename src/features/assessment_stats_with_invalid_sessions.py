"""
Some assessment session don't contain attempts (not sure why).
"""

import os
import numpy as np

from utils.io import read_from_clean, save_features
from utils.common import remove_ext, with_name, prefix_list
from utils.dataframe import apply_funcs, concat_dfs
from features.funcs import filter_assessment, is_assessment_attempt, classify_accuracy


def attempt_stats(df):
    aggs = {
        # keep tiemstamp to see whether or not the temporal order is correct
        'timestamp': [
            with_name(lambda s: s.iloc[-1], 'timestamp'),
        ],

        'attempt_result': [
            with_name(lambda s: s.sum(), 'num_correct'),
            with_name(lambda s: (1 - s).sum(), 'num_incorrect'),
        ],
    }

    # aggregate
    by = ['installation_id', 'title', 'game_session']
    stats = df.groupby(by, sort=False).agg(aggs).reset_index()

    # flatten multi-level columns
    stats.columns = [col[1] if (col[1] != '') else col[0] for col in stats.columns]

    # add accuracy and accuracy_group
    stats['attempts'] = stats['num_correct'] + stats['num_incorrect']
    stats['accuracy'] = (stats['num_correct'] / stats['attempts']).fillna(0)
    stats['accuracy_group'] = stats['accuracy'].map(classify_accuracy).astype(np.int8)
    return stats


def cum_stats(df, is_test=False):

    def process_gdf(df):
        funcs = {
            # operation: columns to apply
            'cumsum': ['num_correct', 'num_incorrect', 'attempts'],
            'cummean': ['accuracy', 'accuracy_group'],
        }

        dfs = []
        drop_cols = []
        for op_name, cols in funcs.items():
            drop_cols += cols

            # for test set, shifting is not necessary.
            periods = 0 if is_test else 1

            if op_name == 'cumsum':
                cum = df[cols].cumsum().shift(periods)
            elif op_name == 'cummean':
                cum = df[cols].expanding().mean().shift(periods)

            cum.columns = prefix_list(cols, op_name)
            dfs.append(cum)

        return concat_dfs([df.drop(drop_cols, axis=1)] + dfs, axis=1)

    return (
        df
        .groupby('installation_id', sort=False)
        .apply(process_gdf)
        .drop(['title', 'timestamp'], axis=1)
    )


def extract_attempt_result(df):
    pattern = r'"correct":(true|false)'
    true_or_false = df['event_data'].str.extract(pattern, expand=False)
    return true_or_false.eq('true').astype(np.int8).where(true_or_false.notnull())


def main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    train['event_data'] = train['event_data'].where(is_assessment_attempt(train), '')
    test['event_data'] = test['event_data'].where(is_assessment_attempt(test), '')

    train = filter_assessment(train)
    train = train.assign(attempt_result=extract_attempt_result)
    train = attempt_stats(train)
    train = cum_stats(train)
    train = train.fillna(0)

    test = filter_assessment(test)
    test = test.assign(attempt_result=extract_attempt_result)
    test = attempt_stats(test)
    test = cum_stats(test, is_test=True)
    test = test.fillna(0)

    name = remove_ext(os.path.basename(__file__))
    save_features(train, name, 'train')
    save_features(test, name, 'test')


if __name__ == '__main__':
    main()
