import os
import numpy as np

from utils.common import remove_ext, with_name
from utils.io import read_features, save_features
from utils.dataframe import apply_funcs
from features.funcs import assign_attempt_result, accuracy_group


def attempt_stats(df):
    aggs = {
        # keep tiemstamp to see whether or not the temporal order is correct
        'timestamp': [
            with_name(lambda s: s.iloc[-1], 'timestamp'),
        ],

        'correct': [
            with_name(lambda s: (s == 1).sum(), 'num_correct'),
            with_name(lambda s: (s == 0).sum(), 'num_incorrect'),
        ],
    }

    # aggregate
    by = ['installation_id', 'title', 'game_session']
    stats = df.groupby(by, sort=False).agg(aggs).reset_index()

    # flatten multi-level columns
    stats.columns = [col[1] if (col[1] != '') else col[0] for col in stats.columns]

    # add accuracy and accuracy_group
    stats['attempts'] = stats['num_correct'] + stats['num_incorrect']
    stats['accuracy'] = stats['num_correct'] / stats['attempts']
    stats['accuracy_group'] = stats['accuracy'].map(accuracy_group).astype(np.int8)
    return stats


def main():
    asm_train, asm_test = read_features('assessment_attempts')

    funcs = [
        assign_attempt_result,
        attempt_stats,
    ]

    attempt_stats_train = apply_funcs(asm_train, funcs)
    attempt_stats_test = apply_funcs(asm_test, funcs)

    name = remove_ext(os.path.basename(__file__))
    save_features(attempt_stats_train, name, 'train')
    save_features(attempt_stats_test, name, 'test')


if __name__ == '__main__':
    main()
