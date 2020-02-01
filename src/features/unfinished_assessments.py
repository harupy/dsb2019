import numpy as np

from utils.io import read_from_clean, save_features, save_feature_meta
from utils.common import remove_dir_ext
from utils.dataframe import apply_funcs
from features.funcs import (filter_assessment_attempt,
                            assign_attempt_result,
                            calc_attempt_stats,
                            cum_by_user)


def main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    funcs = [
        filter_assessment_attempt,
        assign_attempt_result,
        calc_attempt_stats,
    ]

    train = apply_funcs(train, funcs)
    test = apply_funcs(test, funcs)

    train['unfinished'] = train['accuracy_group'].eq(0)
    test['unfinished'] = test['accuracy_group'].eq(0)

    funcs = {'cumsum': ['unfinished']}
    train = cum_by_user(train, funcs)
    test = cum_by_user(test, funcs)

    train = train.fillna({'cumsum_unfinished': 0})
    test = test.fillna({'cumsum_unfinished': 0})

    train['cumsum_unfinished'] = train['cumsum_unfinished'].astype(np.int8)
    test['cumsum_unfinished'] = test['cumsum_unfinished'].astype(np.int8)

    name = remove_dir_ext(__file__)
    save_features(train, name, 'train')
    save_features(test, name, 'test')

    use_cols = ['installation_id', 'game_session', 'cumsum_unfinished']
    meta = {'use_cols': use_cols}
    save_feature_meta(meta, name)


if __name__ == '__main__':
    main()
