from utils.common import remove_dir_ext
from utils.io import read_from_clean, save_features
from utils.dataframe import apply_funcs
from features.funcs import (filter_assessment_attempt,
                            assign_attempt_result,
                            calc_attempt_stats,
                            cumulative_by_user,)


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

    #######################################################

    cum_funcs = {
        'cumsum': ['num_correct', 'num_incorrect', 'attempts'],
        'cummean': ['accuracy', 'accuracy_group'],
    }
    train = cumulative_by_user(train, cum_funcs)
    test = cumulative_by_user(test, cum_funcs, True)

    train['overall_accuracy'] = train['cumsum_num_correct'] / train['cumsum_attempts']
    test['overall_accuracy'] = test['cumsum_num_correct'] / test['cumsum_attempts']

    train = train.fillna(0)
    test = test.fillna(0)

    name = remove_dir_ext(__file__)
    save_features(train, name, 'train')
    save_features(test, name, 'test')


if __name__ == '__main__':
    main()
