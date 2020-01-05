from utils.common import remove_dir_ext
from utils.io import read_from_clean, save_features
from utils.dataframe import prefix_columns
from features.funcs import (is_assessment,
                            cumulative_by_user,
                            assign_attempt_result,
                            calc_attempt_stats)


def main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    is_asm_train = is_assessment(train)
    is_asm_test = is_assessment(test)

    is_4020_train = train['event_code'].eq(4020)
    is_4020_test = test['event_code'].eq(4020)

    # mask event_data
    train['event_data'] = train['event_data'].where(is_4020_train, '')
    test['event_data'] = test['event_data'].where(is_4020_test, '')

    train = assign_attempt_result(train)
    test = assign_attempt_result(test)

    train = train[is_asm_train & is_4020_train]
    test = test[is_asm_test & is_4020_test]

    #######################################################

    train = calc_attempt_stats(train)
    test = calc_attempt_stats(test)

    train = train.drop('accuracy_group', axis=1)
    test = test.drop('accuracy_group', axis=1)

    #######################################################

    funcs = {
        'cumsum': ['num_correct', 'num_incorrect', 'attempts'],
        'cummean': ['accuracy'],
    }
    train = cumulative_by_user(train, funcs)
    test = cumulative_by_user(test, funcs, is_test=True)

    train = train.fillna(0)
    test = test.fillna(0)

    exclude = ['installation_id', 'game_session']
    train = prefix_columns(train, '4020', exclude=exclude)
    test = prefix_columns(test, '4020', exclude=exclude)

    name = remove_dir_ext(__file__)
    save_features(train, name, 'train')
    save_features(test, name, 'test')


if __name__ == '__main__':
    main()
