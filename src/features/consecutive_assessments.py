import pandas as pd

from utils.io import read_from_clean, save_features
from utils.common import remove_dir_ext
from utils.dataframe import apply_funcs
from features.funcs import filter_assessment_attempt, assign_attempt_result, calc_attempt_stats


def assign_prev(df):
    return df.assign(
        type_prev=df['type'].shift(1),
        title_prev=df['title'].shift(1),
        is_valid_prev=df['is_valid'].shift(1),
        accuracy_group_prev=df['accuracy_group'].shift(1),
    )


def filter_consecutive_assessments(df):
    last_events = df.groupby('game_session', sort=False).last().reset_index()

    # shift operation must be performed by installation_id.
    last_events = last_events.groupby('installation_id', sort=False).apply(assign_prev).reset_index(drop=True)
    is_consecutive_assessment = (last_events['type'].eq('Assessment') &
                                 last_events['title'].eq(last_events['title_prev']) &
                                 last_events['is_valid_prev'].eq(1))
    return last_events[is_consecutive_assessment]


def main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    funcs = [
        filter_assessment_attempt,
        assign_attempt_result,
        calc_attempt_stats,
    ]

    attempts_train = apply_funcs(train, funcs)
    attempts_test = apply_funcs(test, funcs)

    # A valid assessment game is one that contains at least one attempt.
    # Some assessment games don't contain attempts.
    # This probably happens when the user exits the game before completeting it.
    attempts_train['is_valid'] = 1
    attempts_test['is_valid'] = 1

    on = ['installation_id', 'game_session']
    cols_merge = on + ['is_valid', 'accuracy_group']
    train = pd.merge(train, attempts_train[cols_merge], how='left', on=on)
    test = pd.merge(test, attempts_test[cols_merge], how='left', on=on)

    train = filter_consecutive_assessments(train).assign(is_consecutive=1)
    test = filter_consecutive_assessments(test).assign(is_consecutive=1)

    cols_save = ['installation_id', 'game_session', 'is_consecutive', 'accuracy_group_prev']
    train = train[cols_save]
    test = test[cols_save]

    name = remove_dir_ext(__file__)
    save_features(train, name, 'train')
    save_features(test, name, 'test')


if __name__ == '__main__':
    main()
