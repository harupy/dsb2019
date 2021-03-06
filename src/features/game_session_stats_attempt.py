"""
game_session_stats.py use all assessments, but some of them don't contain attempts.
This script excludes those assessments when calculating features related to accuracy.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.io import read_from_clean, save_features, save_feature_meta
from utils.common import remove_dir_ext, prefix_list
from utils.dataframe import assert_columns_equal
from features.funcs import (classify_accuracy,
                            filter_assessment,
                            filter_assessment_attempt,
                            move_id_front)


def create_encoders(train, test, cols):
    """
    Create one-hot encoders as a dict where each item represents (col-to-encode, encoder).

    Examples
    --------
    >> > train = pd.DataFrame({'a': ['x', 'y']})
    >> > test = pd.DataFrame({'a': ['z']})
    >> > create_encoders(train, test, ['a'])
    {'a': {'x': 0, 'y': 1, 'z': 2}}

    """
    encoders = {}

    for col in cols:
        unique = np.sort(train[col].append(test[col]).unique())
        encoders[col] = dict(zip(unique, np.arange(len(unique))))
    return encoders


def make_counter(keys, init_value=0):
    """
    Make a counter with given keys and initial value.

    Examples
    --------
    >> > make_counter(['a', 'b'])
    {'a': 0, 'b': 0}

    >> > make_counter(['a', 'b'], -1)
    {'a': -1, 'b': -1}

    """
    return {key: init_value for key in keys}


def safe_div(n, d):
    """
    Return n / d if d is not 0 otherwise 0.

    >> > safe_div(1, 2)
    0.5

    >> > safe_div(1, 0)
    0

    """
    return n / d if d != 0 else 0


def process_user_sample(user_sample, encoders, assess_titles, is_test_set=False):
    last_activity = 0
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_incorrect_attempts = 0
    accumulated_actions = 0
    assessment_count = 0
    durations = []

    user_activities_count = make_counter(encoders['type'].keys())
    accuracy_groups = {f'acg_{acg}': 0 for acg in [0, 1, 2, 3]}
    last_accuracy_title = make_counter(prefix_list(assess_titles, 'acc'), -1)
    event_code_count = make_counter(encoders['event_code'].keys())
    event_id_count = make_counter(encoders['event_id'].keys())
    title_count = make_counter(encoders['title'].keys())
    title_event_code_count = make_counter(encoders['title_event_code'].keys())

    all_assessments = []
    session_count = 0

    for key, session in user_sample.groupby('game_session', sort=False):

        session_type = session['type'].iloc[0]
        title = session['title'].iloc[0]

        # note that this condition contains assessment sessions that don't have attempts.
        if (session_type == 'Assessment') & (is_test_set or len(session) > 1):
            attempts = filter_assessment_attempt(session)
            correct_attempts = attempts['event_data'].str.contains('true').sum()
            incorrect_attempts = attempts['event_data'].str.contains('false').sum()
            total_attempts = correct_attempts + incorrect_attempts
            has_attempt = total_attempts > 0

            features = user_activities_count.copy()
            features.update(last_accuracy_title)
            features.update(event_code_count)
            features.update(event_id_count)
            features.update(title_count)
            features.update(title_event_code_count)
            features.update({'session_count': session_count})

            nonzero_features = [('nonzero_event_code', event_code_count),
                                ('nonzero_event_id', event_id_count),
                                ('nonzero_title', title_count),
                                ('nonzero_title_event_code', title_event_code_count)]

            for key, counter in nonzero_features:
                features[key] = np.count_nonzero(list(counter.values()))

            # add installation_id and game_session to merge other features.
            features['installation_id'] = session['installation_id'].iloc[-1]
            features['game_session'] = session['game_session'].iloc[-1]
            features['title'] = encoders['title'][title]

            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_incorrect_attempts'] = accumulated_incorrect_attempts

            accumulated_correct_attempts += correct_attempts
            accumulated_incorrect_attempts += incorrect_attempts

            features['duration_mean'] = np.mean(durations) if durations else 0
            features['duration_std'] = np.std(durations) if durations else 0
            features['duration_var'] = np.var(durations) if durations else 0
            durations.append((session['timestamp'].iloc[-1] - session['timestamp'].iloc[0]).seconds)

            features['accumulated_accuracy'] = safe_div(accumulated_accuracy, assessment_count)

            # exclude assessments withtout attempts.
            accuracy = safe_div(correct_attempts, total_attempts)
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + title] = accuracy

            accuracy_group = classify_accuracy(accuracy)
            features['accuracy'] = accuracy
            features['accuracy_group'] = accuracy_group
            features.update(accuracy_groups)
            accuracy_groups[f'acg_{accuracy_group}'] += 1 if has_attempt else 0
            features['accumulated_accuracy_group'] = safe_div(accumulated_accuracy_group, assessment_count)
            accumulated_accuracy_group += accuracy_group

            features['accumulated_actions'] = accumulated_actions

            if is_test_set:
                all_assessments.append(features)
            elif total_attempts > 0:
                all_assessments.append(features)

            assessment_count += 1 if has_attempt else 0

        session_count += 1

        def update_counters(counter, col):
            for key, value in session[col].value_counts().items():
                counter[key] += value

        update_counters(event_code_count, 'event_code')
        update_counters(event_id_count, 'event_id')
        update_counters(title_count, 'title')
        update_counters(title_event_code_count, 'title_event_code')

        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1

    if is_test_set:
        return all_assessments[-1]

    return all_assessments


def get_train_and_test(train, test, encoders, assess_titles):
    """
    Create train and test data.
    """
    assessments_train = []
    assessments_test = []
    for ins_id, user_sample in tqdm(train.groupby('installation_id', sort=False)):
        assessments_train += process_user_sample(user_sample, encoders, assess_titles)

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False)):
        test_data = process_user_sample(user_sample, encoders, assess_titles, is_test_set=True)
        assessments_test.append(test_data)

    return pd.DataFrame(assessments_train), pd.DataFrame(assessments_test)


def main():
    # read data
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    # convert timestamp from string to datetime.
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    # add 'title_event_code'.
    train['title_event_code'] = train['title'] + '_' + train['event_code'].astype(str)
    test['title_event_code'] = test['title'] + '_' + test['event_code'].astype(str)

    # build label encoders.
    cols_enc = ['title', 'event_code', 'title_event_code', 'event_id', 'world', 'type']
    encoders = create_encoders(train, test, cols_enc)
    assess_titles = filter_assessment(train)['title'].unique()

    train, test = get_train_and_test(train, test, encoders, assess_titles)

    # move 'installation_id' and 'game_session' to front
    train = move_id_front(train)
    test = move_id_front(test)

    # assert calculated accuracy_group equals to the true values.
    train_labels = read_from_clean('train_labels.ftr')
    cols = ['installation_id', 'game_session', 'accuracy_group']
    merged = pd.merge(train_labels[cols], train[cols],
                      on=cols[:-1], how='left')
    pd.testing.assert_series_equal(merged[cols[-1] + '_x'],
                                   merged[cols[-1] + '_y'],
                                   check_dtype=False, check_names=False)

    name = remove_dir_ext(__file__)
    assert_columns_equal(train, test)
    save_features(train, name, 'train')
    save_features(test, name, 'test')
    save_feature_meta({'drop_cols': ['accuracy', 'accuracy_group']}, name)


if __name__ == '__main__':
    main()
