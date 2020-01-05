import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.io import read_from_clean, save_features
from utils.common import remove_dir_ext, prefix_list
from features.funcs import classify_accuracy, filter_assessment, filter_assessment_attempt


def build_encoders(train, test, cols):
    """
    Examples
    --------
    >>> train = pd.DataFrame({'a': ['x', 'y']})
    >>> test = pd.DataFrame({'a': ['z']})
    >>> build_encoders(train, test, ['a'])
    {'a': {'x': 0, 'y': 1, 'z': 2}}

    """
    encoders = {}

    for col in cols:
        unique = np.sort(train[col].append(test[col]).unique())
        encoders[col] = dict(zip(unique, np.arange(len(unique))))
    return encoders


def build_inverters(encoders):
    """
    Examples
    --------
    >>> encoders = {'a': {'x': 0, 'y': 1, 'z': 2}}
    >>> build_inverters(encoders)
    {'a': {0: 'x', 1: 'y', 2: 'z'}}

    """
    inverters = {}
    for col, encoder in encoders.items():
        inverters[col] = {val: key for key, val in encoder.items()}
    return inverters


def make_counter(lst, init_value=0):
    """
    Examples
    --------
    >>> make_counter(['a', 'b'])
    {'a': 0, 'b': 0}

    >>> make_counter(['a', 'b'], -1)
    {'a': -1, 'b': -1}
    """
    return {x: init_value for x in lst}


def process_user_sample(user_sample, encoders, assess_titles, is_test_set=False):
    last_activity = 0

    user_activities_count = {'Clip': 0, 'Activity': 0, 'Assessment': 0, 'Game': 0}
    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}

    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0  # increment on the assessment session.
    durations = []

    last_accuracy_title = make_counter(prefix_list(assess_titles, 'acc'), -1)
    event_code_count = make_counter(encoders['event_code'].keys())
    event_id_count = make_counter(encoders['event_id'].keys())
    title_count = make_counter(encoders['title'].keys())
    title_event_code_count = make_counter(encoders['title_event_code'].keys())

    all_assessments = []

    for key, session in user_sample.groupby('game_session', sort=False):

        session_type = session['type'].iloc[0]
        title = session['title'].iloc[0]

        if (session_type == 'Assessment') & (is_test_set or len(session) > 1):
            attempts = filter_assessment_attempt(session)
            true_attempts = attempts['event_data'].str.contains('true').sum()
            false_attempts = attempts['event_data'].str.contains('false').sum()
            total_attempts = true_attempts + false_attempts

            features = user_activities_count.copy()
            features.update(last_accuracy_title)
            features.update(event_code_count)
            features.update(event_id_count)
            features.update(title_count)
            features.update(title_event_code_count)

            # get installation_id for aggregation
            features['installation_id'] = session['installation_id'].iloc[-1]
            features['game_session'] = session['game_session'].iloc[-1]
            # features['title'] = title
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts
            accumulated_uncorrect_attempts += false_attempts

            features['duration_mean'] = np.mean(durations) if durations else 0
            durations.append((session['timestamp'].iloc[-1] - session['timestamp'].iloc[0]).seconds)

            features['accumulated_accuracy'] = accumulated_accuracy / counter if counter > 0 else 0
            accuracy = true_attempts / total_attempts if total_attempts != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + title] = accuracy

            accuracy_group = classify_accuracy(accuracy)
            # features['accuracy_group'] = accuracy_group
            features.update(accuracy_groups)
            accuracy_groups[accuracy_group] += 1
            features['accumulated_accuracy_group'] = accumulated_accuracy_group / counter if counter > 0 else 0
            accumulated_accuracy_group += accuracy_group
            features['accumulated_actions'] = accumulated_actions

            if is_test_set:
                all_assessments.append(features)
            elif total_attempts > 0:
                all_assessments.append(features)

            counter += 1

        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter, col):
            value_counts = session[col].value_counts()
            for key in value_counts.keys():
                counter[key] += value_counts[key]
            return counter

        event_code_count = update_counters(event_code_count, 'event_code')
        event_id_count = update_counters(event_id_count, 'event_id')
        title_count = update_counters(title_count, 'title')
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1

    if is_test_set:
        return all_assessments[-1]

    return all_assessments


def get_train_and_test(train, test, encoders, assess_titles):
    compiled_train = []
    compiled_test = []
    for ins_id, user_sample in tqdm(train.groupby('installation_id', sort=False)):
        compiled_train += process_user_sample(user_sample, encoders, assess_titles)

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False)):
        test_data = process_user_sample(user_sample, encoders, assess_titles, is_test_set=True)
        compiled_test.append(test_data)

    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    return reduce_train, reduce_test


def main():
    # read data
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    # convert timestamp
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    # assign title_event_code
    train['title_event_code'] = train['title'] + '_' + train['event_code'].astype(str)
    test['title_event_code'] = test['title'] + '_' + test['event_code'].astype(str)

    # build label encoders
    cols_enc = ['title', 'event_code', 'title_event_code', 'event_id', 'world']
    encoders = build_encoders(train, test, cols_enc)
    assess_titles = filter_assessment(train)['title'].unique()

    session_stats_train, session_stats_test = get_train_and_test(train, test, encoders, assess_titles)

    name = remove_dir_ext(__file__)
    save_features(session_stats_train, name, 'train')
    save_features(session_stats_test, name, 'test')


if __name__ == '__main__':
    main()
