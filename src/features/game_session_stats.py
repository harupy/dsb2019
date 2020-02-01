"""
Cleaner version of this kernel: https://www.kaggle.com/braquino/convert-to-regression.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.io import read_from_raw, save_features, save_feature_meta
from utils.common import remove_dir_ext, prefix_list
from utils.dataframe import (apply_funcs,
                             assert_columns_equal,
                             inspect_columns,
                             highly_correlated_columns)
from features.funcs import (remove_useless_users,
                            classify_accuracy,
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


def get_daytime(hour):
    """
    Get day time of give hour.
    0: morning
    1: afternoon
    2: evening
    3: night
    4: midnight

    Examples
    --------
    >>> hours = [6, 10, 12, 14, 18, 20, 22, 23, 0, 2, 4]
    >>> for h in hours:
    ...     print(h, get_daytime(h))
    6 0
    10 0
    12 1
    14 1
    18 2
    20 2
    22 3
    23 3
    0 3
    2 4
    4 4

    """
    if 6 <= hour < 12:
        return 0
    elif 12 <= hour < 18:
        return 1
    elif 18 <= hour < 22:
        return 2
    elif hour >= 22 or hour == 0:
        return 3
    else:
        return 4


def get_clip_time(clip_title):
    """
    Get the duration of a clip session.
    """
    clip_time = {
        "Welcome to Lost Lagoon!": 19,
        "Tree Top City - Level 1": 17,
        "Ordering Spheres": 61,
        "Costume Box": 61,
        "12 Monkeys": 109,
        "Tree Top City - Level 2": 25,
        "Pirate's Tale": 80,
        "Treasure Map": 156,
        "Tree Top City - Level 3": 26,
        "Rulers": 126,
        "Magma Peak - Level 1": 20,
        "Slop Problem": 60,
        "Magma Peak - Level 2": 22,
        "Crystal Caves - Level 1": 18,
        "Balancing Act": 72,
        "Lifting Heavy Things": 118,
        "Crystal Caves - Level 2": 24,
        "Honey Cake": 142,
        "Crystal Caves - Level 3": 19,
        "Heavy, Heavier, Heaviest": 61
    }
    return clip_time[clip_title]


def get_session_duration(session):
    """
    Get the duration of a session.
    """
    return (session['timestamp'].iloc[-1] - session['timestamp'].iloc[0]).seconds


def convert_timestamp(df):
    """
    Convert string timestamp to datetime.
    """
    return df.assign(timestamp=pd.to_datetime(df['timestamp']))


def assign_round(df):
    """
    Extract `round` from `event_data`.
    """
    return df.assign(round=df['event_data'].str.extract(r'"round":(\d+)', expand=False).astype(float))


def assign_level(df):
    """
    Extract `level` from `event_data`.
    """
    return df.assign(level=df['event_data'].str.extract(r'"level":(\d+)', expand=False).astype(float))


def assign_misses(df):
    """
    Extract `misses` from `event_data`.
    """
    return df.assign(misses=df['event_data'].str.extract(r'"misses":(\d+)', expand=False).astype(float))


def assign_title_event_code(df):
    """
    Concatenate `title` and `event_code`.
    """
    return df.assign(title_event_code=df['title'] + '_' + df['event_code'].astype(str))


def process_user_sample(user_sample, encoders, assess_titles, is_test_set=False):
    """
    Process game sessions of a user.
    """
    last_activity = 0
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_incorrect_attempts = 0
    accumulated_total_attempts = 0
    accumulated_actions = 0
    assessment_count = 0
    durations = []
    clip_durations = []
    game_rounds = []
    game_levels = []
    game_misses = []

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

        if session_type == 'Clip':
            clip_durations.append(get_clip_time(title))

        if session_type == 'Game':
            rnd = session['round'].max()
            if not np.isnan(rnd):
                game_rounds.append(rnd)

            level = session['level'].max()
            if not np.isnan(level):
                game_levels.append(level)

            # append if `session` contains al least one valid value.
            if not session['misses'].isnull().all():
                game_misses.append(session['misses'].sum())

            # note that this condition contains assessment sessions that don't have attempts.
        if (session_type == 'Assessment') & (is_test_set or len(session) > 1):
            attempts = filter_assessment_attempt(session)
            correct_attempts = attempts['event_data'].str.contains('true').sum()
            incorrect_attempts = attempts['event_data'].str.contains('false').sum()
            total_attempts = correct_attempts + incorrect_attempts

            features = user_activities_count.copy()
            features.update(last_accuracy_title)
            features.update(event_code_count)
            features.update(event_id_count)
            features.update(title_count)
            features.update(title_event_code_count)
            features.update({'session_count': session_count})
            features.update({'hour': session['timestamp'].dt.hour.iloc[-1]})
            features.update({'dayofweek': session['timestamp'].dt.dayofweek.iloc[-1]})

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
            features['accumulated_total_attempts'] = accumulated_incorrect_attempts
            features['is_first_assessment'] = (accumulated_total_attempts == 0)
            features['daytime'] = get_daytime(session['timestamp'].dt.hour.iloc[0])

            accumulated_correct_attempts += correct_attempts
            accumulated_incorrect_attempts += incorrect_attempts
            accumulated_total_attempts += total_attempts

            # This decreased the public score.
            # # clip duration
            # features['clip_duration_mean'] = np.mean(clip_durations) if clip_durations else 0
            # features['clip_duration_std'] = np.std(clip_durations) if clip_durations else 0
            # features['clip_duration_var'] = np.std(clip_durations) if clip_durations else 0

            # session duration
            features['duration_mean'] = np.mean(durations) if durations else 0
            features['duration_std'] = np.std(durations) if durations else 0
            features['duration_var'] = np.var(durations) if durations else 0

            # game round.
            features['game_round_mean'] = np.mean(game_rounds) if game_rounds else 0

            # game level.
            features['game_level_mean'] = np.mean(game_levels) if game_levels else 0

            # game misses.
            features['game_misses_sum'] = np.sum(game_misses) if game_misses else 0
            features['game_misses_mean'] = np.mean(game_misses) if game_misses else 0
            features['game_misses_std'] = np.std(game_misses) if game_misses else 0
            features['game_misses_var'] = np.var(game_misses) if game_misses else 0

            durations.append(get_session_duration(session))

            features['accumulated_accuracy'] = safe_div(accumulated_accuracy, assessment_count)
            accuracy = safe_div(correct_attempts, total_attempts)
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + title] = accuracy

            accuracy_group = classify_accuracy(accuracy)
            # don't forget add accuracy and accuracy_group as drop_cols in meta data.
            features['accuracy'] = accuracy
            features['accuracy_group'] = accuracy_group
            features.update(accuracy_groups)
            accuracy_groups[f'acg_{accuracy_group}'] += 1
            features['accumulated_accuracy_group'] = safe_div(accumulated_accuracy_group, assessment_count)
            accumulated_accuracy_group += accuracy_group
            features['accumulated_actions'] = accumulated_actions

            if is_test_set:
                all_assessments.append(features)
            elif total_attempts > 0:
                all_assessments.append(features)

            assessment_count += 1

        session_count += 1

        def update_counter(counter, col):
            for key, value in session[col].value_counts().items():
                counter[key] += value

        update_counter(event_code_count, 'event_code')
        update_counter(event_id_count, 'event_id')
        update_counter(title_count, 'title')
        update_counter(title_event_code_count, 'title_event_code')

        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1

    if is_test_set:
        return all_assessments[-1]

    return all_assessments


def get_train_and_test(train, test, encoders, assess_titles):
    """
    Generate train and test data.
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
    train = read_from_raw('train.csv')
    test = read_from_raw('test.csv')

    # remove useless users who don't take assessments.
    train_labels = read_from_raw('train_labels.csv')
    train = remove_useless_users(train, train_labels)

    # preprocess.
    funcs = [
        convert_timestamp,
        assign_round,
        assign_level,
        assign_misses,
        assign_title_event_code,
    ]

    train = apply_funcs(train, funcs)
    test = apply_funcs(test, funcs)

    # build label encoders.
    cols_enc = ['title', 'event_code', 'title_event_code', 'event_id', 'world', 'type']
    encoders = create_encoders(train, test, cols_enc)
    assess_titles = filter_assessment(train)['title'].unique()

    train, test = get_train_and_test(train, test, encoders, assess_titles)

    # move `installation_id` and `game_session` to front.
    train = move_id_front(train)
    test = move_id_front(test)

    # assert calculated accuracy_group equals to the true values.
    train_labels = read_from_raw('train_labels.csv')
    cols = ['installation_id', 'game_session', 'accuracy_group']
    merged = pd.merge(train_labels[cols], train[cols],
                      on=cols[:-1], how='left')
    pd.testing.assert_series_equal(merged[cols[-1] + '_x'],
                                   merged[cols[-1] + '_y'],
                                   check_dtype=False, check_names=False)

    inspect_columns(train)
    inspect_columns(test)

    to_drop = highly_correlated_columns(train, verbose=True)
    train = train.drop(to_drop, axis=1)
    test = test.drop(to_drop, axis=1)

    name = remove_dir_ext(__file__)
    assert_columns_equal(train, test)
    save_features(train, name, 'train')
    save_features(test, name, 'test')
    save_feature_meta({'drop_cols': ['accuracy', 'accuracy_group']}, name)


if __name__ == '__main__':
    main()
