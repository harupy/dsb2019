"""
Functions to manipulate features.
"""

import numpy as np
from sklearn.metrics import mean_squared_error

from utils.common import with_name
from utils.dataframe import prefix_columns, concat_dfs


def remove_useless_users(train, train_labels):
    """
    The train set contains users that don't take assessments. These users should be removed.
    """
    mask = train['installation_id'].isin(train_labels['installation_id'].unique())
    return train[mask].reset_index(drop=True)


def is_assessment(df):
    """
    Detect assessments.
    """
    return df['type'].eq('Assessment')


def filter_assessment(df):
    """
    Filter assessments.
    """
    return df[is_assessment(df)]


def is_assessment_attempt(df, is_test=False):
    """
    Detect assessment attempts.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'title': [
    ...         'Bird Measurer (Assessment)',
    ...         'Bird Measurer (Assessment)',
    ...         'Mushroom Sorter (Assessment)',
    ...         'Mushroom Sorter (Assessment)',
    ...     ],
    ...     'event_code': [
    ...         4100,
    ...         4110,
    ...         4100,
    ...         4110,
    ...     ],
    ...     'type': ['Assessment' for _ in range(4)],
    ... })

    >>> is_assessment_attempt(df)
    0    False
    1     True
    2     True
    3    False
    dtype: bool

    """
    return (
        (df['title'].eq('Bird Measurer (Assessment)') & df['event_code'].eq(4110)) |
        (df['title'].ne('Bird Measurer (Assessment)') & df['event_code'].eq(4100)) &
        df['type'].eq('Assessment')
        # (is_test & df['installation_id'].ne(df['installation_id'].shift(-1)))
    )


def filter_assessment_attempt(df):
    """
    Filter assessment attempts.
    """
    return df[is_assessment_attempt(df)].reset_index(drop=True)


def get_attempt_result(df):
    """
    Get attempt result (true: 1, false: 0).

    Notes
    -----
    This function might extract attempt results from non-assessment sessions.

    Examples
    --------
    >>> df = pd.DataFrame({'event_data': [
    ...     '{"correct":true, ...}',
    ...     '{"correct":false, ...}',
    ...     '{"correct":test, ...}',
    ... ]})

    >>> get_attempt_result(df)
    0    True
    1    False
    2    False
    Name: event_data, dtype: bool

    """
    pattern = r'"correct":(true|false)'
    return df['event_data'].str.extract(pattern, expand=False).eq('true')


def extract_attempt_result(df):
    """
    Get attempt result (true: 1, false: 0).

    Examples
    --------
    >>> df = pd.DataFrame({'event_data': [
    ...     '{"correct":true, ...}',
    ...     '{"correct":false, ...}',
    ...     '',
    ... ]})

    >>> extract_attempt_result(df)
    0    1.0
    1    0.0
    2    NaN
    Name: event_data, dtype: float64

    """
    pattern = r'"correct":(true|false)'
    true_or_false = df['event_data'].str.extract(pattern, expand=False)
    return true_or_false.eq('true').astype(np.int8).where(true_or_false.notnull())


def assign_attempt_result(df):
    """
    Assign attempt result to the dataframe.
    """
    return df.assign(attempt_result=extract_attempt_result)


def classify_accuracy(acc):
    """
    Classify accuracy into 4 groups.

    >>> [classify_accuracy(acc) for acc in [0, 0.33, 0.5, 1.0]]
    [0, 1, 2, 3]
    """
    if acc == 0:
        return 0
    elif acc == 1:
        return 3
    elif acc == 0.5:
        return 2
    else:
        return 1


def find_highly_correlated_features(df, thresh=0.995):
    counter = 0
    features = df.select_dtypes('number').columns
    result = []
    for feat_a in features:
        for feat_b in features:
            if (feat_a != feat_b) and (feat_a not in result) and (feat_b not in result):
                corr = np.corrcoef(df[feat_a], df[feat_b])[0][1]
                if abs(corr) > thresh:
                    counter += 1
                    result.append(feat_b)
                    print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'
                          .format(counter, feat_a, feat_b, corr))
    return result


def hist_mse(train, test, adjust=False, plot=False):
    n_bins = 10

    if adjust:
        test *= train.mean() / test.mean()
    perc_95 = np.percentile(train, 95)
    train = np.clip(train, 0, perc_95)
    test = np.clip(test, 0, perc_95)
    train_hist = np.histogram(train, bins=n_bins)[0] / len(train)
    test_hist = np.histogram(test, bins=n_bins)[0] / len(test)
    return mean_squared_error(train_hist, test_hist)


def adjust_distribution(train, test):
    """
    Adjust distribution between train and test data.

    Note
    ----
    The private set might have similar distribution to the train data.

    Discussion
    ----------
    https://www.kaggle.com/c/data-science-bowl-2019/discussion/125258
    """
    to_remove = []
    ignore = ['accuracy_group', 'installation_id', 'accuracy_group', 'title']
    test_adjusted = test.copy()
    for col in test.columns:
        if col in ignore:
            continue

        try:
            mean_train = train[col].mean()
            mean_test = test[col].mean()
            mse = hist_mse(train[col], test[col], adjust=True)

            adjust_factor = mean_train / mean_test
            if adjust_factor > 10 or adjust_factor < 0.1:  # or error > 0.01:
                to_remove.append(col)
                print(col, mean_train, mean_test, mse)
            else:
                test_adjusted[col] *= adjust_factor
        except Exception as e:
            print(e)

    return test_adjusted, to_remove


def calc_attempt_stats(df, keep_title=False):
    """
    Calculate the following by assessment game session:
    - num_correct
    - num_incorrect
    - accuracy
    - accuracy group
    """

    aggs = {
        'attempt_result': [
            with_name(lambda s: np.nan if s.isnull().all() else (s == 1).sum(), 'num_correct'),
            with_name(lambda s: np.nan if s.isnull().all() else (s == 0).sum(), 'num_incorrect'),
        ],
    }

    # aggregate
    by = ['installation_id', 'game_session'] + (['title'] if keep_title else [])
    stats = df.groupby(by, sort=False).agg(aggs).reset_index()

    # flatten columns
    stats.columns = [col[1] if (col[1] != '') else col[0] for col in stats.columns]

    # add accuracy and accuracy_group
    stats['attempts'] = stats['num_correct'] + stats['num_incorrect']
    stats['accuracy'] = stats['num_correct'] / stats['attempts']
    stats['accuracy_group'] = stats['accuracy'].map(classify_accuracy).astype(np.int8)
    return stats


def cum_by_user(df, funcs, is_test=False, drop=True):
    """
    Apply cumulative operation by user.
    """
    def take_cum(gdf):
        """
        A function to apply to each group dataframe.
        """
        dfs = []
        drop_cols = []
        for op_name, cols in funcs.items():
            drop_cols += cols

            if op_name == 'cumsum':
                cum = gdf[cols].expanding().sum()
            elif op_name == 'cummean':
                cum = gdf[cols].expanding().mean()
            elif op_name == 'cummax':
                cum = gdf[cols].expanding().max()
            else:
                raise ValueError('Invalid operation name: {}'.format(op_name))

            cum = prefix_columns(cum, op_name)

            # for test, it's not necessary to shift rows by one.
            dfs.append(cum if is_test else cum.shift(1))

        # When multiple cumulative operations are performed on a single column.
        drop_cols = list(set(drop_cols))

        return concat_dfs([gdf.drop(drop_cols if drop else [], axis=1)] + dfs, axis=1)

    return df.groupby('installation_id', sort=False).apply(take_cum)


def move_id_front(df):
    """
    Move installation_id and game_session to front.
    """
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('game_session')))
    cols.insert(0, cols.pop(cols.index('installation_id')))
    return df[cols]
