import numpy as np

from utils.common import with_name
from utils.dataframe import prefix_columns, concat_dfs


def is_assessment(df):
    return df['type'].eq('Assessment')


def filter_assessment(df):
    return df[is_assessment(df)]


def is_assessment_attempt(df):
    """
    Detect assessments.

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
    )


def filter_assessment_attempt(df):
    """
    Filter assessment records.
    """
    return df[is_assessment_attempt(df)].reset_index(drop=True)


def get_attempt_result(df):
    """
    Get attempt result (true: 1, false: 0).

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
    to_remove = []
    for feat_a in features:
        for feat_b in features:
            if (feat_a != feat_b) and (feat_a not in to_remove) and (feat_b not in to_remove):
                corr = np.corrcoef(df[feat_a], df[feat_b])[0][1]
                if abs(corr) > thresh:
                    counter += 1
                    to_remove.append(feat_b)
                    print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'
                          .format(counter, feat_a, feat_b, corr))
    return to_remove


def calc_attempt_stats(df):

    aggs = {
        'attempt_result': [
            with_name(lambda s: np.nan if s.isnull().all() else (s == 1).sum(), 'num_correct'),
            with_name(lambda s: np.nan if s.isnull().all() else (s == 0).sum(), 'num_incorrect'),
        ],
    }

    # aggregate
    by = ['installation_id', 'game_session']
    stats = df.groupby(by, sort=False).agg(aggs).reset_index()

    # flatten columns
    stats.columns = [col[1] if (col[1] != '') else col[0] for col in stats.columns]

    # add accuracy and accuracy_group
    stats['attempts'] = stats['num_correct'] + stats['num_incorrect']
    stats['accuracy'] = stats['num_correct'] / stats['attempts']
    stats['accuracy_group'] = stats['accuracy'].map(classify_accuracy).astype(np.int8)
    return stats


def cumulative_by_user(df, funcs, is_test=False):
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

            cum = prefix_columns(cum, op_name)

            # for test, it's not necessary to shift rows by one.
            dfs.append(cum if is_test else cum.shift(1))

        return concat_dfs([gdf.drop(drop_cols, axis=1)] + dfs, axis=1)

    return df.groupby('installation_id', sort=False).apply(take_cum)
