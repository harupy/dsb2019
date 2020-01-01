import numpy as np


def is_assessment(df):
    return df['type'].eq('Assessment')


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
    ... ]})

    >>> get_attempt_result(df)
    0    True
    1    False
    Name: event_data, dtype: bool

    """
    pattern = r'"correct":([^,]+)'
    return df['event_data'].str.extract(pattern, expand=False).eq('true')


def assign_attempt_result(df):
    """
    Assign attempt result.
    """
    return df.assign(correct=get_attempt_result(df).astype(int))


def accuracy_group(acc):
    """
    Classify accuracy into 4 groups.

    >>> [accuracy_group(acc) for acc in [0, 0.33, 0.5, 1.0]]
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
