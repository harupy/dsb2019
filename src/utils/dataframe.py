"""
Utilities for pandas dataframe.
"""

from functools import reduce
from pprint import pprint
import numpy as np
import pandas as pd


def reduce_mem_usage(df, verbose=True):
    """
    Reduce the memory usage of given dataframe by converting each column into the appropriate data type.

    Examples
    --------
    >>> from datetime import datetime
    >>> df = pd.DataFrame({
    ...     'object': ['a'],
    ...     'bool': [True],
    ...     'datetime': [datetime.now()],
    ...     'timedelta': [datetime.now() - datetime.now()],
    ...     'np.int8': [0],
    ...     'np.int16': [np.iinfo(np.int8).max],
    ...     'np.int32': [np.iinfo(np.int16).max],
    ...     'np.int64': [np.iinfo(np.int32).max],
    ...     'np.float16': [0.0],
    ...     'np.float32': [np.finfo(np.float16).max],
    ...     'np.float64': [np.finfo(np.float64).max],
    ... })

    By default, numeric columns should have `np.int64` or `np.float64` as data types.
    >>> df.dtypes
    object              object
    bool                  bool
    datetime    datetime64[ns]
    timedelta  timedelta64[ns]
    np.int8              int64
    np.int16             int64
    np.int32             int64
    np.int64             int64
    np.float16         float64
    np.float32         float64
    np.float64         float64
    dtype: object

    # After reduction, numeric columns should have the appropriate data types.
    >>> reduce_mem_usage(df, verbose=False).dtypes
    object              object
    bool                  bool
    datetime    datetime64[ns]
    timedelta  timedelta64[ns]
    np.int8               int8
    np.int16             int16
    np.int32             int32
    np.int64             int64
    np.float16         float32
    np.float32         float32
    np.float64         float64
    dtype: object

    """
    numeric_dtypes = ['int8', 'int16', 'int32', 'int64',
                      'float16', 'float32', 'float64']

    dtypes = {}
    for col in df.columns:
        dtype = df[col].dtypes

        if dtype in numeric_dtypes:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(dtype)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dtypes[col] = np.int8
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dtypes[col] = np.int16
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dtypes[col] = np.int32
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dtypes[col] = np.int64
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    # feather doesn't accept float16: https://github.com/wesm/feather/issues/362
                    # df[col] = df[col].astype(np.float16)
                    dtypes[col] = np.float32
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dtypes[col] = np.float32
                else:
                    dtypes[col] = np.float64

    reduced = df.astype(dtypes)
    if verbose:
        mem_before = df.memory_usage().sum() / 1024**2
        mem_after = reduced.memory_usage().sum() / 1024**2
        print('Memory usage reduced to {:5.2f} Mb ({:.1f}% reduction)'
              .format(mem_after, 100 * (mem_before - mem_after) / mem_before))
    return reduced


def apply_funcs(df, funcs):
    """
    Apply multiple functions to given dataframe.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [0]})
    >>> f1 = lambda df: df + 1
    >>> f2 = lambda df: df + 2
    >>> apply_funcs(df, [f1, f2])
       a
    0  3

    """
    return reduce(lambda df, f: f(df), funcs, df)


def concat_dfs(dfs, axis):
    """
    Concatenate multiple dataframes along specified axis.

    Examples
    --------
    >>> df1 = pd.DataFrame({'a': [0]})
    >>> df2 = pd.DataFrame({'b': [0]})
    >>> concat_dfs([df1, df2], axis=1)
       a  b
    0  0  0

    """
    return reduce(lambda left, right: pd.concat([left, right], axis=axis), dfs)


def prefix_columns(df, prefix, sep='_', exclude=None):
    """
    Prefix dataframe columns.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [0], 'b': [0]})
    >>> prefix_columns(df, 'x')
       x_a  x_b
    0    0    0

    >>> prefix_columns(df, 'x', '|')
       x|a  x|b
    0    0    0

    >>> prefix_columns(df, 'x', '|', ['b'])
       x|a  b
    0    0  0

    """
    exclude = [] if exclude is None else exclude
    return df.rename(columns={c: prefix + sep + c for c in df.columns if c not in exclude})


def suffix_columns(df, suffix, sep='_', exclude=None):
    """
    Suffix dataframe columns.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [0], 'b': [0]})
    >>> suffix_columns(df, 'x')
       a_x  b_x
    0    0    0

    >>> suffix_columns(df, 'x', '|')
       a|x  b|x
    0    0    0

    >>> suffix_columns(df, 'x', '|', ['b'])
       a|x  b
    0    0  0

    """
    exclude = [] if exclude is None else exclude
    return df.rename(columns={c: c + sep + suffix for c in df.columns if c not in exclude})


def assert_columns_equal(left, right):
    """
    Assert left and right have the same columns.
    """
    assert left.columns.tolist() == right.columns.tolist()


def all_zero_columns(df):
    """
    Returns all-zero columns of given dataframe.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [0, 0, 0]})
    >>> all_zero_columns(df)
    ['b']
    """
    return df.eq(0).all(axis=0).pipe(lambda s: s[s]).index.tolist()


def all_null_columns(df):
    """
    Returns all-null columns of given dataframe.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, np.nan], 'b': [np.nan, np.nan, np.nan]})
    >>> all_null_columns(df)
    ['b']
    """
    return df.isnull().all(axis=0).pipe(lambda s: s[s]).index.tolist()


def constant_columns(df):
    """
    Returns constant columns of given dataframe.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [0, 0, 0]})
    >>> constant_columns(df)
    ['b']

    """
    return (df == df.iloc[0]).all(axis=0).pipe(lambda s: s[s]).index.tolist()


def highly_correlated_columns(df, thresh=0.995, verbose=False):
    """
    Find highly correlated columns.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'a': [1, 2, 3],
    ...     'b': [1, 2, 3],
    ...     'c': [-1, -2, -3],
    ...     'd': [0.5, 0.1, 0.3],
    ... })
    >>> highly_correlated_columns(df)
    ['b', 'c']
    """
    features = df.select_dtypes('number').columns
    result = []
    counter = 0
    for feat_a in features:
        for feat_b in features:
            if (feat_a == feat_b) or (feat_a in result) or (feat_b in result):
                continue
            corr = np.corrcoef(df[feat_a], df[feat_b])[0][1]
            if abs(corr) > thresh:
                counter += 1
                result.append(feat_b)

                if verbose:
                    print('{} Feature_a: {} Feature_b: {} - correlation: {}'
                          .format(counter, feat_a, feat_b, corr))
    return result


def inspect_columns(df):
    """
    Apply the folllowing functions to a given dataframe and display the results.
    - all_zero_columns
    - all_null_columns
    - constant_columns
    """
    funcs = [
        all_zero_columns,
        all_null_columns,
        constant_columns,
    ]

    for func in funcs:
        cols = func(df)
        print(f'\n---------- {func.__name__} ----------\n')
        pprint(cols)
