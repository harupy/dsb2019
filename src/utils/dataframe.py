from functools import reduce
import numpy as np
import pandas as pd


def reduce_mem_usage(df, verbose=True):
    numeric_dtypes = ['int8', 'int16', 'int32', 'int64',
                      'float16', 'float32', 'float64']
    mem_before = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        dtype = df[col].dtypes

        if dtype in numeric_dtypes:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(dtype)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    # df[col] = df[col].astype(np.float16)
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float132)
                else:
                    df[col] = df[col].astype(np.float64)

    mem_after = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'
              .format(mem_after, 100 * (mem_before - mem_after) / mem_before))


def apply_funcs(df, funcs):
    """
    Apply functions to a dataframe.

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
    Concatenate multiple dataframes.

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
