"""
Utilities for common operations.
"""

import os
from datetime import datetime


def prefix_list(lst, prefix, sep='_'):
    """
    Prefix a list of strings.

    Examples
    --------
    >>> prefix_list(['a', 'b'], 'x')
    ['x_a', 'x_b']

    >>> prefix_list(['a', 'b'], 'x', '|')
    ['x|a', 'x|b']
    """
    return [prefix + sep + x for x in lst]


def suffix_list(lst, suffix, sep='_'):
    """
    Suffix a list of strings.

    Examples
    --------
    >>> prefix_list(['a', 'b'], 'x')
    ['x_a', 'x_b']

    >>> prefix_list(['a', 'b'], 'x', '|')
    ['x|a', 'x|b']
    """
    return [x + sep + suffix for x in lst]


def prefix_dict_keys(dct, prefix, sep='_'):
    """
    Prefix dict keys.

    Examples
    --------
    >>> prefix_dict_keys({'a': 0}, 'x')
    {'x_a': 0}

    >>> prefix_dict_keys({'a': 0}, 'x', '|')
    {'x|a': 0}

    """
    return {prefix + sep + k: v for k, v in dct.items()}


def suffix_dict_keys(dct, suffix, sep='_'):
    """
    Suffix dict keys.

    Examples
    --------
    >>> suffix_dict_keys({'a': 0}, 'x')
    {'a_x': 0}

    >>> suffix_dict_keys({'a': 0}, 'x', '|')
    {'a|x': 0}

    """
    return {k + sep + suffix: v for k, v in dct.items()}


def with_name(func, name):
    """
    Name a function.

    Examples
    --------
    >>> func = with_name(lambda: print('test'), 'test')
    >>> func.__name__
    'test'

    """
    func.__name__ = name
    return func


def get_ext(fpath):
    """
    Get extension of a file.

    Parameters
    ----------
    fpath : str
        file path

    Returns
    -------
    str
        extension

    Examples
    --------
    >>> get_ext('test.txt')
    '.txt'

    >>> get_ext('dir/test.txt')
    '.txt'

    """
    return os.path.splitext(fpath)[1]


def remove_ext(fpath):
    """
    Remove extension of a file.

    Examples
    --------
    >>> remove_ext('test.txt')
    'test'

    >>> remove_ext('dir/test.txt')
    'dir/test'

    """
    return os.path.splitext(fpath)[0]


def replace_ext(fpath, ext):
    """
    Replace file extension.

    Examples
    --------
    >>> replace_ext('test.txt', '.py')
    'test.py'

    >>> replace_ext('dir/test.txt', '.py')
    'dir/test.py'

    """
    return os.path.splitext(fpath)[0] + ext


def remove_dir_ext(fpath):
    """
    Remove the directory component and extension from the specified path.

    Examples
    --------
    >>> remove_dir_ext('test.txt')
    'test'

    >>> remove_dir_ext('dir/test.txt')
    'test'

    >>> remove_dir_ext('test')
    'test'

    """
    return os.path.splitext(os.path.basename(fpath))[0]


def print_divider(text):
    """
    Print a divider.

    Examples
    --------
    >>> print_divider('test')
    <BLANKLINE>
    ---------- test ----------
    <BLANKLINE>

    """
    print('\n---------- {} ----------\n'.format(text))


def get_timestamp():
    """
    Make a timestamp from the current UTC.

    >>> import re
    >>> ts = get_timestamp()
    >>> m = re.match(r'[\d]{8}_[\d]{6}', ts)  # noqa
    >>> bool(m)
    True

    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')
