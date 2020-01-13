"""
Utility functions for numpy array.
"""

import numpy as np


def div_by_sum(x):
    """
    Divide an array by its sum.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4])
    >>> div_by_sum(arr)
    array([0.1, 0.2, 0.3, 0.4])

    """
    return x / x.sum()


def shift_array(arr, num, fill_value=np.nan):
    """
    Shift a numpy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to shift.
    num : int
        Number of periods to shift. can be positive or negative.
    fill_value : object optional
        Scalar value to use for newly introduced missing values.

    Examples
    --------
    1D array
    >>> arr = np.array([1, 2, 3])
    >>> shift_array(arr, 1)
    array([nan, 1., 2.])

    2D array
    >>> arr = np.array(
    ... [[1, 2, 3],
    ...  [4, 5, 6],
    ...  [7, 8, 9]]
    ... )
    >>> shift_array(arr, 1)
    array([[nan, nan, nan],
           [ 1., 2., 3.],
           [ 4., 5., 6.]])

    References
    ----------
    - https://stackoverflow.com/a/42642326

    """
    shifted = np.empty_like(arr.astype(np.float64))
    if num > 0:
        shifted[:num] = fill_value
        shifted[num:] = arr[:-num]
    elif num < 0:
        shifted[num:] = fill_value
        shifted[:num] = arr[-num:]
    else:
        shifted[:] = arr
    return shifted
