import numpy as np


def div_by_sum(x):
    return x / x.sum()


def shift_array(arr, num, fill_value=np.nan):
    """
    Shift a numpy array.

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
