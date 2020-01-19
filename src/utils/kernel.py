import os


def on_kaggle():
    """
    Return True if it's on Kaggle kernel.

    Examples
    --------
    >>> on_kaggle()
    False

    >>> import os
    >>> os.environ['KAGGLE_KERNEL_RUN_TYPE'] = 'Interactive'
    >>> on_kaggle()
    True

    """
    # NOTE: The environment variable `KAGGLE_KERNEL_RUN_TYPE` exists on both notebook and script.
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
