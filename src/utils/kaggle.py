import os


def on_kaggle():
    """
    Detect if it's on a Kaggle kernel.

    Examples
    --------
    >>> on_kaggle()
    False

    >>> os.environ['KAGGLE_KERNEL_RUN_TYPE'] = 'Interactive'
    >>> on_kaggle()
    True

    """
    # NOTE: The environment variable "KAGGLE_KERNEL_RUN_TYPE" exists on both notebook and script.
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
