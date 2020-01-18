import os


def on_kaggle():
    """
    Return True if it's on Kaggle kernel.
    """
    # NOTE: The environment variable `KAGGLE_KERNEL_RUN_TYPE` exists on both notebook and script.
    return not (os.environ['HOME'] == '/Users/harutaka')
