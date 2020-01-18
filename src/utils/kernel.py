import os


def on_kaggle():
    """
    Return True if it's on Kaggle kernel.
    """
    # NOTE: The environment variable `KAGGLE_KERNEL_RUN_TYPE` exists on both notebook and script.
    # Edit: Using `KAGGLE_KERNEL_RUN_TYPE` is a bit risky because it's not clear that it exists
    # on the submit run environment.
    return not (os.environ['HOME'] == '/Users/harutaka')
