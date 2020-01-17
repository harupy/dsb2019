import os


def on_kaggle_kernel():
    # NOTE: The environment variable `KAGGLE_KERNEL_RUN_TYPE` exists on both notebook and script.
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
