import os


def on_kaggle_kernel():
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
