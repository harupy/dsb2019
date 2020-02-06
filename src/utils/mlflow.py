"""
Utilities for MLflow.
"""

import os
import tempfile
import matplotlib.pyplot as plt
import mlflow


from utils.io import to_json, save_data
from contextlib import contextmanager


def set_experiment(expr_name):
    # create an experiment if not exists.
    if mlflow.get_experiment_by_name(expr_name) is None:
        mlflow.create_experiment(expr_name)
    mlflow.set_experiment(expr_name)


@contextmanager
def _artifact_context(fpath):
    """
    Context to make it easier to log files.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, os.path.basename(fpath))
        yield tmp_path
        artifact_path = os.path.dirname(fpath) if len(fpath.split(os.sep)) > 1 else None
        mlflow.log_artifact(tmp_path, artifact_path)


def log_figure(fig, fpath):
    """
    Log a matplotlib figure.
    """
    with _artifact_context(fpath) as tmp_path:
        fig.savefig(tmp_path)
        plt.close(fig)


def log_dict(dct, fpath, fmt='json'):
    """
    Log a dict as JSON.
    """
    with _artifact_context(fpath) as tmp_path:
        to_json(dct, tmp_path)


def log_df(df, fpath):
    """
    Log a pandas dataframe.
    """
    with _artifact_context(fpath) as tmp_path:
        save_data(df, tmp_path)
