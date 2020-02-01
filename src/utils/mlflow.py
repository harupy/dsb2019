import os
import shutil
import tempfile
import mlflow


from utils.io import to_json, save_data
from contextlib import contextmanager


@contextmanager
def artifact_context(fpath):
    tmpdir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmpdir, fpath)
    try:
        yield tmp_path
    finally:
        mlflow.log_artifact(tmp_path)
        shutil.rmtree(tmpdir)


def log_figure(fig, fpath):
    """
    Log a matplotlib figure.
    """
    with artifact_context as tmp_path:
        fig.savefig(tmp_path)


def log_dict(dct, fpath, fmt='json'):
    """
    Log a dict as JSON.
    """
    with artifact_context as tmp_path:
        to_json(dct, tmp_path)


def log_df(df, fpath):
    """
    Log a pandas dataframe.
    """
    with artifact_context as tmp_path:
        save_data(df, tmp_path)
