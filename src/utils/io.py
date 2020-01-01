import os
import json
import yaml
import importlib
import pandas as pd

from utils.common import remove_ext

# relative path from the project root
RAW_DIR = 'data/raw'
CLEAN_DIR = 'data/clean'
FTR_DIR = 'data/features'


def read_config(fpath):
    """
    Read a config dict from the given python script.
    """
    name = remove_ext(fpath.replace('/', '.'))
    return importlib.import_module(name).config


def read_json(fpath):
    """
    Read a json file.
    """
    with open(fpath, 'r') as f:
        return json.load(f)


def read_yaml(fpath):
    """
    Read a yaml file.
    """
    with open("example.yaml", 'r') as f:
        return yaml.safe_load(f)


def read_data(fpath):
    """
    Read tabular data.
    """
    print(f'Reading {fpath}')
    if fpath.endswith('.csv'):
        return pd.read_csv(fpath)
    elif fpath.endswith('.ftr'):
        return pd.read_feather(fpath)
    else:
        raise TypeError('Invalid file type: {}'.format(os.path.splitext(fpath)))


def save_data(df, fpath):
    """
    Save tabular data.
    """
    print(f'Saving {fpath}')
    if fpath.endswith('.csv'):
        return df.to_csv(fpath, index=False)
    elif fpath.endswith('.ftr'):
        return df.reset_index(drop=True).to_feather(fpath)
    else:
        raise TypeError('Invalid file type: {}'.format(os.path.splitext(fpath)))


def read_from_raw(fname):
    """
    Read data from the raw data directory.
    """
    fpath = os.path.join(RAW_DIR, fname)
    return read_data(fpath)


def read_from_clean(fname):
    """
    Read data from the clean data directory.
    """
    fpath = os.path.join(CLEAN_DIR, fname)
    return read_data(fpath)


def save_to_clean(df, fname):
    """
    Save a dataframe in the clean data directory.
    """
    fpath = os.path.join(CLEAN_DIR, fname)
    save_data(df, fpath)


def read_features(name, train_or_test=None, fmt='ftr'):
    """
    Read features from the feature directory.
    """
    if train_or_test is not None and train_or_test not in ['train', 'test']:
        raise ValueError('`train_or_test` must be either "train" or "test".')

    save_dir = os.path.join(FTR_DIR, name)

    # if `train_or_test` is None, return both train and test data.
    if train_or_test is None:
        return (
            read_data(os.path.join(save_dir, f'{name}_train.{fmt}')),
            read_data(os.path.join(save_dir, f'{name}_test.{fmt}')),
        )
    else:
        return read_data(os.path.join(save_dir, f'{name}_{train_or_test}.{fmt}'))


def save_features(df, name, train_or_test, fmt='ftr', reduce_mem=True):
    """
    Save a dataframe to the feature directory.
    """

    if train_or_test not in ['train', 'test']:
        raise ValueError('`train_or_test` must be either "train" or "test".')

    save_dir = os.path.join(FTR_DIR, name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # stringify columns because feather can only accept string column names.
    df.columns = list(map(str, df.columns))
    fname = f'{name}_{train_or_test}.{fmt}'
    save_data(df, os.path.join(save_dir, fname))
