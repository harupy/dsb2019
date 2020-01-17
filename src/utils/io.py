"""
Utility functions for input/output.
"""

import os
import json
import yaml
import importlib
import pandas as pd

# NOTE: `../input` is a read-only directory in Kaggle.
PARENT_DIR = '../input' if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else 'data'
RAW_DIR = f'{PARENT_DIR}/data-science-bowl-2019'
CLEAN_DIR = f'{PARENT_DIR}/clean'
FTR_DIR = f'{PARENT_DIR}/features'

# make directories for clean and features data.
for d in [CLEAN_DIR, FTR_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)


def read_config(fpath):
    """
    Read a config dict from the given python script.
    """
    name = os.path.splitext(fpath)[0].replace('/', '.')
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


def save_dict(dct, fpath):
    """
    Save a dictionary as json.
    """
    with open(fpath, 'w') as f:
        json.dump(dct, f, indent=2)


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


def save_features(df, name, train_or_test, fmt='ftr'):
    """
    Save a dataframe to the feature directory.

    Parameters
    ----------
    df : dataframe
        dataframe containing features.
    name : str
        feature name
    train_or_test : str enum ('train', 'test')
        string to indicate train or test set
    fmt : str default 'ftr'
        format to save features in.

    """
    if train_or_test not in ['train', 'test']:
        raise ValueError('`train_or_test` must be either "train" or "test".')

    save_dir = os.path.join(FTR_DIR, name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # stringify columns because feather can only accept string column names.
    df.columns = list(map(str, df.columns))
    fname = f'{name}_{train_or_test}.{fmt}'
    fpath = os.path.join(save_dir, fname)
    save_data(df, fpath)


def save_features_meta(data, name):
    """
    Save features meta data (dict).
    """
    fpath = os.path.join(FTR_DIR, name, 'meta.json')
    save_dict(data, fpath)


def find_features_meta(name):
    """
    Find features meta data (dict). If the meta data file is not found, return None.

    Notes
    -----
    Meta data is used to select or drop columns when training.
    """
    fpath = os.path.join(FTR_DIR, name, 'meta.json')
    if not os.path.exists(fpath):
        return
    return read_json(fpath)
