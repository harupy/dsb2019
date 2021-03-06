import numpy as np

from utils.common import remove_dir_ext
from utils.io import read_from_raw, save_features, save_feature_meta
from features.funcs import remove_useless_users, cum_by_user


def extract_misses(df):
    """
    Extract misses value from event_data in given dataframe.

    >>> df = pd.DataFrame({
    ...     'event_data': ['"misses":1', '"misses":0', ''],
    ... })
    >>> extract_misses(df)
    0    1.0
    1    0.0
    2    NaN
    Name: event_data, dtype: float64

    """
    pattern = r'"misses":(\d+)'
    return df['event_data'].str.extract(pattern, expand=False).astype(float)


def main():
    train = read_from_raw('train.csv')
    test = read_from_raw('test.csv')

    # remove useless users who don't take assessments.
    train_labels = read_from_raw('train_labels.csv')
    train = remove_useless_users(train, train_labels)

    train = train.assign(misses=extract_misses)
    test = test.assign(misses=extract_misses)

    # For all-NA or empty Series return np.nan otherwise sum.
    aggs = {'misses': lambda s: np.nan if s.isnull().all() else s.sum()}
    by = ['installation_id', 'game_session']
    train = train.groupby(by, sort=False).agg(aggs).reset_index()
    test = test.groupby(by, sort=False).agg(aggs).reset_index()

    funcs = {
        'cumsum': ['misses'],
        'cummean': ['misses'],
    }

    # set `drop` False to keep the column `misses` for confirmation.
    train = cum_by_user(train, funcs, drop=False)
    test = cum_by_user(test, funcs, drop=False)

    train = train.fillna(0)
    test = test.fillna(0)

    name = remove_dir_ext(__file__)
    save_features(train, name, 'train')
    save_features(test, name, 'test')
    save_feature_meta({'drop_cols': ['misses']}, name)


if __name__ == '__main__':
    main()
