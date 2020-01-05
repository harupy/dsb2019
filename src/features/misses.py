import numpy as np

from utils.common import remove_dir_ext
from utils.io import read_from_clean, save_features
from features.funcs import cumulative_by_user


def extract_misses(df):
    """
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
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    train = train.assign(misses=extract_misses)
    test = test.assign(misses=extract_misses)

    # The sum of an all-NA or empty Series is 0 by default.
    aggs = {'misses': lambda s: np.nan if s.isnull().all() else s.sum()}
    by = ['installation_id', 'game_session']
    train = train.groupby(by, sort=False).agg(aggs).reset_index()
    test = test.groupby(by, sort=False).agg(aggs).reset_index()

    funcs = {
        'cumsum': ['misses'],
        'cummean': ['misses'],
    }

    train = cumulative_by_user(train, funcs)
    test = cumulative_by_user(test, funcs)

    train = train.fillna(0)
    test = test.fillna(0)

    name = remove_dir_ext(__file__)
    save_features(train, name, 'train')
    save_features(test, name, 'test')


if __name__ == '__main__':
    main()
