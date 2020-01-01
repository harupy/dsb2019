import os
from functools import reduce
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from utils.common import remove_ext
from utils.io import read_from_clean, save_features


def build_one_hot_encoder(train, test, cols_encode):
    """
    Build a one-hot encoder from the given train and test dataframe.
    """
    merged = train[cols_encode].append(test[cols_encode]).astype(str)
    encoder = OneHotEncoder(dtype=np.int8)
    return encoder.fit(merged)


def get_categories(encoder):
    cats = reduce(lambda lst, cat: lst + cat.tolist(), encoder.categories_, [])

    # find duplicated elements
    seen = set()
    dups = []
    for x in cats:
        if x not in seen:
            seen.add(x)
        else:
            dups.append(x)

    assert len(dups) == 0, 'Find duplicated elements: {}'.format(dups)
    return cats


def apply_one_hot_encoder(df, cols_encode, encoder, drop=True):
    """
    Apply a one-hot encoder to the given dataframe.
    """
    return encoder.transform(df[cols_encode].astype(str)).toarray()


def _sum_game_session(df, cols_encode, encoder):
    def encode_and_sum(df_):
        encoded = apply_one_hot_encoder(df_, cols_encode, encoder, True)
        return encoded.sum(axis=0)

    session_sums = [encode_and_sum(gdf[cols_encode])
                    for _, gdf in df.groupby('game_session', sort=False)]
    return np.vstack(session_sums)


def sum_game_session(df, cols_encode, encoder):
    session_sums = []
    for inst_idx, user_sample in tqdm(df.groupby('installation_id', sort=False)):
        session_sums.append(_sum_game_session(user_sample, cols_encode, encoder))

    keys = (
        df[['installation_id', 'game_session']]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    session_sums = pd.DataFrame(np.vstack(session_sums), columns=get_categories(encoder))

    assert len(keys) == len(session_sums), 'The number of rows must be equal.'
    return pd.concat([keys, session_sums], axis=1)


def main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    cols_encode = ['title', 'event_code']
    encoder = build_one_hot_encoder(train, test, cols_encode)

    sess_sums_train = sum_game_session(train, cols_encode, encoder)
    sess_sums_test = sum_game_session(test, cols_encode, encoder)

    name = remove_ext(os.path.basename(__file__))
    save_features(sess_sums_train, name, 'train')
    save_features(sess_sums_test, name, 'test')


if __name__ == '__main__':
    main()
