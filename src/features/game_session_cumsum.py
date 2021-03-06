import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.common import remove_dir_ext
from utils.io import read_from_clean, save_features
from utils.array import shift_array
from utils.encoding import build_one_hot_encoder, apply_one_hot_encoder, get_categories


def sum_game_session(df, cols_sum):
    return np.vstack([gdf[cols_sum].values.sum(axis=0)
                      for _, gdf in df.groupby('game_session', sort=False)])


def cumsum_game_session(df, cols_encode, encoder):
    categories = get_categories(encoder)
    cumsum = []
    for inst_idx, user_sample in tqdm(df.groupby('installation_id', sort=False)):
        # session_sums = sum_game_session(user_sample, cols_encode, encoder)
        user_sample = apply_one_hot_encoder(user_sample, encoder, cols_encode)
        session_sums = sum_game_session(user_sample, categories)

        # take cumulative sum and shift it.
        cumsum.append(shift_array(np.cumsum(session_sums, axis=0), 1))

    keys = (
        df[['installation_id', 'game_session']]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    cumsum = pd.DataFrame(np.vstack(cumsum), columns=get_categories(encoder))

    assert len(keys) == len(cumsum), 'the number of records must be equal.'
    return pd.concat([keys, cumsum], axis=1)


def main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    train['title_event_code'] = train['title'] + '_' + train['event_code'].astype(str)
    test['title_event_code'] = test['title'] + '_' + test['event_code'].astype(str)

    cols_encode = ['title', 'event_code', 'title_event_code', 'event_id', 'type']
    encoder = build_one_hot_encoder(train, test, cols_encode)

    cumsum_train = cumsum_game_session(train, cols_encode, encoder)
    cumsum_test = cumsum_game_session(test, cols_encode, encoder)

    cumsum_train = cumsum_train.fillna(0)
    cumsum_test = cumsum_test.fillna(0)

    name = remove_dir_ext(__file__)

    # save encoded features separately
    for col_idx, col in enumerate(cols_encode):
        save_cols = ['installation_id', 'game_session'] + encoder.categories_[col_idx].tolist()
        save_features(cumsum_train[save_cols], name + f'_{col}', 'train')
        save_features(cumsum_test[save_cols], name + f'_{col}', 'test')

    save_features(cumsum_train, name, 'train')
    save_features(cumsum_test, name, 'test')


if __name__ == '__main__':
    main()
