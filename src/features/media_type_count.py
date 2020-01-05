import pandas as pd

from utils.io import read_from_clean, save_features
from utils.common import remove_dir_ext
from utils.encoding import build_one_hot_encoder, apply_one_hot_encoder, get_categories


def main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    train = train.groupby(['installation_id', 'game_session']).last().reset_index()
    test = test.groupby(['installation_id', 'game_session']).last().reset_index()

    cols = ['type']
    encoder = build_one_hot_encoder(train, test, cols)
    categories = get_categories(encoder)
    train = apply_one_hot_encoder(train, encoder, cols, drop=True)
    test = apply_one_hot_encoder(test, encoder, cols, drop=True)
    keep = ['installation_id', 'game_session']
    train = train[keep + categories]
    test = test[keep + categories]

    def cumsum_shift(df):
        return pd.concat([df[keep], df.drop(keep, axis=1).cumsum().shift(1, fill_value=0)], axis=1)

    train = train.groupby('installation_id', sort=False).apply(cumsum_shift)
    test = test.groupby('installation_id', sort=False).apply(cumsum_shift)

    name = remove_dir_ext(__file__)
    save_features(train, name, 'train')
    save_features(train, name, 'test')


if __name__ == '__main__':
    main()
