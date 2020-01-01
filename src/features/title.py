import os

from utils.io import read_from_clean, save_features
from utils.common import remove_ext


def main():
    train = read_from_clean('train_labels.ftr')
    test = read_from_clean('test.ftr')

    mapper = {t: idx for idx, t in enumerate(train['title'].unique())}

    cols = ['installation_id', 'game_session', 'title']
    title_train = train[cols]
    title_test = test.groupby('installation_id', sort=False).last().reset_index()[cols]

    title_train['title'] = title_train['title'].map(mapper)
    title_test['title'] = title_test['title'].map(mapper)

    name = remove_ext(os.path.basename(__file__))
    save_features(title_train, name, 'train')
    save_features(title_test, name, 'test')


if __name__ == '__main__':
    main()
