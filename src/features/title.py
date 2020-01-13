from utils.io import read_from_clean, save_features
from utils.common import remove_dir_ext


def main():
    train = read_from_clean('train_labels.ftr')
    test = read_from_clean('test.ftr')

    mapper = {title: idx for idx, title in enumerate(train['title'].unique())}

    cols = ['installation_id', 'game_session', 'title']
    train = train[cols]
    test = test.groupby('installation_id', sort=False).last().reset_index()[cols]

    train['title'] = train['title'].map(mapper)
    test['title'] = test['title'].map(mapper)

    name = remove_dir_ext(__file__)
    save_features(train, name, 'train')
    save_features(test, name, 'test')


if __name__ == '__main__':
    main()
