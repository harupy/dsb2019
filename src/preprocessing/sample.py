from utils.io import read_from_clean, save_to_clean
from utils.sampling import hash_mod_sample


def main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')
    labels = read_from_clean('train_labels.ftr')

    n_splits = 10
    hash_col = 'installation_id'
    train = hash_mod_sample(train, hash_col, n_splits)
    test = hash_mod_sample(test, hash_col, n_splits)
    labels = hash_mod_sample(labels, hash_col, n_splits)

    save_to_clean(train, 'train_sample.ftr')
    save_to_clean(test, 'test_sample.ftr')
    save_to_clean(labels, 'train_labels_sample.ftr')


if __name__ == '__main__':
    main()
