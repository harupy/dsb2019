from utils.io import read_from_raw, save_to_clean
from utils.dataframe import reduce_mem_usage


def main():
    train = read_from_raw('train.csv')
    test = read_from_raw('test.csv')
    labels = read_from_raw('train_labels.csv')

    # remove useless installation ids
    mask = train['installation_id'].isin(labels['installation_id'].unique())
    train = train[mask].reset_index(drop=True)  # reset index to save dataframe as feather

    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    labels = reduce_mem_usage(labels)

    save_to_clean(train, 'train.ftr')
    save_to_clean(test, 'test.ftr')
    save_to_clean(labels, 'train_labels.ftr')


if __name__ == '__main__':
    main()
