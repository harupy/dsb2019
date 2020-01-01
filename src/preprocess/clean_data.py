from utils.io import read_from_raw, save_to_clean
from utils.dataframe import reduce_mem_usage


def main():
    files = ['train.csv', 'test.csv', 'train_labels.csv']
    train, test, labels = read_from_raw(files)

    # remove useless installation ids
    mask = train['installation_id'].isin(labels['installation_id'].unique())
    train = train[mask].reset_index(drop=True)  # reset index to save dataframe as feather

    reduce_mem_usage(train)
    reduce_mem_usage(test)
    reduce_mem_usage(labels)

    save_to_clean(train, 'train.ftr')
    save_to_clean(test, 'test.ftr')
    save_to_clean(labels, 'train_labels.ftr')


if __name__ == '__main__':
    main()
