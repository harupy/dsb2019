"""
Construct unsorted train labels from the train data.
The provided train labels are sorted by `game_session`.
"""

import numpy as np

from utils.io import read_from_raw, read_from_clean, save_to_clean
from utils.common import with_name
from features.funcs import filter_assessment_attempt, assign_attempt_result, classify_accuracy

from pandas.testing import assert_frame_equal


def main():
    train = read_from_clean('train.ftr')
    labels = read_from_raw('train_labels.csv')

    train = filter_assessment_attempt(train)
    train = assign_attempt_result(train)

    aggs = {'correct': [
        with_name(lambda ser: (ser == 1).sum(), 'num_correct'),
        with_name(lambda ser: (ser == 0).sum(), 'num_incorrect'),
        with_name(lambda ser: ser.mean(), 'accuracy'),
    ]}

    by = ['installation_id', 'title', 'game_session']
    train = train.groupby(by, sort=False).agg(aggs).reset_index()
    train.columns = [col[1] if (col[1] != '') else col[0] for col in train.columns]
    train = train.assign(accuracy_group=train['accuracy'].map(classify_accuracy).astype(np.int64))

    assert_frame_equal(
        train.sort_values(by).reset_index(drop=True).sort_index(axis=1),
        labels.sort_values(by).reset_index(drop=True).sort_index(axis=1),
    )

    # the raw train_labels.csv is sorted by game_session.s
    save_to_clean(train, 'train_labels_unsorted.ftr')


if __name__ == '__main__':
    main()
