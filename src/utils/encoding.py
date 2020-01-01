from functools import reduce
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def build_one_hot_encoder(train, test, cols_encode):
    """
    Build a one-hot encoder from the given train and test dataframe.
    """
    merged = train[cols_encode].append(test[cols_encode]).astype(str)
    encoder = OneHotEncoder(dtype=np.int8)
    return encoder.fit(merged)


def get_categories(encoder):
    cats = reduce(lambda lst, cat: lst + cat.tolist(), encoder.categories_, [])

    # find duplicated values
    seen = set()
    dups = []
    for x in cats:
        if x not in seen:
            seen.add(x)
        else:
            dups.append(x)

    # assert no duplicates in the categories
    assert len(dups) == 0, 'Find duplicated elements: {}'.format(dups)
    return cats
