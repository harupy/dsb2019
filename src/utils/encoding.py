"""
Utility functions for encoding.
"""

from functools import reduce
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def build_one_hot_encoder(train, test, cols):
    """
    Build a one-hot encoder from the given train and test dataframe.

    Returns
    -------
    sklearn.preprocessing.OneHotEncoder
        Fitted one-hot encodoer

    """
    merged = train[cols].append(test[cols]).astype(str)
    encoder = OneHotEncoder(dtype=np.int8)
    return encoder.fit(merged)


def apply_one_hot_encoder(df, encoder, cols, drop=False):
    """
    Apply a one-hot encoder to the given dataframe.
    """

    categories = get_categories(encoder)

    # without index, pd.concat does not work properly.
    encoded = pd.DataFrame(encoder.transform(df[cols].astype(str)).toarray(),
                           index=df.index, columns=categories,)

    if drop:
        return pd.concat([df.drop(cols, axis=1), encoded], axis=1)
    else:
        return pd.concat([df, encoded], axis=1)


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
