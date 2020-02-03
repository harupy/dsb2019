import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils.encoding import (AggEncoder,
                            AggDiffEncoder,
                            AggRatioEncoder,
                            AggDiffRatioEncoder,
                            OneHotEncoder,
                            FreqEncoder,
                            TargetEncoder)


def main():
    N = 100
    train = pd.DataFrame({
        'a': np.random.randint(5, size=N),
        'b': np.random.randint(5, size=N),
        'c': np.random.random(N),
        'd': np.random.random(N),
        'target': np.random.randint(2, size=N),
    })

    test = pd.DataFrame({
        'a': np.random.randint(5, size=N),
        'b': np.random.randint(5, size=N),
        'c': np.random.random(N),
        'd': np.random.random(N),
    })

    param_dict = [
        {
            'key': ['a', ['a', 'b']],
            'var': ['c', 'd'],
            'agg': ['mean', 'max', 'std', 'min', 'median', 'nunique'],
        },
    ]

    agg_enc = AggEncoder(param_dict)
    agg_enc.fit(train)
    train = agg_enc.transform(train)
    test = agg_enc.transform(test)

    encs = [
        AggDiffEncoder(param_dict),
        AggRatioEncoder(param_dict),
        AggDiffRatioEncoder(param_dict),
        OneHotEncoder(['a', 'b']),
        FreqEncoder(['a', 'c'])
    ]

    for enc in encs:
        print('Applying', enc.__class__.__name__)
        enc.fit(train)
        train = enc.transform(train)
        test = enc.transform(test)
        del enc

    X_train = train.drop('target', axis=1)
    y_train = train['target']

    target_enc = TargetEncoder(['a', 'b'], n_splits=5, random_state=42)

    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for idx_trn, idx_val in fold.split(X_train, y_train):
        X_trn, X_val = X_train.iloc[idx_trn], X_train.iloc[idx_val]
        y_trn, y_val = y_train.iloc[idx_trn], y_train.iloc[idx_val]

        X_trn, X_val = target_enc.transform_cv(X_trn, y_trn, X_val)

    X_test = target_enc.transform_test(X_train, y_train, test)
    # print(X_test)


if __name__ == '__main__':
    main()
