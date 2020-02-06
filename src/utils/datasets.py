import pandas as pd
from sklearn import datasets


def _get_X_y(data, column_formatter=None):
    """
    Convert scikit-learn's dataset to pandas dataframe and series.
    """
    columns = (data.feature_names if column_formatter is None
               else list(map(column_formatter, data.feature_names)))
    X = pd.DataFrame(data.data, columns=columns)
    y = pd.Series(data.target, name='target')
    return X, y


def iris():
    return _get_X_y(datasets.iris)


def breast_cancer():
    return _get_X_y(datasets.load_breast_cancer(), lambda c: c.replace(' ', '_'))


def wine():
    return _get_X_y(datasets.load_wine)
