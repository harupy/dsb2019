"""
Utilities functions for encoding.
"""

from functools import reduce
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder as ScikitOneHotEncoder
from sklearn.model_selection import KFold


class AggEncoder():
    def __init__(self, param_dict=None):
        param_dict = [param_dict] if isinstance(param_dict, dict) else param_dict
        self.param_dict = [{'key': self._normalize_key(p['key']),
                            'var': self._normalize_var(p['var']),
                            'agg': self._normalize_agg(p['agg'])}
                           for p in param_dict]

    def _normalize_key(self, key):
        if isinstance(key, str):
            return [[key]]
        elif isinstance(key, list) and all(isinstance(k, str) for k in key):
            return [key]
        else:
            return [k if isinstance(k, list) else [k] for k in key]

    def _normalize_var(self, var):
        return var if isinstance(var, list) else [var]

    def _normalize_agg(self, agg):
        return agg if isinstance(agg, list) else [agg]

    def _get_params(self, p_dict):
        return p_dict['key'], p_dict['var'], p_dict['agg']

    def _get_agg_name(self, agg):
        return agg if isinstance(agg, str) else agg.__name__

    def _iter_param_dict(self):
        for p_dict in self.param_dict:
            key, var, agg = self._get_params(p_dict)
            for k in key:
                for v in var:
                    for a in agg:
                        yield k, v, self._get_agg_name(a)

    def _feature_name(self, key, var, agg, prefix=''):
        return '_'.join([prefix + agg, var, 'by'] + key)

    def get_feature_names(self):
        return list(self._feature_name(*item) for item in self._iter_param_dict())

    def _get_agg_feature_names(self, key, var, agg):
        _agg = [self._get_agg_name(a) for a in agg]

        return [self._feature_name(key, v, a)
                for v in var
                for a in _agg]

    def _to_dict_key(self, x):
        return tuple(x) if len(x) > 1 else x[0]

    def fit(self, X, y=None):
        self.features = []
        for p_dict in self.param_dict:
            key, var, agg = self._get_params(p_dict)
            features = {}

            for k in key:
                use_features = list(set(k + var))
                agged = X[use_features].groupby(k)[var].agg(agg).reset_index()
                columns = self._get_agg_feature_names(k, var, agg)
                agged.columns = k + columns
                features[self._to_dict_key(k)] = agged

            self.features.append(features)
        return self

    def transform(self, X, y=None):
        for p_dict, features in zip(self.param_dict, self.features):
            for key in p_dict['key']:
                X = X.merge(features[self._to_dict_key(key)], on=key, how='left')
        return X

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


class AggDiffEncoder(AggEncoder):

    def _feature_name(self, key, var, agg):
        return super()._feature_name(key, var, agg, prefix='diff_')

    def transform(self, X, y=None):
        for k, v, a in self._iter_param_dict():
            agg_feature = super()._feature_name(k, v, a)
            new_feature = self._feature_name(k, v, a)
            X = X.assign(**{new_feature: X[v] - X[agg_feature]})
        return X


class AggRatioEncoder(AggEncoder):

    def _feature_name(self, key, var, agg):
        return super()._feature_name(key, var, agg, prefix='ratio_')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for k, v, a in self._iter_param_dict():
            agg_feature = super()._feature_name(k, v, a)
            new_feature = self._feature_name(k, v, a)
            X = X.assign(**{new_feature: X[v] / X[agg_feature]})
        return X


class AggDiffRatioEncoder(AggEncoder):

    def _feature_name(self, key, var, agg):
        return super()._feature_name(key, var, agg, prefix='diff_ratio_')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for k, v, a in self._iter_param_dict():
            agg_feature = super()._feature_name(k, v, a)
            new_feature = self._feature_name(k, v, a)
            X = X.assign(**{new_feature: (X[v] - X[agg_feature]) / X[agg_feature]})
        return X


class OneHotEncoder():

    def __init__(self, cols):
        self.cols = cols
        self.enc = None

    def get_feature_names(self):
        return reduce(lambda l, c: l + c.tolist(), self.enc.categories_, [])

    def fit(self, X, y=None):
        self.enc = ScikitOneHotEncoder(categories='auto')
        self.enc.fit(X[self.cols].astype(str))
        return self

    def transform(self, X):
        new_columns = self.get_feature_names()
        transformed = pd.DataFrame(self.enc.transform(X[self.cols].astype(str)).toarray(),
                                   index=X.index, columns=new_columns)
        return pd.concat([X, transformed], axis=1)


class FreqEncoder():

    def __init__(self, cols, normalize=True):
        self.cols = cols
        self.normalize = normalize

    def get_feature_names(self):
        return list(self.enc.keys())

    def fit(self, df):
        self.enc = {}
        for col in self.cols:
            self.enc[col] = df[col].value_counts(normalize=self.normalize)
        return self

    def transform(self, df):
        for col in self.cols:
            df = df.assign(**{f'{col}_freq': df[col].map(self.enc[col])})
        return df


class TargetEncoder():

    def __init__(self, cols, n_splits=5, random_state=None):
        self.cols = cols
        self.n_splits = n_splits
        self.random_state = random_state
        self.target_col = '__TARGET__'
        self.feature_col = '__FEATURE__'

    def _feature_name(self, col):
        return f'{col}_target_mean'

    def get_feature_names(self):
        return [self._feature_name(col) for col in self.cols]

    def _validate_columns(self, X):
        assert self.target_col not in X.columns

    def _merge_X_y(self, X, y):
        return pd.DataFrame({self.feature_col: X, self.target_col: y})

    def _target_mean(self, Xy):
        return Xy.groupby(self.feature_col)[self.target_col].mean()

    def _oof_mean(self, Xy):
        # transform train data.
        kf = KFold(n_splits=self.n_splits, shuffle=True,
                   random_state=self.random_state)
        target_mean = np.repeat(np.nan, len(Xy))
        for idx_1, idx_2 in kf.split(Xy):
            tmp = self._target_mean(Xy.iloc[idx_1])
            target_mean[idx_2] = Xy[self.feature_col].iloc[idx_2].map(tmp)
        return target_mean

    def transform(self, X, y):
        for col in self.cols:
            Xy = self._merge_X_y(X[col], y)
            feature_name = self._feature_name(col)
            target_mean = self._oof_mean(Xy)
            X = X.assign(**{feature_name: target_mean})
        return X

    def transform_test(self, X_train, y_train, X_test):
        self._validate_columns(X_train)
        for col in self.cols:
            Xy = self._merge_X_y(X_train[col], y_train)
            feature_name = self._feature_name(col)
            target_mean = self._target_mean(Xy)
            X_test = X_test.assign(**{feature_name: X_test[col].map(target_mean)})
        return X_test

    def transform_cv(self, X_trn, y_trn, X_val):
        self._validate_columns(X_trn)
        for col in self.cols:
            Xy = self._merge_X_y(X_trn[col], y_trn)

            # transform train data.
            feature_name = self._feature_name(col)
            target_mean = self._oof_mean(Xy)
            X_trn = X_trn.assign(**{feature_name: target_mean})

            # transform validation data.
            target_mean = self._target_mean(Xy)
            X_val = X_val.assign(**{feature_name: X_val[col].map(target_mean)})

        return X_trn, X_val
