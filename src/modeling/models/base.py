from abc import abstractmethod, ABCMeta
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def random_truncate(groups, seed):
    """
    Create a mask that samples one assessment from each installation_id.
    """
    assert isinstance(groups, pd.Series)
    return shuffle(groups, random_state=seed).drop_duplicates(keep='first').index


class BaseModel(metaclass=ABCMeta):
    def __init__(self):
        self.models = []
        self.eval_results = []
        self.oof_preds = []

    @abstractmethod
    def fit(self, X_trn, y_trn, X_val, y_val, config):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, X):
        raise NotImplementedError

    @abstractmethod
    def feature_importance(self, model, X):
        raise NotImplementedError

    @abstractmethod
    def feature_name(self, model, X):
        raise NotImplementedError

    def predict_average(self, X):
        """
        Predict average score using given models.
        """
        return np.mean([self.predict(model, X) for model in self.models], axis=0)

    def predict_median(self, X):
        """
        Predict median score using given models.
        """
        return np.median([self.predict(model, X) for model in self.models], axis=0)

    def feature_importance_average(self, importance_type, normalize=True, return_std=False):
        """
        Compute average feature importance.
        """
        imps = []
        for model in self.models:
            imp = self.feature_importance(model, importance_type=importance_type)
            imps.append(imp)

        imps = np.array(imps)

        if normalize:
            imps = imps / imps.sum(axis=1, keepdims=True)

        if return_std:
            return np.mean(imps, axis=0), np.std(imps, axis=0)

        return np.mean(imps, axis=0)

    def cv(self, X, y, groups, cv, config):
        """
        Perform cross-validation.
        """
        num_seeds = len(config.seeds)
        oof_preds = np.zeros((len(X), num_seeds))
        oof_labels = []

        for seed_idx, seed in enumerate(config.seeds):
            config.params.update({'random_state': seed})
            # X, y, groups = shuffle(X, y, groups, random_state=seed)

            for fold_idx, (idx_trn, idx_val) in enumerate(cv.split(X, y, groups)):
                print(f'\n---------- Seed: {seed_idx} / Fold: {fold_idx} ----------\n')
                X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
                y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]
                groups_trn = groups.iloc[idx_trn]
                groups_val = groups.iloc[idx_val]

                # truncate train data.
                mask_trn = random_truncate(groups_trn, seed)
                assert groups[mask_trn].is_unique
                X_trn = X_trn.loc[mask_trn]
                y_trn = y_trn.loc[mask_trn]

                # # truncate validation data.
                # mask_val = random_truncate(groups_val, seed)
                # X_val = X_val.loc[mask_val]
                # y_val = y_val.loc[mask_val]

                model, eval_result = self.fit(X_trn, y_trn, X_val, y_val, config)
                oof_preds[y_val.index.values, seed_idx] = self.predict(model, X_val)

                # # when truncating validation data.
                # oof_preds.append(self.predict(model, X_val))
                # oof_labels.append(y_val)

                self.models.append(model)
                self.eval_results.append(eval_result)

        return np.nanmedian(oof_preds, axis=1)
