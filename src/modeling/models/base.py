from abc import abstractmethod, ABCMeta
import numpy as np
from sklearn.utils import shuffle


def random_truncate(groups, seed):
    """
    Create a mask that samples one assessment from each installation_id.
    """
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

    def feature_importance_average(self, importance_type, normalize=True):
        """
        Compute average feature importance of given models.
        """
        imps = []
        for model in self.models:
            names, imp = self.feature_importance(model, importance_type=importance_type)
            imps.append((imp / imp.sum()) if normalize else imp)

        return np.mean(imps, axis=0)

    def cv(self, X, y, groups, cv, config):
        """
        Perform cross-validation.
        """
        num_seeds = len(config.seeds)
        oof_preds = np.zeros((len(X), num_seeds))
        oof_labels = np.zeros((len(X), num_seeds))

        for seed_idx, seed in enumerate(config.seeds):
            config.params.update({'random_state': seed})

            mask = random_truncate(groups, seed)
            assert groups[mask].is_unique
            X = X.loc[mask]
            y = y.loc[mask]

            for fold_idx, (idx_trn, idx_val) in enumerate(cv.split(X, y, groups)):
                print(f'\n---------- Seed: {seed_idx} / Fold: {fold_idx} ----------\n')
                X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
                y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]

                model, eval_result = self.fit(X_trn, y_trn, X_val, y_val, config)
                self.models.append(model)
                self.eval_results.append(eval_result)
                oof_preds[idx_val, seed_idx] = self.predict(model, X_val)
                oof_labels[idx_val, seed_idx] = y_val

        return oof_preds, oof_labels
