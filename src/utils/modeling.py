"""
Utilities for modeling.
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from utils.array import div_by_sum


def _make_map(classes):
    return {cls.__name__: cls for cls in classes}


def get_cv(config):
    """
    Create a cross-validation splitter with given configuration.
    """

    classes = [KFold, StratifiedKFold, GroupKFold]
    return _make_map(classes)[config.cv.type](**config.cv.params)


def average_feature_importance(models, importance_type, normalize=True):
    """
    Compute average feature importance of given models.
    """
    imps = []
    for model in models:
        imp = model.feature_importance(importance_type=importance_type)
        imps.append(div_by_sum(imp) if normalize else imp)

    return np.mean(imps, axis=0)


def predict_average(models, X):
    """
    Predict average score using given models.
    """
    return np.mean([model.predict(X) for model in models], axis=0)


def predict_median(models, X):
    """
    Predict median score using given models.
    """
    return np.median([model.predict(X) for model in models], axis=0)
