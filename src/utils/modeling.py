import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from utils.array import div_by_sum


def get_cv(config):
    """
    Return a cross-validation splitter with given configuration.
    """
    classes = {
        KFold.__name__: KFold,
        StratifiedKFold.__name__: StratifiedKFold,
        GroupKFold.__name__: GroupKFold,
    }
    return classes[config.cv.type](**config.cv.params)


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
    Compute average predict score using given models.
    """
    return np.mean([model.predict(X) for model in models], axis=0)
