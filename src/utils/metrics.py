
from functools import partial
import numpy as np
import pandas as pd
import scipy
from numba import jit


@jit
def calc_qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
    :param max_rat:
    :return:
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


class OptimizedRounder(object):
    """
    Class to optimize round boundaries to maximize QWK.
    """

    def __init__(self):
        self.coef_ = None

    def _kappa_loss(self, coef, y_true, y_pred):
        return -calc_qwk(y_true, self.round(y_pred, coef))

    def fit(self, y_true, y_pred):
        loss_partial = partial(self._kappa_loss, y_true=y_true, y_pred=y_pred)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = scipy.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def round(self, x, coef):
        # ref.: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html
        bins = [-np.inf] + list(np.sort(coef)) + [np.inf]
        return pd.cut(x, bins, labels=False)

    def coefficients(self):
        return self.coef_['x']
