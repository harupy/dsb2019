
"""
Functions for metrics.
"""


from functools import partial
import numpy as np
import scipy
from numba import jit


@jit
def qwk(a1, a2):
    """
    Compute quadratic weighted kappa.

    https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168
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


def digitize(x, boundaries):
    """
    >>> x = np.array([0.1, 1.1, 2.1])
    >>> boundaries = [0.2, 1.6]
    >>> digitize(x, boundaries)
    array([0, 1, 2])

    """
    bins = [-np.inf] + list(np.sort(boundaries)) + [np.inf]
    return np.digitize(x, bins) - 1


class OptimizedRounder(object):
    """
    Class to optimize round boundaries to maximize QWK.
    """

    def __init__(self):
        self.boundaries = None

    def _kappa_loss(self, coef, y_true, y_pred):
        return -qwk(y_true, digitize(y_pred, coef))

    def fit(self, y_true, y_pred):
        loss_partial = partial(self._kappa_loss, y_true=y_true, y_pred=y_pred)
        self.boundaries = scipy.optimize.minimize(loss_partial, [0.5, 1.5, 2.5],
                                                  method='nelder-mead')['x']

    def predict(self, x):
        return digitize(x, self.boundaries)
