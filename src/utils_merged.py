########################################
# metrics.py
########################################


from zlib import crc32
import importlib
import yaml
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from functools import reduce
from functools import partial
import numpy as np
import scipy
from numba import jit


@jit
def qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
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


########################################
# encoding.py
########################################


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


########################################
# dataframe.py
########################################


def reduce_mem_usage(df, verbose=True):
    numeric_dtypes = ['int8', 'int16', 'int32', 'int64',
                      'float16', 'float32', 'float64']
    mem_before = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        dtype = df[col].dtypes

        if dtype in numeric_dtypes:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(dtype)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    # df[col] = df[col].astype(np.float16)
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float132)
                else:
                    df[col] = df[col].astype(np.float64)

    mem_after = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'
              .format(mem_after, 100 * (mem_before - mem_after) / mem_before))


def apply_funcs(df, funcs):
    """
    Apply functions to a dataframe.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [0]})
    >>> f1 = lambda df: df + 1
    >>> f2 = lambda df: df + 2
    >>> apply_funcs(df, [f1, f2])
       a
    0  3

    """
    return reduce(lambda df, f: f(df), funcs, df)


def concat_dfs(dfs, axis):
    """
    Concatenate multiple dataframes.

    Examples
    --------
    >>> df1 = pd.DataFrame({'a': [0]})
    >>> df2 = pd.DataFrame({'b': [0]})
    >>> concat_dfs([df1, df2], axis=1)
       a  b
    0  0  0

    """
    return reduce(lambda left, right: pd.concat([left, right], axis=axis), dfs)


def prefix_columns(df, prefix, sep='_', exclude=None):
    """
    Prefix dataframe columns.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [0], 'b': [0]})
    >>> prefix_columns(df, 'x')
       x_a  x_b
    0    0    0

    >>> prefix_columns(df, 'x', '|')
       x|a  x|b
    0    0    0

    >>> prefix_columns(df, 'x', '|', ['b'])
       x|a  b
    0    0  0

    """
    exclude = [] if exclude is None else exclude
    return df.rename(columns={c: prefix + sep + c for c in df.columns if c not in exclude})


def suffix_columns(df, suffix, sep='_', exclude=None):
    """
    Suffix dataframe columns.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [0], 'b': [0]})
    >>> suffix_columns(df, 'x')
       a_x  b_x
    0    0    0

    >>> suffix_columns(df, 'x', '|')
       a|x  b|x
    0    0    0

    >>> suffix_columns(df, 'x', '|', ['b'])
       a|x  b
    0    0  0

    """
    exclude = [] if exclude is None else exclude
    return df.rename(columns={c: c + sep + suffix for c in df.columns if c not in exclude})


########################################
# plotting.py
########################################


sns.set()


class JointConfusionMatrix:
    """
    Ref. https://github.com/mwaskom/seaborn/blob/master/seaborn/axisgrid.py#L1551
    """

    def __init__(self, cm, height=6, ratio=5, space=.2,
                 dropna=True, xlim=None, ylim=None, size=None):

        # set up the subplot grid
        f = plt.figure(figsize=(height, height))
        gs = plt.GridSpec(ratio + 1, ratio + 1)

        ax_joint = f.add_subplot(gs[1:, :-1])
        ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
        ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)

        self.fig = f
        self.ax_joint = ax_joint
        self.ax_marg_x = ax_marg_x
        self.ax_marg_y = ax_marg_y
        self.cm = cm

        # turn off tick visibility for the measure axis on the marginal plots
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)

        # utrn off the ticks on the density axis for the marginal plots
        plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
        plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
        plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
        plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
        plt.setp(ax_marg_x.get_yticklabels(), visible=False)
        plt.setp(ax_marg_y.get_xticklabels(), visible=False)

        ax_marg_x.yaxis.grid(False)
        ax_marg_y.xaxis.grid(False)

        if xlim is not None:
            ax_joint.set_xlim(xlim)
        if ylim is not None:
            ax_joint.set_ylim(ylim)

        # make the grid look nice
        sns.utils.despine(f)
        sns.utils.despine(ax=ax_marg_x, left=True)
        sns.utils.despine(ax=ax_marg_y, bottom=True)
        f.tight_layout()
        f.subplots_adjust(hspace=space, wspace=space)

    def make_annotation(self, cm, cm_norm, normalize=True):
        annot = []
        nrows, ncols = cm.shape
        base = '{}\n({:.2f})'
        for ir in range(nrows):
            annot.append([])
            for ic in range(ncols):
                annot[ir].append(base.format(cm[ir, ic], cm_norm[ir, ic]))

        return np.array(annot)

    def plot(self, labels=None, normalize=True):
        labels = [i for i in range(len(self.cm))] if labels is None else labels

        true_dist = self.cm.sum(axis=1)
        pred_dist = self.cm.sum(axis=0)
        pos = np.arange(self.cm.shape[0]) + 0.5

        # normalize
        cm_norm = self.cm / true_dist.reshape(-1, 1)
        annot = self.make_annotation(self.cm, cm_norm)

        FONTSIZE = 20

        # plot confusion matrix as a heatmap
        sns.heatmap(cm_norm, cmap='Blues', vmin=0, vmax=1,
                    annot=annot, fmt='s', annot_kws={'fontsize': FONTSIZE},
                    linewidths=0.2, cbar=False, square=True, ax=self.ax_joint)
        self.ax_joint.set_xlabel('Predicted label', fontsize=FONTSIZE)
        self.ax_joint.set_ylabel('True label', fontsize=FONTSIZE)
        self.ax_joint.set_xticklabels(labels, fontsize=FONTSIZE)
        self.ax_joint.set_yticklabels(labels, fontsize=FONTSIZE)

        props = {'align': 'center'}

        # plot label distribution
        self.ax_marg_x.bar(pos, pred_dist / pred_dist.sum(), **props)
        self.ax_marg_y.barh(pos, true_dist / true_dist.sum(), **props)


def plot_confusion_matrix(cm):
    g = JointConfusionMatrix(cm, height=10)
    g.plot()
    g.fig.tight_layout()
    return g.fig


def plot_feature_importance(feature_names, importance, importance_type, limit=30):
    """
    Plot feature importance and return the figure.
    """
    indices = np.argsort(importance)[-limit:]
    y = np.arange(len(indices))

    fig, ax = plt.subplots()
    ax.barh(y, importance[indices], align='center', height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(feature_names[indices])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Feature Importance: {importance_type}')
    fig.tight_layout()
    return fig


def plot_label_share(labels):
    unique, counts = np.unique(labels, return_counts=True)

    counts_norm = counts / counts.sum()
    fig, ax = plt.subplots()
    bar = sns.barplot(unique, counts_norm)
    for idx, p in enumerate(bar.patches):
        bar.annotate('{:.2f}\n({})'.format(counts_norm[idx], counts[idx]),
                     (p.get_x() + p.get_width() / 2, p.get_height() / 2),
                     ha='center', va='center', color='white', fontsize='large')
    ax.set_xlabel('Label')
    ax.set_ylabel('Share')
    ax.set_title('Label Share')
    fig.tight_layout()
    return fig


def plot_eval_history(eval_results):
    fig, ax = plt.subplots()
    for fold_idx, eval_result in enumerate(eval_results):
        for data_name, metrics in eval_result.items():
            for metric_name, values in metrics.items():
                label = f'{data_name}-{metric_name}-{fold_idx}'
                ax.plot(values, label=label, zorder=1)[0]
                # ax.scatter(data['best_iteration'], data['values'][data['best_iteration'] - 1],
                #            s=60, c=[line.get_color()], edgecolors='k', linewidths=1, zorder=2)
    ax.set_xlabel('Iteration')
    ax.set_title(
        'Evaluation History (marker on each line represents the best iteration)')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    return fig


########################################
# io.py
########################################


PARENT_DIR = '../input' if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else 'data'
RAW_DIR = f'{PARENT_DIR}/data-science-bowl-2019'
CLEAN_DIR = f'{PARENT_DIR}/clean'
FTR_DIR = f'{PARENT_DIR}/features'

for d in [CLEAN_DIR, FTR_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)


def read_config(fpath):
    """
    Read a config dict from the given python script.
    """
    name = os.path.splitext(fpath)[0].replace('/', '.')
    return importlib.import_module(name).config


def read_json(fpath):
    """
    Read a json file.
    """
    with open(fpath, 'r') as f:
        return json.load(f)


def read_yaml(fpath):
    """
    Read a yaml file.
    """
    with open("example.yaml", 'r') as f:
        return yaml.safe_load(f)


def save_dict(dct, fpath):
    """
    Save a dictionary as json.
    """
    with open(fpath, 'w') as f:
        json.dump(dct, f, indent=2)


def read_data(fpath):
    """
    Read tabular data.
    """
    print(f'Reading {fpath}')
    if fpath.endswith('.csv'):
        return pd.read_csv(fpath)
    elif fpath.endswith('.ftr'):
        return pd.read_feather(fpath)
    else:
        raise TypeError('Invalid file type: {}'.format(
            os.path.splitext(fpath)))


def save_data(df, fpath):
    """
    Save tabular data.
    """
    print(f'Saving {fpath}')
    if fpath.endswith('.csv'):
        return df.to_csv(fpath, index=False)
    elif fpath.endswith('.ftr'):
        return df.reset_index(drop=True).to_feather(fpath)
    else:
        raise TypeError('Invalid file type: {}'.format(
            os.path.splitext(fpath)))


def read_from_raw(fname):
    """
    Read data from the raw data directory.
    """
    fpath = os.path.join(RAW_DIR, fname)
    return read_data(fpath)


def read_from_clean(fname):
    """
    Read data from the clean data directory.
    """
    fpath = os.path.join(CLEAN_DIR, fname)
    return read_data(fpath)


def save_to_clean(df, fname):
    """
    Save a dataframe in the clean data directory.
    """
    fpath = os.path.join(CLEAN_DIR, fname)
    save_data(df, fpath)


def read_features(name, train_or_test=None, fmt='ftr'):
    """
    Read features from the feature directory.
    """
    if train_or_test is not None and train_or_test not in ['train', 'test']:
        raise ValueError('`train_or_test` must be either "train" or "test".')

    save_dir = os.path.join(FTR_DIR, name)

    # if `train_or_test` is None, return both train and test data.
    if train_or_test is None:
        return (
            read_data(os.path.join(save_dir, f'{name}_train.{fmt}')),
            read_data(os.path.join(save_dir, f'{name}_test.{fmt}')),
        )
    else:
        return read_data(os.path.join(save_dir, f'{name}_{train_or_test}.{fmt}'))


def save_features(df, name, train_or_test, fmt='ftr'):
    """
    Save a dataframe to the feature directory.

    Parameters
    ----------
    df : dataframe
        dataframe containing features.
    name : str
        feature name
    train_or_test : str enum ('train', 'test')
        string to indicate train or test set
    fmt : str default 'ftr'
        format to save features in.

    """
    if train_or_test not in ['train', 'test']:
        raise ValueError('`train_or_test` must be either "train" or "test".')

    save_dir = os.path.join(FTR_DIR, name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # stringify columns because feather can only accept string column names.
    df.columns = list(map(str, df.columns))
    fname = f'{name}_{train_or_test}.{fmt}'
    fpath = os.path.join(save_dir, fname)
    save_data(df, fpath)


def save_features_meta(data, name):
    """
    Save features meta data (dict).
    """
    fpath = os.path.join(FTR_DIR, name, 'meta.json')
    save_dict(data, fpath)


def find_features_meta(name):
    """
    Find features meta data (dict). If the meta data file is not found, return None.
    """
    fpath = os.path.join(FTR_DIR, name, 'meta.json')
    if not os.path.exists(fpath):
        return
    return read_json(fpath)


########################################
# common.py
########################################


def prefix_list(lst, prefix, sep='_'):
    """
    Prefix a list of strings.

    Examples
    --------
    >>> prefix_list(['a', 'b'], 'x')
    ['x_a', 'x_b']

    >>> prefix_list(['a', 'b'], 'x', '|')
    ['x|a', 'x|b']
    """
    return [prefix + sep + x for x in lst]


def suffix_list(lst, suffix, sep='_'):
    """
    Suffix a list of strings.

    Examples
    --------
    >>> prefix_list(['a', 'b'], 'x')
    ['x_a', 'x_b']

    >>> prefix_list(['a', 'b'], 'x', '|')
    ['x|a', 'x|b']
    """
    return [x + sep + suffix for x in lst]


def prefix_dict_keys(dct, prefix, sep='_'):
    """
    Prefix dict keys.

    Examples
    --------
    >>> prefix_dict_keys({'a': 0}, 'x')
    {'x_a': 0}

    >>> prefix_dict_keys({'a': 0}, 'x', '|')
    {'x|a': 0}

    """
    return {prefix + sep + k: v for k, v in dct.items()}


def suffix_dict_keys(dct, suffix, sep='_'):
    """
    Suffix dict keys.

    Examples
    --------
    >>> suffix_dict_keys({'a': 0}, 'x')
    {'a_x': 0}

    >>> suffix_dict_keys({'a': 0}, 'x', '|')
    {'a|x': 0}

    """
    return {k + sep + suffix: v for k, v in dct.items()}


def with_name(func, name):
    """
    Name a function.

    Examples
    --------
    >>> func = with_name(lambda: print('test'), 'test')
    >>> func.__name__
    'test'

    """
    func.__name__ = name
    return func


def get_ext(fpath):
    """
    Get extension of a file.

    Parameters
    ----------
    fpath : str
        file path

    Returns
    -------
    str
        extension

    Examples
    --------
    >>> get_ext('test.txt')
    '.txt'

    >>> get_ext('dir/test.txt')
    '.txt'

    """
    return os.path.splitext(fpath)[1]


def remove_ext(fpath):
    """
    Remove extension of a file.

    Examples
    --------
    >>> remove_ext('test.txt')
    'test'

    >>> remove_ext('dir/test.txt')
    'dir/test'

    """
    return os.path.splitext(fpath)[0]


def replace_ext(fpath, ext):
    """
    Replace file extension.

    Examples
    --------
    >>> replace_ext('test.txt', '.py')
    'test.py'

    >>> replace_ext('dir/test.txt', '.py')
    'dir/test.py'

    """
    return os.path.splitext(fpath)[0] + ext


def remove_dir_ext(fpath):
    """
    Remove the directory component and extension from the specified path.

    Examples
    --------
    >>> remove_dir_ext('test.txt')
    'test'

    >>> remove_dir_ext('dir/test.txt')
    'test'

    """
    return os.path.splitext(os.path.basename(fpath))[0]


def print_divider(text):
    print('\n---------- {} ----------\n'.format(text))


########################################
# sampling.py
########################################


def hash_mod(x, n_splits):
    return (crc32(x.encode('utf-8')) & 0xffffffff) % n_splits


def hash_mod_sample(df, col, n_splits, keep=[0]):
    hm = df[col].map(lambda x: hash_mod(x, n_splits))
    return df[hm.isin(keep)]


########################################
# array.py
########################################


def div_by_sum(x):
    """
    Divide an array by its sum.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4])
    >>> div_by_sum(arr)
    array([0.1, 0.2, 0.3, 0.4])

    """
    return x / x.sum()


def shift_array(arr, num, fill_value=np.nan):
    """
    Shift a numpy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to shift.
    num : int
        Number of periods to shift. can be positive or negative.
    fill_value : object optional
        Scalar value to use for newly introduced missing values.

    Examples
    --------
    1D array
    >>> arr = np.array([1, 2, 3])
    >>> shift_array(arr, 1)
    array([nan, 1., 2.])

    2D array
    >>> arr = np.array(
    ... [[1, 2, 3],
    ...  [4, 5, 6],
    ...  [7, 8, 9]]
    ... )
    >>> shift_array(arr, 1)
    array([[nan, nan, nan],
           [ 1., 2., 3.],
           [ 4., 5., 6.]])

    References
    ----------
    - https://stackoverflow.com/a/42642326

    """
    shifted = np.empty_like(arr.astype(np.float64))
    if num > 0:
        shifted[:num] = fill_value
        shifted[num:] = arr[:-num]
    elif num < 0:
        shifted[num:] = fill_value
        shifted[:num] = arr[-num:]
    else:
        shifted[:] = arr
    return shifted