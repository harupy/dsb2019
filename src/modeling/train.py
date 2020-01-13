import os
import shutil
import gc
import argparse
import re
import tempfile
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from utils.common import remove_dir_ext, prefix_list, print_divider
from utils.io import (read_config,
                      read_from_clean,
                      read_from_raw,
                      read_features,
                      save_features,
                      find_features_meta)
from utils.dataframe import assert_columns_equal, find_constant_columns, apply_funcs
from utils.array import div_by_sum
from utils.plotting import (plot_importance,
                            plot_label_share,
                            plot_confusion_matrix,
                            plot_eval_results)
from utils.metrics import qwk, digitize, OptimizedRounder
from features.funcs import find_highly_correlated_features, adjust_distribution


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-c', '--config', required=True, help='Config file path')
    return parser.parse_args()


def on_kaggle_kernel():
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def train_cv(config, X, y, inst_ids, cv):
    oof_pred = np.zeros(len(X))
    models = []
    eval_results = []
    num_seeds = len(config['seeds'])
    for seed_idx, seed in enumerate(config['seeds']):
        config['params'].update({'random_state': seed})
        for fold_idx, (idx_trn, idx_val) in enumerate(cv.split(X, y, inst_ids)):
            print_divider(f'Seed: {seed_idx} / Fold: {fold_idx}')
            X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
            y_trn, y_val = y[idx_trn], y[idx_val]

            # for truncation
            inst_ids_trn = inst_ids[idx_trn]
            inst_ids_val = inst_ids[idx_val]

            assert inst_ids_trn.index.equals(X_trn.index)
            assert inst_ids_val.index.equals(X_val.index)

            # some users in the train set have multiple assessments.
            # lines below sample one assessment from each user.
            mask_trn = random_truncation(inst_ids_trn, seed)
            assert inst_ids_trn[mask_trn].is_unique
            X_trn = X_trn.loc[mask_trn]
            y_trn = y_trn.loc[mask_trn]

            # mask_val = random_sample(inst_ids_val, seed)
            # X_val = X_val.loc[mask_val.index]
            # y_val = y_val.loc[mask_val.index]

            trn_set = lgb.Dataset(X_trn, y_trn)
            val_set = lgb.Dataset(X_val, y_val)

            eval_result = {}
            model = lgb.train(config['params'], trn_set,
                              valid_sets=[trn_set, val_set],
                              valid_names=['train', 'valid'],
                              callbacks=[lgb.record_evaluation(eval_result)],
                              **config['fit'])
            oof_pred[idx_val] += model.predict(X_val) / num_seeds
            models.append(model)
            eval_results.append(eval_result)
    return models, eval_results, oof_pred


def get_cv(config):
    """
    Return a cross-validator with the given configuration.
    """
    classes = {
        KFold.__name__: KFold,
        StratifiedKFold.__name__: StratifiedKFold,
        GroupKFold.__name__: GroupKFold,
    }
    return classes[config['cv']['class']](**config['cv']['params'])


def average_feature_importance(models, importance_type, normalize=True):
    imps = []
    for model in models:
        imp = model.feature_importance(importance_type=importance_type)
        imps.append(div_by_sum(imp) if normalize else imp)

    return np.mean(imps, axis=0)


def predict_average(models, X):
    return np.mean([model.predict(X) for model in models], axis=0)


def flatten_features(features):
    """
    >>> flatten_features(['a', 'b'])
    ['a', 'b']

    >>> flatten_features([{'a': ['b', 'c']}])
    ['a_b', 'a_c']

    >>> flatten_features(['a', {'b': ['c', 'd']}])
    ['a', 'b_c', 'b_d']

    >>> flatten_features([0])
    Traceback (most recent call last):
        ...
    TypeError: Invalid type: <class 'int'>.

    """
    result = []

    for feat in features:
        if isinstance(feat, str):
            result.append(feat)
        elif isinstance(feat, dict):
            for parent, children in feat.items():
                result.extend(prefix_list(children, parent))
        else:
            raise TypeError('Invalid type: {}.'.format(type(feat)))
    return result


def random_truncation(inst_ids, seed):
    """
    Return a mask for truncation.
    """
    return shuffle(inst_ids, random_state=seed).drop_duplicates(keep='first').index


def percentile_boundaries(accuracy_group, pred):
    freq_norm = accuracy_group.value_counts(normalize=True).sort_index()
    acc = 0
    bounds = []
    for key, val in list(freq_norm.items())[:-1]:  # ignore the last item.
        acc += val
        bounds.append(np.percentile(pred, 100 * acc))

    return bounds


def apply_meta(df, meta):
    funcs = []

    if 'use_cols' in meta:
        funcs.append(lambda df_: df_[meta['use_cols']])

    if 'drop_cols' in meta:
        funcs.append(lambda df_: df_.drop(meta['drop_cols'], axis=1))

    return apply_funcs(df, funcs)


def main():
    args = parse_args()
    # labels = read_from_raw('train_labels.csv')
    labels = read_from_clean('train_labels_pseudo.ftr')
    train = labels.copy()
    train = train[['installation_id', 'game_session', 'accuracy_group']]

    sbm = read_from_raw('sample_submission.csv')
    test = sbm.copy()
    test = test[['installation_id']]

    config = read_config(args.config)

    # merge features
    for feature_name in flatten_features(config['features']):
        train_ft, test_ft = read_features(feature_name)
        meta = find_features_meta(feature_name)

        if meta:
            train_ft = apply_meta(train_ft, meta)
            test_ft = apply_meta(test_ft, meta)

        # train
        train = pd.merge(train, train_ft, how='left', on=['installation_id', 'game_session'])

        # number of rows should not change before and after merge.
        assert len(train) == len(labels)

        # test
        test_ft = test_ft.drop('game_session', axis=1).groupby('installation_id').last().reset_index()
        test = pd.merge(test, test_ft, how='left', on='installation_id')
        assert len(test) == len(sbm)

        del train_ft, test_ft
        gc.collect()

    train = train.fillna(0)
    test = test.fillna(0)

    constant_columns = find_constant_columns(train)
    train = train.drop(constant_columns, axis=1)
    test = test.drop(constant_columns, axis=1)

    # remove highly correlated features
    to_remove = find_highly_correlated_features(train)
    train = train.drop(to_remove, axis=1)
    test = test.drop(to_remove, axis=1)

    # adjust distribution between train and test
    test, to_remove = adjust_distribution(train, test)
    train = train.drop(to_remove, axis=1)
    test = test.drop(to_remove, axis=1)

    config_name = remove_dir_ext(args.config)
    save_features(train, 'final', 'train')
    save_features(test, 'final', 'test')

    # replace non-alphanumeric characters with '_'
    # to prevent LightGBM from raising an error on invalid column names.
    non_alphanumeric = r'[^a-zA-Z\d]+'
    train.columns = [re.sub(non_alphanumeric, '_', c).strip('_') for c in train.columns]
    test.columns = [re.sub(non_alphanumeric, '_', c).strip('_') for c in test.columns]

    # prepare train and test data
    inst_ids_train = train['installation_id']  # keep installation_id for group k fold
    X_train = train.drop(['installation_id', 'game_session', 'accuracy_group'], axis=1)
    y_train = train['accuracy_group']
    X_test = test.drop('installation_id', axis=1)

    del train, test

    # assert train and test data have the same columns.
    assert_columns_equal(X_train, X_test)

    # prepare cv generator
    cv = get_cv(config)

    models, eval_results, oof_pred = train_cv(config, X_train, y_train, inst_ids_train, cv)

    # feature importance
    feature_names = np.array(models[0].feature_name())
    imp_split = average_feature_importance(models, 'split')
    imp_gain = average_feature_importance(models, 'gain')

    # optimize round boundaries
    opt = OptimizedRounder()
    opt.fit(y_train, oof_pred)
    oof_pred_round = opt.predict(oof_pred)
    QWK = qwk(y_train, oof_pred_round)
    bounds = opt.boundaries.tolist()
    print('boundaries (OptimizedRounder):', bounds)
    print('QWK:', QWK)

    avg_proba = predict_average(models, X_test)
    # pred = opt.predict(avg_proba)

    bounds = percentile_boundaries(y_train, avg_proba)
    pred = digitize(avg_proba, bounds)
    print('boundaries (percentile_boundaries):', bounds)

    # assert pred does not contain invalid values
    assert (~np.isnan(pred)).all(), 'NaN should not be included.'
    assert np.isin(pred, [0, 1, 2, 3]).all(), 'Predicted value must be one of [0, 1, 2, 3]'

    # make submission file
    sbm['accuracy_group'] = pred
    sbm.to_csv('submission.csv', index=False)
    assert os.path.exists('submission.csv')

    # on kaggle kernel, ignore mlflow logging.
    if on_kaggle_kernel():
        return

    import mlflow

    def log_figure(fig, fname):
        """
        Log a matplotlib figure in the artifact store.
        """
        tmpdir = tempfile.mkdtemp()
        fpath = os.path.join(tmpdir, fname)
        fig.savefig(fpath, dpi=200)
        mlflow.log_artifact(fpath)
        shutil.rmtree(tmpdir)  # clean up tmpdir

    # create mlflow experiment using the config name if not exists.
    if mlflow.get_experiment_by_name(config_name) is None:
        mlflow.create_experiment(config_name)
    mlflow.set_experiment(config_name)

    with mlflow.start_run():
        mlflow.log_artifact(args.config)

        mlflow.log_params({'boundaries': bounds})
        mlflow.log_metrics({'qwk': QWK})

        # log plots
        cm = confusion_matrix(y_train, oof_pred_round)
        log_figure(plot_confusion_matrix(cm), 'confusion_matrix.png')
        log_figure(plot_eval_results(eval_results), 'eval_history.png')

        # log label share
        log_figure(plot_label_share(y_train), 'train_label_share.png')
        log_figure(plot_label_share(pred), 'pred_label_share.png')

        for imp, imp_type in zip([imp_split, imp_gain], ['split', 'gain']):
            fig = plot_importance(feature_names, imp, imp_type, 30)
            log_figure(fig, f'feature_importance_{imp_type}.png')


if __name__ == '__main__':
    main()
