import os
import gc
import argparse
import re
import tempfile
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from utils.common import remove_dir_ext, prefix_list, print_divider
from utils.io import (read_config,
                      read_from_clean,
                      read_from_raw,
                      read_features,
                      save_features,
                      find_feature_meta)
from utils.dataframe import (assert_columns_equal,
                             constant_columns,
                             all_zero_columns,
                             all_null_columns,
                             highly_correlated_columns,
                             apply_funcs)
from utils.plotting import (plot_importance,
                            plot_label_share,
                            plot_confusion_matrix,
                            plot_eval_results)
from utils.metrics import qwk, digitize, OptimizedRounder
from utils.kaggle import on_kaggle
from utils.modeling import (get_cv,
                            average_feature_importance,
                            predict_average,
                            predict_median)
from utils.config_dict import ConfigDict

from features.funcs import adjust_distribution
from modeling.models import LgbModel, XgbModel


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-c', '--config', required=True, help='Config file path')
    return parser.parse_args()


def train_cv(config, X, y, inst_ids, cv):
    """
    Perform cross-validation with given configuration.
    """
    models = []
    eval_results = []
    num_seeds = len(config.seeds)
    oof_pred = np.zeros((len(X), num_seeds)) * np.nan

    for seed_idx, seed in enumerate(config.seeds):
        config.params.update({'random_state': seed})
        # X, y, inst_ids = shuffle(X, y, inst_ids, random_state=seed)

        for fold_idx, (idx_trn, idx_val) in enumerate(cv.split(X, y, inst_ids)):
            print_divider(f'Seed: {seed_idx} / Fold: {fold_idx}')
            X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
            y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]

            # For truncation
            inst_ids_trn = inst_ids.iloc[idx_trn]
            inst_ids_val = inst_ids.iloc[idx_val]

            assert len(set(inst_ids_trn).intersection(set(inst_ids_val))) == 0
            assert inst_ids_trn.index.equals(X_trn.index)
            assert inst_ids_val.index.equals(X_val.index)

            # Some users in the train set have multiple assessments.
            # This section samples one assessment from each user.
            # This is called "truncation" on the discussion.
            # It seems there are multiple ways to perform this.
            # - Truncate only the train set.
            # - Truncate only the validation set.
            # - Truncate both the train and validation sets.
            # Ref.: https://www.kaggle.com/poteman/sampling-train-data-and-use-prediction-as-feature  # noqa
            mask_trn = random_truncate(inst_ids_trn, seed)
            assert inst_ids_trn[mask_trn].is_unique
            X_trn = X_trn.loc[mask_trn]
            y_trn = y_trn.loc[mask_trn]

            mask_val = random_truncate(inst_ids_val, seed)
            X_val = X_val.loc[mask_val.index]
            y_val = y_val.loc[mask_val.index]

            trn_set = lgb.Dataset(X_trn, y_trn)
            val_set = lgb.Dataset(X_val, y_val)

            eval_result = {}
            model = lgb.train(config.params, trn_set,
                              valid_sets=[trn_set, val_set],
                              valid_names=['train', 'valid'],
                              callbacks=[lgb.record_evaluation(eval_result)],
                              **config.fit)

            # oof_pred[y_val.index.values] += model.predict(X_val) / num_seeds
            oof_pred[y_val.index.values, seed_idx] = model.predict(X_val)
            models.append(model)
            eval_results.append(eval_result)

    print(oof_pred)
    return models, eval_results, np.nanmedian(oof_pred, axis=1)


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


def random_truncate(inst_ids, seed):
    """
    Create a mask that samples one assessment from each installation_id.
    """
    return shuffle(inst_ids, random_state=seed).drop_duplicates(keep='first').index


def percentile_boundaries(accuracy_group, pred):
    """
    Compute round boundaries from the percentile of training accuracy_group.

    Note
    ----
    This function
    """
    freq_norm = accuracy_group.value_counts(normalize=True).sort_index()
    acc = 0
    bounds = []
    for key, val in list(freq_norm.items())[:-1]:  # ignore the last item.
        acc += val
        bounds.append(np.percentile(pred, 100 * acc))

    return bounds


def apply_meta(df, meta):
    """
    Apply operations specified in meta data to given dataframe.
    """
    funcs = []

    if 'use_cols' in meta:
        funcs.append(lambda df_: df_[meta['use_cols']])

    if 'drop_cols' in meta:
        funcs.append(lambda df_: df_.drop(meta['drop_cols'], axis=1))

    return apply_funcs(df, funcs)


def main():
    args = parse_args()
    labels = read_from_raw('train_labels.csv')
    train = labels.copy()
    train = train[['installation_id', 'game_session', 'accuracy_group']]

    sbm = read_from_raw('sample_submission.csv')
    test = sbm.copy()
    test = test[['installation_id']]

    config_dict = read_config(args.config)
    config = ConfigDict(config_dict)

    # merge features
    for feature_name in flatten_features(config.features):
        train_ft, test_ft = read_features(feature_name)
        meta = find_feature_meta(feature_name)

        # Some features don't have meta data.
        if meta:
            train_ft = apply_meta(train_ft, meta)
            test_ft = apply_meta(test_ft, meta)

        # train
        train = pd.merge(train, train_ft, how='left', on=['installation_id', 'game_session'])

        # number of rows should not change before and after merge.
        assert len(train) == len(labels)

        # test
        test_ft = (
            test_ft.drop('game_session', axis=1)
            .groupby('installation_id').last()
            .reset_index()
        )
        test = pd.merge(test, test_ft, how='left', on='installation_id')
        assert len(test) == len(sbm)

        del train_ft, test_ft
        gc.collect()

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    to_drop = []
    funcs = [all_zero_columns, all_null_columns, constant_columns]
    for func in funcs:
        to_drop += func(train)
        to_drop += func(test)

    # remove duplicates.
    to_drop = list(set(to_drop))

    train = train.drop(to_drop, axis=1)
    test = test.drop(to_drop, axis=1)

    # # remove highly correlated features.
    to_drop = highly_correlated_columns(train, verbose=True)
    train = train.drop(to_drop, axis=1)
    test = test.drop(to_drop, axis=1)

    # adjust distribution between train and test.
    test, to_drop = adjust_distribution(train, test)
    train = train.drop(to_drop, axis=1)
    test = test.drop(to_drop, axis=1)

    # Saving data on Kaggle might cause IOError.
    # if not on_kaggle():
    #     save_features(train, 'final', 'train')
    #     save_features(test, 'final', 'test')

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

    # prepare cross-validation splitter.
    cv = get_cv(config)

    # models, eval_results, oof_pred = train_cv(config, X_train, y_train, inst_ids_train, cv)
    lgb_model = LgbModel()
    xgb_model = XgbModel()

    # perform cross-validation.
    oof_pred = 0.8 * lgb_model.cv(X_train, y_train, inst_ids_train, cv, config.lightgbm)
    oof_pred += 0.2 * xgb_model.cv(X_train, y_train, inst_ids_train, cv, config.xgboost)

    opt = OptimizedRounder()
    opt.fit(y_train, oof_pred)
    oof_pred = opt.predict(oof_pred)
    QWK = qwk(y_train, oof_pred)

    # Note that boundaries might be unsorted.
    bounds = np.sort(opt.boundaries).tolist()

    # Some top public kernels use this method, but this is dangerous because
    # the label distribution of the private test set might be different from the train set.
    # bounds = percentile_boundaries(y_train, test_pred)
    # oof_round = digitize(oof_preds, bounds)
    # QWK = qwk(y_train, oof_round)
    # print('boundaries:', bounds)
    # print('QWK:', QWK)

    print('boundaries:', bounds)
    print('QWK:', QWK)

    test_pred = lgb_model.predict_median(X_test)
    pred_sbm = digitize(test_pred, bounds)

    # post process (not sure if this works.)
    cum_best = read_features('cum_best_accuracy_group', 'test')  # only use test data.
    cum_best = cum_best[['installation_id', 'cum_best_accuracy_group']]
    cum_best = cum_best[cum_best['cum_best_accuracy_group'] == 3.0]  # use only 3.0
    cum_best = cum_best.groupby('installation_id', sort=False).last().reset_index()
    cum_best = pd.merge(sbm, cum_best, on=['installation_id'], how='left')
    cum_best = cum_best['cum_best_accuracy_group'].values
    print('Nan count:', np.isnan(cum_best).sum())

    pred_sbm = np.where(np.isnan(cum_best), pred_sbm, cum_best)
    pred_sbm = pred_sbm.astype(np.int8)

    # assert pred does not contain invalid values
    assert (~np.isnan(pred_sbm)).all()
    assert np.isin(pred_sbm, [0, 1, 2, 3]).all()
    assert len(pred_sbm) == len(sbm)

    # make a submission file.
    sbm['accuracy_group'] = pred_sbm
    sbm.to_csv('submission.csv', index=False)

    # assert the submission file exists.
    assert os.path.exists('submission.csv')

    # on kaggle kernel, ignore mlflow logging.
    if on_kaggle():
        return

    # On local, logging the parameters and metrics with MLflow.
    import mlflow

    def log_figure(fig, fname):
        """
        Log a matplotlib figure in the artifact store.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = tempfile.mkdtemp()
            fpath = os.path.join(tmpdir, fname)
            fig.savefig(fpath, dpi=200)
            mlflow.log_artifact(fpath)

    # create MLflow experiment using the config name if not exists.
    config_name = remove_dir_ext(args.config)
    if mlflow.get_experiment_by_name(config_name) is None:
        mlflow.create_experiment(config_name)
    mlflow.set_experiment(config_name)

    with mlflow.start_run():
        mlflow.log_artifact(args.config)

        mlflow.log_params({'boundaries': bounds})
        mlflow.log_metrics({'qwk': QWK})

        # log plots
        cm = confusion_matrix(y_train, oof_pred)
        log_figure(plot_confusion_matrix(cm), 'confusion_matrix.png')
        log_figure(plot_eval_results(lgb_model.eval_results), 'eval_history.png')

        # log label share
        log_figure(plot_label_share(y_train), 'train_label_share.png')
        log_figure(plot_label_share(oof_pred), 'pred_label_share.png')

        # feature importance
        feature_names = lgb_model.feature_name()
        for imp_type in ['split', 'gain']:
            imp, imp_std = lgb_model.feature_importance_average(imp_type, return_std=True)
            fig = plot_importance(feature_names, imp, imp_type, imp_std, 30)
            log_figure(fig, f'feature_importance_{imp_type}.png')


if __name__ == '__main__':
    main()
