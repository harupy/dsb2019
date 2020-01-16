import os
import gc
import argparse
import re
import tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import lightgbm as lgb


from utils.common import remove_dir_ext, prefix_list, print_divider
from utils.io import (read_config,
                      read_from_clean,
                      read_from_raw,
                      read_features,
                      save_features,
                      find_features_meta)
from utils.dataframe import find_constant_columns, apply_funcs
from utils.plotting import (plot_importance,
                            plot_label_share,
                            plot_confusion_matrix,
                            plot_eval_results,
                            plot_tree)
from features.funcs import find_highly_correlated_features, adjust_distribution
from utils.modeling import (get_cv,
                            average_feature_importance)
from utils.config_dict import ConfigDict


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-c', '--config', required=True, help='Config file path')
    return parser.parse_args()


def train_cv(config, X, y, inst_ids, cv):
    oof_proba = np.zeros(len(X))
    models = []
    eval_results = []
    num_seeds = len(config.seeds)

    for seed_idx, seed in enumerate(config.seeds):
        config.params.update({'random_state': seed})
        for fold_idx, (idx_trn, idx_val) in enumerate(cv.split(X, y, inst_ids)):
            print_divider(f'Seed: {seed_idx} / Fold: {fold_idx}')
            X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
            y_trn, y_val = y[idx_trn], y[idx_val]

            trn_set = lgb.Dataset(X_trn, y_trn)
            val_set = lgb.Dataset(X_val, y_val)

            eval_result = {}
            model = lgb.train(config.params, trn_set,
                              valid_sets=[trn_set, val_set],
                              valid_names=['train', 'valid'],
                              callbacks=[lgb.record_evaluation(eval_result)],
                              **config.fit)
            oof_proba[idx_val] += model.predict(X_val) / num_seeds
            models.append(model)
            eval_results.append(eval_result)

    return models, eval_results, oof_proba


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
    # labels = read_from_clean('train_labels_pseudo.ftr')
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

    # remove constant columns.
    constant_columns = find_constant_columns(train)
    train = train.drop(constant_columns, axis=1)
    test = test.drop(constant_columns, axis=1)

    # remove highly correlated features.
    to_remove = find_highly_correlated_features(train)
    train = train.drop(to_remove, axis=1)
    test = test.drop(to_remove, axis=1)

    # adjust distribution between train and test.
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
    train = pd.concat([train.sample(frac=0.5).assign(target=0), test.assign(target=1)],
                      axis=0, ignore_index=True)
    inst_ids_train = train['installation_id']  # keep installation_id for group k fold
    X_train = train.drop(['installation_id', 'game_session', 'accuracy_group', 'target'], axis=1)
    y_train = train['target']

    del train, test

    # prepare cv generator
    cv = get_cv(config)

    models, eval_results, oof_proba = train_cv(config, X_train, y_train, inst_ids_train, cv)
    oof_pred = (oof_proba > 0.5).astype(np.int8)

    # feature importance
    feature_names = np.array(models[0].feature_name())
    imp_split = average_feature_importance(models, 'split')
    imp_gain = average_feature_importance(models, 'gain')

    import mlflow

    def log_figure(fig, fname):
        """
        Log a matplotlib figure in the artifact store.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = tempfile.mkdtemp()
            fpath = os.path.join(tmpdir, fname)
            fig.savefig(fpath)
            mlflow.log_artifact(fpath)

    # create mlflow experiment using the config name if not exists.
    if mlflow.get_experiment_by_name(config_name) is None:
        mlflow.create_experiment(config_name)
    mlflow.set_experiment(config_name)

    with mlflow.start_run():
        mlflow.log_artifact(args.config)

        # log plots
        cm = confusion_matrix(y_train, oof_pred)
        log_figure(plot_confusion_matrix(cm), 'confusion_matrix.png')
        log_figure(plot_eval_results(eval_results), 'eval_history.png')
        log_figure(plot_label_share(y_train), 'train_label_share.png')
        log_figure(plot_label_share(oof_pred), 'pred_label_share.png')
        log_figure(plot_tree(models[0]), 'tree.png')

        for imp, imp_type in zip([imp_split, imp_gain], ['split', 'gain']):
            fig = plot_importance(feature_names, imp, imp_type, 30)
            log_figure(fig, f'feature_importance_{imp_type}.png')


if __name__ == '__main__':
    main()
