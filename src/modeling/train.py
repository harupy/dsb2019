import os
import shutil
import gc
import argparse
import tempfile
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import confusion_matrix
import mlflow

from utils.common import remove_ext, print_divider
from utils.io import read_config, read_from_raw, read_features
from utils.array import div_by_sum
from utils.plotting import plot_feature_importance, plot_label_share, plot_confusion_matrix
from utils.metrics import calc_qwk, OptimizedRounder


def parse_args():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-c', '--config', required=True, help='Config file path')
    return parser.parse_args()


def is_booster(model):
    """
    Return True if the given model is an instance of `lightgbm.Booster`.

    Parameters
    ----------
    model : any type
        model to check.

    Examples
    --------
    >>> model = lgb.LGBMClassifier().fit([[0], [0]], [0, 0])
    >>> is_booster(model.booster_)
    True

    >>> is_booster(model)
    False

    """
    return isinstance(model, lgb.Booster)


def cross_validation(config, X, y, index_sets):
    models = []
    for fold_idx, (idx_trn, idx_val) in enumerate(index_sets):
        print_divider(f'Fold: {fold_idx}')
        X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
        y_trn, y_val = y[idx_trn], y[idx_val]

        dtrn = lgb.Dataset(X_trn, y_trn)
        dval = lgb.Dataset(X_val, y_val)

        model = lgb.train(config['params'], dtrn,
                          valid_sets=[dtrn, dval],
                          **config['fit_params'])
        models.append(model)
    return models


def get_fold(config):
    classes = {
        KFold.__name__: KFold,
        StratifiedKFold.__name__: StratifiedKFold,
        GroupKFold.__name__: GroupKFold,
    }
    return classes[config['fold']['class']](**config['fold']['params'])


def feature_importance_avg(models, importance_type, normalize=True):
    imps = []
    for model in models:
        imp = model.feature_importance(importance_type=importance_type)
        imps.append(div_by_sum(imp) if normalize else imp)

    return np.mean(imps, axis=0)


def oof_predict(models, X, index_sets):
    oof_pred = np.zeros(len(X))
    for fold_idx, (_, idx_val) in enumerate(index_sets):
        X_val = X.iloc[idx_val]
        model = models[fold_idx]
        if is_booster(model):
            oof_pred[idx_val] = model.predict(X_val)
        else:
            oof_pred[idx_val] = model.predict_proba(X_val)[:, 1]

    return oof_pred


def predict_avg(models, X):
    return np.mean([model.predict(X) for model in models], axis=0)


def log_figure(fig, fname):
    """
    Log a matplotlib figure as an artifact.
    """
    tmpdir = tempfile.mkdtemp()
    fpath = os.path.join(tmpdir, fname)
    fig.savefig(fpath, dpi=200)
    mlflow.log_artifact(fpath)
    shutil.rmtree(tmpdir)  # clean up tmpdir


def main():
    args = parse_args()
    labels = read_from_raw('train_labels.csv')
    train = labels.copy()
    train = train[['installation_id', 'game_session', 'accuracy_group']]

    sbm = read_from_raw('sample_submission.csv')
    test = sbm.copy()
    test = test[['installation_id']]

    config = read_config(args.config)

    # merge features
    for feature_name in config['features']:
        train_ft, test_ft = read_features(feature_name)

        # test
        train = pd.merge(train, train_ft, how='left', on=['installation_id', 'game_session'])

        # number of rows should not change before and after merge.
        assert len(train) == len(labels)

        # test
        test_ft = test_ft.drop('game_session', axis=1).groupby('installation_id').last().reset_index()
        test = pd.merge(test, test_ft, how='left', on='installation_id')
        assert len(test) == len(sbm)

        del train_ft, test_ft
        gc.collect()

    # prepare train data
    inst_ids_train = train['installation_id']  # keep installation_id for group k fold
    X_train = train.drop(['installation_id', 'game_session', 'accuracy_group'], axis=1)
    y_train = train['accuracy_group']

    # prepare test data
    X_test = test.drop('installation_id', axis=1)

    # assert train and test data have the same columns.
    assert X_train.columns.tolist() == X_test.columns.tolist()

    # prepare fold
    fold = get_fold(config)
    index_sets = list(fold.split(X_train, y_train, inst_ids_train))

    models = cross_validation(config, X_train, y_train, index_sets)
    oof_pred = oof_predict(models, X_train, index_sets)

    # feature importance
    feature_names = np.array(models[0].feature_name())
    imp_split = feature_importance_avg(models, 'split')
    imp_gain = feature_importance_avg(models, 'gain')

    # optimize round boundaries
    opt = OptimizedRounder()
    opt.fit(y_train, oof_pred)
    coef = opt.coefficients()
    oof_pred_round = opt.round(oof_pred, coef)
    qwk = calc_qwk(y_train, oof_pred_round)
    print('QWK:', qwk)

    # predict
    proba = predict_avg(models, X_test)
    pred = opt.round(proba, coef)

    # assert pred does not contain invalid values
    assert (~np.isnan(pred)).all()
    assert np.isin(pred, [0, 1, 2, 3]).all()

    # make submission file
    sbm['accuracy_group'] = pred
    sbm.to_csv('submission.csv', index=False)

    # log results with mlflow
    expr_name = remove_ext(os.path.basename(args.config))

    # create mlflow experiment if not exists
    if mlflow.get_experiment_by_name(expr_name) is None:
        mlflow.create_experiment(expr_name)

    mlflow.set_experiment(expr_name)
    with mlflow.start_run():
        # log config file
        mlflow.log_artifact(args.config)

        mlflow.log_metrics({'qwk': qwk})

        # log plots
        cm = confusion_matrix(y_train, oof_pred_round)
        log_figure(plot_confusion_matrix(cm), 'confusion_matrix.png')

        # predicted label share
        log_figure(plot_label_share(pred), 'pred_label_share.png')

        for imp, imp_type in zip([imp_split, imp_gain], ['split', 'gain']):
            fig = plot_feature_importance(feature_names, imp, imp_type)
            log_figure(fig, f'feature_importance_{imp_type}.png')


if __name__ == '__main__':
    main()
