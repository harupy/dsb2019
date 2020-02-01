import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils.common import flatten_dict
from utils.datasets import breast_cancer
from utils.plotting import plot_feature_importance
from utils.mlflow import set_experiment, log_dict, log_figure


def adversarial_validation(X_train, X_test, drop=None):
    """
    Perform adversarial validation.
    """
    # prepare train data.
    drop = [] if drop is None else drop
    target_col = 'target'
    X = pd.concat([
        X_train.drop(drop, axis=1),
        X_test.drop(drop, axis=1),
    ], ignore_index=True)
    y = pd.Series(np.hstack([np.zeros(len(X_train)), np.ones(len(X_test))]),
                  name=target_col)

    # train model.
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': 42,
    }

    train_params = {
        'num_boost_round': 1000,
        'early_stopping_rounds': 50,
    }

    fold_params = {
        'n_splits': 5,
        'shuffle': True,
        'random_state': 42,
    }

    fold = StratifiedKFold(**fold_params)

    feature_names = X.columns.values
    importance = np.zeros((X.shape[1], fold.get_n_splits()))
    importance_type = 'gain'

    for idx_fold, (idx_trn, idx_val) in enumerate(fold.split(X, y)):
        X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
        y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]

        trn_set = lgb.Dataset(X_trn, y_trn)
        val_set = lgb.Dataset(X_val, y_val)
        model = lgb.train(params, trn_set, **train_params,
                          valid_sets=[trn_set, val_set],
                          valid_names=['train', 'valid'])

        imp = model.feature_importance(importance_type=importance_type)
        importance[:, idx_fold] = (imp / imp.sum())

    # sort feature importances
    importance_std = np.std(importance, axis=1)
    importance = np.mean(importance, axis=1)
    indices = np.argsort(importance)[::-1]
    feature_names = feature_names[indices]
    importance = importance[indices]
    importance_std = importance_std[indices]

    expr_name = 'adversarial-validation'
    set_experiment(expr_name)

    with mlflow.start_run():
        mlflow.log_params(flatten_dict({
            'params': params,
            'train': train_params,
            'fold': fold_params,
        }))
        mlflow.log_metrics(flatten_dict(model.best_score, sep='-'))
        fig = plot_feature_importance(feature_names, importance, importance_type,
                                      std=importance_std)
        log_figure(fig, f'feature_importance_{importance_type}.png')
        log_dict({
            'drop': drop,
            'feature_importance': list(zip(feature_names.tolist(), importance.tolist()))
        }, 'result.json')


def main():
    X, y = breast_cancer()
    X_train, X_test = train_test_split(X, random_state=42, shuffle=True)
    adversarial_validation(X_train, X_test)


if __name__ == '__main__':
    main()
