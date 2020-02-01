import pandas as pd
import lightgbm as lgb
import mlflow

from utils.plotting import plot_feature_importance
from utils.mlflow import log_figure, create_experiment_if_not_exists


def adversarial_validation(train, test, to_drop=None):
    """
    """

    params = {
        'objective': 'binary',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': 42,
    }

    # prepare train data.
    to_drop = [] if to_drop is None else to_drop
    train_set = lgb.Dataset(pd.concat([
        train.drop(to_drop, axis=1).assign(target=0),
        test.drop(to_drop, axis=1).assign(target=0),
    ]), ignore_index=True)

    # train model
    model = lgb.train(params, train_set, num_boost_round=100)

    imp_type = 'gain'
    importance = model.feature_importance(importance_type=imp_type)
    feature_names = model.feature_name()

    expr_name = 'adversarial-validation'
    if mlflow.get_experiment_by_name(expr_name) is None:
        mlflow.create_experiment(expr_name)
    mlflow.set_experiment(expr_name)

    fig = plot_feature_importance(feature_names, importance, imp_type)
    log_figure(fig, f'feature_importance_{imp_type}.png')
