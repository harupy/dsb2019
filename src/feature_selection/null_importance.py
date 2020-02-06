import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import mlflow

from utils.datasets import breast_cancer
from utils.mlflow import set_experiment, log_figure
from utils.plotting import plot_feature_importance

sns.set()


def get_feature_importance(X, y, shuffle=False, random_state=None):
    train_set = lgb.Dataset(X, y.sample(frac=1.0) if shuffle else y)
    params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 127,
        'max_depth': 8,
        'bagging_freq': 1,
        'random_state': random_state,
    }

    model = lgb.train(params, train_set, num_boost_round=200)
    auc = roc_auc_score(y, model.predict(X))

    return pd.DataFrame({
        'feature_name': X.columns.tolist(),
        'importance_split': model.feature_importance(importance_type='split'),
        'importance_gain': model.feature_importance(importance_type='gain'),
        'train_auc': [auc for _ in range(len(X.columns))]
    }).sort_values('importance_gain', ascending=False)


def null_importance(X, y, repeat, random_state=None):
    null_imp = pd.DataFrame()
    for i in range(repeat):
        imp = get_feature_importance(X, y, shuffle=True, random_state=random_state)
        null_imp = pd.concat([null_imp, imp])
    return null_imp.reset_index(drop=True)


def inspect_columns(X, y, repeat=100, thresh=0.8, random_state=None):
    actual_imp = get_feature_importance(X, y, shuffle=False, random_state=random_state)
    null_imp = null_importance(X, y, repeat, random_state=random_state)
    useful = []
    useless = []
    for feature_name in actual_imp['feature_name']:
        query = f'feature_name == "{feature_name}"'
        actual_value = actual_imp.query(query)['importance_gain'].values
        null_value = null_imp.query(query)['importance_gain'].values
        percentage = (null_value < actual_value).mean()
        if percentage >= thresh:
            useful.append(feature_name)
        else:
            useless.append(feature_name)

    return useful, useless, actual_imp, null_imp


def plot_dist(actual_imp, null_imp, feature_name):
    query = f'feature_name == "{feature_name}"'
    actual_imp = actual_imp.query(query)['importance_gain'].mean()
    null_imp = null_imp.query(query)['importance_gain']

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    freq = ax.hist(null_imp, label="Null Importance")
    ax.vlines(x=actual_imp, ymin=0, ymax=np.max(freq[0]),
              color='r', linewidth=10, label='Actual Importance')
    ax.legend(loc='upper right')
    ax.set_title(f'Importance of {feature_name}')
    ax.set_xlabel(f'Null Importance Distribution for {feature_name}')
    ax.set_ylabel("Importance")
    return fig


def main():
    X, y = breast_cancer()
    useful, useless, actual_imp, null_imp = inspect_columns(X, y, random_state=42)

    set_experiment('null-importance')

    with mlflow.start_run():
        for feature_name in useful:
            fig = plot_dist(actual_imp, null_imp, feature_name)
            log_figure(fig, f'useful/{feature_name}.png')

        for feature_name in useless[:10]:
            fig = plot_dist(actual_imp, null_imp, feature_name)
            log_figure(fig, f'useless/{feature_name}.png')

        for imp_type in ['split', 'gain']:
            fig = plot_feature_importance(actual_imp['feature_name'],
                                          actual_imp[f'importance_{imp_type}'],
                                          imp_type)
            log_figure(fig, f'feature_importance_{imp_type}.png')


if __name__ == '__main__':
    main()
