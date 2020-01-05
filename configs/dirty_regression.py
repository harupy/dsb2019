config = {
  # note: lgb.train(params, ..., **fit_params)
  'params': {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'subsample': 0.75,
    'subsample_freq': 1,
    'learning_rate': 0.04,
    'feature_fraction': 0.9,
    'max_depth': -1,
    'lambda_l1': 1,
    'lambda_l2': 1,
    'verbose': -1,
    'random_state': 42,
  },


  'fit_params': {
      'num_boost_round': 10000,
      'verbose_eval': 100,
      'early_stopping_rounds': 100,
      'categorical_feature': ['title']
  },

  'fold': {
    'class': 'GroupKFold',
    'params': {
      'n_splits': 5
    },
  },

  'features': [
    'title',
    'game_session_stats',
    '4020_attempts_stats_cum',
  ]
}
