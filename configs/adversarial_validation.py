config = {
  # note: lgb.train(params, ..., **fit_params)
  'params': {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'is_unbalance': True,
    'random_state': 42,
  },

  'fit': {
      'num_boost_round': 10000,
      'verbose_eval': 20,
      'early_stopping_rounds': 20,
      'categorical_feature': ['title']  # important
  },

  'cv': {
    'class': 'GroupKFold',
    'params': {
      'n_splits': 5
    },
  },

  'seeds': [42],

  'features': [
    'game_session_stats',

  ]
}
