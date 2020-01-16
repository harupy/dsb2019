config = {
  # note: lgb.train(params, ..., **fit_params)
  'params': {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'subsample': 0.75,
    'subsample_freq': 1,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'max_depth': 15,
    'lambda_l1': 1,
    'lambda_l2': 1,
    'random_state': 42,
  },

  'fit': {
      'num_boost_round': 10000,
      'verbose_eval': 100,
      'early_stopping_rounds': 100,
      'categorical_feature': ['title']  # important
  },

  'cv': {
    'type': 'GroupKFold',
    'params': {
      'n_splits': 5
    },
  },

  'seeds': [0, 1, 2, 3, 4],

  'features': [
    # can be str or dict.
    # 'title',
    # {'game_session_cumsum': [
    #   'title',
    #   'event_code',
    #   'title_event_code',
    #   'type',
    # ]},
    'game_session_stats',
    # 'media_type_count',
    # '4020_attempts_stats_cum',
    # 'game_time_stats',
    # 'assessment_stats_with_invalid_sessions'
    # 'assessment_attempts_stats_cum',
    # 'assessment_attempts_before',
    # 'misses',
    # 'consecutive_assessments',
    # 'unfinished_assessments',
  ]
}
