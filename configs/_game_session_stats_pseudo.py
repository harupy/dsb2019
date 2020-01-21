SEEDS = list(range(5))

config = {
  'lightgbm': {
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    'params': {
      'boosting_type': 'gbdt',
      'objective': 'regression',
      'metric': 'rmse',
      'learning_rate': 0.01,
      'subsample': 0.75,  # sample rows
      'colsample_bytree': 0.9,  # sample columns
      'subsample_freq': 1,
      'max_depth': 15,
      'reg_alpha': 1,
      'reg_lambda': 1,
      'random_state': SEEDS[0],
    },
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html#lightgbm.train
    'fit': {
        'num_boost_round': 10000,
        'verbose_eval': 100,
        'early_stopping_rounds': 100,
        'categorical_feature': ['title']  # important
    },
    'seeds': SEEDS,
  },

  'xgboost': {
    # https://xgboost.readthedocs.io/en/latest/parameter.html#xgboost-parameters
    'params': {
      'objective': 'reg:squarederror',
      'eval_metric': 'rmse',
      'learning_rate': 0.01,
      'subsample': 0.75,  # sample rows
      'colsample_bytree': 0.9,  # sample columns
      'max_depth': 10,
      'reg_alpha': 1,
      'reg_lambda': 1,
      'seed': SEEDS[0],
    },
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train
    'fit': {
        'num_boost_round': 10000,
        'verbose_eval': 100,
        'early_stopping_rounds': 100,
    },
    'seeds': SEEDS,
  },

  'cv': {
    'type': 'GroupKFold',
    'params': {
      'n_splits': 5
    },
  },

  'features': [
    # can be str or dict.
    # 'title',
    # {'game_session_cumsum': [
    #   'title',
    #   'event_code',
    #   'title_event_code',
    #   'type',
    # ]},
    'game_session_stats_pseudo',
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
