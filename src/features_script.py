########################################
# title
########################################

def title_main():
    train = read_from_clean('train_labels.ftr')
    test = read_from_clean('test.ftr')

    mapper = {title: idx for idx, title in enumerate(train['title'].unique())}

    cols = ['installation_id', 'game_session', 'title']
    title_train = train[cols]
    title_test = test.groupby('installation_id', sort=False).last().reset_index()[cols]

    title_train['title'] = title_train['title'].map(mapper)
    title_test['title'] = title_test['title'].map(mapper)

    name = remove_dir_ext('title')
    save_features(title_train, name, 'train')
    save_features(title_test, name, 'test')


if __name__ == '__main__':
    title_main()


########################################
# game_session_cumsum
########################################

def sum_game_session(df, cols_sum):
    return np.vstack([gdf[cols_sum].values.sum(axis=0)
                      for _, gdf in df.groupby('game_session', sort=False)])


def cumsum_game_session(df, cols_encode, encoder):
    categories = get_categories(encoder)
    cumsum = []
    for inst_idx, user_sample in tqdm(df.groupby('installation_id', sort=False)):
        # session_sums = sum_game_session(user_sample, cols_encode, encoder)
        user_sample = apply_one_hot_encoder(user_sample, encoder, cols_encode)
        session_sums = sum_game_session(user_sample, categories)

        # take cumulative sum and shift it.
        cumsum.append(shift_array(np.cumsum(session_sums, axis=0), 1))

    keys = (
        df[['installation_id', 'game_session']]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    cumsum = pd.DataFrame(np.vstack(cumsum), columns=get_categories(encoder))

    assert len(keys) == len(cumsum), 'the number of records must be equal.'
    return pd.concat([keys, cumsum], axis=1)


def game_session_cumsum_main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    train['title_event_code'] = train['title'] + '_' + train['event_code'].astype(str)
    test['title_event_code'] = test['title'] + '_' + test['event_code'].astype(str)

    cols_encode = ['title', 'event_code', 'title_event_code', 'event_id', 'type']
    encoder = build_one_hot_encoder(train, test, cols_encode)

    cumsum_train = cumsum_game_session(train, cols_encode, encoder)
    cumsum_test = cumsum_game_session(test, cols_encode, encoder)

    cumsum_train = cumsum_train.fillna(0)
    cumsum_test = cumsum_test.fillna(0)

    name = remove_dir_ext('game_session_cumsum')

    # save encoded features separately
    for col_idx, col in enumerate(cols_encode):
        save_cols = ['installation_id', 'game_session'] + encoder.categories_[col_idx].tolist()
        save_features(cumsum_train[save_cols], name + f'_{col}', 'train')
        save_features(cumsum_test[save_cols], name + f'_{col}', 'test')

    save_features(cumsum_train, name, 'train')
    save_features(cumsum_test, name, 'test')


if __name__ == '__main__':
    game_session_cumsum_main()


########################################
# 4020_attempts_stats_cum
########################################

def _4020_attempts_stats_cum_main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    is_asm_train = is_assessment(train)
    is_asm_test = is_assessment(test)

    is_4020_train = train['event_code'].eq(4020)
    is_4020_test = test['event_code'].eq(4020)

    # mask event_data
    train['event_data'] = train['event_data'].where(is_4020_train, '')
    test['event_data'] = test['event_data'].where(is_4020_test, '')

    train = assign_attempt_result(train)
    test = assign_attempt_result(test)

    name = remove_dir_ext('4020_attempts_stats_cum')
    train = train[is_asm_train & is_4020_train]
    test = test[is_asm_test & is_4020_test]

    #######################################################

    train = calc_attempt_stats(train)
    test = calc_attempt_stats(test)

    train = train.drop('accuracy_group', axis=1)
    test = test.drop('accuracy_group', axis=1)

    #######################################################

    funcs = {
        'cumsum': ['num_correct', 'num_incorrect', 'attempts'],
        'cummean': ['accuracy'],
    }
    train = cumulative_by_user(train, funcs)
    test = cumulative_by_user(test, funcs, is_test=True)

    train = train.fillna(0)
    test = test.fillna(0)

    exclude = ['installation_id', 'game_session']
    train = prefix_columns(train, '4020', exclude=exclude)
    test = prefix_columns(test, '4020', exclude=exclude)

    name = remove_dir_ext('4020_attempts_stats_cum')
    save_features(train, name, 'train')
    save_features(test, name, 'test')


if __name__ == '__main__':
    _4020_attempts_stats_cum_main()


########################################
# assessment_attempts_stats_cum
########################################

def assessment_attempts_stats_cum_main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')
    funcs = [
        filter_assessment_attempt,
        assign_attempt_result,
        calc_attempt_stats,
    ]

    train = apply_funcs(train, funcs)
    test = apply_funcs(test, funcs)

    #######################################################

    cum_funcs = {
        'cumsum': ['num_correct', 'num_incorrect', 'attempts'],
        'cummean': ['accuracy', 'accuracy_group'],
    }
    train = cumulative_by_user(train, cum_funcs)
    test = cumulative_by_user(test, cum_funcs, True)

    train['overall_accuracy'] = train['cumsum_num_correct'] / train['cumsum_attempts']
    test['overall_accuracy'] = test['cumsum_num_correct'] / test['cumsum_attempts']

    train = train.fillna(0)
    test = test.fillna(0)

    name = remove_dir_ext('assessment_attempts_stats_cum')
    save_features(train, name, 'train')
    save_features(test, name, 'test')


if __name__ == '__main__':
    assessment_attempts_stats_cum_main()


########################################
# misses
########################################

def extract_misses(df):
    """
    >>> df = pd.DataFrame({
    ...     'event_data': ['"misses":1', '"misses":0', ''],
    ... })
    >>> extract_misses(df)
    0    1.0
    1    0.0
    2    NaN
    Name: event_data, dtype: float64

    """
    pattern = r'"misses":(\d+)'
    return df['event_data'].str.extract(pattern, expand=False).astype(float)


def misses_main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    train = train.assign(misses=extract_misses)
    test = test.assign(misses=extract_misses)

    # The sum of an all-NA or empty Series is 0 by default.
    aggs = {'misses': lambda s: np.nan if s.isnull().all() else s.sum()}
    by = ['installation_id', 'game_session']
    train = train.groupby(by, sort=False).agg(aggs).reset_index()
    test = test.groupby(by, sort=False).agg(aggs).reset_index()

    funcs = {
        'cumsum': ['misses'],
        'cummean': ['misses'],
    }

    train = cumulative_by_user(train, funcs)
    test = cumulative_by_user(test, funcs)

    train = train.fillna(0)
    test = test.fillna(0)

    name = remove_dir_ext('misses')
    save_features(train, name, 'train')
    save_features(test, name, 'test')


if __name__ == '__main__':
    misses_main()


########################################
# consecutive_assessments
########################################

def assign_prev(df):
    return df.assign(
        type_prev=df['type'].shift(1),
        title_prev=df['title'].shift(1),
        is_valid_prev=df['is_valid'].shift(1),
        accuracy_group_prev=df['accuracy_group'].shift(1),
    )


def filter_consecutive_assessments(df):
    last_events = df.groupby('game_session', sort=False).last().reset_index()

    # shift operation must be performed by installation_id.
    last_events = last_events.groupby('installation_id', sort=False).apply(assign_prev).reset_index(drop=True)
    is_consecutive_assessment = (last_events['type'].eq('Assessment') &
                                 last_events['title'].eq(last_events['title_prev']) &
                                 last_events['is_valid_prev'].eq(1))
    return last_events[is_consecutive_assessment]


def consecutive_assessments_main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    funcs = [
        filter_assessment_attempt,
        assign_attempt_result,
        calc_attempt_stats,
    ]

    attempts_train = apply_funcs(train, funcs)
    attempts_test = apply_funcs(test, funcs)

    # A valid assessment game is one that contains at least one attempt.
    # Some assessment games don't contain attempts.
    # This probably happens when the user exits the game before completeting it.
    attempts_train['is_valid'] = 1
    attempts_test['is_valid'] = 1

    on = ['installation_id', 'game_session']
    cols_merge = on + ['is_valid', 'accuracy_group']
    train = pd.merge(train, attempts_train[cols_merge], how='left', on=on)
    test = pd.merge(test, attempts_test[cols_merge], how='left', on=on)

    train = filter_consecutive_assessments(train).assign(is_consecutive=1)
    test = filter_consecutive_assessments(test).assign(is_consecutive=1)

    cols_save = ['installation_id', 'game_session', 'is_consecutive', 'accuracy_group_prev']
    train = train[cols_save]
    test = test[cols_save]

    name = remove_dir_ext('consecutive_assessments')
    save_features(train, name, 'train')
    save_features(test, name, 'test')


if __name__ == '__main__':
    consecutive_assessments_main()


########################################
# unfinished_assessments
########################################

def unfinished_assessments_main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    funcs = [
        filter_assessment_attempt,
        assign_attempt_result,
        calc_attempt_stats,
    ]

    train = apply_funcs(train, funcs)
    test = apply_funcs(test, funcs)

    train['unfinished'] = train['accuracy_group'].eq(0)
    test['unfinished'] = test['accuracy_group'].eq(0)

    funcs = {'cumsum': ['unfinished']}
    train = cumulative_by_user(train, funcs)
    test = cumulative_by_user(test, funcs)

    train = train.fillna({'cumsum_unfinished': 0})
    test = test.fillna({'cumsum_unfinished': 0})

    train['cumsum_unfinished'] = train['cumsum_unfinished'].astype(np.int8)
    test['cumsum_unfinished'] = test['cumsum_unfinished'].astype(np.int8)

    name = remove_dir_ext('unfinished_assessments')
    save_features(train, name, 'train')
    save_features(test, name, 'test')

    use_cols = ['installation_id', 'game_session', 'cumsum_unfinished']
    meta = {'use_cols': use_cols}
    save_features_meta(meta, name)


if __name__ == '__main__':
    unfinished_assessments_main()
