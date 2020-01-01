import os

from utils.common import prefix_list, remove_ext
from utils.io import read_features, save_features
from utils.dataframe import concat_dfs, prefix_columns


def cum_stats_4020(df, is_test=False):

    def process_gdf(df):
        funcs = {
            'cumsum': ['num_correct', 'num_incorrect', 'attempts'],
            'cummean': ['accuracy'],
        }

        dfs = []
        drop_cols = []
        for func, cols in funcs.items():
            drop_cols += cols

            # for test, it's not necessary to shift rows
            periods = 0 if is_test else 1

            if func == 'cumsum':
                cum = df[cols].cumsum().shift(periods)
            elif func == 'cummean':
                cum = df[cols].expanding().mean().shift(periods)

            cum.columns = prefix_list(cols, func)
            dfs.append(cum)

        return concat_dfs([df.drop(drop_cols, axis=1)] + dfs, axis=1)

    return (
        df
        .groupby('installation_id', sort=False)
        .apply(process_gdf)
        .drop(['title', 'timestamp'], axis=1)
    )


def main():
    stats_train, stats_test = read_features('4020_attempts_stats')

    cum_stats_train = cum_stats_4020(stats_train)
    cum_stats_test = cum_stats_4020(stats_test, True)

    exclude = ['installation_id', 'game_session']
    cum_stats_train = prefix_columns(cum_stats_train, '4020', exclude=exclude)
    cum_stats_test = prefix_columns(cum_stats_test, '4020', exclude=exclude)

    name = remove_ext(os.path.basename(__file__))
    save_features(cum_stats_train, name, 'train')
    save_features(cum_stats_test, name, 'test')


if __name__ == '__main__':
    main()
