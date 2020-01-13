from utils.io import read_features


def main():
    train1 = read_features('game_session_stats', 'train')
    train2 = read_features('convert_to_regression', 'train')

    for col in train2.columns:
        if col not in train1.columns:
            print(f'{col} not found in convert_to_regression')
            continue

        is_equal = train1[col].equals(train2[col])
        if not is_equal:
            print(col, is_equal)


if __name__ == '__main__':
    main()
