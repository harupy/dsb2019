import os

from utils.io import read_from_clean, save_features
from utils.common import remove_ext
from features.funcs import filter_assessment_attempt


def main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    asm_train = filter_assessment_attempt(train)
    asm_test = filter_assessment_attempt(test)

    name = remove_ext(os.path.basename(__file__))
    save_features(asm_train, name, 'train')
    save_features(asm_test, name, 'test')


if __name__ == '__main__':
    main()
