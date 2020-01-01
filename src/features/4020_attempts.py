"""
Ref.
https://www.kaggle.com/bhavikapanara/2019-dsb-with-more-features-qwk-0-549
"""

import os
import numpy as np

from utils.common import remove_ext
from utils.io import read_from_clean, save_features
from features.funcs import is_assessment, get_attempt_result


def main():
    train = read_from_clean('train.ftr')
    test = read_from_clean('test.ftr')

    is_asm_train = is_assessment(train)
    is_asm_test = is_assessment(test)

    is_correct_train = get_attempt_result(train)
    is_correct_test = get_attempt_result(test)

    is_4020_train = train['event_code'].eq(4020)
    is_4020_test = test['event_code'].eq(4020)

    train['4020_correct'] = (is_asm_train & is_4020_train & is_correct_train).astype(np.int8)
    test['4020_correct'] = (is_asm_test & is_4020_test & is_correct_test).astype(np.int8)

    name = remove_ext(os.path.basename(__file__))
    save_features(train[is_asm_train & is_4020_train], name, 'train')
    save_features(test[is_asm_test & is_4020_test], name, 'test')


if __name__ == '__main__':
    main()
