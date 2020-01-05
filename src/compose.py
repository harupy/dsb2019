import os
import re
import argparse

from utils.io import read_config
from utils.common import remove_dir_ext

FTR_SCRIPT_DIR = 'src/features'


def parse_args():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-c', '--config', required=True, help='Config file path')
    return parser.parse_args()


def get_scripts(features):
    """
    >>> get_scripts(['a', 'b'])
    ['src/features/a.py', 'src/features/b.py']

    >>> get_scripts(['a', {'b': ['c']}])
    ['src/features/a.py', 'src/features/b.py']

    """
    result = []

    for feat in features:
        if isinstance(feat, str):
            result.append(os.path.join(FTR_SCRIPT_DIR, f'{feat}.py'))
        elif isinstance(feat, dict):
            result.extend([os.path.join(FTR_SCRIPT_DIR, f'{k}.py') for k in feat.keys()])
    return result


def remove_imports(lines):
    cnt = -1
    for line in lines:
        cnt += 1
        if line.startswith('def '):
            break

    return lines[cnt:]


def main():
    args = parse_args()
    config = read_config(args.config)

    feat_scripts = get_scripts(config['features'])

    merged = ''

    divider_base = """

########################################
# {}
########################################

"""

    for fpath in feat_scripts:
        with open(fpath, 'r') as f:
            feat_name = remove_dir_ext(fpath)
            src = ''.join(remove_imports(f.readlines()))
            suffix = '_' if re.match(r'^\d', feat_name) else ''
            src = src.replace('main()', suffix + f'{feat_name}_main()')
            src = src.replace('__file__', f"'{feat_name}'")
            merged += divider_base.format(feat_name) + src

    merged = merged.lstrip()
    with open('src/features_script.py', 'w') as f:
        f.write(merged)


if __name__ == '__main__':
    main()
