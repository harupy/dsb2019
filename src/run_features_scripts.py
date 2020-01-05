import os
import subprocess
import argparse

from utils.io import read_config


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
    SCRIPTS_DIR = 'src/features'
    result = []

    for feat in features:
        if isinstance(feat, str):
            result.append(os.path.join(SCRIPTS_DIR, f'{feat}.py'))
        elif isinstance(feat, dict):
            result.extend([os.path.join(SCRIPTS_DIR, f'{k}.py') for k in feat.keys()])
    return result


def run(command):
    with_export = 'export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src" && ' + command
    p = subprocess.Popen([with_export], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = p.communicate()
    print('----- stdout -----')
    print(stdout.decode('utf8'))
    print('----- stderr -----')
    print(stderr.decode('utf8'))


def main():
    args = parse_args()
    config = read_config(args.config)

    feat_scripts = get_scripts(config['features'])

    for fpath in feat_scripts:
        run(f'python {fpath}')


if __name__ == "__main__":
    main()
