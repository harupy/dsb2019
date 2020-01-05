import os
import subprocess
import gzip
import base64
from pathlib import Path


SCRIPTS = {{scripts}}

CONFIG = {{config}}


def get_feature_scripts(features):
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
    stdout = stdout.decode('utf8')
    stderr = stderr.decode('utf8')

    if stdout:
        print('----- stdout -----\n', stdout)

    # note that warnings will be displayed as stderr.
    if stderr:
        print('----- stderr -----\n', stderr)


def decode_scripts(scripts):
    for path, encoded in scripts.items():
        print(path)
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def main():
    decode_scripts(SCRIPTS)
    feature_scripts = get_feature_scripts(CONFIG['features'])
    feature_scripts = ['src/preprocess/clean_data.py'] + feature_scripts

    for feature_script in feature_scripts:
        run(f'python {feature_script}')

    run('python src/modeling/train.py -c configs/baseline.py')


if __name__ == '__main__':
    main()
