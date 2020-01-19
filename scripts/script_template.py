import os
import subprocess
import gzip
import base64
from pathlib import Path
import pandas as pd

# encoded source code.
SCRIPTS = {{scripts}}  # noqa

# training configuration.
CONFIG = {{config}}  # noqa

# commit hash.
COMMIT_HASH = "{{commit_hash}}"


def skip_commit_run():
    """
    Terminate the commit run immediately.
    """
    # skip the commit run (https://www.kaggle.com/onodera/skip-1st-run).
    sbm = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

    # len(sbm) returns 1000 for the public test set, but > 1000 for the private test set.
    if len(sbm) == 1000:
        sbm.to_csv('submission.csv', index=False)
        exit(0)


def get_feature_scripts(features):
    """
    Get features scripts files.
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
    """
    Execute the command (a string) in a subshell.
    """
    print('Executing:', command)
    with_export = 'export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src" && ' + command
    p = subprocess.Popen([with_export], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = p.communicate()
    stdout = stdout.decode('utf8')
    stderr = stderr.decode('utf8')

    if stdout:
        print('\n---------- stdout ----------\n')
        print(stdout)

    # note that warnings will be included in stderr.
    if stderr:
        print('\n---------- stderr ----------\n')
        print(stderr)


def decode_scripts(scripts):
    """
    Decode encoded scripts (file path -> encoded code).
    """
    for path, encoded_src in scripts.items():
        print(path)
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(gzip.decompress(base64.b64decode(encoded_src)))


def main():
    decode_scripts(SCRIPTS)

    # print out the source code of train.py
    run('cat src/modeling/train.py')
    skip_commit_run()

    # find features scripts to run from the training configuration.
    feature_scripts = get_feature_scripts(CONFIG['features'])

    # generate features.
    for features_script in feature_scripts:
        run(f'python {features_script}')

    # run training
    run('python src/modeling/train.py -c {{config_path}}')


if __name__ == '__main__':
    main()
