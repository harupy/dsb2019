"""
Build a runner script and push it to Kaggle as a kernel.
Note that Kaggle automatically executes the pushed script.
"""

import os
import subprocess
import argparse
import json
from functools import reduce
import base64
import gzip
from pathlib import Path

from utils.common import get_timestamp
from utils.io import read_config, read_json, save_dict
from utils.git import get_commit_hash, is_changed

# directory to store scripts to submit.
SCRIPTS_DIR = 'scripts'


def parse_args():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-c', '--config', required=True, help='Config file path')
    return parser.parse_args()


def encode_src(path):
    """
    Return encoded source of given file.
    """
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')


def search_by_ext(top, ext):
    """
    Search files by extension.
    """
    ret = []
    for root, dirs, files in os.walk(top):
        for fname in files:
            if os.path.splitext(fname)[1] not in ext:
                continue
            ret.append(os.path.join(root, fname))
    return ret


def build_kernel_meta(kernel_id, code_file):
    """
    Create kernel metadata with given kernel id
    """
    meta = read_json(f'{SCRIPTS_DIR}/kernel-metadata-base.json')
    meta.update({
        'id': meta['id'].format(kernel_id),
        'title': meta['title'].format(kernel_id),
        'code_file': meta['code_file'].format(code_file),
    })
    return meta


def run(command):
    """
    Execute the command (a string) in a subshell.
    """
    p = subprocess.Popen([command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = p.communicate()
    stdout = stdout.decode('utf8')
    stderr = stderr.decode('utf8')

    if stdout:
        print('---------- stdout ----------')
        print(stdout)

    # Note: warnings will be included in stderr.
    if stderr:
        print('---------- stderr ----------')
        print(stderr)


def push_kernel(directory):
    """
    Push a kernel (script and metadat set) to Kaggle.
    """
    run(f'kaggle kernels push -p {directory}')


def main():
    if is_changed():
        print('Detect changes. Please commit them before building the script.')
        exit()

    args = parse_args()
    config = read_config(args.config)

    # encode scripts under specified directories.
    dirs = ['src', 'configs']
    to_encode = reduce(lambda l, d: l + search_by_ext(d, ['.py']), dirs, [])
    scripts = {str(path): encode_src(Path(path)) for path in to_encode}

    # make a directory using and the current timestamp and commit hash.
    timestamp = get_timestamp()
    commit_hash = get_commit_hash()
    save_dir = f'{SCRIPTS_DIR}/{timestamp}_{commit_hash}'
    filename = f'{commit_hash}.py'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # build a runner script.
    template = Path(f'{SCRIPTS_DIR}/script_template.py').read_text('utf8')
    script_path = f'{save_dir}/{filename}'
    Path(script_path).write_text(
        (template
         .replace('{{commit_hash}}', get_commit_hash())
         .replace('{{scripts}}', json.dumps(scripts, indent=4))
         .replace('{{config}}', json.dumps(config, indent=2))
         .replace('{{config_path}}', args.config)
         ),
        encoding='utf8')

    # build kernel meta data.
    meta = build_kernel_meta(commit_hash, filename)
    save_dict(meta, os.path.join(save_dir, 'kernel-metadata.json'))
    push_kernel(save_dir)


if __name__ == '__main__':
    main()
