import os
import argparse
import json
from functools import reduce
import base64
import gzip
from pathlib import Path
import git

from utils.common import get_timestamp
from utils.io import read_config, read_json, save_dict


def parse_args():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-c', '--config', required=True, help='Config file path')
    return parser.parse_args()


def encode_file(path):
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')


def search_by_ext(top, ext):
    ret = []
    for root, dirs, files in os.walk(top):
        for fname in files:
            if os.path.splitext(fname)[1] not in ext:
                continue
            ret.append(os.path.join(root, fname))
    return ret


def get_commit_hash():
    return git.Repo().head.object.hexsha


def is_changed():
    changes = [item.a_path for item in git.Repo().index.diff(None)]
    return len(changes) != 0


def create_kernel_meta(kernel_id, save_dir):
    """
    Create kernel metadata with given kernel id
    """
    meta = read_json('kernel-metadata-base.json')
    meta.update({
        'id': meta['id'].format(kernel_id),
        'title': meta['title'].format(kernel_id),
    })
    save_dict(meta, os.path.join(save_dir, 'kernel-metadata.json'))


def build_script():
    if is_changed():
        print('Detect changes. Please commit them before building the script.')
        exit()

    args = parse_args()
    config = read_config(args.config)
    to_encode = reduce(lambda l, d: l + search_by_ext(d, ['.py']), ['src', 'configs'], [])
    scripts = {str(path): encode_file(Path(path)) for path in to_encode}
    template = Path('script_template.py').read_text('utf8')
    timestamp = get_timestamp()
    commit_hash = get_commit_hash()
    save_dir = f'scripts/{timestamp}_{commit_hash}'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    Path(f'{save_dir}/{commit_hash}.py').write_text(
        (template
         .replace('{{git_hash}}', get_commit_hash())
         .replace('{{scripts}}', json.dumps(scripts, indent=4))
         .replace('{{config}}', json.dumps(config, indent=2))
         .replace('{{config_path}}', args.config)
         ),
        encoding='utf8')
    create_kernel_meta(commit_hash, save_dir)


if __name__ == '__main__':
    build_script()
