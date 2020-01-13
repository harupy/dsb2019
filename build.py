import os
import argparse
import json
from functools import reduce
import base64
import gzip
from pathlib import Path

from utils.io import read_config


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


def build_script():
    args = parse_args()
    config = read_config(args.config)
    to_encode = reduce(lambda l, d: l + search_by_ext(d, ['.py']), ['src', 'configs'], [])
    scripts = {str(path): encode_file(Path(path)) for path in to_encode}
    template = Path('script_template.py').read_text('utf8')
    Path('script.py').write_text(
        (template
         .replace('{{scripts}}', json.dumps(scripts, indent=4))
         .replace('{{config}}', json.dumps(config, indent=2))
         .replace('{{config_path}}', args.config)
         ),
        encoding='utf8')


if __name__ == '__main__':
    build_script()
