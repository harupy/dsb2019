import os
from functools import reduce
import autopep8


def remove_utils(src):
    """
    Remove lines that import utils.
    """
    lines = src.split('\n')
    lines = list(filter(lambda l: not l.startswith('from utils.'), lines))
    return '\n'.join(lines)


def main():
    # relative path from the root.

    src = ''
    exclude = ['profiling.py']

    UTILS_DIR = 'src/utils'
    TEMPLATE = """

########################################
# {}
########################################

"""

    for fname in os.listdir(UTILS_DIR):
        fpath = os.path.join(UTILS_DIR, fname)

        if os.path.isdir(fpath) or \
           (fname == '__init__.py') or \
           (not fname.endswith('.py')) or \
           (fname in exclude):
            continue

        print(f'Processing {fname}')

        src += TEMPLATE.format(fname)

        with open(fpath, 'r') as f:
            src += f.read()

    out_path = os.path.join(os.path.dirname(UTILS_DIR), 'utils_merged.py')
    with open(out_path, 'w') as f:
        funcs = [
            remove_utils,
            autopep8.fix_code,
            str.strip
        ]
        f.write(reduce(lambda s, f: f(s), funcs, src))


if __name__ == '__main__':
    main()
