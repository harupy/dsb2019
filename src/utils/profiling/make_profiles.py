"""
A script to make a profile notebook.
"""

import os
import argparse

from utils.common import get_ext, replace_ext


# relative from the root.
TEMPLATE_PATH = 'src/utils/profiling/profile_template.ipynb'


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Make a profile')
    parser.add_argument('-d', '--data-dir', required=True, help='Data directory')
    return parser.parse_args()


def search_by_ext(top, ext):
    """
    Search files by extension under given directory.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     names = ['test.txt', 'test.py', 'test.sh']
    ...     files = [os.path.join(tmpdir, f) for f in names]
    ...     for f in files:
    ...         open(f, 'w').close()
    ...     matched = search_by_ext(tmpdir, ['.txt', '.py'])
    ...     set(matched) == set(files[:-1])
    True

    """
    result = []
    for root, dirs, files in os.walk(top):
        for fname in files:
            if get_ext(fname) not in ext:
                continue
            result.append(os.path.join(root, fname))
    return result


def execute_notebook(nb_path):
    """
    Execute a jupyter notebook.
    """
    command = f'jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --inplace {nb_path}'
    os.system(command)


def make_profiles(dir_or_path):
    # search target data to make profiles of.
    ext = ['.csv', '.ftr']
    if os.path.isdir(dir_or_path):
        files = search_by_ext(dir_or_path, ext)
    else:
        assert os.path.splitext(dir_or_path)[1] in ['.csv', '.ftr']
        files = [dir_or_path]

    num_files = len(files)

    # read notebook template.
    with open(TEMPLATE_PATH, 'r') as f:
        template_src = f.read()

    for file_idx, fpath in enumerate(files):
        print('\n---------- Executing {} ({}/{}) ----------\n'
              .format(os.path.basename(fpath), file_idx + 1, num_files))

        # replace <DATA_DIR> and <DATA_PATH> in the template source.
        # note that the cloned notebook will be executed in the same directory as the input data.
        nb_src = template_src.replace('<DATA_DIR>', os.path.dirname(fpath))
        nb_src = nb_src.replace('<DATA_PATH>', os.path.basename(fpath))

        # save as a new notebook.
        nb_path = replace_ext(fpath, '.ipynb')

        with open(nb_path, 'w') as f:
            f.write(nb_src)

        # execute the cloned notebook.
        execute_notebook(nb_path)


def main():
    args = parse_args()
    make_profiles(args.data_dir)


if __name__ == '__main__':
    main()
