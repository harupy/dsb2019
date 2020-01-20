
import git


def get_commit_hash():
    """
    Get the current commit hash.
    """
    return git.Repo().head.object.hexsha


def is_changed():
    """
    Return True if Git detects changes.
    """
    changes = [item.a_path for item in git.Repo().index.diff(None)]
    return len(changes) != 0
