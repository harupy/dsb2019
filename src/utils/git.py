
import git


def get_commit_hash():
    """
    Get the current commit hash.
    """
    return git.Repo().head.object.hexsha


def get_active_branch():
    """
    Get the active branch name.
    """
    return git.Repo().active_branch.name


def contain_changes():
    """
    Detect code changes.
    """
    changes = [item.a_path for item in git.Repo().index.diff(None)]
    return len(changes) != 0
