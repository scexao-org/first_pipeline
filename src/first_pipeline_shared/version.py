"""
Single source of truth for the FIRST pipeline version.

Everything else (setup.py, the package __init__ files, the FITS headers of the
products) reads the version from here. Bump __version__ when releasing and tag
the commit (git tag v<version>).
"""

import functools
import os
import subprocess

__version__ = "2.1.0"


@functools.lru_cache(maxsize=None)
def get_git_info():
    """
    (short commit hash, dirty flag) of the pipeline source tree, or None when
    the code is not run from a git checkout (or git is unavailable).

    'dirty' means uncommitted changes in src/, setup.py or requirements.txt
    (untracked files in src/ included). git is run with GIT_OPTIONAL_LOCKS=0
    so that it never leaves an index.lock behind.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def git(*args):
        return subprocess.run(["git", "-C", here, *args], capture_output=True,
                              text=True, env=env, timeout=10)
    try:
        commit = git("rev-parse", "--short=8", "HEAD")
        if commit.returncode != 0:
            return None
        top = git("rev-parse", "--show-toplevel").stdout.strip()
        status = git("status", "--porcelain", "--", os.path.join(top, "src"),
                     os.path.join(top, "setup.py"), os.path.join(top, "requirements.txt"))
        return commit.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def get_version_string():
    """'2.1.0', or '2.1.0+g<commit>' / '2.1.0+g<commit>.dirty' in a git checkout."""
    info = get_git_info()
    if info is None:
        return __version__
    commit, dirty = info
    return f"{__version__}+g{commit}{'.dirty' if dirty else ''}"


def add_version_keywords(header):
    """
    Stamp a FITS header with the pipeline version that produced the file:
      Q_PIPVER = '2.1.0'              pipeline version
      Q_PIPGIT = '00e6b37a-dirty'     git commit ('-dirty': uncommitted changes)
    Returns the header.
    """
    header['Q_PIPVER'] = (__version__, 'FIRST pipeline version')
    info = get_git_info()
    if info is None:
        git_str = 'unknown'
    else:
        git_str = info[0] + ('-dirty' if info[1] else '')
    header['Q_PIPGIT'] = (git_str, 'git commit of the pipeline (-dirty: uncommitted)')
    return header
