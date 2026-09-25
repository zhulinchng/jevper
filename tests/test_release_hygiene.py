"""The version a checkout carries has to describe the library a release would ship.

Two facts are read together by everyone who installs the package: the number in
``pip install jevper==…`` and the documentation built from the branch. When ``src/`` has moved since
the tag that carried the published version, the working tree already *is* the next release, and
keeping the old number makes the docs describe behaviour nobody can install — the truncation advice
naming ``max_completion_tokens`` on Chat Completions while the published wheel still names
``max_tokens`` is exactly that, and a bug report saying "0.7.3" cannot be read either.

The check needs the tag history, so it skips where there is none: an installed wheel in the runtime
matrix, a source tarball, a fresh clone with no tags fetched.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import jevper

REPO = Path(__file__).resolve().parents[1]
VERSION = re.compile(r'^__version__ = "([^"]+)"', re.MULTILINE)


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def test_the_declared_version_is_one_number_everywhere() -> None:
    """The three places that carry it — metadata, the module, the lock — must not disagree."""
    pyproject = (REPO / "pyproject.toml").read_text()
    lock = (REPO / "uv.lock").read_text()
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    assert declared is not None, "pyproject.toml declares no version"
    locked = re.search(r'name = "jevper"\nversion = "([^"]+)"', lock)

    assert declared.group(1) == jevper.__version__ == (locked.group(1) if locked else declared.group(1))
    source = (REPO / "src" / "jevper" / "__init__.py").read_text()
    exported = VERSION.search(source)
    assert exported is not None and exported.group(1) == jevper.__version__


def test_a_changed_library_is_not_still_the_published_version() -> None:
    """``src/`` has moved since the last release tag: say so, or revert the change."""
    tag = _git("describe", "--tags", "--abbrev=0", "--match", "v*")
    if tag is None:
        pytest.skip("no release tag is reachable from this checkout")
    tagged = _git("show", f"{tag}:src/jevper/__init__.py")
    if tagged is None:
        pytest.skip(f"{tag} predates src/jevper/__init__.py")
    shipped = VERSION.search(tagged)
    if shipped is None:
        pytest.skip(f"{tag} declares no __version__")
    # The tag against the working tree, not against HEAD: a change that is staged but uncommitted
    # still describes a library no release carries.
    changed = bool(_git("diff", "--name-only", tag, "--", "src"))

    assert not (changed and jevper.__version__ == shipped.group(1)), (
        f"src/ has changed since {tag} ({shipped.group(1)}), but the version is still "
        f"{jevper.__version__}; bump it so the published package and this branch stop claiming to be "
        "the same release"
    )
