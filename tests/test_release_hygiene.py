"""The version a checkout carries has to describe the library a release would ship.

Three numbers are read together by everyone who installs the package: the one in
``pip install jevper==…``, the one the documentation states, and the one in the code. When they
disagree, a bug report naming a version cannot be read — the truncation advice naming
``max_completion_tokens`` on Chat Completions while the published wheel still names ``max_tokens`` is
exactly that, and it passed review because the prose reads the same in both.

What is not checked is that ``src/`` has moved since the last tag. A commit that changes the library
is the next release being written; the number it carries is the maintainer's decision at release
time, and the guard that keeps the pages and the code in step until then is the site-version check
below.

The published version is read from the release tag rather than from PyPI, so the checks need no
network and skip where there is no tag history: an installed wheel in the runtime matrix, a source
tarball, a fresh clone with no tags fetched.
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


def test_the_published_release_is_the_one_the_docs_describe() -> None:
    """The version on PyPI and the version the site states must be the same number.

    The release tag is what PyPI carries, so it is the published version read from the checkout
    rather than over the network: a test that asked PyPI would fail on a laptop offline, and one that
    read ``pip index`` would answer a different question on every mirror. A reader who installed
    0.7.3 has to be able to tell whether the page in front of them is the one that still advises
    ``max_tokens`` on Chat Completions or the one that advises ``max_completion_tokens`` — the prose
    reads the same either way, which is how that divergence passed review in the first place.

    What is deliberately *not* checked here is that ``src/`` has moved since the tag. A commit that
    changes the library is the next release being written, and refusing to let that be committed is
    how a version stops meaning anything; the number a release carries is the maintainer's decision
    at release time, and the site-version guard above keeps the pages and the code in step until
    then.
    """
    tag = _git("describe", "--tags", "--abbrev=0", "--match", "v*")
    if tag is None:
        pytest.skip("no release tag is reachable from this checkout")
    tagged = _git("show", f"{tag}:src/jevper/__init__.py")
    if tagged is None:
        pytest.skip(f"{tag} predates src/jevper/__init__.py")
    shipped = VERSION.search(tagged)
    if shipped is None:
        pytest.skip(f"{tag} declares no __version__")
    if shipped.group(1) != tag.removeprefix("v"):
        pytest.fail(f"the tag {tag} and the version inside it ({shipped.group(1)}) disagree")

    landing = (REPO / "docs" / "index.md").read_text()
    config = (REPO / "mkdocs.yml").read_text()
    stated = re.findall(r"jevper (\d+\.\d+\.\d+)", landing)
    footer = re.search(r'^copyright: "[^"]*?jevper (\d+\.\d+\.\d+)"', config, re.MULTILINE)
    documented = {*stated, footer.group(1) if footer is not None else None} - {None}

    assert documented, "the site states no version to compare against"
    for number in sorted(documented):
        assert number == shipped.group(1), (
            f"the site documents jevper {number} but the published release is "
            f"{shipped.group(1)} ({tag}); a reader cannot tell which one the pages describe unless "
            "they are the same release"
        )


def test_the_site_states_the_release_it_documents() -> None:
    """The site says which release it describes, and that is the release this branch is.

    A reader who installed 0.7.3 cannot otherwise tell whether the page in front of them is the one
    that still advises ``max_tokens`` on Chat Completions or the one that advises
    ``max_completion_tokens`` — the prose reads the same either way, which is what let the 0.7.3
    divergence pass review. The number appears twice, in the footer and on the landing page, and
    both are checked here because a stale one is as misleading as none.
    """
    landing = (REPO / "docs" / "index.md").read_text()
    config = (REPO / "mkdocs.yml").read_text()

    stated = re.findall(r"jevper (\d+\.\d+\.\d+)", landing)
    assert stated, "docs/index.md does not say which release it documents"
    footer = re.search(r'^copyright: "[^"]*?jevper (\d+\.\d+\.\d+)"', config, re.MULTILINE)
    assert footer is not None, "mkdocs.yml footer states no version"

    for number in (*stated, footer.group(1)):
        assert number == jevper.__version__, (
            f"the site says jevper {number} but the package is {jevper.__version__}; the pages and "
            "the installed library have to be the same release for either to be worth reading"
        )
