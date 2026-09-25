"""The documentation is checked the way a reader meets it.

The programs it publishes run: ``examples/complete_call.py`` — the program the "Complete example"
page embeds — is executed against the stub server with the request fields it documents checked on
the wire, and every public option is required to appear in it, so a new parameter cannot merge
without a documented example. The pages themselves are held to what a renderer needs: every table
well-formed, every relative link and heading fragment resolving.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fakes import StubServer, chat_body

from jevper import AsyncSystemOneClient, SystemOneClient

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "complete_call.py"
DUCK = REPO / "examples" / "duck_client.py"
PAGES = sorted((REPO / "docs").glob("*.md")) + [REPO / "README.md"]

DISTRIBUTIONS = {
    "jevper_choice": {"billing": 0.82, "technical": 0.12, "sales": 0.06},
    "jevper_score": {"0": 0.0, "1": 0.15, "2": 0.85},
}

def _public_options(owner: Any) -> set[str]:
    """The keyword-settable options of a public callable. ``client`` is the positional client object,
    and a program shows that by handing one in, so it is not part of the keyword inventory."""
    return set(inspect.signature(owner).parameters) - {"self", "client"}


def _load_example(path: Path = EXAMPLE) -> ModuleType:
    """A documented program, imported as written rather than copied into the test."""
    spec = importlib.util.spec_from_file_location(f"jevper_example_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _answer_for(body: dict[str, Any]) -> str:
    """The answer the stub model gives, read off the schema the request asked for."""
    schema = body["response_format"]["json_schema"]
    name: str = schema["name"]
    if name == "jevper_noul":
        return json.dumps({"noul": 0.93})
    keys = list(schema["schema"]["properties"]["probabilities"]["properties"])
    wanted = DISTRIBUTIONS[name]
    return json.dumps({"probabilities": {key: wanted[key] for key in keys}})


@pytest.mark.parametrize(
    "owner",
    [SystemOneClient.__init__, SystemOneClient.system_one, AsyncSystemOneClient.system_one],
    ids=["client", "system_one", "async_system_one"],
)
def test_the_complete_example_sets_every_public_option(owner: Any) -> None:
    source = EXAMPLE.read_text()
    missing = sorted(
        name
        for name in _public_options(owner)
        if not re.search(rf"\b{re.escape(name)}\s*=", source)
    )

    assert not missing, (
        f"{EXAMPLE.name} does not set {missing}; the complete example is how a new public option "
        "reaches a reader, so add it there (docs/complete-example.md embeds this file)"
    )


def test_the_complete_example_page_embeds_the_programs() -> None:
    page = (REPO / "docs" / "complete-example.md").read_text()

    for path in (EXAMPLE, DUCK):
        assert f'--8<-- "{path.relative_to(REPO).as_posix()}"' in page, (
            f"the page must embed {path.name}, not a copy of it, or the two can drift"
        )


def test_the_duck_typed_client_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    _load_example(DUCK).main()

    assert capsys.readouterr().out.split() == ["0.8", "1", "chat_completions"]


def test_the_complete_example_runs_and_sends_what_it_documents(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def chat(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return 200, chat_body(content=_answer_for(body))

    with StubServer(chat=chat) as stub:
        monkeypatch.setenv("JEVPER_BASE_URL", stub.base_url)
        monkeypatch.setenv("JEVPER_API_KEY", "test")
        monkeypatch.setenv("JEVPER_MODEL", "stub-model")

        _load_example().main()

    out = capsys.readouterr().out
    assert "intent choice" in out
    assert "needs_human noul" in out
    assert "sentiment score" in out
    assert "usage:" in out

    # One provider call per question: native reasoning is one call, and both the method and the
    # surface are pinned, so nothing is probed first.
    assert len(stub.requests) == 3
    for body, headers in zip(stub.requests, stub.headers):
        assert body["model"] == "stub-model"
        assert body["temperature"] == 0.0
        assert body["reasoning_effort"] == "low"
        assert body["max_completion_tokens"] == 2048  # the budget, in the surface's own name
        assert body["prompt_cache_key"] == "complete-example"
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert {name.lower(): value for name, value in headers.items()}.get("x-jevper-example") == (
            "complete"
        )


def _prose(text: str) -> list[str]:
    """The markdown lines outside every fenced code block, which is where tables and links live."""
    outside: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            outside.append(line)
    return outside


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


@pytest.mark.parametrize("path", PAGES, ids=lambda path: path.name)
def test_every_documented_table_is_well_formed(path: Path) -> None:
    """A table whose rows disagree on their columns does not render as a table, and a strict build
    does not notice: the reader gets a paragraph of pipes."""
    run: list[str] = []
    problems: list[str] = []

    def check(rows: list[str]) -> None:
        if len(rows) < 2:
            problems.append(f"a table with {len(rows)} row(s) and no delimiter row: {rows[0]!r}")
            return
        if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in _cells(rows[1])):
            problems.append(f"no delimiter row under {rows[0]!r}: {rows[1]!r}")
            return
        widths = {len(_cells(row)) for row in rows}
        if len(widths) != 1:
            problems.append(f"rows disagree on column count under {rows[0]!r}: {sorted(widths)}")

    for line in [*_prose(path.read_text()), ""]:
        if line.strip().startswith("|"):
            run.append(line)
        elif run:
            check(run)
            run = []
    if run:
        check(run)

    assert not problems, f"{path.name}: " + "; ".join(problems)


def _slug(heading: str) -> str:
    """The anchor MkDocs gives a heading: lower case, punctuation dropped, spaces as hyphens."""
    text = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return re.sub(r"\s+", "-", text)


def _anchors(path: Path) -> set[str]:
    return {
        _slug(match.group(1))
        for line in _prose(path.read_text())
        if (match := re.match(r"^#{1,6}\s+(.*?)\s*#*$", line))
    }


@pytest.mark.parametrize("path", PAGES, ids=lambda path: path.name)
def test_every_relative_link_and_fragment_resolves(path: Path) -> None:
    """A link to a page that moved, or to a heading that was renamed, is a dead end in the rendered
    docs — and a strict build only notices a missing file, never a missing anchor."""
    broken: list[str] = []
    for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)", "\n".join(_prose(path.read_text()))):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        file_part, _, fragment = target.partition("#")
        target_path = (path.parent / file_part).resolve()
        if not target_path.exists():  # a directory link is fine on GitHub, where the README lives
            broken.append(target)
            continue
        if fragment and fragment not in _anchors(target_path):
            broken.append(target)

    assert not broken, f"{path.name}: dead links {sorted(set(broken))}"
