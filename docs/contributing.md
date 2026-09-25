# Contributing to the documentation

The documentation source lives in `docs/`. The generated `site/` directory is ignored and must not be committed.

## Edit the site

1. Edit the relevant Markdown page under `docs/`.
2. Keep the explicit navigation in `mkdocs.yml` complete and ordered by task.
3. Use relative links between pages, such as `[API reference](api.md)`.
4. Use a fenced ` ```mermaid ` block for a new diagram and keep labels readable.
5. Run the strict build from the repository root:

   ```sh
   uv run --extra docs mkdocs build --strict
   ```

6. Preview the rendered site and check search, navigation, light/dark mode, code blocks, and every diagram:

   ```sh
   uv run --extra docs mkdocs serve
   ```

Open the local URL printed by MkDocs. The build must pass before a documentation change is submitted. For a focused change, update only the page that owns the information; keep the public API, provider caveats, and Mermaid examples in their existing canonical pages.

## Claims and releases

A sentence about what jevper sends, reads or raises is a claim about `src/jevper`, and the test that
pins it is the proof: `ruff check src tests` and the suite run offline, so a claim that no test
backs is a claim nothing will catch drifting. A sentence about a *released* version is a claim about
that tag, not about the branch — read it with `git show v<version>:src/jevper/<module>.py` and say
which one you read.

Any change under `src/` changes the library, so it needs a version bump: a branch whose `src/` has
moved since the last release tag must not carry that tag's number, or the documentation and
`pip install jevper==<version>` describe different libraries. `tests/test_release_hygiene.py` fails
the build when it does.
