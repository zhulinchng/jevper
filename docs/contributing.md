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
