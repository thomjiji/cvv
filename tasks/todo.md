# TUI Throughput Dashboard

## Plan
- [x] Inspect the current copy engine, CLI processor, and test surface.
- [x] Create a dedicated feature branch for the TUI work.
- [x] Refactor the CLI layer so UI selection is explicit and renderer-driven.
- [x] Add a Rich-powered TUI with rolling copy and verify throughput history.
- [x] Extend tests for throughput tracking, UI selection, and parser behavior.
- [x] Run the test suite and capture verification notes.

## Review
- `./.venv/bin/python -m compileall src tests`
- `env UV_CACHE_DIR=/tmp/uv-cache uv sync`
- `./.venv/bin/python -m pytest tests/test_cvv.py`
- `./.venv/bin/python -m cvv.main --help`
- `./.venv/bin/python -c "from rich.console import Console; print('rich-ok')"`
- `./.venv/bin/cvv "$src" "$dest" --ui tui` smoke-tested via PTY on a temporary 0.5 MB file
- Notes: parser shows the new `--ui {tui,text}` flag; pytest passed with `24 passed, 1 skipped`.
