# CLAUDE.md

## Project

cvv — a Python CLI tool for copying files from one source to multiple destinations simultaneously, with hash-based verification. Inspired by Pomfort Offload Manager's core workflow for on-set DIT use.

## Build & run

```bash
uv run cvv <source> <dest1> [dest2 ...]      # run directly
uv run pytest tests/ -v                       # tests
uv run ruff check . && uv run ruff format .   # lint + format
```

## Architecture

Single file: `src/cvv/main.py`. Key classes:

- `HashCalculator` — streaming hash (default xxh3_64, also xxh64/md5/sha1/sha256)
- `CopyEngine` — one-source-to-N-destinations copy via reader thread + per-destination writer threads with queue-based fan-out; yields events (progress, result)
- `CLIProcessor` — CLI orchestration, directory walking, progress display

## Verification modes

- **transfer** — size-only check (fastest, no hash)
- **source** — in-flight source hash + post-copy source re-read hash comparison (detects bad cards/readers)
- **full** — source verification + destination hash verification (maximum integrity)

## Conventions

- Python ≥3.9, type hints throughout, NumPy-style docstrings
- `uv` for dependency management (not pip)
- Ruff for linting and formatting

## Planned features

- Real-time copy speed display
- Progress bar (rich/tqdm)
- Per-run log file
- MHL / ASC MHL output
