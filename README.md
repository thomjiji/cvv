This project is highly experimental, don't use it in production enviroment.

# cvv

Copy files to multiple destinations simultaneously with (optionally) integrity verification.

The name “cvv” comes from the simplest copy shortcut — Ctrl-C / Cmd-C and Ctrl-V / Cmd-V. We call it copy using "cv", or just "cv" it. But why two V’s? The idea is to highlight “one read, multiple writes” — in other words, a single Ctrl-C that can perform multiple Ctrl-V’s simultaneously.

## What it does

cvv copies files or directories to multiple destinations simultaneously and verifies the copies are correct using checksums.

## Prerequisites

This project is managed with **uv**. You need to have [uv installed](https://docs.astral.sh/uv/getting-started/installation/) before proceeding.

## Installation

```bash
# Install the project and its dependencies
uv sync

# Run cvv using uv
uv run cvv --help
```

Or install it in development mode:
```bash
uv pip install -e .
```

## Usage

After installation, you can run cvv directly:

```bash
cvv source.mov /backup1/dest.mov /backup2/dest.mov
```

Or without installing, use `uv run`:

```bash
uv run cvv source.mov /backup1/dest.mov /backup2/dest.mov
```

### Basic Examples

Copy a single file to multiple destinations:

```bash
cvv source.mov /backup1/dest.mov /backup2/dest.mov
```

Copy a directory to multiple destinations:

```bash
cvv /source/folder /backup1/folder /backup2/folder
```

## Options

- `-m, --mode` - Verification mode: `transfer` (size only), `source` (verify source), `full` (verify all) (default: full)
- `--hash-algorithm` - Hash algorithm: `xxh64be`, `md5`, `sha1`, `sha256` (default: xxh64be)

## Examples

Copy with transfer mode (fastest, size check only):

```bash
cvv -m transfer video.mov /backup1/video.mov /backup2/video.mov
```

Copy with full verification using xxh64be:

```bash
cvv -m full --hash-algorithm xxh64be video.mov /backup1/video.mov /backup2/video.mov
```

Copy a directory:

```bash
cvv /source/project /backup1/project /backup2/project
```

## Requirements

- Python 3.9+
- uv (see Prerequisites section above)

Dependencies (automatically installed by uv):
- xxhash library
