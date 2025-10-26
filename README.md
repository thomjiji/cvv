# cvv

Copy files to multiple destinations with integrity verification.

## What it does

cvv copies files or directories to multiple destinations simultaneously and verifies the copies are correct using hash checksums.

## Installation

```bash
pip install -e .
```

## Usage

Copy a single file to multiple destinations:

```bash
cvv source.mov /backup1/dest.mov /backup2/dest.mov
```

Copy a directory to multiple destinations:

```bash
cvv /source/folder /backup1/folder /backup2/folder
```

## Options

- `-v` - Show verification mode (transfer/source/full)
- `-a` - Hash algorithm (xxh64be, md5, sha256)

## Examples

Copy with full verification:

```bash
cvv -v full -a xxh64be video.mov /backup1/video.mov /backup2/video.mov
```

Copy a directory:

```bash
cvv /source/project /backup1/project /backup2/project
```

## Requirements

- Python 3.8+
- xxhash library
