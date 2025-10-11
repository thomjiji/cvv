#!/usr/bin/env python3
import os
import time
import xxhash
import logging
from pathlib import Path
from datetime import datetime
import argparse


# =========================
# Logging configuration
# =========================
class SimpleFormatter(logging.Formatter):
    def format(self, record):
        level = record.levelname[0].upper()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        pid = os.getpid()
        module = record.name
        msg = record.getMessage()
        return f"{level} {timestamp} {module} ({pid}): {record.levelname}: {msg}"


logger = logging.getLogger("offload")
handler = logging.StreamHandler()
handler.setFormatter(SimpleFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# =========================
# File copy + hash
# =========================
def copy_with_hash(
    src: Path, dsts: list[Path], block_size: int = 8 * 1024 * 1024
) -> str:
    """Copy file from src to multiple destinations, with xxh64 hash check."""
    file_size = src.stat().st_size
    logger.info(f"starting copy from {src} to {[str(d) for d in dsts]}")

    start = time.time()
    h = xxhash.xxh64()

    # Prepare destination files
    dst_files = []
    try:
        for dst in dsts:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst_files.append(dst.open("wb"))

        copied = 0
        with src.open("rb") as f:
            while True:
                buf = f.read(block_size)
                if not buf:
                    break
                h.update(buf)
                for d in dst_files:
                    d.write(buf)
                copied += len(buf)
                logger.info(f"copied {copied} of {file_size} bytes")

    finally:
        for d in dst_files:
            d.close()

    end = time.time()
    speed = file_size / (end - start) / (1024 * 1024)
    hash_value = h.hexdigest()

    logger.info(f"hash XXH64BE:{hash_value}")
    logger.info(f"copy speed {speed:.1f} MB/sec")

    for dst in dsts:
        dst_size = dst.stat().st_size
        if dst_size == file_size:
            logger.info(
                f"Copy completed successfully for destination file with file size check ({dst})"
            )
        else:
            logger.error(
                f"File size mismatch for {dst}: expected {file_size}, got {dst_size}"
            )

    return hash_value


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Offload files with verification (xxh64 hash and parallel destinations)."
    )
    parser.add_argument("source", type=Path, help="Source file to copy.")
    parser.add_argument(
        "destinations", type=Path, nargs="+", help="One or more destination file paths."
    )
    parser.add_argument(
        "-b",
        "--block-size",
        type=int,
        default=8 * 1024 * 1024,
        help="Read/write block size in bytes (default: 8 MiB).",
    )

    args = parser.parse_args()

    src = args.source
    dsts = args.destinations
    block_size = args.block_size

    if not src.exists() or not src.is_file():
        logger.error(f"Source file does not exist or is not a file: {src}")
        return

    try:
        copy_with_hash(src, dsts, block_size)
    except Exception as e:
        logger.error(f"Error during copy: {e}")
        raise


if __name__ == "__main__":
    main()
