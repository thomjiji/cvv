import argparse
from pathlib import Path

import xxhash


def copy_and_hash(src: Path, dst: Path, chunk_size=8192):
    """
    Copies a file from src to dst and returns its xxhash64 checksum.
    """
    h = xxhash.xxh64()
    try:
        # Ensure the destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as fsrc, dst.open("wb") as fdst:
            while chunk := fsrc.read(chunk_size):
                fdst.write(chunk)
                h.update(chunk)
        checksum = h.hexdigest()
        print(f"Copied '{src}' to '{dst}'\nxxhash64: {checksum}")
        return checksum
    except OSError as e:
        print(f"Error copying file {src}: {e}")
        return None


def process_paths(src_path: Path, dst_path: Path):
    """
    Processes the source and destination paths, copying files or directories.
    """
    if not src_path.exists():
        print(f"Error: Source path '{src_path}' does not exist.")
        return

    if src_path.is_file():
        # If destination is a directory, copy file inside it, otherwise use the name
        final_dst = dst_path / src_path.name if dst_path.is_dir() else dst_path
        copy_and_hash(src_path, final_dst)
    elif src_path.is_dir():
        # Recursively walk the source directory
        for src_file in src_path.rglob("*"):
            if src_file.is_file():
                # Create the corresponding destination path
                relative_path = src_file.relative_to(src_path)
                dest_file = dst_path / relative_path
                copy_and_hash(src_file, dest_file)


def main():
    parser = argparse.ArgumentParser(
        description="Copy files or directories and calculate xxhash64 checksums."
    )
    parser.add_argument("source", type=Path, help="Source file or directory path.")
    parser.add_argument(
        "destination", type=Path, help="Destination file or directory path."
    )
    args = parser.parse_args()

    process_paths(args.source, args.destination)


if __name__ == "__main__":
    main()
