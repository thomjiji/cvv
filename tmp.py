import argparse
from pathlib import Path

import xxhash


def calculate_hash(file_path: Path, chunk_size=8192):
    """Calculates and returns the xxhash64 checksum of a file."""
    h = xxhash.xxh64()
    try:
        with file_path.open("rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None


def secure_copy_file(src: Path, dst: Path, verify_source: bool, chunk_size=8192):
    """
    Securely copies a file with multiple verification steps.
    """
    print(f"--- Starting copy for: {src} ---")

    # 1. Initial source hash
    src_hash_1 = calculate_hash(src)
    if not src_hash_1:
        return
    print(f"1. Initial source hash: {src_hash_1}")

    # 2. Copy file and get destination hash
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as fsrc, dst.open("wb") as fdst:
            while chunk := fsrc.read(chunk_size):
                fdst.write(chunk)
    except OSError as e:
        print(f"Error copying file from {src} to {dst}: {e}")
        return

    dst_hash = calculate_hash(dst)
    if not dst_hash:
        return
    print(f"2. Destination hash:    {dst_hash}")

    # 3. Destination Verification
    if src_hash_1 != dst_hash:
        print(f"!!! VERIFICATION FAILED: Hashes do not match for {src} and {dst}")
        return
    else:
        print("✅ Destination verification successful.")

    # 4. Source Verification (optional)
    if verify_source:
        print("--- Performing source verification ---")
        src_hash_2 = calculate_hash(src)
        if not src_hash_2:
            return
        print(f"3. Second source hash:  {src_hash_2}")
        if src_hash_1 != src_hash_2:
            print(f"!!! SOURCE VERIFICATION FAILED: Source file {src} may be unstable.")
            return
        else:
            print("✅ Source verification successful.")

    print(f"--- Successfully copied '{src}' to '{dst}' ---\n")


def process_paths(src_path: Path, dst_path: Path, verify_source: bool):
    """
    Processes the source and destination paths, copying files or directories.
    """
    if not src_path.exists():
        print(f"Error: Source path '{src_path}' does not exist.")
        return

    if src_path.is_file():
        final_dst = dst_path / src_path.name if dst_path.is_dir() else dst_path
        secure_copy_file(src_path, final_dst, verify_source)
    elif src_path.is_dir():
        for src_file in sorted(src_path.rglob("*")):
            if src_file.is_file():
                relative_path = src_file.relative_to(src_path)
                dest_file = dst_path / relative_path
                secure_copy_file(src_file, dest_file, verify_source)


def main():
    parser = argparse.ArgumentParser(
        description="Copy files or directories with xxhash64 checksum verification."
    )
    parser.add_argument("source", type=Path, help="Source file or directory path.")
    parser.add_argument(
        "destination", type=Path, help="Destination file or directory path."
    )
    parser.add_argument(
        "--verify-source",
        action="store_true",
        help="Enable source verification by re-reading the source file after copy.",
    )
    args = parser.parse_args()

    process_paths(args.source, args.destination, args.verify_source)


if __name__ == "__main__":
    main()
