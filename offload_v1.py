import argparse
import contextlib
from pathlib import Path
from string import printable

import xxhash


def calculate_hash(file_path: Path, chunk_size: int = 8192):
    """Calculates and returns the xxhash64 checksum of a file."""
    h = xxhash.xxh64()
    try:
        with file_path.open("rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
                print(h.hexdigest())
        return h.hexdigest()
    except OSError as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None


def secure_copy_file(
    src: Path, dsts: list[Path], verify_source: bool, chunk_size: int = 8192
):
    """
    Securely copies a file to multiple destinations, reading the source only once.
    """
    print(f"--- Starting copy for: {src} to {len(dsts)} destination(s) ---")

    # 1. Initial source hash
    src_hash_1 = calculate_hash(src)
    if not src_hash_1:
        return
    print(f"1. Initial source hash: {src_hash_1}")

    # 2. Copy file to all destinations, reading source only once
    try:
        with contextlib.ExitStack() as stack:
            # Open source file for reading
            fsrc = stack.enter_context(src.open("rb"))

            # Create parent directories and open all destination files for writing
            fdsts = []
            for dst in dsts:
                dst.parent.mkdir(parents=True, exist_ok=True)
                fdsts.append(stack.enter_context(dst.open("wb")))

            # Read from source and write to all destinations
            while chunk := fsrc.read(chunk_size):
                for fdst in fdsts:
                    fdst.write(chunk)
        print(f"✅ File data written to {len(dsts)} destination(s).")

    except OSError as e:
        print(f"!!! FATAL ERROR during copy: {e}")
        return

    # 3. Destination Verification for each destination
    all_dest_ok = True
    for i, dst in enumerate(dsts, 1):
        print(f"--- Verifying destination {i}/{len(dsts)}: {dst} ---")
        dst_hash = calculate_hash(dst)
        if not dst_hash:
            all_dest_ok = False
            continue
        print(f"2. Destination hash:    {dst_hash}")

        if src_hash_1 != dst_hash:
            all_dest_ok = False
            print(f"!!! VERIFICATION FAILED: Hashes do not match for {dst}")
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
        else:
            print("✅ Source verification successful.")

    if all_dest_ok:
        print(f"--- Successfully copied '{src}' to all destinations ---\n")
    else:
        print(f"--- Copy for '{src}' finished with one or more errors ---\n")


def process_paths(src_path: Path, dst_paths: list[Path], verify_source: bool):
    """
    Processes the source and destination paths, copying files or directories.
    """
    if not src_path.exists():
        print(f"Error: Source path '{src_path}' does not exist.")
        return

    if src_path.is_file():
        final_dsts = [dst / src_path.name if dst.is_dir() else dst for dst in dst_paths]
        secure_copy_file(src_path, final_dsts, verify_source)
    elif src_path.is_dir():
        for src_file in sorted(src_path.rglob("*")):
            if src_file.is_file():
                relative_path = src_file.relative_to(src_path)
                dest_files = [dst / relative_path for dst in dst_paths]
                secure_copy_file(src_file, dest_files, verify_source)


def main():
    parser = argparse.ArgumentParser(
        description="Copy a source to multiple destinations with xxhash64 verification."
    )
    parser.add_argument("source", type=Path, help="Source file or directory path.")
    parser.add_argument(
        "destination",
        type=Path,
        nargs="+",
        help="One or more destination file or directory paths.",
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
