#!/usr/bin/env python3
"""
cvv - Professional file copying tool with integrity verification.

This tool is designed to provide reliable file copying with hash verification,
progress monitoring, and multi-destination support, emulating best practices
from professional DIT (Digital Imaging Technician) software.
"""

import argparse
import contextlib
import hashlib
import logging
import queue
import shutil
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

try:
    import xxhash
except ImportError:
    logging.error("The 'xxhash' library is required but not installed.")
    logging.error("Please install it using: pip install xxhash")
    sys.exit(1)

# Constants
BUFFER_SIZE = 8 * 1024 * 1024  # 8MB


class CopyJobError(Exception):
    """Custom exception for errors during a copy job."""

    pass


@dataclass
class CopyConfig:
    """Configuration for file copy operations."""

    buffer_size: int = BUFFER_SIZE
    paranoid_verification: bool = False  # Verify destination hashes after copy
    source_verification_hash: str = "xxh64be"


@dataclass
class FileCopyResult:
    """Result of a single file copy operation."""

    source_path: Path
    destinations: dict[Path, bool]
    source_hash: str | None = None
    destination_hashes: dict[Path, str] = field(default_factory=dict)
    size: int = 0
    duration: float = 0.0
    speed_mb_sec: float = 0.0
    verified: bool = True
    error: str | None = None


class ProgressTracker:
    """Track and report copy progress."""

    def __init__(
        self, total_size: int, source_path: Path, callback: Callable | None = None
    ):
        self.total_size = total_size
        self.source_path = source_path
        self.callback = callback
        self.bytes_copied = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def update(self, chunk_size: int):
        with self.lock:
            self.bytes_copied += chunk_size
            if self.callback:
                self.callback(self.bytes_copied, self.total_size)

    def get_stats(self) -> tuple[float, float]:
        duration = time.time() - self.start_time
        speed = (self.bytes_copied / (1024 * 1024)) / duration if duration > 0 else 0
        return duration, speed


class HashCalculator:
    """Calculate file hashes."""

    def __init__(self, algorithm: str = "xxh64be"):
        self.algorithm = algorithm.lower()
        if self.algorithm not in ["xxh64be", "md5", "sha1", "sha256"]:
            raise ValueError(f"Unsupported hash algorithm: {self.algorithm}")
        self.hasher = self._get_hasher()

    def _get_hasher(self):
        if self.algorithm == "xxh64be":
            return xxhash.xxh64()
        return hashlib.new(self.algorithm)

    def update(self, data: bytes):
        self.hasher.update(data)

    def hexdigest(self) -> str:
        return self.hasher.hexdigest()

    @staticmethod
    def hash_file(
        file_path: Path, algorithm: str = "xxh64be", buffer_size: int = BUFFER_SIZE
    ) -> str:
        hasher = HashCalculator(algorithm)
        with open(file_path, "rb") as f:
            while chunk := f.read(buffer_size):
                hasher.update(chunk)
        return hasher.hexdigest()


class CopyJob:
    """Manages a single file copy operation from one source to multiple destinations."""

    def __init__(self, config: CopyConfig | None = None):
        self._source: Path | None = None
        self._destinations: list[Path] = []
        self.config = config or CopyConfig()
        self._progress_callback: Callable | None = None
        self._abort_event = threading.Event()

    def source(self, path: Path) -> "CopyJob":
        self._source = path
        return self

    def add_destination(self, path: Path) -> "CopyJob":
        self._destinations.append(path)
        return self

    def on_progress(self, callback: Callable) -> "CopyJob":
        self._progress_callback = callback
        return self

    def abort(self):
        """Signals the copy operation to abort."""
        logging.warning("Abort signal received. Attempting to stop...")
        self._abort_event.set()

    def execute(self) -> FileCopyResult:
        """Executes the configured copy job."""
        if not self._source or not self._destinations:
            raise CopyJobError("Source and at least one destination must be set.")

        start_time = time.time()
        source_size = self._source.stat().st_size
        result = FileCopyResult(
            source_path=self._source, destinations={}, size=source_size
        )

        try:
            self._check_disk_space(source_size)
            self._prepare_destination_dirs()

            source_hash = self._run_pipeline(source_size)
            result.source_hash = source_hash

            self._compare_sizes(source_size)

            if self.config.paranoid_verification:
                verified, dest_hashes = self._verify_destinations(source_hash)
                result.verified = verified
                result.destination_hashes = dest_hashes
                if not verified:
                    raise CopyJobError(
                        "Paranoid verification failed: Hashes do not match."
                    )

            result.destinations = {dest: True for dest in self._destinations}
            result.verified = True

        except (CopyJobError, OSError) as e:
            logging.error(f"Failed to copy {self._source.name}: {e}")
            result.error = str(e)
            result.verified = False
            # Mark all destinations as failed
            for dest in self._destinations:
                result.destinations[dest] = False
        finally:
            duration = time.time() - start_time
            result.duration = duration
            result.speed_mb_sec = (
                (source_size / (1024 * 1024) / duration) if duration > 0 else 0
            )

        return result

    def _check_disk_space(self, source_size: int):
        """Checks if all destination volumes have enough space."""
        logging.info("Checking disk space...")
        for dest in self._destinations:
            dest_parent = dest.parent
            dest_parent.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(dest_parent)
            if usage.free < source_size:
                raise CopyJobError(
                    f"Not enough space on {dest_parent}. "
                    f"Required: {source_size / 1e9:.2f} GB, "
                    f"Available: {usage.free / 1e9:.2f} GB"
                )

    def _prepare_destination_dirs(self):
        """Creates parent directories for all destination files."""
        for dest in self._destinations:
            dest.parent.mkdir(parents=True, exist_ok=True)

    def _run_pipeline(self, source_size: int) -> str:
        """Sets up and runs the stream pipeline for copy and hashing."""
        source_hasher = HashCalculator(self.config.source_verification_hash)
        progress = ProgressTracker(source_size, self._source, self._progress_callback)

        writer_threads: list[threading.Thread] = []
        chunk_queue = queue.Queue(
            maxsize=10
        )  # Bounded queue to prevent high memory usage

        # Setup and start writer threads
        for dest_path in self._destinations:
            thread = threading.Thread(
                target=self._writer_thread, args=(dest_path, chunk_queue)
            )
            thread.start()
            writer_threads.append(thread)

        try:
            # Reader thread
            with open(self._source, "rb") as f_src:
                while True:
                    if self._abort_event.is_set():
                        raise CopyJobError("Operation aborted by user.")

                    chunk = f_src.read(self.config.buffer_size)
                    if not chunk:
                        break

                    source_hasher.update(chunk)
                    chunk_queue.put(chunk)
                    progress.update(len(chunk))
        finally:
            # Signal writers to finish
            for _ in writer_threads:
                chunk_queue.put(None)

            # Wait for all writers to complete
            for t in writer_threads:
                t.join()

        return source_hasher.hexdigest()

    def _writer_thread(self, dest_path: Path, chunk_queue: queue.Queue):
        """Target for writer threads. Writes chunks from queue to a destination file."""
        temp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp")
        try:
            with open(temp_path, "wb") as f_dest:
                while True:
                    chunk = chunk_queue.get()
                    if chunk is None:
                        break
                    if self._abort_event.is_set():
                        break
                    f_dest.write(chunk)

            if not self._abort_event.is_set():
                shutil.move(temp_path, dest_path)

        except OSError as e:
            logging.error(f"Error writing to {dest_path}: {e}")
            self._abort_event.set()  # Signal other threads to stop
        finally:
            # Cleanup temp file if it still exists
            if temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()

    def _compare_sizes(self, source_size: int):
        """Compares source and destination file sizes."""
        logging.info("Verifying file sizes...")
        for dest in self._destinations:
            dest_size = dest.stat().st_size
            if source_size != dest_size:
                raise CopyJobError(
                    f"File size mismatch for {dest.name}. "
                    f"Source: {source_size}, Destination: {dest_size}"
                )
        logging.info("File sizes match.")

    def _verify_destinations(self, source_hash: str) -> tuple[bool, dict[Path, str]]:
        """Hashes all destination files and compares to the source hash."""
        logging.info("Paranoid verification enabled. Hashing destination files...")
        all_match = True
        dest_hashes: dict[Path, str] = {}
        for dest in self._destinations:
            if self._abort_event.is_set():
                raise CopyJobError("Verification aborted.")

            logging.info(f"Hashing {dest.name}...")
            dest_hash = HashCalculator.hash_file(
                dest, self.config.source_verification_hash
            )
            dest_hashes[dest] = dest_hash
            if dest_hash != source_hash:
                all_match = False
                logging.error(
                    f"HASH MISMATCH: {dest.name} hash {dest_hash} != source hash {source_hash}"
                )

        if all_match:
            logging.info("All destination hashes match the source hash.")

        return all_match, dest_hashes


def discover_files(source: Path) -> list[Path]:
    """Discovers all files in a source path, whether it's a file or directory."""
    if source.is_file():
        return [source]
    if source.is_dir():
        return [f for f in source.rglob("*") if f.is_file()]
    raise FileNotFoundError(
        f"Source path {source} does not exist or is not a file/directory."
    )


def main():
    """Main function for the cvv command-line tool."""
    parser = argparse.ArgumentParser(description="Professional file copying tool.")
    parser.add_argument("source", type=Path, help="Source file or directory.")
    parser.add_argument(
        "destinations",
        type=Path,
        nargs="+",
        help="One or more destination files or directories.",
    )
    parser.add_argument(
        "-p",
        "--paranoid",
        action="store_true",
        help="Verify destination file hashes after copy.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        source_files = discover_files(args.source)
        total_files = len(source_files)
        logging.info(f"Found {total_files} file(s) to copy.")

        config = CopyConfig(paranoid_verification=args.paranoid)

        overall_success = True
        for i, file_path in enumerate(source_files):
            logging.info("-" * 60)
            logging.info(f"Copying file {i + 1}/{total_files}: {file_path.name}")

            # Determine destination paths
            dest_paths = []
            if args.source.is_dir():
                relative_path = file_path.relative_to(args.source)
                for dest_root in args.destinations:
                    dest_paths.append(dest_root / relative_path)
            else:  # source is a file
                # If one dest is given and it's a dir, place file inside it
                if len(args.destinations) == 1 and args.destinations[0].is_dir():
                    dest_paths.append(args.destinations[0] / file_path.name)
                else:
                    dest_paths = args.destinations

            job = CopyJob(config).source(file_path)
            for dest in dest_paths:
                job.add_destination(dest)

            # Simple progress bar callback
            def progress_callback(copied, total):
                percent = (copied / total * 100) if total > 0 else 0
                bar = "█" * int(percent / 2) + "-" * (50 - int(percent / 2))
                sys.stdout.write(f"\rProgress: |{bar}| {percent:.2f}%")
                sys.stdout.flush()

            job.on_progress(progress_callback)

            result = job.execute()
            sys.stdout.write("\n")  # Newline after progress bar

            if result.error:
                logging.error(f"Result for {file_path.name}: FAILED")
                overall_success = False
            else:
                logging.info(f"Result for {file_path.name}: SUCCESS")
                logging.info(f"  - Size: {result.size / 1e6:.2f} MB")
                logging.info(f"  - Speed: {result.speed_mb_sec:.2f} MB/s")
                logging.info(
                    f"  - Source Hash ({config.source_verification_hash}): {result.source_hash}"
                )
                if result.verified:
                    logging.info("  - Verification: PASSED")

        logging.info("=" * 60)
        if overall_success:
            logging.info("All copy operations completed successfully.")
            return 0
        else:
            logging.error("One or more copy operations failed.")
            return 1

    except (CopyJobError, FileNotFoundError) as e:
        logging.error(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        logging.warning("\nOperation interrupted by user.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
