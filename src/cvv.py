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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    import xxhash
except ImportError:
    logging.error("The 'xxhash' library is required but not installed.")
    logging.error("Please install it using: pip install xxhash")
    sys.exit(1)

# Constants
BUFFER_SIZE = 8 * 1024 * 1024  # 8MB


class VerificationMode(Enum):
    """Defines the verification strategy for the copy operation."""

    # fmt: off
    TRANSFER = "transfer"  # Verify by file size only.
    SOURCE = "source"      # Verify source read integrity and destination file sizes.
    FULL = "full"          # Verify source and all destinations with checksums.
    # fmt: on


class CopyJobError(Exception):
    """Custom exception for errors during a copy job."""

    pass


@dataclass
class CopyConfig:
    """Configuration for file copy operations."""

    buffer_size: int = BUFFER_SIZE
    verification_mode: VerificationMode = VerificationMode.FULL
    hash_algorithm: str = "xxh64be"


@dataclass
class FileCopyResult:
    """Result of a single file copy operation."""

    source_path: Path
    destinations: dict[Path, bool] = field(default_factory=dict)
    source_hash_inflight: str | None = None
    source_hash_post: str | None = None
    destination_hashes_post: dict[Path, str] = field(default_factory=dict)
    size: int = 0
    duration: float = 0.0
    speed_mb_sec: float = 0.0
    verified: bool = True
    error: str | None = None


class ProgressTracker:
    """Track and report copy progress."""

    def __init__(self, total_size: int, callback: Callable | None = None):
        self.total_size = total_size
        self.callback = callback
        self.bytes_copied = 0
        self.lock = threading.Lock()

    def update(self, chunk_size: int):
        with self.lock:
            self.bytes_copied += chunk_size
            if self.callback:
                self.callback(self.bytes_copied, self.total_size)


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
    def hash_file(file_path: Path, algorithm: str, buffer_size: int) -> str:
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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="CopyJob")

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
        self._abort_event.set()

    def execute(self) -> Future[FileCopyResult]:
        """Executes the copy job asynchronously and returns a Future."""
        if not self._source or not self._destinations:
            raise CopyJobError("Source and at least one destination must be set.")
        return self._executor.submit(self._run_and_get_result)

    def _run_and_get_result(self) -> FileCopyResult:
        """The actual workhorse method that runs in a background thread."""
        start_time = time.time()
        source_size = self._source.stat().st_size
        result = FileCopyResult(source_path=self._source, size=source_size)

        try:
            self._check_disk_space(source_size)
            self._prepare_destination_dirs()

            if self.config.verification_mode != VerificationMode.TRANSFER:
                result.source_hash_inflight = self._run_pipeline(
                    source_size, enable_hashing=True
                )
            else:
                self._run_pipeline(source_size, enable_hashing=False)

            self._compare_sizes(source_size)
            self._perform_post_copy_verification(result)

            result.destinations = {dest: True for dest in self._destinations}

        except (CopyJobError, OSError) as e:
            logging.error(f"Failed to copy {self._source.name}: {e}")
            result.error = str(e)
            result.verified = False
            for dest in self._destinations:
                result.destinations[dest] = False
        finally:
            duration = time.time() - start_time
            result.duration = duration
            result.speed_mb_sec = (
                (source_size / (1024 * 1024) / duration) if duration > 0 else 0
            )
            self._executor.shutdown(wait=False)

        return result

    def _perform_post_copy_verification(self, result: FileCopyResult):
        mode = self.config.verification_mode
        logging.info(f"Starting '{mode.value}' verification...")

        if mode == VerificationMode.TRANSFER:
            logging.info("File sizes verified successfully.")
            return

        if mode == VerificationMode.SOURCE:
            self._verify_source_post_copy(result)
            logging.info("Source integrity verified successfully.")

        elif mode == VerificationMode.FULL:
            self._verify_full_post_copy(result)
            logging.info("Source and all destinations verified successfully.")

    def _check_disk_space(self, source_size: int):
        for dest in self._destinations:
            dest.parent.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(dest.parent)
            if usage.free < source_size:
                raise CopyJobError(
                    f"Not enough space on {dest.parent}. Required: {source_size / 1e9:.2f}GB, Available: {usage.free / 1e9:.2f}GB"
                )

    def _prepare_destination_dirs(self):
        for dest in self._destinations:
            dest.parent.mkdir(parents=True, exist_ok=True)

    def _run_pipeline(self, source_size: int, enable_hashing: bool) -> str | None:
        source_hasher = (
            HashCalculator(self.config.hash_algorithm) if enable_hashing else None
        )
        progress = ProgressTracker(source_size, self._progress_callback)

        chunk_queues = [queue.Queue(maxsize=10) for _ in self._destinations]
        writer_threads = []
        for i, dest_path in enumerate(self._destinations):
            thread = threading.Thread(
                target=self._writer_thread, args=(dest_path, chunk_queues[i])
            )
            thread.start()
            writer_threads.append(thread)

        try:
            with open(self._source, "rb") as f_src:
                while True:
                    if self._abort_event.is_set():
                        raise CopyJobError("Operation aborted")
                    chunk = f_src.read(self.config.buffer_size)
                    if not chunk:
                        break
                    if source_hasher:
                        source_hasher.update(chunk)
                    progress.update(len(chunk))
                    for q in chunk_queues:
                        q.put(chunk)
        finally:
            for q in chunk_queues:
                q.put(None)
            for t in writer_threads:
                t.join()

        return source_hasher.hexdigest() if source_hasher else None

    def _writer_thread(self, dest_path: Path, chunk_queue: queue.Queue):
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
            self._abort_event.set()
        finally:
            if temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()

    def _compare_sizes(self, source_size: int):
        for dest in self._destinations:
            dest_size = dest.stat().st_size
            if source_size != dest_size:
                raise CopyJobError(
                    f"File size mismatch for {dest.name}. Source: {source_size}, Dest: {dest_size}"
                )

    def _verify_source_post_copy(self, result: FileCopyResult):
        logging.info(f"Re-hashing source file {self._source.name}...")
        post_hash = HashCalculator.hash_file(
            self._source, self.config.hash_algorithm, self.config.buffer_size
        )
        result.source_hash_post = post_hash
        if post_hash != result.source_hash_inflight:
            result.verified = False
            raise CopyJobError(
                f"Source file changed during copy. In-flight: {result.source_hash_inflight}, Post-copy: {post_hash}"
            )

    def _verify_full_post_copy(self, result: FileCopyResult):
        logging.info("Re-hashing source and all destinations in parallel...")
        files_to_hash = [self._source] + self._destinations
        hashes = {}

        with ThreadPoolExecutor(max_workers=len(files_to_hash)) as executor:
            future_to_path = {
                executor.submit(
                    HashCalculator.hash_file,
                    path,
                    self.config.hash_algorithm,
                    self.config.buffer_size,
                ): path
                for path in files_to_hash
            }
            for future in future_to_path:
                path = future_to_path[future]
                try:
                    hashes[path] = future.result()
                except Exception as e:
                    raise CopyJobError(f"Failed to hash {path.name} post-copy: {e}")

        result.source_hash_post = hashes[self._source]
        for dest in self._destinations:
            result.destination_hashes_post[dest] = hashes[dest]

        if result.source_hash_post != result.source_hash_inflight:
            result.verified = False
            raise CopyJobError(
                f"Source file changed during copy. In-flight: {result.source_hash_inflight}, Post-copy: {result.source_hash_post}"
            )

        for dest in self._destinations:
            if result.destination_hashes_post[dest] != result.source_hash_inflight:
                result.verified = False
                logging.error(
                    f"HASH MISMATCH: {dest.name} hash {result.destination_hashes_post[dest]} != source hash {result.source_hash_inflight}"
                )

        if not result.verified:
            raise CopyJobError("Full verification failed: Hashes do not match.")


class BatchProcessor:
    """Manages a batch of copy jobs from a single command-line invocation."""

    def __init__(self, args):
        self.args = args
        self.config = self._create_config()
        self.jobs: list[tuple[CopyJob, Future[FileCopyResult]]] = []

    def _create_config(self) -> CopyConfig:
        return CopyConfig(
            verification_mode=VerificationMode(self.args.mode),
            hash_algorithm=self.args.hash_algorithm,
        )

    def run(self) -> bool:
        source_files = self._discover_files()
        self._execute_jobs(source_files)
        results = self._process_results()
        return self._final_summary(results)

    def _discover_files(self) -> list[Path]:
        source = self.args.source
        if source.is_file():
            return [source]
        if source.is_dir():
            return sorted([f for f in source.rglob("*") if f.is_file()])
        raise FileNotFoundError(f"Source path {source} does not exist.")

    def _execute_jobs(self, source_files: list[Path]):
        total_files = len(source_files)
        for i, file_path in enumerate(source_files):
            logging.info("-" * 60)
            logging.info(f"Starting file {i + 1}/{total_files}: {file_path.name}")

            dest_paths = self._get_dest_paths(file_path)
            job = CopyJob(self.config).source(file_path)
            for dest in dest_paths:
                job.add_destination(dest)

            def progress_callback(copied, total):
                percent = (copied / total * 100) if total > 0 else 0
                bar = "█" * int(percent / 2) + "–" * (50 - int(percent / 2))
                sys.stdout.write(f"\rProgress: |{bar}| {percent:.2f}%")
                sys.stdout.flush()

            job.on_progress(progress_callback)
            future = job.execute()
            self.jobs.append((job, future))

    def _get_dest_paths(self, source_file: Path) -> list[Path]:
        if self.args.source.is_dir():
            relative_path = source_file.relative_to(self.args.source)
            return [dest_root / relative_path for dest_root in self.args.destinations]

        if len(self.args.destinations) == 1 and self.args.destinations[0].is_dir():
            return [self.args.destinations[0] / source_file.name]

        return self.args.destinations

    def _process_results(self) -> list[FileCopyResult]:
        results = []
        for _job, future in self.jobs:
            try:
                result = future.result()  # Blocks until this job is done
                sys.stdout.write("\n")
                if result.error:
                    logging.error(
                        f"Result for {result.source_path.name}: FAILED ({result.error})"
                    )
                else:
                    logging.info(f"Result for {result.source_path.name}: SUCCESS")
                    logging.info(
                        f"  - Speed: {result.speed_mb_sec:.2f} MB/s, Size: {result.size / 1e6:.2f} MB"
                    )
                    if result.source_hash_inflight:
                        logging.info(
                            f"  - Hash ({self.config.hash_algorithm}): {result.source_hash_inflight}"
                        )
                results.append(result)
            except Exception as e:
                logging.error(f"A job failed with an unexpected exception: {e}")
        return results

    def _final_summary(self, results: list[FileCopyResult]) -> bool:
        logging.info("=" * 60)
        success_count = sum(1 for r in results if not r.error and r.verified)
        total_count = len(results)

        if success_count == total_count:
            logging.info(f"All {total_count} operations completed successfully.")
            return True
        else:
            logging.error(
                f"{total_count - success_count} out of {total_count} operations failed."
            )
            return False


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
        "-m",
        "--mode",
        type=str,
        default="full",
        choices=["transfer", "source", "full"],
        help="Verification mode.",
    )
    parser.add_argument(
        "--hash-algorithm",
        type=str,
        default="xxh64be",
        choices=["xxh64be", "md5", "sha1", "sha256"],
        help="Hash algorithm for verification.",
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
        processor = BatchProcessor(args)
        success = processor.run()
        return 0 if success else 1
    except (CopyJobError, FileNotFoundError) as e:
        logging.error(f"Critical error: {e}")
        return 1
    except KeyboardInterrupt:
        logging.warning("\nOperation interrupted by user.")
        # TODO: Gracefully abort jobs
        return 1


if __name__ == "__main__":
    sys.exit(main())
