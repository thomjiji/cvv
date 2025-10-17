#!/usr/bin/env python3
"""
cvv - Professional file copying tool with integrity verification.

This tool implements the core functionality of Offload Manager's pfndispatchcopy,
providing reliable file copying with hash verification, progress monitoring,
and multi-destination support for professional DIT workflows.
"""

import argparse
import contextlib
import hashlib
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from io import BufferedWriter
from pathlib import Path
from typing import Any

try:
    import xxhash
except ImportError:
    logging.error("The 'xxhash' library is required but not installed.")
    logging.error("Please install it using: pip install xxhash")
    import sys

    sys.exit(1)

# Buffer size constants
BUFFER_SIZE = 8 * 1024 * 1024  # 8MB in bytes


class VerificationMode(Enum):
    """Source verification modes."""

    NONE = "none"
    PER_FILE = "per_file"
    AFTER_ALL = "after_all"


@dataclass
class VerificationResult:
    """Result of source verification for a file."""

    file_path: Path
    verified: bool
    initial_hash: str
    final_source_hash: str | None = None
    error: str | None = None


@dataclass
class CopyConfig:
    """Configuration for file copy operations."""

    buffer_size: int = BUFFER_SIZE
    source_verification: VerificationMode = VerificationMode.NONE
    source_verification_hash: str = "xxh64be"
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.buffer_size <= 0:
            raise ValueError(f"Buffer size must be positive, got {self.buffer_size}")

        valid_algorithms = ["xxh64be", "md5", "sha1", "sha256"]
        if self.source_verification_hash.lower() not in valid_algorithms:
            raise ValueError(f"Invalid hash algorithm: {self.source_verification_hash}")
        self.source_verification_hash = self.source_verification_hash.lower()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CopyConfig":
        """Create config from command-line arguments."""
        verify_mode_map = {
            "none": VerificationMode.NONE,
            "per_file": VerificationMode.PER_FILE,
            "after_all": VerificationMode.AFTER_ALL,
        }
        source_verify = verify_mode_map.get(args.source_verify, VerificationMode.NONE)

        return cls(
            buffer_size=BUFFER_SIZE,
            source_verification=source_verify,
            source_verification_hash=args.source_verify_hash or "xxh64be",
            verbose=args.verbose,
        )


class SourceVerificationManager:
    """Unified manager for all source verification workflows."""

    def __init__(self, config: CopyConfig) -> None:
        """
        Initialize source verification manager.

        Parameters
        ----------
        config : CopyConfig
            Configuration containing verification settings
        """
        self.config = config
        self.mode = config.source_verification
        self.hash_algorithm = config.source_verification_hash
        self.buffer_size = config.buffer_size
        self.initial_hashes: dict[Path, str] = {}
        self.verification_results: dict[Path, VerificationResult] = {}
        self.lock = threading.Lock()

    def should_verify(self) -> bool:
        """
        Check if any verification is enabled.

        Returns
        -------
        bool
            True if verification is enabled
        """
        return self.mode != VerificationMode.NONE

    def should_verify_immediately(self) -> bool:
        """
        Check if we should verify immediately after each file.

        Returns
        -------
        bool
            True if PER_FILE mode is enabled
        """
        return self.mode == VerificationMode.PER_FILE

    def should_verify_at_end(self) -> bool:
        """
        Check if we should verify at the end of all operations.

        Returns
        -------
        bool
            True if AFTER_ALL mode is enabled
        """
        return self.mode == VerificationMode.AFTER_ALL

    def store_initial_hash(self, file_path: Path, hash_value: str) -> None:
        """
        Store initial hash for a file.

        Parameters
        ----------
        file_path : Path
            Path to the file
        hash_value : str
            Initial hash value
        """
        with self.lock:
            self.initial_hashes[file_path] = hash_value

    def prepare_file_for_verification(self, file_path: Path) -> None:
        """
        Calculate and store initial hash for a file.

        Parameters
        ----------
        file_path : Path
            Path to the file to prepare for verification
        """
        if not self.should_verify():
            return

        hash_calc = HashCalculator(self.hash_algorithm)
        initial_hash = hash_calc.calculate(file_path, self.buffer_size)
        self.store_initial_hash(file_path, initial_hash)

    def verify_source(
        self, file_path: Path, buffer_size: int | None = None
    ) -> VerificationResult:
        """
        Verify source file by recalculating its hash.

        Parameters
        ----------
        file_path : Path
            Path to source file
        buffer_size : int | None
            Buffer size for reading (uses config default if None)

        Returns
        -------
        VerificationResult
            Verification result
        """
        if buffer_size is None:
            buffer_size = self.buffer_size

        if file_path not in self.initial_hashes:
            return VerificationResult(
                file_path=file_path,
                verified=False,
                initial_hash="",
                error="No initial hash stored",
            )

        initial_hash = self.initial_hashes[file_path]

        try:
            hash_calc = HashCalculator(self.hash_algorithm)
            final_hash = hash_calc.calculate(file_path, buffer_size)

            verified = initial_hash == final_hash

            result = VerificationResult(
                file_path=file_path,
                verified=verified,
                initial_hash=initial_hash,
                final_source_hash=final_hash,
                error=None if verified else "Hash mismatch - source file changed",
            )

        except Exception as e:
            result = VerificationResult(
                file_path=file_path,
                verified=False,
                initial_hash=initial_hash,
                final_source_hash=None,
                error=str(e),
            )

        with self.lock:
            self.verification_results[file_path] = result

        return result

    def verify_file_immediately(self, file_path: Path) -> VerificationResult | None:
        """
        Perform immediate verification (PER_FILE mode).

        Parameters
        ----------
        file_path : Path
            Path to the file to verify

        Returns
        -------
        VerificationResult | None
            Verification result or None if verification is not enabled
        """
        if not self.should_verify():
            return None

        logging.info("")
        logging.info("=" * 60)
        logging.info("Starting source verification...")
        logging.info("=" * 60)

        verify_result = self.verify_source(file_path, self.buffer_size)

        if verify_result.verified:
            logging.info(f"✓ Source verified: {file_path.name}")
            logging.info("")
            logging.info("Source verification summary:")
            logging.info("  Verified: 1")
            logging.info("  Failed: 0")
        else:
            logging.error(
                f"✗ Source verification FAILED: {file_path.name} - "
                f"{verify_result.error}"
            )
            logging.info("")
            logging.info("Source verification summary:")
            logging.info("  Verified: 0")
            logging.info("  Failed: 1")

        return verify_result

    def verify_all_sources(self) -> dict[Path, VerificationResult]:
        """
        Verify all stored source files.

        Returns
        -------
        dict[Path, VerificationResult]
            Verification results for all files
        """
        results = {}

        for file_path in self.initial_hashes:
            result = self.verify_source(file_path, self.buffer_size)
            results[file_path] = result

        return results

    def verify_all_pending(
        self, source_root: Path | None = None
    ) -> dict[Path, VerificationResult]:
        """
        Verify all pending files (AFTER_ALL mode).

        Parameters
        ----------
        source_root : Path | None
            Root path for relative path calculation in logging

        Returns
        -------
        dict[Path, VerificationResult]
            Verification results for all files
        """
        if not self.should_verify():
            return {}

        logging.info("")
        logging.info("=" * 60)
        logging.info("Starting source verification for all files...")
        logging.info("=" * 60)

        verification_results = self.verify_all_sources()

        for source_file, verify_result in verification_results.items():
            if source_root and source_root.is_dir():
                rel_path = source_file.relative_to(source_root)
            else:
                rel_path = source_file.name

            if verify_result.verified:
                logging.info(f"✓ Source verified: {rel_path}")
            else:
                logging.error(
                    f"✗ Source verification FAILED: {rel_path} - {verify_result.error}"
                )

        summary = self.get_summary()
        logging.info("")
        logging.info("Source verification summary:")
        logging.info(f"  Verified: {summary['verified']}")
        logging.info(f"  Failed: {summary['failed']}")
        logging.info(f"  Pending: {summary['pending']}")

        return verification_results

    def get_summary(self) -> dict[str, int]:
        """
        Get verification summary statistics.

        Returns
        -------
        dict[str, int]
            Summary with counts of verified, failed, and pending
        """
        verified = sum(1 for r in self.verification_results.values() if r.verified)
        failed = sum(1 for r in self.verification_results.values() if not r.verified)
        pending = len(self.initial_hashes) - len(self.verification_results)

        return {
            "verified": verified,
            "failed": failed,
            "pending": pending,
            "total": len(self.initial_hashes),
        }


class ProgressTracker:
    """Track and report copy progress across multiple threads."""

    def __init__(
        self, total_size: int, source_path: Path, update_interval: float = 2.0
    ) -> None:
        """
        Initialize progress tracker.

        Parameters
        ----------
        total_size : int
            Total file size in bytes
        source_path : Path
            Path to the source file
        update_interval : float
            Interval between progress updates in seconds
        """
        self.total_size: int = total_size
        self.source_path: Path = source_path
        self.update_interval: float = update_interval
        self.bytes_copied: int = 0
        self.start_time: float = 0.0
        self.last_update = 0
        self.last_percentage = 0
        self.min_bytes_threshold = max(
            1024 * 1024 * 10, total_size // 50
        )  # 10MB or 2% of file
        self.lock = threading.Lock()

    def start(self) -> None:
        """Log the start of the copy operation."""
        logging.info(f"copying {self.source_path}...")
        self.start_time = time.time()

    def update(self, bytes_copied: int) -> None:
        """
        Update progress and log if needed.

        Parameters
        ----------
        bytes_copied : int
            Total bytes copied so far
        """
        with self.lock:
            self.bytes_copied = bytes_copied
            current_time = time.time()
            current_percentage = (
                (bytes_copied / self.total_size * 100) if self.total_size > 0 else 0
            )

            # Update if enough time has passed AND (enough bytes copied OR significant
            # percentage change)
            time_threshold = current_time - self.last_update >= self.update_interval
            bytes_threshold = (
                bytes_copied - (self.total_size * self.last_percentage / 100)
                >= self.min_bytes_threshold
            )
            percentage_threshold = (
                current_percentage - self.last_percentage >= 5
            )  # Every 5%

            if time_threshold and (bytes_threshold or percentage_threshold):
                logging.info(
                    f"copied {bytes_copied} of {self.total_size} bytes "
                    f"({current_percentage:.1f}%)"
                )
                self.last_update = current_time
                self.last_percentage = current_percentage

    def get_stats(self) -> tuple[float, float]:
        """
        Get copy statistics.

        Returns
        -------
        Tuple[float, float]
            Duration in seconds and speed in MB/sec
        """
        duration = time.time() - self.start_time
        speed_mb_sec = (
            (self.total_size / (1024 * 1024)) / duration if duration > 0 else 0
        )
        return duration, speed_mb_sec

    def finish(self, file_hash: str | None = None) -> None:
        """Log the completion of the copy operation and final stats."""
        duration, speed_mb_sec = self.get_stats()
        logging.info(
            f"copy speed {self.total_size} bytes in {duration:.5f} sec ({speed_mb_sec:.1f} MB/sec)"
        )
        if file_hash:
            logging.info(f"hash XXH64BE:{file_hash}")
        logging.info("done.")


class HashCalculator:
    """Calculate file hashes for integrity verification."""

    def __init__(self, algorithm: str = "xxh64be") -> None:
        """
        Initialize hash calculator.

        Parameters
        ----------
        algorithm : str
            Hash algorithm to use ('xxh64be', 'md5', 'sha1', 'sha256')

        Raises
        ------
        ValueError
            If unsupported hash algorithm is specified
        """
        self.algorithm = algorithm.lower()
        if self.algorithm not in ["xxh64be", "md5", "sha1", "sha256"]:
            raise ValueError(f"Unsupported hash algorithm: {self.algorithm}")

    def calculate(self, file_path: Path, buffer_size: int = BUFFER_SIZE) -> str:
        """
        Calculate hash of file.

        Parameters
        ----------
        file_path : Path
            Path to file to hash
        buffer_size : int
            Buffer size for reading file

        Returns
        -------
        str
            Hexadecimal hash digest

        Raises
        ------
        ValueError
            If unsupported hash algorithm
        FileNotFoundError
            If file doesn't exist
        """
        if self.algorithm == "xxh64be":
            hasher = xxhash.xxh64()
        elif self.algorithm == "md5":
            hasher = hashlib.md5()
        elif self.algorithm == "sha1":
            hasher = hashlib.sha1()
        elif self.algorithm == "sha256":
            hasher = hashlib.sha256()

        with open(file_path, "rb") as f:
            while chunk := f.read(buffer_size):
                hasher.update(chunk)

        return hasher.hexdigest()


@dataclass
class ChunkData:
    """Data structure for passing chunks between threads."""

    chunk_id: int
    data: bytes


class ParallelWriter:
    """Handles parallel writing to multiple destinations."""

    def __init__(
        self,
        destinations: list[Path],
        temp_files: dict[Path, Path],
        progress_tracker: ProgressTracker,
        config: CopyConfig,
    ) -> None:
        """Initialize parallel writer."""
        self.destinations = destinations
        self.temp_files = temp_files
        self.progress_tracker = progress_tracker
        self.config = config
        self.chunk_queues = {dest: queue.Queue() for dest in destinations}
        self.file_handles: dict[Path, BufferedWriter] = {}
        self.write_errors = {}
        self.chunks_written = {dest: 0 for dest in destinations}
        self.total_bytes_written = 0
        self.write_lock = threading.Lock()
        self.exit_stack = contextlib.ExitStack()

        # Initialize hash calculator - always use xxh64be for data stream hashing
        self.hasher = xxhash.xxh64()

    def __enter__(self) -> "ParallelWriter":
        """Enter context manager - open all destination files for writing."""
        for dest in self.destinations:
            # Use ExitStack to manage multiple file context managers
            file_handle = self.exit_stack.enter_context(
                open(self.temp_files[dest], "wb")
            )
            self.file_handles[dest] = file_handle
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - close all destination files."""
        self.exit_stack.close()

    def open_files(self) -> None:
        """
        Open all destination files for writing using context managers.

        Deprecated: Use ParallelWriter as a context manager instead.
        This method is kept for backward compatibility.
        """
        self.__enter__()

    def close_files(self) -> None:
        """
        Close all destination files using the exit stack.

        Deprecated: Use ParallelWriter as a context manager instead.
        This method is kept for backward compatibility.
        """
        self.exit_stack.close()

    def writer_thread(self, destination: Path) -> None:
        """
        Writer thread function for a single destination.

        Parameters
        ----------
        destination : Path
            The destination this thread is responsible for
        """
        try:
            chunk_queue = self.chunk_queues[destination]
            handle = self.file_handles[destination]

            while True:
                try:
                    chunk = chunk_queue.get(timeout=1.0)
                    if chunk is None:  # Sentinel to stop
                        break

                    # Write chunk
                    handle.write(chunk.data)

                    # Update progress (thread-safe)
                    with self.write_lock:
                        self.chunks_written[destination] += 1
                        # Only update progress when all destinations have
                        # written this chunk
                        min_chunks = min(self.chunks_written.values())
                        expected_bytes = min_chunks * len(chunk.data)
                        if expected_bytes > self.total_bytes_written:
                            self.total_bytes_written = expected_bytes
                            self.progress_tracker.update(self.total_bytes_written)

                    chunk_queue.task_done()

                except queue.Empty:
                    continue

        except Exception as e:
            self.write_errors[destination] = str(e)

    def start_writers(self) -> list[threading.Thread]:
        """Start all writer threads."""
        threads = []
        for dest in self.destinations:
            thread = threading.Thread(
                target=self.writer_thread, args=(dest,), name=f"Writer-{dest.name}"
            )
            thread.start()
            threads.append(thread)
        return threads

    def update_hash(self, data: bytes) -> None:
        """Update hash with chunk data."""
        if self.hasher is not None:
            self.hasher.update(data)

    def distribute_chunk(self, chunk: ChunkData) -> None:
        """Distribute chunk to all writers."""
        for dest in self.destinations:
            self.chunk_queues[dest].put(chunk)

    def stop_writers(self, threads: list[threading.Thread]) -> None:
        """Stop all writer threads and wait for completion."""
        # Send stop sentinel to all queues
        for dest in self.destinations:
            self.chunk_queues[dest].put(None)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    def get_hash(self) -> str | None:
        """Get calculated hash digest."""
        if self.hasher is not None:
            return self.hasher.hexdigest()
        return None


class CopyTaskWrapper:
    """
    Wrapper class that handles both file and directory copying operations.

    This class mimics the behavior of Offload Manager's PSTaskWrapper,
    providing unified interface for copying single files or entire directory trees
    to multiple destinations while preserving directory structure.
    """

    def __init__(self, config: CopyConfig) -> None:
        """Initialize PSTaskWrapper."""
        self.config: CopyConfig = config
        self.verbose: bool = config.verbose
        self.total_files: int = 0
        self.completed_files: int = 0
        self.failed_files: int = 0
        self.total_bytes: int = 0
        self.copied_bytes: int = 0
        self.verification_manager: SourceVerificationManager = (
            SourceVerificationManager(config)
        )

    def discover_files(self, source_path: Path) -> list[Path]:
        """
        Discover all files in source path (file or directory).

        Parameters
        ----------
        source_path : Path
            Source file or directory path

        Returns
        -------
        list[Path]
            List of all files to be copied
        """
        if source_path.is_dir():
            files = []
            for item in source_path.rglob("*"):
                if item.is_file():
                    files.append(item)
            return files
        else:
            raise FileNotFoundError(f"Source path does not exist: {source_path}")

    def calculate_relative_path(self, file_path: Path, source_root: Path) -> Path:
        """
        Calculate relative path from source root.

        Parameters
        ----------
        file_path : Path
            Absolute path to file
        source_root : Path
            Source root directory

        Returns
        -------
        Path
            Relative path from source root
        """
        if source_root.is_file():
            return Path(file_path.name)
        else:
            return file_path.relative_to(source_root)

    def map_destinations(
        self, source_file: Path, source_root: Path, destination_roots: list[Path]
    ) -> list[Path]:
        """
        Map source file to corresponding destination paths.

        Parameters
        ----------
        source_file : Path
            Source file path
        source_root : Path
            Source root path (file or directory)
        destination_roots : list[Path]
            List of destination root paths

        Returns
        -------
        list[Path]
            List of mapped destination file paths
        """
        relative_path = self.calculate_relative_path(source_file, source_root)
        destinations = []

        for dest_root in destination_roots:
            if source_root.is_file():
                # For single file, destination should be the target file
                destinations.append(dest_root)
            else:
                # For directory, append relative path to destination root
                destinations.append(dest_root / relative_path)

        return destinations

    def copy_directory_structure(
        self,
        source: Path,
        destinations: list[Path],
    ) -> dict[str, Any]:
        """Copy directory structure to multiple destinations."""
        logging.info(f"PSTaskWrapper launching directory copy from {source}")

        # check if source verify and log the source verification mode: per_file or
        # after_all
        if self.verification_manager.should_verify():
            mode = self.config.source_verification.value
            logging.info(f"Source verification enabled: {mode} mode")

        # Discover all files in source
        source_files = self.discover_files(source)
        self.total_files = len(source_files)

        if self.total_files == 0:
            logging.warning(f"No files found in source: {source}")
            return {"success": True, "files_copied": 0, "files_failed": 0}

        logging.info(f"Found {self.total_files} files to copy")

        # Calculate total size
        self.total_bytes = sum(f.stat().st_size for f in source_files)

        # TODO: This is too shitty...`results`
        results = {
            "success": True,
            "files_copied": 0,
            "files_failed": 0,
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "file_results": {},
            "source_verification_enabled": self.verification_manager.should_verify(),
            "source_verification_mode": self.config.source_verification.value,
            "source_verification_results": {},
        }

        # Process each file
        for source_file in source_files:
            try:
                # Map to destination paths
                dest_files = self.map_destinations(source_file, source, destinations)

                file_result = self.copy_file(
                    source=source_file,
                    destinations=dest_files,
                )

                if file_result["success"]:
                    self.completed_files += 1
                    results["files_copied"] += 1
                    progress = f"{self.completed_files}/{self.total_files}"
                    logging.info(f"✓ Copied {source_file.name} ({progress})")
                else:
                    self.failed_files += 1
                    results["files_failed"] += 1
                    results["success"] = False
                    logging.error(f"✗ Failed {source_file.name}")

                results["file_results"][str(source_file)] = file_result

            except Exception as e:
                self.failed_files += 1
                results["files_failed"] += 1
                results["success"] = False
                results["file_results"][str(source_file)] = {
                    "success": False,
                    "error": str(e),
                }
                logging.error(f"✗ Error copying {source_file}: {e}")

        # Final statistics
        success_rate = (
            (self.completed_files / self.total_files * 100)
            if self.total_files > 0
            else 0
        )

        progress = f"{self.completed_files}/{self.total_files}"
        logging.info(
            f"Directory copy completed: {progress} files ({success_rate:.1f}%)"
        )

        if self.failed_files > 0:
            logging.warning(f"Failed files: {self.failed_files}")
            results["success"] = False

        # Perform source verification after all files if needed
        if self.verification_manager.should_verify_at_end() and results["success"]:
            verification_results = self.verification_manager.verify_all_pending(source)

            for source_file, verify_result in verification_results.items():
                if not verify_result.verified:
                    results["success"] = False

                results["source_verification_results"][str(source_file)] = {
                    "verified": verify_result.verified,
                    "error": verify_result.error,
                    "initial_hash": verify_result.initial_hash,
                    "final_hash": verify_result.final_source_hash,
                }

            summary = self.verification_manager.get_summary()
            if summary["failed"] > 0:
                results["success"] = False

        return results

    def copy_file(
        self,
        source: Path,
        destinations: list[Path],
    ) -> dict[str, Any]:
        """Unified file copy function with parallel I/O."""
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        config = self.config
        verification_manager = self.verification_manager

        actual_size = source.stat().st_size

        progress_tracker = ProgressTracker(actual_size, source)
        progress_tracker.start()

        results = {"success": True, "destinations": {}}
        temp_files: dict[Path, Path] = {}

        try:
            # Create temporary files for all destinations
            for dest in destinations:
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp_file = dest.parent / f".{dest.name}.tmp"
                temp_files[dest] = temp_file

            # Initialize parallel writer and use it as a context manager
            with ParallelWriter(
                destinations, temp_files, progress_tracker, config
            ) as parallel_writer:
                # Start writer threads
                writer_threads = parallel_writer.start_writers()

                # Read source and distribute chunks to writers
                chunk_id = 0
                with open(source, "rb") as src:
                    while chunk := src.read(config.buffer_size):
                        chunk_data = ChunkData(chunk_id, chunk)
                        parallel_writer.update_hash(chunk_data.data)
                        parallel_writer.distribute_chunk(chunk_data)
                        chunk_id += 1

                # Stop all writer threads
                parallel_writer.stop_writers(writer_threads)

                # Check for write errors
                if parallel_writer.write_errors:
                    results["success"] = False
                    for dest, error in parallel_writer.write_errors.items():
                        results["destinations"][str(dest)] = f"Write error: {error}"
                        logging.error(f"Write error for {dest}: {error}")
                    return results

                # Get hash and stats before files are closed
                file_hash = parallel_writer.get_hash()
                duration, speed = progress_tracker.get_stats()

            # Files are now closed, safe to verify and move them
            # Verify and move files
            for dest in destinations:
                temp_file = temp_files[dest]
                if temp_file.stat().st_size != actual_size:
                    results["success"] = False
                    results["destinations"][str(dest)] = "File size mismatch"
                    continue

                temp_file.rename(dest)
                logging.info(f"Copy completed successfully: {dest}")
                results["destinations"][str(dest)] = "success"

            # Finalize progress and store results
            progress_tracker.finish(file_hash)
            results["duration"] = duration
            results["speed_mb_sec"] = speed
            if file_hash:
                results["hash"] = file_hash

            # TODO: Should I put the source verification here?
            # Source verification if requested
            if (
                verification_manager
                and verification_manager.should_verify()
                and verification_manager.should_verify_immediately()
            ):
                # verification_manager.prepare_file_for_verification(source)

                verify_result = verification_manager.verify_file_immediately(source)
                if verify_result:
                    results["source_verified"] = verify_result.verified
                    if not verify_result.verified:
                        results["success"] = False

            return results

        except Exception as e:
            # Clean up temp files if they were created
            for temp_file in temp_files.values():
                if temp_file.exists():
                    temp_file.unlink()
            raise OSError(f"Copy operation failed: {e}") from e

    def launch_copy(
        self,
        source: Path | str,
        destinations: list[Path | str],
    ) -> dict[str, Any]:
        """Launch copy operation for file or directory."""
        source_path = Path(source)
        dest_paths = [Path(dest) for dest in destinations]

        if source_path.is_file():
            # File copy
            result = self.copy_file(
                source=source_path,
                destinations=dest_paths,
            )

            # TODO: The verification mode is not clear
            # Perform AFTER_ALL verification for single file if needed
            if self.verification_manager.should_verify_at_end() and result.get(
                "success"
            ):
                verification_results = self.verification_manager.verify_all_pending()

                if verification_results:
                    verify_result = list(verification_results.values())[0]
                    result["source_verified"] = verify_result.verified

                    if not verify_result.verified:
                        result["success"] = False

            return result
        elif source_path.is_dir():
            # TODO: PENDING (Unwrap the copy_directory_structure inline here cause
            # we...)
            # Directory copy
            return self.copy_directory_structure(
                source=source_path,
                destinations=dest_paths,
            )
        else:
            raise FileNotFoundError(f"Source path does not exist: {source_path}")


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.

    Parameters
    ----------
    verbose : bool
        Enable verbose logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Configure logging format similar to the original
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Professional file copying tool with integrity verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
   %(prog)s -v source.mov dest1.mov dest2.mov
   %(prog)s --source-verify per_file \\
     --source-verify-hash md5 source.mov dest1.mov dest2.mov
        """,
    )

    # Verbose output
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    # Source verification
    parser.add_argument(
        "--source-verify",
        type=str,
        default="none",
        choices=["none", "per_file", "after_all"],
        help="Source verification mode: none (default), per_file (verify each "
        "file after copy), after_all (verify all files after all copies complete)",
    )

    # Hash algorithm for source verification
    # (only used when --source-verify is not 'none')
    parser.add_argument(
        "--source-verify-hash",
        type=str,
        default="xxh64be",
        choices=["xxh64be", "md5", "sha1", "sha256"],
        help="Hash algorithm for source verification (default: xxh64be). "
        "Only used when --source-verify is enabled.",
    )

    # Source file or directory (required)
    parser.add_argument("source", type=Path, help="Source file or directory path")

    # Destination files or directories (one or more required)
    parser.add_argument(
        "destinations", nargs="+", type=Path, help="Destination file or directory paths"
    )

    return parser.parse_args()


def main() -> int:
    """Main function for cvv."""
    args = None
    try:
        args = parse_arguments()

        # Setup logging
        setup_logging(args.verbose)

        # Create config
        config = CopyConfig.from_args(args)

        # Create task wrapper
        task_wrapper = CopyTaskWrapper(config)

        # Launch copy
        results = task_wrapper.launch_copy(
            source=args.source,
            destinations=args.destinations,
        )

        if results["success"]:
            if "files_copied" in results:
                # Directory copy results
                file_count = results["files_copied"]
                logging.info(
                    f"All copy operations completed successfully ({file_count} files)"
                )
            else:
                # File copy results
                logging.info("All copy operations completed successfully")
            return 0
        else:
            if "files_failed" in results:
                # Directory copy results
                logging.error(
                    f"Some copy operations failed ({results['files_failed']} failures)"
                )
            else:
                # File copy results
                logging.error("Some copy operations failed")
                for dest, status in results.get("destinations", {}).items():
                    if status != "success":
                        logging.error(f"Failed: {dest} - {status}")
            return 1

    except KeyboardInterrupt:
        logging.error("Operation interrupted by user")
        return 1
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logging.error(f"Invalid parameter: {e}")
        return 1
    except OSError as e:
        logging.error(f"I/O error: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if args and args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
