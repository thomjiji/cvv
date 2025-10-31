#!/usr/bin/env python3
"""
cvv - Professional file copying tool with integrity verification.

A clean, professional implementation of multi-destination file copying with
configurable verification modes, designed for DIT (Digital Imaging Technician)
workflows.

Architecture:
- Core logic is completely UI-agnostic (yields events, never touches stdout)
- Generator pattern for natural progress reporting
- Per-destination error tracking
- Clean separation between business logic and presentation
"""

import argparse
import asyncio
import contextlib
import hashlib
import shutil
import signal
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import BinaryIO

import aiofiles
import aiofiles.os

try:
    import xxhash
except ImportError:
    print("ERROR: The 'xxhash' library is required but not installed.", file=sys.stderr)
    print("Please install it using: pip install xxhash", file=sys.stderr)
    sys.exit(1)

# Constants
BUFFER_SIZE = 8 * 1024 * 1024  # 8MB


# ============================================================================
# Data Models
# ============================================================================


class VerificationMode(Enum):
    """
    Verification strategy for copy operations.

    Attributes
    ----------
    TRANSFER : str
        File size comparison only
    SOURCE : str
        Hash source in-flight and post-copy
    FULL : str
        Hash source and all destinations post-copy
    """

    TRANSFER = "transfer"
    SOURCE = "source"
    FULL = "full"


class EventType(Enum):
    """
    Events emitted during copy operations.

    Attributes
    ----------
    COPY_START : str
        Copy operation started
    COPY_PROGRESS : str
        Copy operation progress update
    COPY_COMPLETE : str
        Copy operation completed
    VERIFY_START : str
        Verification started
    VERIFY_PROGRESS : str
        Verification progress update
    VERIFY_COMPLETE : str
        Verification completed
    """

    COPY_START = "copy_start"
    COPY_PROGRESS = "copy_progress"
    COPY_COMPLETE = "copy_complete"
    VERIFY_START = "verify_start"
    VERIFY_PROGRESS = "verify_progress"
    VERIFY_COMPLETE = "verify_complete"


@dataclass
class CopyEvent:
    """
    Event emitted during copy/verification operations.

    Attributes
    ----------
    type : EventType
        Type of event
    bytes_processed : int, default=0
        Number of bytes processed so far
    total_bytes : int, default=0
        Total bytes to process
    message : str, default=""
        Optional message describing the event
    """

    type: EventType
    bytes_processed: int = 0
    total_bytes: int = 0
    message: str = ""


@dataclass
class DestinationResult:
    """
    Result for a single destination.

    Attributes
    ----------
    path : Path
        Destination file path
    success : bool
        Whether the copy operation succeeded
    bytes_written : int, default=0
        Number of bytes written to destination
    hash_post : str | None, default=None
        Post-copy hash of destination file
    error : str | None, default=None
        Error message if operation failed
    """

    path: Path
    success: bool
    bytes_written: int = 0
    hash_post: str | None = None
    error: str | None = None


@dataclass
class CopyResult:
    """
    Complete result of a multi-destination copy operation.

    Attributes
    ----------
    source_path : Path
        Source file path
    source_size : int
        Size of source file in bytes
    destinations : list[DestinationResult], default=[]
        Results for each destination
    source_hash_inflight : str | None, default=None
        Hash computed during copy (in-flight)
    source_hash_post : str | None, default=None
        Hash computed after copy (post-copy verification)
    duration : float, default=0.0
        Total operation duration in seconds
    verification_mode : VerificationMode, default=VerificationMode.TRANSFER
        Verification mode used for the operation
    """

    source_path: Path
    source_size: int
    destinations: list[DestinationResult] = field(default_factory=list)
    source_hash_inflight: str | None = None
    source_hash_post: str | None = None
    duration: float = 0.0
    verification_mode: VerificationMode = VerificationMode.TRANSFER

    @property
    def success(self) -> bool:
        """
        Check if all destinations succeeded.

        Returns
        -------
        bool
            True if all destination copies succeeded, False otherwise
        """
        return all(d.success for d in self.destinations)

    @property
    def speed_mb_sec(self) -> float:
        """
        Calculate transfer speed in MB/s.

        Returns
        -------
        float
            Transfer speed in megabytes per second
        """
        if self.duration > 0:
            return (self.source_size / (1024 * 1024)) / self.duration
        return 0.0


# ============================================================================
# Core Copy Engine (UI-agnostic)
# ============================================================================


class HashCalculator:
    """
    Thread-safe hash calculator supporting multiple algorithms.

    Parameters
    ----------
    algorithm : str, default="xxh64be"
        Hash algorithm to use. Supported: xxh64be, md5, sha1, sha256
    """

    def __init__(self, algorithm: str = "xxh64be"):
        self.algorithm = algorithm.lower()
        if self.algorithm == "xxh64be":
            self._hasher = xxhash.xxh64()
        elif self.algorithm in ["md5", "sha1", "sha256"]:
            self._hasher = hashlib.new(self.algorithm)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    def update(self, data: bytes) -> None:
        """
        Update hash with new data.

        Parameters
        ----------
        data : bytes
            Data chunk to add to the hash
        """
        self._hasher.update(data)

    def hexdigest(self) -> str:
        """
        Get final hex digest.

        Returns
        -------
        str
            Hexadecimal string representation of the hash
        """
        return self._hasher.hexdigest()

    @staticmethod
    async def hash_file_async(
        path: Path,
        algorithm: str = "xxh64be",
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[tuple[int, str]]:
        """
        Hash a file asynchronously and yield progress.

        Parameters
        ----------
        path : Path
            Path to file to hash
        algorithm : str, default="xxh64be"
            Hash algorithm to use
        abort_event : asyncio.Event | None, default=None
            Event to check for abort signal

        Yields
        ------
        tuple[int, str]
            (bytes_hashed, final_hash_or_empty_string)
            Progress updates yield empty string, final yield contains complete hash

        Raises
        ------
        InterruptedError
            If abort_event is set during hashing
        """
        hasher = HashCalculator(algorithm)
        total_bytes = 0

        async with aiofiles.open(path, "rb") as f:
            while chunk := await f.read(BUFFER_SIZE):
                # Check for abort
                if abort_event and abort_event.is_set():
                    raise InterruptedError("Hash operation interrupted")
                hasher.update(chunk)
                total_bytes += len(chunk)
                yield (total_bytes, "")

        # Final yield with complete hash
        yield (total_bytes, hasher.hexdigest())

    @staticmethod
    def hash_file(
        path: Path,
        algorithm: str = "xxh64be",
        abort_event: threading.Event | None = None,
    ) -> Iterator[tuple[int, str]]:
        """
        Hash a file and yield progress (sync version for backward compatibility).

        Parameters
        ----------
        path : Path
            Path to file to hash
        algorithm : str, default="xxh64be"
            Hash algorithm to use
        abort_event : threading.Event | None, default=None
            Event to check for abort signal

        Yields
        ------
        tuple[int, str]
            (bytes_hashed, final_hash_or_empty_string)
            Progress updates yield empty string, final yield contains complete hash

        Raises
        ------
        InterruptedError
            If abort_event is set during hashing
        """
        hasher = HashCalculator(algorithm)
        total_bytes = 0

        with open(path, "rb") as f:
            while chunk := f.read(BUFFER_SIZE):
                # Check for abort
                if abort_event and abort_event.is_set():
                    raise InterruptedError("Hash operation interrupted")
                hasher.update(chunk)
                total_bytes += len(chunk)
                yield (total_bytes, "")

        # Final yield with complete hash
        yield (total_bytes, hasher.hexdigest())


class CopyEngine:
    """
    Core copy engine: reads source once, writes to multiple destinations.

    This is completely UI-agnostic - it yields events and never touches stdout.

    Parameters
    ----------
    source : Path
        Source file path
    destinations : list[Path]
        List of destination file paths
    verification_mode : VerificationMode, default=VerificationMode.FULL
        Verification strategy to use
    hash_algorithm : str, default="xxh64be"
        Hash algorithm for verification
    abort_event : threading.Event | None, default=None
        Optional custom abort event (uses shared event if None)
    """

    # Class-level shared abort event (persists across all instances)
    _shared_abort_event = threading.Event()
    _signal_handler_installed = False

    def __init__(
        self,
        source: Path,
        destinations: list[Path],
        verification_mode: VerificationMode = VerificationMode.FULL,
        hash_algorithm: str = "xxh64be",
        abort_event: threading.Event | None = None,
    ):
        self.source = source
        self.destinations = destinations
        self.verification_mode = verification_mode
        self.hash_algorithm = hash_algorithm
        # Use provided abort event, or fall back to shared one
        self._abort_event = (
            abort_event if abort_event else CopyEngine._shared_abort_event
        )
        self._interrupted = False

        # Install signal handler once (not per instance) - only for shared event
        if abort_event is None and not CopyEngine._signal_handler_installed:
            signal.signal(signal.SIGINT, self._handle_interrupt)
            CopyEngine._signal_handler_installed = True

    @classmethod
    def reset_shared_state(cls) -> None:
        """
        Reset the shared abort event (useful for testing).

        Notes
        -----
        This is primarily used in test suites to ensure clean state between tests.
        """
        cls._shared_abort_event.clear()

    def _handle_interrupt(self, signum, frame):
        """
        Handle Ctrl+C gracefully - stops all current and future operations.

        Parameters
        ----------
        signum : int
            Signal number
        frame : frame
            Current stack frame
        """
        if not CopyEngine._shared_abort_event.is_set():
            CopyEngine._shared_abort_event.set()
            self._interrupted = True
            print("\n\nCopy interrupted.", file=sys.stderr)

    async def copy(self) -> AsyncIterator[CopyEvent | CopyResult]:
        """
        Execute the copy operation.

        Yields
        ------
        CopyEvent | CopyResult
            CopyEvent objects during operation, final yield is CopyResult
        """
        start_time = time.time()

        # Initialize result with defaults (will be updated if source exists)
        result = CopyResult(
            source_path=self.source,
            source_size=0,
            verification_mode=self.verification_mode,
        )

        try:
            # Get source size (may fail if source doesn't exist)
            source_size = self.source.stat().st_size
            result.source_size = source_size
            # Pre-flight checks
            self._check_source_exists()
            self._check_disk_space(source_size)
            self._prepare_destinations()

            # Copy phase
            yield CopyEvent(
                type=EventType.COPY_START,
                total_bytes=source_size,
                message=f"Copying {self.source.name} to {len(self.destinations)} destination(s)",
            )

            bytes_copied = 0
            # Always enable in-flight hashing - it's essentially free
            # We just won't do post-copy verification in TRANSFER mode
            enable_hashing = True

            async for event_or_hash in self._stream_to_destinations(enable_hashing):
                if isinstance(event_or_hash, str):
                    # Final hash result
                    result.source_hash_inflight = event_or_hash
                else:
                    # Progress event
                    bytes_copied = event_or_hash
                    yield CopyEvent(
                        type=EventType.COPY_PROGRESS,
                        bytes_processed=bytes_copied,
                        total_bytes=source_size,
                    )

            # Verify all destinations were written
            dest_results = self._verify_destinations_written()
            result.destinations = dest_results

            yield CopyEvent(
                type=EventType.COPY_COMPLETE,
                bytes_processed=source_size,
                total_bytes=source_size,
                message="Copy phase complete",
            )

            # Verification phase
            if self.verification_mode == VerificationMode.TRANSFER:
                # Already verified sizes during _verify_destinations_written
                pass
            elif self.verification_mode == VerificationMode.SOURCE:
                async for event in self._verify_source_only(result):
                    yield event
            elif self.verification_mode == VerificationMode.FULL:
                async for event in self._verify_full(result):
                    yield event

        except InterruptedError:
            # User pressed Ctrl+C - exit gracefully
            # .tmp files are preserved for resume
            pass
        except Exception as e:
            # Mark all destinations as failed
            for dest in self.destinations:
                result.destinations.append(
                    DestinationResult(path=dest, success=False, error=str(e))
                )

        finally:
            result.duration = time.time() - start_time

        yield result

    def abort(self) -> None:
        """
        Signal the copy operation to abort.

        Notes
        -----
        Sets the abort event which causes all threads to exit gracefully.
        """
        self._abort_event.set()

    # ------------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------------

    def _check_source_exists(self) -> None:
        """
        Verify source file exists and is readable.

        Raises
        ------
        FileNotFoundError
            If source file does not exist
        ValueError
            If source is not a file (e.g., is a directory)
        """
        if not self.source.exists():
            raise FileNotFoundError(f"Source file not found: {self.source}")
        if not self.source.is_file():
            raise ValueError(f"Source is not a file: {self.source}")

    def _check_disk_space(self, required_bytes: int) -> None:
        """
        Verify sufficient disk space on all destinations.

        Parameters
        ----------
        required_bytes : int
            Number of bytes required for the copy operation

        Raises
        ------
        OSError
            If any destination has insufficient free space
        """
        for dest in self.destinations:
            dest.parent.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(dest.parent)
            if usage.free < required_bytes:
                raise OSError(
                    f"Insufficient space on {dest.parent}: "
                    f"need {required_bytes / 1e9:.2f} GB, "
                    f"have {usage.free / 1e9:.2f} GB"
                )

    def _prepare_destinations(self) -> None:
        """
        Create destination directories.

        Notes
        -----
        Creates parent directories for all destinations if they don't exist.
        """
        for dest in self.destinations:
            dest.parent.mkdir(parents=True, exist_ok=True)

    async def _stream_to_destinations(self, enable_hashing: bool) -> AsyncIterator[int | str]:
        """
        Read source once, write to multiple destinations concurrently.

        Parameters
        ----------
        enable_hashing : bool
            Whether to compute hash during copy

        Yields
        ------
        int | str
            bytes_copied (int) during copy, then final hash (str) if enabled

        Raises
        ------
        OSError
            If write operations encounter errors
        InterruptedError
            If operation is aborted
        """
        # Open all destination files asynchronously
        dest_files = []
        temp_paths = []

        try:
            for dest_path in self.destinations:
                temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
                # Open file with explicit large buffer for better performance
                dest_file = await aiofiles.open(temp_path, 'wb', buffering=BUFFER_SIZE)
                dest_files.append(dest_file)
                temp_paths.append((dest_path, temp_path))

            # Create hasher if needed
            hasher = HashCalculator(self.hash_algorithm) if enable_hashing else None
            bytes_read = 0
            last_progress_time = time.time()
            progress_interval = 0.1  # Throttle progress to max 10 updates/second

            # Read source and write to all destinations concurrently
            async with aiofiles.open(self.source, 'rb') as f_source:
                while not self._abort_event.is_set():
                    chunk = await f_source.read(BUFFER_SIZE)
                    if not chunk:
                        break

                    if hasher:
                        hasher.update(chunk)

                    bytes_read += len(chunk)

                    # Write to ALL destinations concurrently (no queue bottleneck!)
                    if not self._abort_event.is_set():
                        try:
                            await asyncio.gather(*[
                                dest_file.write(chunk)
                                for dest_file in dest_files
                            ])
                        except Exception as e:
                            raise OSError(f"Write error: {e}")

                    # Throttle progress updates to reduce CPU usage
                    current_time = time.time()
                    if current_time - last_progress_time >= progress_interval:
                        yield bytes_read
                        last_progress_time = current_time

            # Yield final progress to ensure 100% is shown
            if bytes_read > 0:
                yield bytes_read

            # Return final hash if enabled
            if hasher:
                yield hasher.hexdigest()

        finally:
            # Close all destination files
            for dest_file in dest_files:
                try:
                    await dest_file.close()
                except Exception:
                    pass

            # Atomic rename on success (only if not aborted)
            if not self._abort_event.is_set():
                for dest_path, temp_path in temp_paths:
                    try:
                        temp_path.replace(dest_path)
                    except Exception as e:
                        raise OSError(f"Failed to rename {temp_path} to {dest_path}: {e}")
            else:
                # On abort, clean up temp files if not interrupted by Ctrl+C
                if not self._interrupted:
                    for _, temp_path in temp_paths:
                        if temp_path.exists():
                            with contextlib.suppress(Exception):
                                temp_path.unlink()

    def _verify_destinations_written(self) -> list[DestinationResult]:
        """
        Verify all destination files exist and have correct size.

        Returns
        -------
        list[DestinationResult]
            List of destination results with size verification status
        """
        results = []
        source_size = self.source.stat().st_size

        for dest in self.destinations:
            if not dest.exists():
                results.append(
                    DestinationResult(
                        path=dest,
                        success=False,
                        error="Destination file not created",
                    )
                )
                continue

            dest_size = dest.stat().st_size
            if dest_size != source_size:
                results.append(
                    DestinationResult(
                        path=dest,
                        success=False,
                        bytes_written=dest_size,
                        error=f"Size mismatch: expected {source_size}, got {dest_size}",
                    )
                )
                continue

            # Success (so far)
            results.append(
                DestinationResult(
                    path=dest,
                    success=True,
                    bytes_written=dest_size,
                )
            )

        return results

    async def _verify_source_only(self, result: CopyResult) -> AsyncIterator[CopyEvent]:
        """
        SOURCE mode: Re-hash source file to ensure it didn't change during copy.

        Parameters
        ----------
        result : CopyResult
            Copy result to update with verification information

        Yields
        ------
        CopyEvent
            Verification progress events
        """
        yield CopyEvent(
            type=EventType.VERIFY_START,
            total_bytes=result.source_size,
            message="Verifying source file integrity",
        )

        bytes_hashed = 0
        final_hash = ""

        # Note: We still use threading.Event for abort_event (signal handling)
        # but check it in async context - this is fine
        async for bytes_hashed, final_hash in HashCalculator.hash_file_async(
            self.source, self.hash_algorithm, None  # Don't pass threading.Event to async
        ):
            # Check for abort manually
            if self._abort_event.is_set():
                break

            if final_hash:
                result.source_hash_post = final_hash
            else:
                yield CopyEvent(
                    type=EventType.VERIFY_PROGRESS,
                    bytes_processed=bytes_hashed,
                    total_bytes=result.source_size,
                )

        # Check if source changed
        if result.source_hash_post != result.source_hash_inflight:
            for dest_result in result.destinations:
                dest_result.success = False
                dest_result.error = "Source file changed during copy"

        yield CopyEvent(
            type=EventType.VERIFY_COMPLETE,
            bytes_processed=result.source_size,
            total_bytes=result.source_size,
            message="Source verification complete",
        )

    async def _verify_full(self, result: CopyResult) -> AsyncIterator[CopyEvent]:
        """
        FULL mode: Hash source and all destinations in parallel with real-time progress.

        Parameters
        ----------
        result : CopyResult
            Copy result to update with verification information

        Yields
        ------
        CopyEvent
            Verification progress events
        """
        files_to_hash = [self.source] + self.destinations
        total_bytes = result.source_size * len(files_to_hash)

        yield CopyEvent(
            type=EventType.VERIFY_START,
            total_bytes=total_bytes,
            message=f"Verifying source + {len(self.destinations)} destination(s)",
        )

        # Shared progress counter (thread-safe for asyncio tasks)
        shared_progress = {"bytes_hashed": 0}
        progress_lock = asyncio.Lock()

        async def hash_file_with_progress(path: Path) -> tuple[Path, str]:
            """
            Hash a file and update shared progress counter in real-time.

            Parameters
            ----------
            path : Path
                File to hash

            Returns
            -------
            tuple[Path, str]
                (file_path, hash_digest)
            """
            final_hash = ""
            last_bytes = 0

            async for bytes_hashed, final_hash in HashCalculator.hash_file_async(
                path, self.hash_algorithm, None
            ):
                if self._abort_event.is_set():
                    break

                if not final_hash:
                    # Progress update - report delta since last update
                    delta = bytes_hashed - last_bytes
                    last_bytes = bytes_hashed

                    async with progress_lock:
                        shared_progress["bytes_hashed"] += delta

            # Return final hash
            return (path, final_hash)

        # Hash all files in parallel using asyncio.gather
        hashes = {}

        try:
            # Create tasks for all files
            tasks = [hash_file_with_progress(path) for path in files_to_hash]

            # Create a task to monitor progress while hashing happens
            async def monitor_progress():
                last_reported = 0
                while True:
                    async with progress_lock:
                        current_bytes = shared_progress["bytes_hashed"]

                    if current_bytes > last_reported:
                        yield CopyEvent(
                            type=EventType.VERIFY_PROGRESS,
                            bytes_processed=current_bytes,
                            total_bytes=total_bytes,
                        )
                        last_reported = current_bytes

                    await asyncio.sleep(0.1)  # Check every 100ms

            # Run hashing and progress monitoring concurrently
            monitor_task = asyncio.create_task(monitor_progress().__anext__())

            # Wait for all hashing to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Cancel monitoring
            monitor_task.cancel()

            # Process results
            for path, result_or_exc in zip(files_to_hash, results):
                if isinstance(result_or_exc, Exception):
                    # Mark this destination as failed
                    if path != self.source:
                        for dest_result in result.destinations:
                            if dest_result.path == path:
                                dest_result.success = False
                                dest_result.error = f"Hash failed: {result_or_exc}"
                elif isinstance(result_or_exc, tuple):
                    path_result, file_hash = result_or_exc
                    hashes[path_result] = file_hash

        except asyncio.CancelledError:
            # Interrupted - just return without setting hashes
            return

        # Store hashes in result
        result.source_hash_post = hashes.get(self.source)

        for dest_result in result.destinations:
            dest_hash = hashes.get(dest_result.path)
            dest_result.hash_post = dest_hash

            # Verify hash matches source
            if dest_result.success and dest_hash != result.source_hash_inflight:
                dest_result.success = False
                dest_result.error = (
                    f"Hash mismatch: {dest_hash} != {result.source_hash_inflight}"
                )

        # Check if source changed during copy
        if result.source_hash_post != result.source_hash_inflight:
            for dest_result in result.destinations:
                if dest_result.success:
                    dest_result.success = False
                    dest_result.error = "Source file changed during copy"

        yield CopyEvent(
            type=EventType.VERIFY_COMPLETE,
            bytes_processed=total_bytes,
            total_bytes=total_bytes,
            message="Full verification complete",
        )

    def _hash_file_to_completion(self, path: Path) -> str:
        """
        Hash a file completely and return the final digest.

        Parameters
        ----------
        path : Path
            File to hash

        Returns
        -------
        str
            Final hash digest
        """
        final_hash = ""
        for _, final_hash in HashCalculator.hash_file(path, self.hash_algorithm):
            if final_hash:
                return final_hash
        return final_hash


# ============================================================================
# CLI Layer (Presentation)
# ============================================================================


class CLIProcessor:
    """
    Handles CLI orchestration and presentation.

    This layer is completely separate from the core copy engine.

    Parameters
    ----------
    source : Path
        Source file or directory path
    destinations : list[Path]
        List of destination paths
    verification_mode : VerificationMode
        Verification strategy to use
    hash_algorithm : str
        Hash algorithm for verification
    """

    def __init__(
        self,
        source: Path,
        destinations: list[Path],
        verification_mode: VerificationMode,
        hash_algorithm: str,
    ):
        self.source = source
        self.destinations = destinations
        self.verification_mode = verification_mode
        self.hash_algorithm = hash_algorithm

    async def run(self) -> bool:
        """
        Execute copy jobs for all source files.

        Returns
        -------
        bool
            True if all operations succeeded, False otherwise
        """
        # Discover files to copy
        source_files = self._discover_files()

        if not source_files:
            print("No files to copy")
            return True

        # Execute each copy job
        results = []
        for i, source_file in enumerate(source_files, 1):
            # Check if user pressed Ctrl+C (before starting next file)
            if CopyEngine._shared_abort_event.is_set():
                break

            print(f"\nFile {i}/{len(source_files)}: {source_file.name}")

            dest_paths = self._calculate_destinations(source_file)
            result = await self._execute_single_copy(source_file, dest_paths)
            results.append(result)

            # Show result summary
            self._show_result_summary(result)

            if not result.success:
                print("Operation failed, stopping.")
                break

        # Final summary
        self._show_final_summary(results)

        return all(r.success for r in results)

    def _discover_files(self) -> list[Path]:
        """
        Discover all source files to copy.

        Returns
        -------
        list[Path]
            List of source files to copy

        Raises
        ------
        FileNotFoundError
            If source does not exist
        """
        if self.source.is_file():
            return [self.source]
        elif self.source.is_dir():
            return sorted([f for f in self.source.rglob("*") if f.is_file()])
        else:
            raise FileNotFoundError(f"Source not found: {self.source}")

    def _calculate_destinations(self, source_file: Path) -> list[Path]:
        """
        Calculate destination paths for a source file.

        Parameters
        ----------
        source_file : Path
            Source file to calculate destinations for

        Returns
        -------
        list[Path]
            List of destination paths
        """
        dest_paths = []

        # If source is a directory, preserve structure
        if self.source.is_dir():
            relative_path = source_file.relative_to(self.source)
            return [dest_root / relative_path for dest_root in self.destinations]

        # Single file source
        for dest in self.destinations:
            if dest.is_dir():
                dest_paths.append(dest / source_file.name)
            else:
                dest_paths.append(dest)

        return dest_paths

    def _check_duplicates_and_cleanup(
        self, source: Path, destinations: list[Path]
    ) -> list[Path]:
        """
        Check for duplicate files and clean up incomplete .tmp files.

        Parameters
        ----------
        source : Path
            Source file path
        destinations : list[Path]
            List of destination paths

        Returns
        -------
        list[Path]
            List of destinations that need to be copied (excluding duplicates)

        Notes
        -----
        Priority 1: Skip already-completed files (duplication detection)
        Priority 2: Clean up incomplete .tmp files (restart from beginning)
        """
        source_size = source.stat().st_size
        destinations_to_copy = []

        for dest in destinations:
            # Priority 1: Check if destination file already exists and is complete
            if dest.exists():
                dest_size = dest.stat().st_size
                if dest_size == source_size:
                    print(f"  ✓ {dest.name} already exists (skipping)")
                    continue
                else:
                    print(
                        f"  ! {dest.name} exists but wrong size "
                        f"({dest_size} vs {source_size}), will overwrite"
                    )
                    dest.unlink()

            # Priority 2: Check for incomplete .tmp file - just delete it and restart
            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            if tmp_path.exists():
                tmp_size = tmp_path.stat().st_size
                mb_done = tmp_size / (1024 * 1024)
                source_mb = source_size / (1024 * 1024)
                print(
                    f"  ! Found incomplete {tmp_path.name} "
                    f"({mb_done:.1f}/{source_mb:.1f} MB), restarting from beginning"
                )
                tmp_path.unlink()

            destinations_to_copy.append(dest)

        return destinations_to_copy

    async def _execute_single_copy(
        self, source: Path, destinations: list[Path]
    ) -> CopyResult:
        """
        Execute a single copy operation with simple text progress.

        Parameters
        ----------
        source : Path
            Source file path
        destinations : list[Path]
            List of destination paths

        Returns
        -------
        CopyResult
            Result of the copy operation
        """
        # Check for duplicates and cleanup incomplete .tmp files
        destinations_to_copy = self._check_duplicates_and_cleanup(source, destinations)

        # If all destinations already exist, return success result
        if not destinations_to_copy:
            print("  ✓ All destinations already complete (nothing to copy)")
            result = CopyResult(
                source_path=source,
                source_size=source.stat().st_size,
                verification_mode=self.verification_mode,
            )
            # Create success results for all destinations
            for dest in destinations:
                result.destinations.append(DestinationResult(path=dest, success=True))
            return result

        engine = CopyEngine(
            source=source,
            destinations=destinations_to_copy,
            verification_mode=self.verification_mode,
            hash_algorithm=self.hash_algorithm,
        )

        result = None
        copy_total = 0
        verify_total = 0

        async for event in engine.copy():
            if isinstance(event, CopyResult):
                result = event
                sys.stdout.write("\n")  # Newline after progress
                sys.stdout.flush()
                break

            elif event.type == EventType.COPY_START:
                copy_total = event.total_bytes

            elif event.type == EventType.COPY_PROGRESS:
                percent = (
                    (event.bytes_processed / copy_total * 100) if copy_total else 0
                )
                mb_done = event.bytes_processed / (1024 * 1024)
                mb_total = copy_total / (1024 * 1024)
                # Clear line and write progress
                sys.stdout.write(
                    f"\rCopying: {percent:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)".ljust(
                        80
                    )
                )
                sys.stdout.flush()

            elif event.type == EventType.VERIFY_START:
                sys.stdout.write("\n")
                sys.stdout.flush()
                verify_total = event.total_bytes

            elif event.type == EventType.VERIFY_PROGRESS:
                percent = (
                    (event.bytes_processed / verify_total * 100) if verify_total else 0
                )
                mb_done = event.bytes_processed / (1024 * 1024)
                mb_total = verify_total / (1024 * 1024)
                # Clear line and write progress
                sys.stdout.write(
                    f"\rVerifying: {percent:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)".ljust(
                        80
                    )
                )
                sys.stdout.flush()

        return result

    def _show_result_summary(self, result: CopyResult) -> None:
        """
        Display a summary of a single copy operation.

        Parameters
        ----------
        result : CopyResult
            Result of the copy operation to summarize
        """
        if result.success:
            print(
                f"✓ Success "
                f"({result.speed_mb_sec:.2f} MB/s, "
                f"{result.source_size / (1024 * 1024):.2f} MB)"
            )

            # Display hash information based on verification mode
            if result.verification_mode == VerificationMode.TRANSFER:
                # TRANSFER: Show in-flight hash
                if result.source_hash_inflight:
                    print(
                        f"  Source hash ({self.hash_algorithm}): {result.source_hash_inflight}"
                    )

            elif result.verification_mode == VerificationMode.SOURCE:
                # SOURCE: Show both in-flight and post-copy hashes
                if result.source_hash_inflight:
                    print(f"  Source hash (in-flight):  {result.source_hash_inflight}")
                if result.source_hash_post:
                    match_indicator = (
                        "✓"
                        if result.source_hash_post == result.source_hash_inflight
                        else "✗"
                    )
                    print(
                        f"  Source hash (post-copy):  {result.source_hash_post} [{match_indicator}]"
                    )

            elif result.verification_mode == VerificationMode.FULL:
                # FULL: Show source hashes and all destination hashes
                if result.source_hash_inflight:
                    print(f"  Source hash (in-flight):  {result.source_hash_inflight}")
                if result.source_hash_post:
                    match_indicator = (
                        "✓"
                        if result.source_hash_post == result.source_hash_inflight
                        else "✗"
                    )
                    print(
                        f"  Source hash (post-copy):  {result.source_hash_post} [{match_indicator}]"
                    )

                # Show each destination hash
                for dest_result in result.destinations:
                    if dest_result.hash_post:
                        match_indicator = (
                            "✓"
                            if dest_result.hash_post == result.source_hash_inflight
                            else "✗"
                        )
                        print(
                            f"  {dest_result.path.name}: {dest_result.hash_post} [{match_indicator}]"
                        )
        else:
            print("✗ Failed")
            for dest_result in result.destinations:
                if not dest_result.success:
                    print(f"  ✗ {dest_result.path.name}: {dest_result.error}")

    def _show_final_summary(self, results: list[CopyResult]) -> None:
        """
        Display final summary of all operations.

        Parameters
        ----------
        results : list[CopyResult]
            List of all copy operation results
        """
        print("\n" + "=" * 60)

        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success

        if failed == 0:
            print(f"All {total} operation(s) completed successfully")
        else:
            print(f"{failed} of {total} operation(s) failed")


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> int:
    """
    CLI entry point.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for failure, 130 for keyboard interrupt
    """
    parser = argparse.ArgumentParser(
        description="Professional file copying tool with integrity verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cvv -m transfer /source /dest1 /dest2         # Fast copy with size check only
  cvv -m full /source_dir /dest_dir             # Copy entire directory with full integrity verification
  cvv source.mp4 /dest1 /dest2                  # Copy single file is also ok, default use full verification mode
        """,
    )

    parser.add_argument(
        "source",
        type=Path,
        help="Source file or directory to copy",
    )

    parser.add_argument(
        "destinations",
        type=Path,
        nargs="+",
        help="One or more destination paths",
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="full",
        choices=["transfer", "source", "full"],
        help="Verification mode (default: full)",
    )

    parser.add_argument(
        "--hash-algorithm",
        type=str,
        default="xxh64be",
        choices=["xxh64be", "md5", "sha1", "sha256"],
        help="Hash algorithm for verification (default: xxh64be)",
    )

    args = parser.parse_args()

    # Run the CLI processor
    try:
        processor = CLIProcessor(
            source=args.source,
            destinations=args.destinations,
            verification_mode=VerificationMode(args.mode),
            hash_algorithm=args.hash_algorithm,
        )

        success = asyncio.run(processor.run())
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
