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
import contextlib
import hashlib
import logging
import queue
import shutil
import signal
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    import xxhash
except ImportError:
    print("ERROR: The 'xxhash' library is required but not installed.", file=sys.stderr)
    print("Please install it using: pip install xxhash", file=sys.stderr)
    sys.exit(1)

# Constants
BUFFER_SIZE = 8 * 1024 * 1024  # 8MB
QUEUE_SIZE = 10  # Max chunks buffered per destination


# ============================================================================
# Data Models
# ============================================================================


class VerificationMode(Enum):
    """Verification strategy for copy operations."""

    TRANSFER = "transfer"  # File size comparison only
    SOURCE = "source"  # Hash source in-flight and post-copy
    FULL = "full"  # Hash source and all destinations post-copy


class EventType(Enum):
    """Events emitted during copy operations."""

    COPY_START = "copy_start"
    COPY_PROGRESS = "copy_progress"
    COPY_COMPLETE = "copy_complete"
    VERIFY_START = "verify_start"
    VERIFY_PROGRESS = "verify_progress"
    VERIFY_COMPLETE = "verify_complete"


@dataclass
class CopyEvent:
    """Event emitted during copy/verification operations."""

    type: EventType
    bytes_processed: int = 0
    total_bytes: int = 0
    message: str = ""


@dataclass
class DestinationResult:
    """Result for a single destination."""

    path: Path
    success: bool
    bytes_written: int = 0
    hash_post: str | None = None
    error: str | None = None


@dataclass
class ResumeInfo:
    """Information about resumable .tmp files."""

    dest_path: Path
    tmp_path: Path
    bytes_written: int
    progress_percent: float


@dataclass
class CopyResult:
    """Complete result of a multi-destination copy operation."""

    source_path: Path
    source_size: int
    destinations: list[DestinationResult] = field(default_factory=list)
    source_hash_inflight: str | None = None
    source_hash_post: str | None = None
    duration: float = 0.0
    verification_mode: VerificationMode = VerificationMode.TRANSFER

    @property
    def success(self) -> bool:
        """Overall success if all destinations succeeded."""
        return all(d.success for d in self.destinations)

    @property
    def speed_mb_sec(self) -> float:
        """Calculate transfer speed in MB/s."""
        if self.duration > 0:
            return (self.source_size / (1024 * 1024)) / self.duration
        return 0.0


# ============================================================================
# Core Copy Engine (UI-agnostic)
# ============================================================================


class HashCalculator:
    """Thread-safe hash calculator supporting multiple algorithms."""

    def __init__(self, algorithm: str = "xxh64be"):
        self.algorithm = algorithm.lower()
        if self.algorithm == "xxh64be":
            self._hasher = xxhash.xxh64()
        elif self.algorithm in ["md5", "sha1", "sha256"]:
            self._hasher = hashlib.new(self.algorithm)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    def update(self, data: bytes) -> None:
        """Update hash with new data."""
        self._hasher.update(data)

    def hexdigest(self) -> str:
        """Get final hex digest."""
        return self._hasher.hexdigest()

    @staticmethod
    def hash_file(
        path: Path,
        algorithm: str = "xxh64be",
        abort_event: threading.Event | None = None,
    ) -> Iterator[tuple[int, str]]:
        """
        Hash a file and yield progress.

        Yields: (bytes_hashed, final_hash_or_empty_string)
        Final yield contains the complete hash.
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
    """

    def __init__(
        self,
        source: Path,
        destinations: list[Path],
        verification_mode: VerificationMode = VerificationMode.FULL,
        hash_algorithm: str = "xxh64be",
        resume_info: dict[Path, ResumeInfo] | None = None,
    ):
        self.source = source
        self.destinations = destinations
        self.verification_mode = verification_mode
        self.hash_algorithm = hash_algorithm
        self.resume_info = resume_info or {}
        self._abort_event = threading.Event()
        self._interrupted = False

        # Install signal handler for graceful Ctrl+C
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully - preserve .tmp files for resume."""
        if not self._interrupted:
            self._interrupted = True
            self._abort_event.set()
            print(
                "\n\nCopy interrupted. Re-run the same command to resume.",
                file=sys.stderr,
            )

    def copy(self) -> Iterator[CopyEvent | CopyResult]:
        """
        Execute the copy operation.

        Yields CopyEvent objects during operation, final yield is CopyResult.
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

            for event_or_hash in self._stream_to_destinations(
                source_size, enable_hashing
            ):
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
                yield from self._verify_source_only(result)
            elif self.verification_mode == VerificationMode.FULL:
                yield from self._verify_full(result)

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
        """Signal the copy operation to abort."""
        self._abort_event.set()

    # ------------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------------

    def _check_source_exists(self) -> None:
        """Verify source file exists and is readable."""
        if not self.source.exists():
            raise FileNotFoundError(f"Source file not found: {self.source}")
        if not self.source.is_file():
            raise ValueError(f"Source is not a file: {self.source}")

    def _check_disk_space(self, required_bytes: int) -> None:
        """Verify sufficient disk space on all destinations."""
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
        """Create destination directories."""
        for dest in self.destinations:
            dest.parent.mkdir(parents=True, exist_ok=True)

    def _restore_hash_state(
        self, dest_path: Path, bytes_to_hash: int
    ) -> HashCalculator:
        """
        Restore hash state by re-hashing existing .tmp file content.

        Returns: HashCalculator with state matching the resume position.
        """
        hasher = HashCalculator(self.hash_algorithm)
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

        if bytes_to_hash == 0:
            return hasher

        # Re-hash the existing .tmp content
        bytes_hashed = 0
        with open(tmp_path, "rb") as f:
            while bytes_hashed < bytes_to_hash:
                # Check for abort
                if self._abort_event.is_set():
                    raise InterruptedError("Hash restoration interrupted")

                chunk_size = min(BUFFER_SIZE, bytes_to_hash - bytes_hashed)
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
                bytes_hashed += len(chunk)

        return hasher

    def _stream_to_destinations(
        self, source_size: int, enable_hashing: bool
    ) -> Iterator[int | str]:
        """
        Read source once, fan out to multiple writers.

        Yields: bytes_copied (int) during copy, then final hash (str) if enabled.
        """
        # Check if we need independent resume (different byte positions)
        resume_positions = [
            self.resume_info.get(dest, ResumeInfo(dest, dest, 0, 0)).bytes_written
            for dest in self.destinations
        ]
        min_position = min(resume_positions)
        max_position = max(resume_positions)

        # If all destinations at same position, use parallel fan-out
        # Otherwise, use sequential copying for simplicity
        if min_position == max_position:
            yield from self._stream_parallel(source_size, enable_hashing, min_position)
        else:
            yield from self._stream_sequential(source_size, enable_hashing)

    def _stream_parallel(
        self, enable_hashing: bool, resume_from: int
    ) -> Iterator[int | str]:
        """
        Parallel fan-out streaming (all destinations at same position).
        """
        # Create queues and writer threads
        chunk_queues = [queue.Queue(maxsize=QUEUE_SIZE) for _ in self.destinations]
        writer_threads = []
        writer_errors = {}

        for i, dest_path in enumerate(self.destinations):
            thread = threading.Thread(
                target=self._writer_thread,
                args=(dest_path, chunk_queues[i], writer_errors, resume_from > 0),
                daemon=True,
            )
            thread.start()
            writer_threads.append(thread)

        # Restore hash state if resuming
        hasher = None
        if enable_hashing and resume_from > 0:
            # Restore hash state by re-hashing first .tmp file
            print(f"Restoring hash state ({resume_from / (1024 * 1024):.1f} MB)...")
            hasher = self._restore_hash_state(self.destinations[0], resume_from)
        elif enable_hashing:
            hasher = HashCalculator(self.hash_algorithm)

        bytes_read = resume_from
        last_progress_time = time.time()
        progress_interval = 0.1  # Throttle progress to max 10 updates/second

        try:
            with open(self.source, "rb") as f:
                # Seek to resume position
                if resume_from > 0:
                    f.seek(resume_from)

                while not self._abort_event.is_set():
                    chunk = f.read(BUFFER_SIZE)
                    if not chunk:
                        break

                    if hasher:
                        hasher.update(chunk)

                    bytes_read += len(chunk)

                    # Distribute to all queues (with timeout to allow interrupt)
                    for q in chunk_queues:
                        while not self._abort_event.is_set():
                            try:
                                q.put(chunk, timeout=0.1)
                                break  # Successfully queued
                            except queue.Full:
                                # Queue full, retry after checking abort
                                continue
                        if self._abort_event.is_set():
                            break

                    # Throttle progress updates to reduce CPU usage
                    current_time = time.time()
                    if current_time - last_progress_time >= progress_interval:
                        yield bytes_read
                        last_progress_time = current_time

                    # Check for writer errors
                    if writer_errors:
                        raise OSError(f"Writer errors: {writer_errors}")

        finally:
            # Signal all writers to stop (with timeout to avoid blocking)
            for q in chunk_queues:
                try:
                    q.put(None, timeout=0.5)
                except queue.Full:
                    # Queue full - thread likely already exiting
                    pass

            # Wait for all writers to finish (with timeout if interrupted)
            timeout_per_thread = 0.5 if self._abort_event.is_set() else None
            for t in writer_threads:
                t.join(timeout=timeout_per_thread)

            # Check for any writer errors (only if not interrupted)
            if writer_errors and not self._abort_event.is_set():
                raise OSError(f"Writer thread errors: {writer_errors}")

        # Yield final progress to ensure 100% is shown
        if bytes_read > 0:
            yield bytes_read

        # Return final hash if enabled
        if hasher:
            yield hasher.hexdigest()

    def _stream_sequential(
        self, source_size: int, enable_hashing: bool
    ) -> Iterator[int | str]:
        """
        Sequential streaming (destinations at different resume positions).
        """
        # Process each destination independently
        hashers = {}

        for dest_path in self.destinations:
            resume_info = self.resume_info.get(dest_path)
            resume_from = resume_info.bytes_written if resume_info else 0

            # Restore or create hasher for this destination
            if enable_hashing and resume_from > 0:
                print(
                    f"Restoring {dest_path.name} hash state ({resume_from / (1024 * 1024):.1f} MB)..."
                )
                hasher = self._restore_hash_state(dest_path, resume_from)
            elif enable_hashing:
                hasher = HashCalculator(self.hash_algorithm)
            else:
                hasher = None

            hashers[dest_path] = hasher

            # Copy remaining bytes to this destination
            temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
            mode = "ab" if resume_from > 0 else "wb"

            bytes_written = resume_from
            last_progress_time = time.time()
            progress_interval = 0.1

            try:
                with open(self.source, "rb") as f_in, open(temp_path, mode) as f_out:
                    if resume_from > 0:
                        f_in.seek(resume_from)

                    while not self._abort_event.is_set():
                        chunk = f_in.read(BUFFER_SIZE)
                        if not chunk:
                            break

                        f_out.write(chunk)
                        if hasher:
                            hasher.update(chunk)

                        bytes_written += len(chunk)

                        # Throttle progress
                        current_time = time.time()
                        if current_time - last_progress_time >= progress_interval:
                            yield bytes_written
                            last_progress_time = current_time

                # Atomic rename
                if not self._abort_event.is_set():
                    temp_path.replace(dest_path)

            except Exception as e:
                raise OSError(f"Error writing {dest_path}: {e}")

        # Yield final progress
        yield source_size

        # Return hash from first destination (they should all match)
        if enable_hashing:
            first_hasher = hashers[self.destinations[0]]
            if first_hasher:
                yield first_hasher.hexdigest()

    def _writer_thread(
        self,
        dest_path: Path,
        chunk_queue: queue.Queue,
        error_dict: dict,
        append_mode: bool = False,
    ) -> None:
        """
        Writer thread: receives chunks and writes to temp file.

        On success, atomically renames temp to final destination.
        """
        temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
        mode = "ab" if append_mode else "wb"

        try:
            with open(temp_path, mode) as f:
                while True:
                    # Get chunk with timeout to allow checking abort
                    try:
                        chunk = chunk_queue.get(timeout=0.1)
                    except queue.Empty:
                        # Check if we should exit
                        if self._abort_event.is_set():
                            return
                        continue  # Keep waiting for chunks

                    if chunk is None:  # Sentinel
                        break
                    if self._abort_event.is_set():
                        return
                    f.write(chunk)

            # Atomic rename on success
            if not self._abort_event.is_set():
                temp_path.replace(dest_path)

        except Exception as e:
            error_dict[dest_path] = str(e)
            self._abort_event.set()
        finally:
            # Clean up temp file if interrupted (keep .tmp on Ctrl+C for resume)
            if self._abort_event.is_set() and not self._interrupted:
                # Only delete on real error, not on Ctrl+C
                if temp_path.exists():
                    with contextlib.suppress(Exception):
                        temp_path.unlink()

    def _verify_destinations_written(self) -> list[DestinationResult]:
        """
        Verify all destination files exist and have correct size.

        Returns list of DestinationResult objects.
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

    def _verify_source_only(self, result: CopyResult) -> Iterator[CopyEvent]:
        """
        SOURCE mode: Re-hash source file to ensure it didn't change during copy.
        """
        yield CopyEvent(
            type=EventType.VERIFY_START,
            total_bytes=result.source_size,
            message="Verifying source file integrity",
        )

        bytes_hashed = 0
        final_hash = ""

        for bytes_hashed, final_hash in HashCalculator.hash_file(
            self.source, self.hash_algorithm, self._abort_event
        ):
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

    def _verify_full(self, result: CopyResult) -> Iterator[CopyEvent]:
        """
        FULL mode: Hash source and all destinations in parallel with real-time progress.
        """
        files_to_hash = [self.source] + self.destinations
        total_bytes = result.source_size * len(files_to_hash)

        yield CopyEvent(
            type=EventType.VERIFY_START,
            total_bytes=total_bytes,
            message=f"Verifying source + {len(self.destinations)} destination(s)",
        )

        # Shared progress counter (thread-safe) tracks bytes hashed across all threads
        progress_lock = threading.Lock()
        shared_progress = {"bytes_hashed": 0}

        def hash_file_with_progress(path: Path) -> tuple[Path, str]:
            """Hash a file and update shared progress counter in real-time."""
            final_hash = ""
            last_bytes = 0

            for bytes_hashed, final_hash in HashCalculator.hash_file(
                path, self.hash_algorithm, self._abort_event
            ):
                if not final_hash:
                    # Progress update - report delta since last update
                    delta = bytes_hashed - last_bytes
                    last_bytes = bytes_hashed

                    with progress_lock:
                        shared_progress["bytes_hashed"] += delta

            # Return final hash
            return (path, final_hash)

        # Hash all files in parallel
        hashes = {}

        with ThreadPoolExecutor(max_workers=len(files_to_hash)) as executor:
            # Submit all hash jobs
            future_to_path = {
                executor.submit(hash_file_with_progress, path): path
                for path in files_to_hash
            }

            # Poll shared progress and yield events
            last_reported = 0
            all_done = False

            while not all_done:
                # Check for abort
                if self._abort_event.is_set():
                    # Cancel all running futures
                    for future in future_to_path:
                        future.cancel()
                    # Exit immediately without completing verification
                    return

                # Check current progress
                with progress_lock:
                    current_bytes = shared_progress["bytes_hashed"]

                # Yield progress if changed
                if current_bytes > last_reported:
                    yield CopyEvent(
                        type=EventType.VERIFY_PROGRESS,
                        bytes_processed=current_bytes,
                        total_bytes=total_bytes,
                    )
                    last_reported = current_bytes

                # Check if all futures are done
                all_done = all(f.done() for f in future_to_path)

                if not all_done:
                    time.sleep(0.1)  # Poll every 100ms (reduces CPU usage)

            # Collect results
            for future, path in future_to_path.items():
                try:
                    path_result, file_hash = future.result()
                    hashes[path_result] = file_hash
                except InterruptedError:
                    # Interrupted - just return without setting hashes
                    return
                except Exception as e:
                    # Mark this destination as failed
                    if path != self.source:
                        for dest_result in result.destinations:
                            if dest_result.path == path:
                                dest_result.success = False
                                dest_result.error = f"Hash failed: {e}"

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
        """Hash a file completely and return the final digest."""
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
    Handles CLI orchestration and presentation using Rich library.

    This layer is completely separate from the core copy engine.
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

    def run(self) -> bool:
        """
        Execute copy jobs for all source files.

        Returns: True if all operations succeeded.
        """
        # Discover files to copy
        source_files = self._discover_files()

        if not source_files:
            print("No files to copy")
            return True

        # Execute each copy job
        results = []
        for i, source_file in enumerate(source_files, 1):
            print(f"\nFile {i}/{len(source_files)}: {source_file.name}")

            dest_paths = self._calculate_destinations(source_file)
            result = self._execute_single_copy(source_file, dest_paths)
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
        """Discover all source files to copy."""
        if self.source.is_file():
            return [self.source]
        elif self.source.is_dir():
            return sorted([f for f in self.source.rglob("*") if f.is_file()])
        else:
            raise FileNotFoundError(f"Source not found: {self.source}")

    def _calculate_destinations(self, source_file: Path) -> list[Path]:
        """Calculate destination paths for a source file."""
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

    def _check_resume(
        self, source: Path, destinations: list[Path]
    ) -> tuple[bool, list[ResumeInfo]]:
        """
        Check for incomplete .tmp files and prompt user to resume.

        Returns: (should_resume, resume_info_list)
        """
        resume_infos = []
        source_stat = source.stat()

        # Check each destination for .tmp file
        for dest in destinations:
            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            if tmp_path.exists():
                try:
                    tmp_stat = tmp_path.stat()
                    tmp_size = tmp_stat.st_size
                    source_size = source_stat.st_size

                    # Validate .tmp file size
                    if tmp_size > source_size:
                        print(
                            f"Warning: {tmp_path.name} is larger than source, will restart",
                            file=sys.stderr,
                        )
                        tmp_path.unlink()
                        continue

                    # Check if source was modified after .tmp was created
                    if source_stat.st_mtime > tmp_stat.st_mtime:
                        print(
                            f"Warning: Source file modified since {tmp_path.name} was created, will restart",
                            file=sys.stderr,
                        )
                        tmp_path.unlink()
                        continue

                    if tmp_size > 0:
                        progress = (
                            (tmp_size / source_size * 100) if source_size > 0 else 0
                        )
                        resume_infos.append(
                            ResumeInfo(
                                dest_path=dest,
                                tmp_path=tmp_path,
                                bytes_written=tmp_size,
                                progress_percent=progress,
                            )
                        )
                except (OSError, FileNotFoundError) as e:
                    print(f"Warning: Could not check {tmp_path}: {e}", file=sys.stderr)

        # If no resumable files found, return False
        if not resume_infos:
            return False, []

        # Show prompt
        print("\nFound incomplete copy:")
        for info in resume_infos:
            mb_done = info.bytes_written / (1024 * 1024)
            source_mb = source.stat().st_size / (1024 * 1024)
            print(
                f"  {info.tmp_path.name} ({mb_done:.1f}/{source_mb:.1f} MB - {info.progress_percent:.1f}%)"
            )

        # Interactive prompt with timeout
        try:
            response = input("Resume from where you left off? [y/n]: ").strip().lower()
            should_resume = response in ["y", "yes"]
        except (EOFError, KeyboardInterrupt):
            print("\nDefaulting to NO (fresh start)")
            should_resume = False

        # If not resuming, clean up .tmp files
        if not should_resume:
            for info in resume_infos:
                info.tmp_path.unlink()
            return False, []

        return True, resume_infos

    def _execute_single_copy(
        self, source: Path, destinations: list[Path]
    ) -> CopyResult:
        """Execute a single copy operation with simple text progress."""
        # Check for resume opportunity
        should_resume, resume_infos = self._check_resume(source, destinations)

        # Create resume info dict for engine
        resume_map = {}
        if should_resume:
            resume_map = {info.dest_path: info for info in resume_infos}

        engine = CopyEngine(
            source=source,
            destinations=destinations,
            verification_mode=self.verification_mode,
            hash_algorithm=self.hash_algorithm,
            resume_info=resume_map,
        )

        result = None
        copy_total = 0
        verify_total = 0

        for event in engine.copy():
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
        """Display a summary of a single copy operation."""
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
        """Display final summary of all operations."""
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
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Professional file copying tool with integrity verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cvv source.mp4 /dest1 /dest2              # Copy to 2 destinations with full verification
  cvv source.mp4 /dest1 -m transfer         # Fast copy with size check only
  cvv /source_dir /dest_dir -m full         # Copy entire directory with verification
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

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # Run the CLI processor
    try:
        processor = CLIProcessor(
            source=args.source,
            destinations=args.destinations,
            verification_mode=VerificationMode(args.mode),
            hash_algorithm=args.hash_algorithm,
        )

        success = processor.run()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
