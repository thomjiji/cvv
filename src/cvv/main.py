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
import queue
import shutil
import signal
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

try:
    import xxhash
except ImportError:
    print("ERROR: The 'xxhash' library is required but not installed.", file=sys.stderr)
    print("Please install it using: pip install xxhash", file=sys.stderr)
    sys.exit(1)

# Constants
BUFFER_SIZE = 8 * 1024 * 1024  # 8MB
QUEUE_SIZE = 10  # Max chunks buffered per destination

HASH_FILE_EXTENSIONS = {
    "xxh3_64": ".xxh3",
    "xxh64": ".xxh",
    "md5": ".md5",
    "sha1": ".sha1",
    "sha256": ".sha256",
}


# ============================================================================
# Data Models
# ============================================================================


class VerificationMode(Enum):
    """Verification strategy for copy operations."""

    TRANSFER = "transfer"
    SOURCE = "source"
    FULL = "full"


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
        return all(d.success for d in self.destinations)

    @property
    def speed_mb_sec(self) -> float:
        if self.duration > 0:
            return (self.source_size / (1024 * 1024)) / self.duration
        return 0.0


# ============================================================================
# Core Copy Engine (UI-agnostic)
# ============================================================================


class HashCalculator:
    """Thread-safe hash calculator supporting multiple algorithms."""

    def __init__(self, algorithm: str = "xxh3_64"):
        self.algorithm = algorithm.lower()
        if self.algorithm == "xxh3_64":
            self._hasher = xxhash.xxh3_64()
        elif self.algorithm == "xxh64":
            self._hasher = xxhash.xxh64()
        elif self.algorithm in ["md5", "sha1", "sha256"]:
            self._hasher = hashlib.new(self.algorithm)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    def update(self, data: bytes) -> None:
        self._hasher.update(data)

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()

    @staticmethod
    def hash_file(
        path: Path,
        algorithm: str = "xxh3_64",
        abort_event: threading.Event | None = None,
    ) -> Iterator[tuple[int, str]]:
        """Hash a file and yield (bytes_hashed, hash_or_empty) progress tuples."""
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
    """Core copy engine: reads source once, writes to N destinations. Yields events, never touches stdout."""

    # Class-level shared abort event (persists across all instances)
    _shared_abort_event = threading.Event()
    _signal_handler_installed = False

    def __init__(
        self,
        source: Path,
        destinations: list[Path],
        verification_mode: VerificationMode = VerificationMode.FULL,
        hash_algorithm: str = "xxh3_64",
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
        """Reset the shared abort event (for testing)."""
        cls._shared_abort_event.clear()

    def _handle_interrupt(self, signum, frame):
        if not CopyEngine._shared_abort_event.is_set():
            CopyEngine._shared_abort_event.set()
            self._interrupted = True
            print("\n\nCopy interrupted.", file=sys.stderr)

    def copy(self) -> Iterator[CopyEvent | CopyResult]:
        """Execute the copy operation, yielding progress events and a final CopyResult."""
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

            for event_or_hash in self._stream_to_destinations(enable_hashing):
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
        self._abort_event.set()

    def _check_source_exists(self) -> None:
        if not self.source.exists():
            raise FileNotFoundError(f"Source file not found: {self.source}")
        if not self.source.is_file():
            raise ValueError(f"Source is not a file: {self.source}")

    def _check_disk_space(self, required_bytes: int) -> None:
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
        for dest in self.destinations:
            dest.parent.mkdir(parents=True, exist_ok=True)

    def _stream_to_destinations(self, enable_hashing: bool) -> Iterator[int | str]:
        """Read source once, fan out to N writer threads via queues."""
        # Create queues and writer threads
        chunk_queues = [queue.Queue(maxsize=QUEUE_SIZE) for _ in self.destinations]
        writer_threads = []
        writer_errors = {}

        for i, dest_path in enumerate(self.destinations):
            thread = threading.Thread(
                target=self._writer_thread,
                args=(dest_path, chunk_queues[i], writer_errors),
                daemon=True,
            )
            thread.start()
            writer_threads.append(thread)

        # Create hasher if needed
        hasher = HashCalculator(self.hash_algorithm) if enable_hashing else None
        bytes_read = 0
        last_progress_time = time.time()
        progress_interval = 0.1  # Throttle progress to max 10 updates/second

        try:
            with open(self.source, "rb") as f:
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

    def _writer_thread(
        self,
        dest_path: Path,
        chunk_queue: queue.Queue,
        error_dict: dict,
    ) -> None:
        """Write chunks from queue to a .tmp file, then atomically rename on success."""
        temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
        mode = "wb"  # Always start fresh (no resume from exact position)

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
        """Verify all destination files exist and have correct size."""
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
        """SOURCE mode: re-hash source to detect changes during copy."""
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
        """FULL mode: hash source and all destinations in parallel."""
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

    def verify(self, result: CopyResult) -> Iterator[CopyEvent | CopyResult]:
        """Run verification on an already-copied file. Yields events and updated CopyResult."""
        start_time = time.time()
        try:
            if self.verification_mode == VerificationMode.SOURCE:
                yield from self._verify_source_only(result)
            elif self.verification_mode == VerificationMode.FULL:
                yield from self._verify_full(result)
        except InterruptedError:
            pass
        finally:
            result.duration += time.time() - start_time
        yield result

    def _hash_file_to_completion(self, path: Path) -> str:
        final_hash = ""
        for _, final_hash in HashCalculator.hash_file(path, self.hash_algorithm):
            if final_hash:
                return final_hash
        return final_hash


# ============================================================================
# Hash File Writers
# ============================================================================


class HashFileWriter:
    """Generates hash files in TeraCopy (.xxh) and ASC MHL formats."""

    ALGO_DISPLAY = {
        "xxh3_64": "xxHash3-64",
        "xxh64": "xxHash-64",
        "md5": "MD5",
        "sha1": "SHA-1",
        "sha256": "SHA-256",
    }

    @staticmethod
    def write_xxh(
        entries: list[tuple[Path, str, int]],
        output_path: Path,
        hash_algorithm: str,
    ) -> Path:
        """Write a TeraCopy-compatible hash file (.xxh/.md5/.sha1/.sha256)."""
        algo_name = HashFileWriter.ALGO_DISPLAY.get(hash_algorithm, hash_algorithm)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"; {algo_name} checksums created by cvv\n")
            f.write(";\n\n")
            for rel_path, hash_hex, _size in entries:
                f.write(f"{hash_hex.upper()} *{rel_path.as_posix()}\n")
        return output_path

    @staticmethod
    def write_mhl(
        entries: list[tuple[Path, str, int]],
        output_dir: Path,
        hash_algorithm: str,
        source_name: str,
    ) -> Path:
        """Write an ASC MHL XML file in ascmhl/ subdirectory."""
        ascmhl_dir = output_dir / "ascmhl"
        ascmhl_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%d_%H%M%S")
        output_path = ascmhl_dir / f"0001_{source_name}_{timestamp}.mhl"

        root = ET.Element("hashlist", version="2.0")

        creator = ET.SubElement(root, "creatorinfo")
        ET.SubElement(creator, "creationtool").text = "cvv 0.0.1"
        ET.SubElement(creator, "creationdate").text = now.isoformat()

        hashes_elem = ET.SubElement(root, "hashes")
        for rel_path, hash_hex, size in entries:
            hash_elem = ET.SubElement(hashes_elem, "hash")
            path_elem = ET.SubElement(hash_elem, "path", size=str(size))
            path_elem.text = rel_path.as_posix()
            ET.SubElement(hash_elem, hash_algorithm).text = hash_hex.lower()

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="UTF-8", xml_declaration=True)

        return output_path


# ============================================================================
# CLI Layer (Presentation)
# ============================================================================


class CLIProcessor:
    """CLI orchestration and presentation layer."""

    def __init__(
        self,
        source: Path,
        destinations: list[Path],
        verification_mode: VerificationMode,
        hash_algorithm: str,
        hash_file_formats: list[str] | None = None,
        hash_file_dest: str = "dest",
        verify_deferred: bool = False,
    ):
        self.source = source
        self.destinations = destinations
        self.verification_mode = verification_mode
        self.hash_algorithm = hash_algorithm
        self.hash_file_formats = hash_file_formats or []
        self.hash_file_dest = hash_file_dest
        self.verify_deferred = verify_deferred
        self.console = Console()

    def run(self) -> bool:
        """Execute copy jobs for all source files."""
        source_files = self._discover_files()
        if not source_files:
            self.console.print("No files to copy")
            return True

        total_bytes = sum(f.stat().st_size for f in source_files)
        copy_mode = VerificationMode.TRANSFER if self.verify_deferred else self.verification_mode
        results: list[CopyResult] = []
        engines: list[CopyEngine] = []
        bytes_completed = 0

        # Phase 1: Copy all files
        with self._make_progress() as progress:
            overall_task = progress.add_task(
                f"[bold]Copying [0/{len(source_files)}]",
                total=total_bytes,
            )

            for i, source_file in enumerate(source_files, 1):
                if CopyEngine._shared_abort_event.is_set():
                    break

                file_size = source_file.stat().st_size
                dest_paths = self._calculate_destinations(source_file)
                destinations_to_copy = self._check_duplicates_and_cleanup(
                    source_file, dest_paths
                )

                if not destinations_to_copy:
                    self.console.print(
                        f"[dim]⊘ {source_file.name} (already exists)[/dim]"
                    )
                    result = CopyResult(
                        source_path=source_file,
                        source_size=file_size,
                        verification_mode=self.verification_mode,
                    )
                    for dest in dest_paths:
                        result.destinations.append(
                            DestinationResult(path=dest, success=True)
                        )
                    results.append(result)
                    bytes_completed += file_size
                    progress.update(
                        overall_task,
                        completed=bytes_completed,
                        description=f"[bold]Copying [{i}/{len(source_files)}]",
                    )
                    continue

                engine = CopyEngine(
                    source=source_file,
                    destinations=destinations_to_copy,
                    verification_mode=copy_mode,
                    hash_algorithm=self.hash_algorithm,
                )

                file_task = progress.add_task(
                    f"Copying {source_file.name}", total=file_size
                )

                result = None
                for event in engine.copy():
                    if isinstance(event, CopyResult):
                        result = event
                        break
                    elif event.type == EventType.COPY_PROGRESS:
                        progress.update(
                            file_task, completed=event.bytes_processed
                        )
                        progress.update(
                            overall_task,
                            completed=bytes_completed + event.bytes_processed,
                        )
                    elif event.type == EventType.COPY_COMPLETE:
                        bytes_completed += file_size
                        progress.update(
                            overall_task, completed=bytes_completed
                        )
                    elif event.type == EventType.VERIFY_START:
                        progress.reset(
                            file_task,
                            total=event.total_bytes,
                            description=f"Verifying {source_file.name}",
                        )
                    elif event.type == EventType.VERIFY_PROGRESS:
                        progress.update(
                            file_task, completed=event.bytes_processed
                        )

                progress.remove_task(file_task)
                progress.update(
                    overall_task,
                    description=f"[bold]Copying [{i}/{len(source_files)}]",
                )

                results.append(result)
                engines.append(engine)
                self._print_file_result(result)

                if not result.success:
                    break

        # Phase 2: Deferred verification
        if self.verify_deferred and self.verification_mode != VerificationMode.TRANSFER:
            all_copied = all(r.success for r in results)
            if all_copied and results and not CopyEngine._shared_abort_event.is_set():
                self.console.print(
                    f"\n[bold]All {len(results)} file(s) copied.[/bold] "
                    f"Verification mode: [cyan]{self.verification_mode.value}[/cyan]"
                )
                answer = input("Start verification? [Y/n] ").strip().lower()
                if answer in ("", "y", "yes"):
                    self._run_deferred_verify(results, engines)
                else:
                    self.console.print("[dim]Verification skipped.[/dim]")

        # Generate hash files
        all_success = all(r.success for r in results)
        if self.hash_file_formats and all_success and results:
            self._generate_hash_files(results)

        self._show_final_summary(results)
        return all_success

    def _run_deferred_verify(
        self, results: list[CopyResult], engines: list[CopyEngine]
    ) -> None:
        """Run verification on all copied files."""
        total_bytes = 0
        verify_pairs: list[tuple[CopyEngine, CopyResult]] = []
        for engine, result in zip(engines, results):
            if not result.success:
                continue
            engine.verification_mode = self.verification_mode
            result.verification_mode = self.verification_mode
            if self.verification_mode == VerificationMode.SOURCE:
                total_bytes += result.source_size
            elif self.verification_mode == VerificationMode.FULL:
                total_bytes += result.source_size * (1 + len(result.destinations))
            verify_pairs.append((engine, result))

        with self._make_progress() as progress:
            overall_task = progress.add_task(
                f"[bold]Verifying [0/{len(verify_pairs)}]", total=total_bytes
            )
            bytes_completed = 0

            for i, (engine, result) in enumerate(verify_pairs, 1):
                if CopyEngine._shared_abort_event.is_set():
                    break

                file_task = progress.add_task(
                    f"Verifying {result.source_path.name}",
                    total=result.source_size,
                )

                for event in engine.verify(result):
                    if isinstance(event, CopyResult):
                        break
                    elif event.type == EventType.VERIFY_START:
                        progress.reset(file_task, total=event.total_bytes)
                    elif event.type == EventType.VERIFY_PROGRESS:
                        progress.update(
                            file_task, completed=event.bytes_processed
                        )
                        progress.update(
                            overall_task,
                            completed=bytes_completed + event.bytes_processed,
                        )

                if self.verification_mode == VerificationMode.SOURCE:
                    bytes_completed += result.source_size
                elif self.verification_mode == VerificationMode.FULL:
                    bytes_completed += result.source_size * (1 + len(result.destinations))

                progress.remove_task(file_task)
                progress.update(
                    overall_task,
                    completed=bytes_completed,
                    description=f"[bold]Verifying [{i}/{len(verify_pairs)}]",
                )
                self._print_file_result(result)

    def _make_progress(self) -> Progress:
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            console=self.console,
        )

    def _discover_files(self) -> list[Path]:
        if self.source.is_file():
            return [self.source]
        elif self.source.is_dir():
            return sorted([f for f in self.source.rglob("*") if f.is_file()])
        else:
            raise FileNotFoundError(f"Source not found: {self.source}")

    def _calculate_destinations(self, source_file: Path) -> list[Path]:
        if self.source.is_dir():
            relative_path = source_file.relative_to(self.source)
            return [dest_root / relative_path for dest_root in self.destinations]

        dest_paths = []
        for dest in self.destinations:
            if dest.is_dir():
                dest_paths.append(dest / source_file.name)
            else:
                dest_paths.append(dest)
        return dest_paths

    def _check_duplicates_and_cleanup(
        self, source: Path, destinations: list[Path]
    ) -> list[Path]:
        """Skip already-completed files and clean up incomplete .tmp files."""
        source_size = source.stat().st_size
        destinations_to_copy = []

        for dest in destinations:
            if dest.exists():
                dest_size = dest.stat().st_size
                if dest_size == source_size:
                    self.console.print(
                        f"  [green]✓[/green] {dest.name} already exists [dim](skipping)[/dim]"
                    )
                    continue
                else:
                    self.console.print(
                        f"  [yellow]![/yellow] {dest.name} wrong size "
                        f"({dest_size} vs {source_size}), will overwrite"
                    )
                    dest.unlink()

            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            if tmp_path.exists():
                tmp_size = tmp_path.stat().st_size
                self.console.print(
                    f"  [yellow]![/yellow] Found incomplete {tmp_path.name} "
                    f"({tmp_size / (1024 * 1024):.1f}/{source_size / (1024 * 1024):.1f} MB), restarting"
                )
                tmp_path.unlink()

            destinations_to_copy.append(dest)

        return destinations_to_copy

    def _print_file_result(self, result: CopyResult) -> None:
        name = result.source_path.name
        size = self._format_size(result.source_size)
        speed = f"{result.speed_mb_sec:.1f} MB/s"

        if not result.success:
            errors = [dr.error for dr in result.destinations if dr.error]
            self.console.print(
                f"[red]✗[/red] {name}  [red]{errors[0] if errors else 'Failed'}[/red]"
            )
            return

        parts = [f"[green]✓[/green] {name}", f"[dim]{size}[/dim]", f"[dim]{speed}[/dim]"]

        if result.source_hash_inflight:
            parts.append(f"[cyan]{result.source_hash_inflight}[/cyan]")

        if result.source_hash_post:
            ok = result.source_hash_post == result.source_hash_inflight
            parts.append(f"[{'green' if ok else 'red'}]src {'✓' if ok else '✗'}[/]")

        if result.verification_mode == VerificationMode.FULL:
            for dr in result.destinations:
                if dr.hash_post:
                    ok = dr.hash_post == result.source_hash_inflight
                    parts.append(
                        f"[{'green' if ok else 'red'}]dst {'✓' if ok else '✗'}[/]"
                    )

        self.console.print("  ".join(parts))

    def _show_final_summary(self, results: list[CopyResult]) -> None:
        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success
        total_size = sum(r.source_size for r in results)
        total_duration = sum(r.duration for r in results)

        self.console.print()
        if failed == 0:
            avg_speed = (
                f"{(total_size / (1024 ** 2)) / total_duration:.1f} MB/s"
                if total_duration > 0
                else ""
            )
            self.console.print(
                f"[bold green]All {total} file(s) completed[/bold green]  "
                f"[dim]{self._format_size(total_size)}  {avg_speed}[/dim]"
            )
        else:
            self.console.print(
                f"[bold red]{failed} of {total} file(s) failed[/bold red]"
            )

    def _generate_hash_files(self, results: list[CopyResult]) -> None:
        """Write hash files at source and/or destination roots."""
        source_name = self.source.stem if self.source.is_file() else self.source.name
        ext = HASH_FILE_EXTENSIONS.get(self.hash_algorithm, ".xxh")

        dirs_to_write: list[tuple[Path, list[tuple[Path, str, int]]]] = []

        if self.hash_file_dest in ("dest", "both"):
            for dest_root in self.destinations:
                hash_dir = dest_root if self.source.is_dir() or dest_root.is_dir() else dest_root.parent
                entries = self._collect_entries(results, hash_dir)
                if entries:
                    dirs_to_write.append((hash_dir, entries))

        if self.hash_file_dest in ("source", "both"):
            source_dir = self.source if self.source.is_dir() else self.source.parent
            entries = self._collect_source_entries(results, source_dir)
            if entries:
                dirs_to_write.append((source_dir, entries))

        for hash_dir, entries in dirs_to_write:
            if "xxh" in self.hash_file_formats:
                path = HashFileWriter.write_xxh(
                    entries, hash_dir / (source_name + ext), self.hash_algorithm
                )
                self.console.print(f"  [dim]Hash file:[/dim] {path}")
            if "mhl" in self.hash_file_formats:
                path = HashFileWriter.write_mhl(
                    entries, hash_dir, self.hash_algorithm, source_name
                )
                self.console.print(f"  [dim]MHL file:[/dim] {path}")

    def _collect_entries(
        self, results: list[CopyResult], hash_dir: Path
    ) -> list[tuple[Path, str, int]]:
        """Collect (relative_path, hash, size) for a destination hash_dir."""
        entries: list[tuple[Path, str, int]] = []
        for result in results:
            hash_hex = result.source_hash_inflight
            if not hash_hex:
                continue
            for dr in result.destinations:
                try:
                    rel = dr.path.relative_to(hash_dir)
                except ValueError:
                    continue
                entries.append((rel, dr.hash_post or hash_hex, result.source_size))
                break
        return entries

    def _collect_source_entries(
        self, results: list[CopyResult], source_dir: Path
    ) -> list[tuple[Path, str, int]]:
        """Collect (relative_path, hash, size) relative to source directory."""
        entries: list[tuple[Path, str, int]] = []
        for result in results:
            hash_hex = result.source_hash_inflight
            if not hash_hex:
                continue
            try:
                rel = result.source_path.relative_to(source_dir)
            except ValueError:
                continue
            entries.append((rel, hash_hex, result.source_size))
        return entries

    @staticmethod
    def _format_size(n: int) -> str:
        if n >= 1024**3:
            return f"{n / 1024 ** 3:.1f} GB"
        if n >= 1024**2:
            return f"{n / 1024 ** 2:.1f} MB"
        if n >= 1024:
            return f"{n / 1024:.1f} KB"
        return f"{n} B"


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> int:
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
        default="xxh3_64",
        choices=["xxh3_64", "xxh64", "md5", "sha1", "sha256"],
        help="Hash algorithm for verification (default: xxh3_64)",
    )

    parser.add_argument(
        "--hash-file",
        type=str,
        nargs="+",
        choices=["xxh", "mhl"],
        help="Generate hash file(s) (xxh: TeraCopy format, mhl: ASC MHL)",
    )

    parser.add_argument(
        "--hash-file-dest",
        type=str,
        default="dest",
        choices=["source", "dest", "both"],
        help="Where to write hash files: source, dest, or both (default: dest)",
    )

    parser.add_argument(
        "--verify-after",
        action="store_true",
        help="Defer verification until all files are copied, then prompt",
    )

    args = parser.parse_args()

    # Run the CLI processor
    try:
        processor = CLIProcessor(
            source=args.source,
            destinations=args.destinations,
            verification_mode=VerificationMode(args.mode),
            hash_algorithm=args.hash_algorithm,
            hash_file_formats=args.hash_file or [],
            hash_file_dest=args.hash_file_dest,
            verify_deferred=args.verify_after,
        )

        success = processor.run()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
