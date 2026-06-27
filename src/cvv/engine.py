"""Core copy engine — reads source once, writes to N destinations."""

import contextlib
import hashlib
import queue
import shutil
import signal
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import xxhash
except ImportError:
    print("ERROR: The 'xxhash' library is required but not installed.", file=sys.stderr)
    print("Please install it using: pip install xxhash", file=sys.stderr)
    sys.exit(1)

from .models import (
    BUFFER_SIZE,
    QUEUE_SIZE,
    CopyEvent,
    CopyResult,
    DestinationResult,
    EventType,
    VerificationMode,
)


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
                if abort_event and abort_event.is_set():
                    raise InterruptedError("Hash operation interrupted")
                hasher.update(chunk)
                total_bytes += len(chunk)
                yield (total_bytes, "")

        yield (total_bytes, hasher.hexdigest())


class CopyEngine:
    """Core copy engine: reads source once, writes to N destinations. Yields events, never touches stdout."""

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
        self._abort_event = (
            abort_event if abort_event else CopyEngine._shared_abort_event
        )
        self._interrupted = False

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

        result = CopyResult(
            source_path=self.source,
            source_size=0,
            verification_mode=self.verification_mode,
        )

        try:
            source_size = self.source.stat().st_size
            result.source_size = source_size
            self._check_source_exists()
            self._check_disk_space(source_size)
            self._prepare_destinations()

            yield CopyEvent(
                type=EventType.COPY_START,
                total_bytes=source_size,
                message=f"Copying {self.source.name} to {len(self.destinations)} destination(s)",
            )

            bytes_copied = 0
            enable_hashing = True

            for event_or_hash in self._stream_to_destinations(enable_hashing):
                if isinstance(event_or_hash, str):
                    result.source_hash_inflight = event_or_hash
                else:
                    bytes_copied = event_or_hash
                    yield CopyEvent(
                        type=EventType.COPY_PROGRESS,
                        bytes_processed=bytes_copied,
                        total_bytes=source_size,
                    )

            dest_results = self._verify_destinations_written()
            result.destinations = dest_results

            yield CopyEvent(
                type=EventType.COPY_COMPLETE,
                bytes_processed=source_size,
                total_bytes=source_size,
                message="Copy phase complete",
            )

            if self.verification_mode == VerificationMode.TRANSFER:
                pass
            elif self.verification_mode == VerificationMode.SOURCE:
                yield from self._verify_source_only(result)
            elif self.verification_mode == VerificationMode.FULL:
                yield from self._verify_full(result)

        except InterruptedError:
            pass
        except Exception as e:
            for dest in self.destinations:
                result.destinations.append(
                    DestinationResult(path=dest, success=False, error=str(e))
                )

        finally:
            result.duration = time.time() - start_time

        yield result

    def abort(self) -> None:
        self._abort_event.set()

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

        hasher = HashCalculator(self.hash_algorithm) if enable_hashing else None
        bytes_read = 0
        last_progress_time = time.time()
        progress_interval = 0.1

        try:
            with open(self.source, "rb") as f:
                while not self._abort_event.is_set():
                    chunk = f.read(BUFFER_SIZE)
                    if not chunk:
                        break

                    if hasher:
                        hasher.update(chunk)

                    bytes_read += len(chunk)

                    for q in chunk_queues:
                        while not self._abort_event.is_set():
                            try:
                                q.put(chunk, timeout=0.1)
                                break
                            except queue.Full:
                                continue
                        if self._abort_event.is_set():
                            break

                    current_time = time.time()
                    if current_time - last_progress_time >= progress_interval:
                        yield bytes_read
                        last_progress_time = current_time

                    if writer_errors:
                        raise OSError(f"Writer errors: {writer_errors}")

        finally:
            for q in chunk_queues:
                try:
                    q.put(None, timeout=0.5)
                except queue.Full:
                    pass

            timeout_per_thread = 0.5 if self._abort_event.is_set() else None
            for t in writer_threads:
                t.join(timeout=timeout_per_thread)

            if writer_errors and not self._abort_event.is_set():
                raise OSError(f"Writer thread errors: {writer_errors}")

        if bytes_read > 0:
            yield bytes_read

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
        mode = "wb"

        try:
            with open(temp_path, mode) as f:
                while True:
                    try:
                        chunk = chunk_queue.get(timeout=0.1)
                    except queue.Empty:
                        if self._abort_event.is_set():
                            return
                        continue

                    if chunk is None:
                        break
                    if self._abort_event.is_set():
                        return
                    f.write(chunk)

            if not self._abort_event.is_set():
                temp_path.replace(dest_path)
                shutil.copystat(self.source, dest_path)

        except Exception as e:
            error_dict[dest_path] = str(e)
            self._abort_event.set()
        finally:
            if self._abort_event.is_set() and not self._interrupted:
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

        progress_lock = threading.Lock()
        shared_progress = {"bytes_hashed": 0}

        def hash_file_with_progress(path: Path) -> tuple[Path, str]:
            final_hash = ""
            last_bytes = 0

            for bytes_hashed, final_hash in HashCalculator.hash_file(
                path, self.hash_algorithm, self._abort_event
            ):
                if not final_hash:
                    delta = bytes_hashed - last_bytes
                    last_bytes = bytes_hashed

                    with progress_lock:
                        shared_progress["bytes_hashed"] += delta

            return (path, final_hash)

        hashes = {}

        with ThreadPoolExecutor(max_workers=len(files_to_hash)) as executor:
            future_to_path = {
                executor.submit(hash_file_with_progress, path): path
                for path in files_to_hash
            }

            last_reported = 0
            all_done = False

            while not all_done:
                if self._abort_event.is_set():
                    for future in future_to_path:
                        future.cancel()
                    return

                with progress_lock:
                    current_bytes = shared_progress["bytes_hashed"]

                if current_bytes > last_reported:
                    yield CopyEvent(
                        type=EventType.VERIFY_PROGRESS,
                        bytes_processed=current_bytes,
                        total_bytes=total_bytes,
                    )
                    last_reported = current_bytes

                all_done = all(f.done() for f in future_to_path)

                if not all_done:
                    time.sleep(0.1)

            for future, path in future_to_path.items():
                try:
                    path_result, file_hash = future.result()
                    hashes[path_result] = file_hash
                except InterruptedError:
                    return
                except Exception as e:
                    if path != self.source:
                        for dest_result in result.destinations:
                            if dest_result.path == path:
                                dest_result.success = False
                                dest_result.error = f"Hash failed: {e}"

        result.source_hash_post = hashes.get(self.source)

        for dest_result in result.destinations:
            dest_hash = hashes.get(dest_result.path)
            dest_result.hash_post = dest_hash

            if dest_result.success and dest_hash != result.source_hash_inflight:
                dest_result.success = False
                dest_result.error = (
                    f"Hash mismatch: {dest_hash} != {result.source_hash_inflight}"
                )

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
        final_hash = ""
        for _, final_hash in HashCalculator.hash_file(path, self.hash_algorithm):
            if final_hash:
                return final_hash
        return final_hash
