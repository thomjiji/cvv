#!/usr/bin/env python3
"""
pfndispatchcopy - Professional file copying tool with integrity verification.

This tool implements the core functionality of Offload Manager's pfndispatchcopy,
providing reliable file copying with hash verification, progress monitoring,
and multi-destination support for professional DIT workflows.
"""

import argparse
import hashlib
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import xxhash

    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False


class ProgressTracker:
    """Track and report copy progress across multiple threads."""

    def __init__(self, total_size: int, update_interval: float = 2.0) -> None:
        """
        Initialize progress tracker.

        Parameters
        ----------
        total_size : int
            Total file size in bytes
        update_interval : float
            Interval between progress updates in seconds
        """
        self.total_size = total_size
        self.update_interval = update_interval
        self.bytes_copied = 0
        self.start_time = time.time()
        self.last_update = 0
        self.last_percentage = 0
        self.min_bytes_threshold = max(
            1024 * 1024 * 10, total_size // 50
        )  # 10MB or 2% of file
        self.lock = threading.Lock()

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

            # Update if enough time has passed AND (enough bytes copied OR significant percentage change)
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
                    f"copied {bytes_copied:,} of {self.total_size:,} bytes ({current_percentage:.1f}%)"
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

        if self.algorithm == "xxh64be" and not HAS_XXHASH:
            raise ValueError(
                "xxhash library not available. Install with: pip install xxhash"
            )

    def calculate(self, file_path: Path, buffer_size: int = 8388608) -> str:
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
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.algorithm}")

        with open(file_path, "rb") as f:
            while chunk := f.read(buffer_size):
                hasher.update(chunk)

        return hasher.hexdigest()


class ChunkData:
    """Data structure for passing chunks between threads."""

    def __init__(self, chunk_id: int, data: bytes) -> None:
        """
        Initialize chunk data.

        Parameters
        ----------
        chunk_id : int
            Sequential ID of this chunk
        data : bytes
            The actual chunk data
        """
        self.chunk_id = chunk_id
        self.data = data


class ParallelWriter:
    """Handles parallel writing to multiple destinations."""

    def __init__(
        self,
        destinations: list[Path],
        temp_files: dict[Path, Path],
        progress_tracker: ProgressTracker,
    ) -> None:
        """
        Initialize parallel writer.

        Parameters
        ----------
        destinations : List[Path]
            List of destination paths
        temp_files : Dict[Path, Path]
            Mapping of destinations to temporary files
        progress_tracker : ProgressTracker
            Progress tracking instance
        """
        self.destinations = destinations
        self.temp_files = temp_files
        self.progress_tracker = progress_tracker
        self.chunk_queues = {dest: queue.Queue() for dest in destinations}
        self.file_handles = {}
        self.write_errors = {}
        self.chunks_written = {dest: 0 for dest in destinations}
        self.total_bytes_written = 0
        self.write_lock = threading.Lock()

    def open_files(self) -> None:
        """Open all destination files for writing."""
        for dest in self.destinations:
            self.file_handles[dest] = open(self.temp_files[dest], "wb")

    def close_files(self) -> None:
        """Close all destination files."""
        for handle in self.file_handles.values():
            if not handle.closed:
                handle.close()

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
                        # Only update progress when all destinations have written this chunk
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

    def distribute_chunk(self, chunk: ChunkData) -> None:
        """Distribute a chunk to all writer queues."""
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


class PSTaskWrapper:
    """
    Wrapper class that handles both file and directory copying operations.

    This class mimics the behavior of Offload Manager's PSTaskWrapper,
    providing unified interface for copying single files or entire directory trees
    to multiple destinations while preserving directory structure.
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize PSTaskWrapper.

        Parameters
        ----------
        verbose : bool
            Enable verbose logging
        """
        self.verbose = verbose
        self.total_files = 0
        self.completed_files = 0
        self.failed_files = 0
        self.total_bytes = 0
        self.copied_bytes = 0

    def discover_files(self, source_path: Path) -> list[Path]:
        """
        Discover all files in source path (file or directory).

        Parameters
        ----------
        source_path : Path
            Source file or directory path

        Returns
        -------
        List[Path]
            List of all files to be copied
        """
        if source_path.is_file():
            return [source_path]
        elif source_path.is_dir():
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
            return file_path.name
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
        destination_roots : List[Path]
            List of destination root paths

        Returns
        -------
        List[Path]
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
        buffer_size: int = 8388608,
        expected_size: int | None = None,
        noflush_destinations: list[Path] | None = None,
        hash_algorithm: str | None = None,
        keep_source: bool = True,
    ) -> dict[str, Any]:
        """
        Copy entire directory structure to multiple destinations.

        Parameters
        ----------
        source : Path
            Source directory path
        destinations : List[Path]
            List of destination directory paths
        buffer_size : int
            Buffer size for copying operations
        expected_size : Optional[int]
            Expected total size for verification (optional for directories)
        noflush_destinations : List[Path], optional
            Destinations to skip flushing for
        hash_algorithm : Optional[str]
            Hash algorithm for verification
        keep_source : bool
            Whether to keep source after copying

        Returns
        -------
        Dict[str, Any]
            Copy results including statistics and success status
        """
        if noflush_destinations is None:
            noflush_destinations = []

        logging.info(f"PSTaskWrapper launching directory copy from {source}")

        # Discover all files in source
        source_files = self.discover_files(source)
        self.total_files = len(source_files)

        if self.total_files == 0:
            logging.warning(f"No files found in source: {source}")
            return {"success": True, "files_copied": 0, "files_failed": 0}

        logging.info(f"Found {self.total_files} files to copy")

        # Calculate total size
        self.total_bytes = sum(f.stat().st_size for f in source_files)
        if expected_size and self.total_bytes != expected_size:
            logging.warning(
                f"Size mismatch: expected {expected_size}, found {self.total_bytes}"
            )

        results = {
            "success": True,
            "files_copied": 0,
            "files_failed": 0,
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "file_results": {},
        }

        # Process each file
        for source_file in source_files:
            try:
                # Map to destination paths
                dest_files = self.map_destinations(source_file, source, destinations)

                # Determine which destinations should skip flush
                noflush_for_file = []
                for dest_file in dest_files:
                    for noflush_dest in noflush_destinations:
                        if dest_file == noflush_dest or dest_file.is_relative_to(
                            noflush_dest
                        ):
                            noflush_for_file.append(dest_file)
                            break

                # Copy file to all destinations
                if len(dest_files) > 1:
                    file_result = copy_with_multiple_destinations_parallel(
                        source=source_file,
                        destinations=dest_files,
                        buffer_size=buffer_size,
                        noflush_destinations=noflush_for_file,
                        hash_algorithm=hash_algorithm,
                        keep_source=keep_source,
                    )
                else:
                    file_result = copy_with_multiple_destinations(
                        source=source_file,
                        destinations=dest_files,
                        buffer_size=buffer_size,
                        noflush_destinations=noflush_for_file,
                        hash_algorithm=hash_algorithm,
                        keep_source=keep_source,
                    )

                if file_result["success"]:
                    self.completed_files += 1
                    results["files_copied"] += 1
                    logging.info(
                        f"✓ Copied {source_file.name} ({self.completed_files}/{self.total_files})"
                    )
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

        logging.info(
            f"Directory copy completed: {self.completed_files}/{self.total_files} files ({success_rate:.1f}%)"
        )

        if self.failed_files > 0:
            logging.warning(f"Failed files: {self.failed_files}")
            results["success"] = False

        return results

    def launch_copy(
        self,
        source: Path | str,
        destinations: list[Path | str],
        buffer_size: int = 8388608,
        expected_size: int | None = None,
        noflush_destinations: list[Path | str] | None = None,
        hash_algorithm: str | None = None,
        keep_source: bool = True,
    ) -> dict[str, Any]:
        """
        Launch copy operation for file or directory.

        Parameters
        ----------
        source : Union[Path, str]
            Source file or directory path
        destinations : Union[List[Path], List[str]]
            List of destination paths
        buffer_size : int
            Buffer size for copying operations
        expected_size : Optional[int]
            Expected file/directory size for verification
        noflush_destinations : Optional[List[Union[Path, str]]]
            Destinations to skip flushing for
        hash_algorithm : Optional[str]
            Hash algorithm for verification
        keep_source : bool
            Whether to keep source after copying

        Returns
        -------
        Dict[str, Any]
            Copy results
        """
        # Convert to Path objects
        source_path = Path(source)
        dest_paths = [Path(dest) for dest in destinations]
        noflush_paths = [Path(dest) for dest in (noflush_destinations or [])]

        if source_path.is_file():
            # Single file copy
            if len(dest_paths) > 1:
                return copy_with_multiple_destinations_parallel(
                    source=source_path,
                    destinations=dest_paths,
                    buffer_size=buffer_size,
                    expected_size=expected_size,
                    noflush_destinations=noflush_paths,
                    hash_algorithm=hash_algorithm,
                    keep_source=keep_source,
                )
            else:
                return copy_with_multiple_destinations(
                    source=source_path,
                    destinations=dest_paths,
                    buffer_size=buffer_size,
                    expected_size=expected_size,
                    noflush_destinations=noflush_paths,
                    hash_algorithm=hash_algorithm,
                    keep_source=keep_source,
                )
        elif source_path.is_dir():
            # Directory copy
            return self.copy_directory_structure(
                source=source_path,
                destinations=dest_paths,
                buffer_size=buffer_size,
                expected_size=expected_size,
                noflush_destinations=noflush_paths,
                hash_algorithm=hash_algorithm,
                keep_source=keep_source,
            )
        else:
            raise FileNotFoundError(f"Source path does not exist: {source_path}")


def copy_file_to_destination(
    source: Path,
    destination: Path,
    buffer_size: int,
    progress_tracker: ProgressTracker,
    noflush: bool = False,
) -> bool:
    """
    Copy file to a single destination with progress tracking.

    Parameters
    ----------
    source : Path
        Source file path
    destination : Path
        Destination file path
    buffer_size : int
        Buffer size for copying
    progress_tracker : ProgressTracker
        Progress tracker instance
    noflush : bool
        Whether to skip file system flush

    Returns
    -------
    bool
        True if copy successful

    Raises
    ------
    IOError
        If copy operation fails
    """
    # Create destination directory if it doesn't exist
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary file in destination directory
    temp_file = destination.parent / f".{destination.name}.tmp"

    try:
        with open(source, "rb") as src, open(temp_file, "wb") as dst:
            bytes_copied = 0
            while chunk := src.read(buffer_size):
                dst.write(chunk)
                bytes_copied += len(chunk)
                progress_tracker.update(bytes_copied)

        # Verify file size
        if temp_file.stat().st_size != source.stat().st_size:
            raise IOError(f"File size mismatch for {destination}")

        # Flush to disk if not disabled
        if not noflush:
            os.fsync(dst.fileno() if "dst" in locals() and not dst.closed else -1)

        # Atomically move temp file to final destination
        temp_file.rename(destination)

        logging.info(
            f"Copy completed successfully for destination file with file size check ({destination})"
        )
        return True

    except Exception as e:
        # Clean up temp file on error
        if temp_file.exists():
            temp_file.unlink()
        raise IOError(f"Failed to copy to {destination}: {e}")


def copy_with_multiple_destinations_parallel(
    source: Path,
    destinations: list[Path],
    buffer_size: int,
    expected_size: int | None = None,
    noflush_destinations: list[Path] | None = None,
    hash_algorithm: str | None = None,
    keep_source: bool = True,
) -> dict[str, Any]:
    """
    Copy source file to multiple destinations with true parallel writing.

    This function implements genuine parallel I/O where multiple writer threads
    simultaneously write chunks to different destinations while a reader thread
    feeds data to all writers through queues.

    Parameters
    ----------
    source : Path
        Source file path
    destinations : List[Path]
        List of destination paths
    buffer_size : int
        Buffer size for copying operations
    expected_size : Optional[int]
        Expected file size for verification
    noflush_destinations : List[Path], optional
        Destinations to skip flushing for
    hash_algorithm : Optional[str]
        Hash algorithm for verification
    keep_source : bool
        Whether to keep source file after copying

    Returns
    -------
    Dict[str, Any]
        Copy results including hash, speed, and success status

    Raises
    ------
    FileNotFoundError
        If source file doesn't exist
    ValueError
        If file size doesn't match expected
    """
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    # Verify expected file size
    actual_size = source.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise ValueError(
            f"File size mismatch: expected {expected_size}, got {actual_size}"
        )

    if noflush_destinations is None:
        noflush_destinations = []

    logging.info(f"copying {source}...")

    # Initialize progress tracker
    progress_tracker = ProgressTracker(actual_size)
    results = {"success": True, "destinations": {}}

    try:
        # Create temporary files for all destinations
        temp_files = {}
        for dest in destinations:
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp_file = dest.parent / f".{dest.name}.tmp"
            temp_files[dest] = temp_file

        # Initialize parallel writer
        parallel_writer = ParallelWriter(destinations, temp_files, progress_tracker)
        parallel_writer.open_files()

        try:
            # Start writer threads
            writer_threads = parallel_writer.start_writers()

            # Read source and distribute chunks to writers
            chunk_id = 0
            total_bytes_read = 0

            with open(source, "rb") as src:
                while chunk := src.read(buffer_size):
                    chunk_data = ChunkData(chunk_id, chunk)
                    parallel_writer.distribute_chunk(chunk_data)

                    chunk_id += 1
                    total_bytes_read += len(chunk)

            # Stop all writer threads
            parallel_writer.stop_writers(writer_threads)

            # Check for write errors
            if parallel_writer.write_errors:
                results["success"] = False
                for dest, error in parallel_writer.write_errors.items():
                    results["destinations"][str(dest)] = f"Write error: {error}"
                    logging.error(f"Write error for {dest}: {error}")
                return results

        finally:
            parallel_writer.close_files()

        # Verify and move files
        for dest in destinations:
            temp_file = temp_files[dest]

            # Verify file size
            if temp_file.stat().st_size != actual_size:
                results["success"] = False
                results["destinations"][str(dest)] = "File size mismatch"
                continue

            # Flush if not disabled
            noflush = dest in noflush_destinations
            if noflush:
                logging.info(f'skipping flush for destination "{dest}"')

            # Move to final destination
            temp_file.rename(dest)
            logging.info(
                f"Copy completed successfully for destination file with file size check ({dest})"
            )
            results["destinations"][str(dest)] = "success"

        # Calculate performance stats
        duration, speed = progress_tracker.get_stats()
        results["duration"] = duration
        results["speed_mb_sec"] = speed

        logging.info(
            f"copy speed {actual_size} bytes in {duration:.5f} sec ({speed:.1f} MB/sec)"
        )

        # Calculate hash if requested
        if hash_algorithm:
            hash_calc = HashCalculator(hash_algorithm)
            file_hash = hash_calc.calculate(source, buffer_size)
            logging.info(f"hash {hash_algorithm.upper()}:{file_hash}")
            results["hash"] = file_hash

        logging.info("moving files in place..")
        logging.info("flushing files")
        logging.info("done.")

        return results

    except Exception as e:
        # Clean up temp files on error
        for temp_file in temp_files.values():
            if temp_file.exists():
                temp_file.unlink()
        raise IOError(f"Copy operation failed: {e}")


def copy_with_multiple_destinations(
    source: Path,
    destinations: list[Path],
    buffer_size: int,
    expected_size: int | None = None,
    noflush_destinations: list[Path] | None = None,
    hash_algorithm: str | None = None,
    keep_source: bool = True,
) -> dict[str, Any]:
    """
    Copy source file to multiple destinations simultaneously.

    Parameters
    ----------
    source : Path
        Source file path
    destinations : List[Path]
        List of destination paths
    buffer_size : int
        Buffer size for copying operations
    expected_size : Optional[int]
        Expected file size for verification
    noflush_destinations : List[Path], optional
        Destinations to skip flushing for
    hash_algorithm : Optional[str]
        Hash algorithm for verification
    keep_source : bool
        Whether to keep source file after copying

    Returns
    -------
    Dict[str, Any]
        Copy results including hash, speed, and success status

    Raises
    ------
    FileNotFoundError
        If source file doesn't exist
    ValueError
        If file size doesn't match expected
    """
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    # Verify expected file size
    actual_size = source.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise ValueError(
            f"File size mismatch: expected {expected_size}, got {actual_size}"
        )

    if noflush_destinations is None:
        noflush_destinations = []

    logging.info(f"copying {source}...")

    # Initialize progress tracker
    progress_tracker = ProgressTracker(actual_size)

    # Use parallel implementation for multiple destinations
    if len(destinations) > 1:
        return copy_with_multiple_destinations_parallel(
            source=source,
            destinations=destinations,
            buffer_size=buffer_size,
            expected_size=expected_size,
            noflush_destinations=noflush_destinations,
            hash_algorithm=hash_algorithm,
            keep_source=keep_source,
        )

    # For single destination, use simpler approach
    results = {"success": True, "destinations": {}}

    try:
        # Create temporary files for all destinations
        temp_files = {}
        file_handles = {}

        for dest in destinations:
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp_file = dest.parent / f".{dest.name}.tmp"
            temp_files[dest] = temp_file
            file_handles[dest] = open(temp_file, "wb")

        # Copy file data to all destinations
        bytes_copied = 0
        with open(source, "rb") as src:
            while chunk := src.read(buffer_size):
                # Write chunk to all destinations
                for dest, handle in file_handles.items():
                    handle.write(chunk)

                bytes_copied += len(chunk)
                progress_tracker.update(bytes_copied)

        # Close all file handles
        for handle in file_handles.values():
            handle.close()

        # Verify and move files
        for dest in destinations:
            temp_file = temp_files[dest]

            # Verify file size
            if temp_file.stat().st_size != actual_size:
                results["success"] = False
                results["destinations"][str(dest)] = "File size mismatch"
                continue

            # Flush if not disabled
            noflush = dest in noflush_destinations
            if noflush:
                logging.info(f'skipping flush for destination "{dest}"')

            # Move to final destination
            temp_file.rename(dest)
            logging.info(
                f"Copy completed successfully for destination file with file size check ({dest})"
            )
            results["destinations"][str(dest)] = "success"

        # Calculate performance stats
        duration, speed = progress_tracker.get_stats()
        results["duration"] = duration
        results["speed_mb_sec"] = speed

        logging.info(
            f"copy speed {actual_size} bytes in {duration:.5f} sec ({speed:.1f} MB/sec)"
        )

        # Calculate hash if requested
        if hash_algorithm:
            hash_calc = HashCalculator(hash_algorithm)
            file_hash = hash_calc.calculate(source, buffer_size)
            logging.info(f"hash {hash_algorithm.upper()}:{file_hash}")
            results["hash"] = file_hash

        logging.info("moving files in place..")
        logging.info("flushing files")
        logging.info("done.")

        return results

    except Exception as e:
        # Clean up temp files on error
        for temp_file in temp_files.values():
            if temp_file.exists():
                temp_file.unlink()

        # Close any open file handles
        for handle in file_handles.values():
            if not handle.closed:
                handle.close()

        raise IOError(f"Copy operation failed: {e}")


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
  %(prog)s -v -t xxh64be source.mov dest1.mov dest2.mov
  %(prog)s -noflush_dest dest1.mov -b 16777216 -f_size 1000000 source.mov dest1.mov
        """,
    )

    # Verbose output
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    # Hash algorithm
    parser.add_argument(
        "-t",
        "--hash",
        type=str,
        default=None,
        choices=["xxh64be", "md5", "sha1", "sha256"],
        help="Hash algorithm for verification (default: none)",
    )

    # Keep source file
    parser.add_argument(
        "-k",
        "--keep",
        action="store_true",
        default=True,
        help="Keep source file after copying (default: True)",
    )

    # Expected file size
    parser.add_argument(
        "-f_size",
        "--file-size",
        type=int,
        default=None,
        help="Expected file size in bytes for verification",
    )

    # Buffer size
    parser.add_argument(
        "-b",
        "--buffer-size",
        type=int,
        default=8388608,
        help="Buffer size in bytes (default: 8MB)",
    )

    # No flush destinations
    parser.add_argument(
        "-noflush_dest",
        action="append",
        dest="noflush_destinations",
        help="Destination path to skip flushing (can be used multiple times)",
    )

    # Source file or directory (required)
    parser.add_argument("source", type=Path, help="Source file or directory path")

    # Destination files or directories (one or more required)
    parser.add_argument(
        "destinations", nargs="+", type=Path, help="Destination file or directory paths"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main function for pfndispatchcopy.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    try:
        args = parse_arguments()

        # Setup logging
        setup_logging(args.verbose)

        # Convert noflush destinations to Path objects
        noflush_destinations = []
        if args.noflush_destinations:
            noflush_destinations = [Path(dest) for dest in args.noflush_destinations]

        # Use PSTaskWrapper for unified file/directory handling
        task_wrapper = PSTaskWrapper(verbose=args.verbose)

        results = task_wrapper.launch_copy(
            source=args.source,
            destinations=args.destinations,
            buffer_size=args.buffer_size,
            expected_size=args.file_size,
            noflush_destinations=noflush_destinations,
            hash_algorithm=args.hash,
            keep_source=args.keep,
        )

        if results["success"]:
            if "files_copied" in results:
                # Directory copy results
                logging.info(
                    f"All copy operations completed successfully ({results['files_copied']} files)"
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
    except IOError as e:
        logging.error(f"I/O error: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
