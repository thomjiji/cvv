#!/usr/bin/env python3
"""
Simple test script for cvv implementation.
This script performs basic functional tests to verify that the cvv
implementation works correctly.
"""

import hashlib
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cvv import (
    BatchProcessor,
    CopyConfig,
    CopyJob,
    FileCopyResult,
    HashCalculator,
    ProgressTracker,
    VerificationMode,
)


class TestHashCalculator(unittest.TestCase):
    """Test cases for HashCalculator class."""

    def test_md5_hash_calculation(self) -> None:
        """Test MD5 hash calculation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            test_data = b"Hello, World!"
            tmp_file.write(test_data)
            tmp_file.flush()

            # Calculate hash
            result = HashCalculator.hash_file(
                Path(tmp_file.name), "md5", buffer_size=1024
            )

            # Calculate expected hash
            expected = hashlib.md5(test_data).hexdigest()
            self.assertEqual(result, expected)

            # Clean up
            os.unlink(tmp_file.name)

    def test_sha256_hash_calculation(self) -> None:
        """Test SHA256 hash calculation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            test_data = b"Test data for SHA256"
            tmp_file.write(test_data)
            tmp_file.flush()

            # Calculate hash
            result = HashCalculator.hash_file(
                Path(tmp_file.name), "sha256", buffer_size=1024
            )

            # Calculate expected hash
            expected = hashlib.sha256(test_data).hexdigest()
            self.assertEqual(result, expected)

            # Clean up
            os.unlink(tmp_file.name)

    def test_unsupported_algorithm(self) -> None:
        """Test error handling for unsupported hash algorithm."""
        with self.assertRaises(ValueError):
            HashCalculator("unsupported_algorithm")


class TestProgressTracker(unittest.TestCase):
    """Test cases for ProgressTracker class."""

    def test_progress_tracker_initialization(self) -> None:
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker(total_size=1000, callback=None)
        self.assertEqual(tracker.total_size, 1000)
        self.assertEqual(tracker.bytes_copied, 0)

    def test_update(self) -> None:
        """Test the update method."""
        mock_callback = MagicMock()
        tracker = ProgressTracker(total_size=1000, callback=mock_callback)
        tracker.update(100)
        self.assertEqual(tracker.bytes_copied, 100)
        mock_callback.assert_called_once_with(100, 1000)


class TestCopyJob(unittest.TestCase):
    """Test cases for the CopyJob class."""

    def setUp(self) -> None:
        """Set up a temporary directory for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.source_file = self.test_path / "source.txt"
        self.test_data = b"This is a test file for the CopyJob class."
        with open(self.source_file, "wb") as f:
            f.write(self.test_data)

    def tearDown(self) -> None:
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_successful_copy_transfer_mode(self) -> None:
        """Test a successful copy in TRANSFER mode."""
        dest1 = self.test_path / "dest1.txt"
        config = CopyConfig(verification_mode=VerificationMode.TRANSFER)
        job = CopyJob(config).source(self.source_file).add_destination(dest1)

        future = job.execute()
        result: FileCopyResult = future.result()

        self.assertTrue(result.verified)
        self.assertIsNone(result.error)
        self.assertTrue(dest1.exists())
        self.assertEqual(dest1.read_bytes(), self.test_data)
        self.assertIsNone(result.source_hash_inflight)

    def test_successful_copy_full_mode(self) -> None:
        """Test a successful copy in FULL mode."""
        dest1 = self.test_path / "dest1.txt"
        config = CopyConfig(
            verification_mode=VerificationMode.FULL, hash_algorithm="xxh64be"
        )
        job = CopyJob(config).source(self.source_file).add_destination(dest1)

        future = job.execute()
        result: FileCopyResult = future.result()

        self.assertTrue(result.verified)
        self.assertIsNotNone(result.source_hash_inflight)
        self.assertEqual(result.source_hash_inflight, result.source_hash_post)
        self.assertEqual(
            result.source_hash_inflight, result.destination_hashes_post[dest1]
        )

    def test_full_verification_catches_corruption(self) -> None:
        """Test that FULL verification mode catches a corrupted destination file."""
        dest1 = self.test_path / "dest1.txt"
        config = CopyConfig(verification_mode=VerificationMode.FULL)

        original_move = shutil.move

        def tampered_move(src, dst):
            original_move(src, dst)
            with open(dst, "ab") as f:
                f.write(b"corruption")

        with patch("shutil.move", side_effect=tampered_move):
            job = CopyJob(config).source(self.source_file).add_destination(dest1)
            future = job.execute()
            result: FileCopyResult = future.result()

            self.assertFalse(result.verified)
            self.assertIn("Full verification failed", result.error)

    def test_source_verification_catches_source_corruption(self) -> None:
        """Test that SOURCE verification mode catches a corrupted source file."""
        dest1 = self.test_path / "dest1.txt"
        config = CopyConfig(verification_mode=VerificationMode.SOURCE)

        original_stream_and_dispatch = CopyJob._stream_and_dispatch_chunks

        def tampered_stream_and_dispatch(job_instance, source_size, enable_hashing):
            # Call the original method to get the in-flight hash
            hash_inflight = original_stream_and_dispatch(
                job_instance, source_size, enable_hashing
            )
            # Tamper with the source file after it has been read
            with open(job_instance._source, "ab") as f:
                f.write(b"source corruption")
            return hash_inflight

        with patch.object(
            CopyJob,
            "_stream_and_dispatch_chunks",
            side_effect=tampered_stream_and_dispatch,
            autospec=True,
        ):
            job = CopyJob(config).source(self.source_file).add_destination(dest1)
            future = job.execute()
            result: FileCopyResult = future.result()

            self.assertFalse(result.verified)
            self.assertIn("Source file changed during copy", result.error)

    def test_not_enough_disk_space(self) -> None:
        """Test error handling when there is not enough disk space."""
        dest1 = self.test_path / "dest1.txt"
        config = CopyConfig()
        job = CopyJob(config).source(self.source_file).add_destination(dest1)

        with patch("shutil.disk_usage") as mock_disk_usage:
            mock_disk_usage.return_value.free = 0
            future = job.execute()
            result: FileCopyResult = future.result()

            self.assertFalse(result.verified)
            self.assertIn("Not enough space", result.error)

    def test_abort_operation(self) -> None:
        """Test aborting a copy operation."""
        dest1 = self.test_path / "dest1.txt"
        config = CopyConfig()
        job = CopyJob(config).source(self.source_file).add_destination(dest1)

        # Abort the job shortly after it starts
        job.abort()
        future = job.execute()
        result: FileCopyResult = future.result()

        self.assertFalse(result.verified)
        self.assertIn("Operation aborted", result.error)


class TestBatchProcessor(unittest.TestCase):
    """Test cases for the BatchProcessor class."""

    def setUp(self) -> None:
        """Set up a temporary directory for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.source_dir = self.test_path / "source"
        self.source_dir.mkdir()
        self.dest_dir = self.test_path / "destination"
        self.dest_dir.mkdir()

        # Create some test files
        (self.source_dir / "file1.txt").write_text("file1")
        (self.source_dir / "file2.txt").write_text("file2")

    def tearDown(self) -> None:
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_run_batch_processor(self) -> None:
        """Test running the batch processor with a directory source."""
        args = MagicMock()
        args.source = self.source_dir
        args.destinations = [self.dest_dir]
        args.mode = "full"
        args.hash_algorithm = "xxh64be"
        args.verbose = False

        processor = BatchProcessor(args)
        success = processor.run()

        self.assertTrue(success)
        self.assertTrue((self.dest_dir / "file1.txt").exists())
        self.assertTrue((self.dest_dir / "file2.txt").exists())
        self.assertEqual((self.dest_dir / "file1.txt").read_text(), "file1")
        self.assertEqual((self.dest_dir / "file2.txt").read_text(), "file2")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)
