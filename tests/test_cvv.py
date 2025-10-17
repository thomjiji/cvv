#!/usr/bin/env python3
"""
Simple test script for cvv implementation.

This script performs basic functional tests to verify that the cvv
implementation works correctly.
"""

import hashlib
import logging
import os

# Import the modules we want to test
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cvv import (
    CopyConfig,
    CopyTaskWrapper,
    HashCalculator,
    ProgressTracker,
    parse_arguments,
    setup_logging,
)


class TestProgressTracker(unittest.TestCase):
    """Test cases for ProgressTracker class."""

    def setUp(self) -> None:
        self.source_path = Path("/fake/source.txt")

    def test_progress_tracker_initialization(self) -> None:
        """Test ProgressTracker initialization."""
        total_size = 1000000
        tracker = ProgressTracker(total_size, self.source_path, update_interval=0.1)

        self.assertEqual(tracker.total_size, total_size)
        self.assertEqual(tracker.source_path, self.source_path)
        self.assertEqual(tracker.update_interval, 0.1)
        self.assertEqual(tracker.bytes_copied, 0)
        self.assertEqual(tracker.start_time, 0.0)

    def test_start(self) -> None:
        """Test the start method."""
        tracker = ProgressTracker(1000, self.source_path)
        with self.assertLogs(level="INFO") as log:
            tracker.start()
        self.assertGreater(tracker.start_time, 0)
        self.assertIn(f"copying {self.source_path}...", log.output[0])

    def test_progress_update(self) -> None:
        """Test progress updates."""
        tracker = ProgressTracker(
            1000000, self.source_path, update_interval=0.0
        )  # Immediate updates
        tracker.start()

        with self.assertLogs(level="INFO") as log:
            tracker.update(500000)

        self.assertEqual(tracker.bytes_copied, 500000)
        self.assertTrue(
            any("copied 500000 of 1000000 bytes" in message for message in log.output)
        )

    def test_get_stats(self) -> None:
        """Test statistics calculation."""
        tracker = ProgressTracker(1000000, self.source_path)
        tracker.start()
        tracker.bytes_copied = 1000000

        duration, speed = tracker.get_stats()

        self.assertIsInstance(duration, float)
        self.assertIsInstance(speed, float)
        self.assertGreater(duration, 0)
        # Speed can be zero if duration is very small, so we don't assert on it being greater than 0.

    def test_finish(self) -> None:
        """Test the finish method."""
        tracker = ProgressTracker(1000000, self.source_path)
        tracker.start()
        with self.assertLogs(level="INFO") as log:
            tracker.finish(file_hash="test_hash")

        self.assertIn("copy speed 1000000 bytes", log.output[0])
        self.assertIn("hash XXH64BE:test_hash", log.output[1])
        self.assertIn("done.", log.output[2])


class TestHashCalculator(unittest.TestCase):
    """Test cases for HashCalculator class."""

    def test_md5_hash_calculation(self) -> None:
        """Test MD5 hash calculation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            test_data = b"Hello, World!"
            tmp_file.write(test_data)
            tmp_file.flush()

            calc = HashCalculator("md5")
            result = calc.calculate(Path(tmp_file.name))

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

            calc = HashCalculator("sha256")
            result = calc.calculate(Path(tmp_file.name))

            # Calculate expected hash
            expected = hashlib.sha256(test_data).hexdigest()
            self.assertEqual(result, expected)

            # Clean up
            os.unlink(tmp_file.name)

    def test_unsupported_algorithm(self) -> None:
        """Test error handling for unsupported hash algorithm."""
        with self.assertRaises(ValueError):
            HashCalculator("unsupported_algorithm")


class TestFileCopyOperations(unittest.TestCase):
    """Test cases for file copying operations."""

    def setUp(self) -> None:
        """Set up test environment."""
        # Create temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        # Create test source file
        self.source_file = self.test_path / "source.txt"
        self.test_data = b"This is test data for copying operations."
        with open(self.source_file, "wb") as f:
            f.write(self.test_data)

    def tearDown(self) -> None:
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_single_destination_copy(self) -> None:
        """Test copying to a single destination."""
        dest1 = self.test_path / "dest1.txt"

        config = CopyConfig(buffer_size=1024, verbose=False)
        wrapper = CopyTaskWrapper(config)
        result = wrapper.launch_copy(source=self.source_file, destinations=[dest1])

        self.assertTrue(result["success"])
        self.assertTrue(dest1.exists())
        self.assertEqual(dest1.read_bytes(), self.test_data)
        self.assertIn("hash", result)

    def test_multiple_destinations_copy(self) -> None:
        """Test copying to multiple destinations."""
        dest1 = self.test_path / "backup1" / "file.txt"
        dest2 = self.test_path / "backup2" / "file.txt"

        config = CopyConfig(buffer_size=512, verbose=False)
        wrapper = CopyTaskWrapper(config)
        result = wrapper.launch_copy(
            source=self.source_file, destinations=[dest1, dest2]
        )

        self.assertTrue(result["success"])
        self.assertTrue(dest1.exists())
        self.assertTrue(dest2.exists())
        self.assertEqual(dest1.read_bytes(), self.test_data)
        self.assertEqual(dest2.read_bytes(), self.test_data)

    def test_file_size_verification(self) -> None:
        """Test file size verification."""
        dest1 = self.test_path / "dest_size_check.txt"

        config = CopyConfig(buffer_size=1024, verbose=False)
        wrapper = CopyTaskWrapper(config)
        result = wrapper.launch_copy(source=self.source_file, destinations=[dest1])

        self.assertTrue(result["success"])
        self.assertTrue(dest1.exists())
        self.assertEqual(dest1.stat().st_size, len(self.test_data))

    def test_nonexistent_source(self) -> None:
        """Test error handling for nonexistent source file."""
        nonexistent = self.test_path / "does_not_exist.txt"
        dest1 = self.test_path / "dest.txt"

        config = CopyConfig(buffer_size=1024, verbose=False)
        wrapper = CopyTaskWrapper(config)
        with self.assertRaises(FileNotFoundError):
            wrapper.launch_copy(source=nonexistent, destinations=[dest1])


class TestArgumentParsing(unittest.TestCase):
    """Test cases for command-line argument parsing."""

    def test_basic_argument_parsing(self) -> None:
        """Test basic argument parsing."""
        with patch("sys.argv", ["cvv.py", "source.txt", "dest.txt"]):
            args = parse_arguments()

            self.assertEqual(args.source, Path("source.txt"))
            self.assertEqual(args.destinations, [Path("dest.txt")])
            self.assertFalse(args.verbose)
            self.assertEqual(args.source_verify, "none")
            self.assertEqual(args.source_verify_hash, "xxh64be")

    def test_verbose_flag(self) -> None:
        """Test verbose flag parsing."""
        with patch("sys.argv", ["cvv.py", "-v", "source.txt", "dest.txt"]):
            args = parse_arguments()
            self.assertTrue(args.verbose)

    def test_source_verification_options(self) -> None:
        """Test source verification option parsing."""
        with patch(
            "sys.argv",
            [
                "cvv.py",
                "--source-verify",
                "per_file",
                "--source-verify-hash",
                "md5",
                "source.txt",
                "dest.txt",
            ],
        ):
            args = parse_arguments()
            self.assertEqual(args.source_verify, "per_file")
            self.assertEqual(args.source_verify_hash, "md5")

    def test_multiple_destinations(self) -> None:
        """Test parsing multiple destinations."""
        with patch("sys.argv", ["cvv.py", "source.txt", "dest1.txt", "dest2.txt"]):
            args = parse_arguments()
            self.assertEqual(len(args.destinations), 2)
            self.assertEqual(args.destinations[0], Path("dest1.txt"))
            self.assertEqual(args.destinations[1], Path("dest2.txt"))


class TestLoggingSetup(unittest.TestCase):
    """Test cases for logging setup."""

    def setUp(self) -> None:
        """Reset logging configuration."""
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)


    def test_setup_logging_info_level(self) -> None:
        """Test logging setup with INFO level."""
        setup_logging(verbose=False)
        self.assertEqual(logging.getLogger().level, logging.INFO)

    def test_setup_logging_debug_level(self) -> None:
        """Test logging setup with DEBUG level."""
        setup_logging(verbose=True)
        self.assertEqual(logging.getLogger().level, logging.DEBUG)


def create_integration_test() -> None:
    """
    Run a simple integration test.
    This function creates temporary files and tests the complete workflow.
    """
    print("Running integration test...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source_file = temp_path / "test_source.bin"
        test_data = b"A" * 1024 * 1024  # 1MB of data

        with open(source_file, "wb") as f:
            f.write(test_data)

        dest1 = temp_path / "backup1" / "test_file.bin"
        dest2 = temp_path / "backup2" / "test_file.bin"

        setup_logging(verbose=True)

        try:
            config = CopyConfig(buffer_size=64 * 1024, verbose=False)
            wrapper = CopyTaskWrapper(config)
            result = wrapper.launch_copy(
                source=source_file, destinations=[dest1, dest2]
            )

            assert result["success"], "Copy operation failed"
            assert dest1.exists(), "Destination 1 does not exist"
            assert dest2.exists(), "Destination 2 does not exist"
            assert dest1.stat().st_size == len(test_data), "Destination 1 size mismatch"
            assert dest2.stat().st_size == len(test_data), "Destination 2 size mismatch"

            print("✓ Integration test passed!")
            print(f"  - Copied {len(test_data)} bytes")
            print(f"  - Speed: {result.get('speed_mb_sec', 0):.1f} MB/sec")
            print(f"  - Hash: {result.get('hash', 'N/A')}")

        except Exception as e:
            print(f"✗ Integration test failed: {e}")
            raise


def main() -> None:
    """Run all tests."""
    print("=" * 60)
    print("cvv Test Suite")
    print("=" * 60)

    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(verbosity=2, exit=False, argv=[""])

    # Run integration test
    print("\n" + "=" * 60)
    create_integration_test()

    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    main()

