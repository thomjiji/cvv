#!/usr/bin/env python3
"""
Simple test script for pfndispatchcopy implementation.

This script performs basic functional tests to verify that the pfndispatchcopy
implementation works correctly.
"""

import hashlib
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Import the modules we want to test
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pfndispatchcopy import (
    HashCalculator,
    ProgressTracker,
    copy_with_multiple_destinations,
    parse_arguments,
    setup_logging,
)


class TestProgressTracker(unittest.TestCase):
    """Test cases for ProgressTracker class."""

    def test_progress_tracker_initialization(self) -> None:
        """Test ProgressTracker initialization."""
        total_size = 1000000
        tracker = ProgressTracker(total_size, update_interval=0.1)

        self.assertEqual(tracker.total_size, total_size)
        self.assertEqual(tracker.update_interval, 0.1)
        self.assertEqual(tracker.bytes_copied, 0)

    def test_progress_update(self) -> None:
        """Test progress updates."""
        tracker = ProgressTracker(1000000, update_interval=0.0)  # Immediate updates

        with self.assertLogs(level="INFO") as log:
            tracker.update(500000)

        self.assertEqual(tracker.bytes_copied, 500000)
        self.assertTrue(
            any("copied 500000 of 1000000 bytes" in message for message in log.output)
        )

    def test_get_stats(self) -> None:
        """Test statistics calculation."""
        tracker = ProgressTracker(1000000)
        tracker.bytes_copied = 1000000

        duration, speed = tracker.get_stats()

        self.assertIsInstance(duration, float)
        self.assertIsInstance(speed, float)
        self.assertGreater(duration, 0)
        self.assertGreater(speed, 0)


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

        result = copy_with_multiple_destinations(
            source=self.source_file,
            destinations=[dest1],
            buffer_size=1024,
            hash_algorithm="md5",
        )

        self.assertTrue(result["success"])
        self.assertTrue(dest1.exists())
        self.assertEqual(dest1.read_bytes(), self.test_data)
        self.assertIn("hash", result)

    def test_multiple_destinations_copy(self) -> None:
        """Test copying to multiple destinations."""
        dest1 = self.test_path / "backup1" / "file.txt"
        dest2 = self.test_path / "backup2" / "file.txt"

        result = copy_with_multiple_destinations(
            source=self.source_file,
            destinations=[dest1, dest2],
            buffer_size=512,
            hash_algorithm="sha1",
        )

        self.assertTrue(result["success"])
        self.assertTrue(dest1.exists())
        self.assertTrue(dest2.exists())
        self.assertEqual(dest1.read_bytes(), self.test_data)
        self.assertEqual(dest2.read_bytes(), self.test_data)

    def test_file_size_verification(self) -> None:
        """Test file size verification."""
        dest1 = self.test_path / "dest_size_check.txt"

        # Test with correct size
        result = copy_with_multiple_destinations(
            source=self.source_file,
            destinations=[dest1],
            buffer_size=1024,
            expected_size=len(self.test_data),
        )

        self.assertTrue(result["success"])

        # Test with incorrect size
        with self.assertRaises(ValueError):
            copy_with_multiple_destinations(
                source=self.source_file,
                destinations=[dest1],
                buffer_size=1024,
                expected_size=999999,  # Wrong size
            )

    def test_nonexistent_source(self) -> None:
        """Test error handling for nonexistent source file."""
        nonexistent = self.test_path / "does_not_exist.txt"
        dest1 = self.test_path / "dest.txt"

        with self.assertRaises(FileNotFoundError):
            copy_with_multiple_destinations(
                source=nonexistent, destinations=[dest1], buffer_size=1024
            )


class TestArgumentParsing(unittest.TestCase):
    """Test cases for command-line argument parsing."""

    def test_basic_argument_parsing(self) -> None:
        """Test basic argument parsing."""
        with patch("sys.argv", ["pfndispatchcopy.py", "source.txt", "dest.txt"]):
            args = parse_arguments()

            self.assertEqual(args.source, Path("source.txt"))
            self.assertEqual(args.destinations, [Path("dest.txt")])
            self.assertFalse(args.verbose)
            self.assertIsNone(args.hash)

    def test_verbose_flag(self) -> None:
        """Test verbose flag parsing."""
        with patch("sys.argv", ["pfndispatchcopy.py", "-v", "source.txt", "dest.txt"]):
            args = parse_arguments()

            self.assertTrue(args.verbose)

    def test_hash_algorithm_option(self) -> None:
        """Test hash algorithm option parsing."""
        with patch(
            "sys.argv", ["pfndispatchcopy.py", "-t", "sha256", "source.txt", "dest.txt"]
        ):
            args = parse_arguments()

            self.assertEqual(args.hash, "sha256")

    def test_multiple_destinations(self) -> None:
        """Test parsing multiple destinations."""
        with patch(
            "sys.argv", ["pfndispatchcopy.py", "source.txt", "dest1.txt", "dest2.txt"]
        ):
            args = parse_arguments()

            self.assertEqual(len(args.destinations), 2)
            self.assertEqual(args.destinations[0], Path("dest1.txt"))
            self.assertEqual(args.destinations[1], Path("dest2.txt"))

    def test_buffer_size_option(self) -> None:
        """Test buffer size option parsing."""
        with patch(
            "sys.argv",
            ["pfndispatchcopy.py", "-b", "16777216", "source.txt", "dest.txt"],
        ):
            args = parse_arguments()

            self.assertEqual(args.buffer_size, 16777216)

    def test_file_size_option(self) -> None:
        """Test file size option parsing."""
        with patch(
            "sys.argv",
            ["pfndispatchcopy.py", "-f_size", "1000000", "source.txt", "dest.txt"],
        ):
            args = parse_arguments()

            self.assertEqual(args.file_size, 1000000)

    def test_noflush_destinations(self) -> None:
        """Test noflush destinations option parsing."""
        with patch(
            "sys.argv",
            [
                "pfndispatchcopy.py",
                "-noflush_dest",
                "dest1.txt",
                "-noflush_dest",
                "dest2.txt",
                "source.txt",
                "dest1.txt",
                "dest2.txt",
            ],
        ):
            args = parse_arguments()

            self.assertEqual(len(args.noflush_destinations), 2)
            self.assertIn("dest1.txt", args.noflush_destinations)
            self.assertIn("dest2.txt", args.noflush_destinations)


class TestLoggingSetup(unittest.TestCase):
    """Test cases for logging setup."""

    def test_setup_logging_info_level(self) -> None:
        """Test logging setup with INFO level."""
        setup_logging(verbose=False)

        # Check that logging is configured
        logger = logging.getLogger()
        self.assertEqual(logger.level, logging.INFO)

    def test_setup_logging_debug_level(self) -> None:
        """Test logging setup with DEBUG level."""
        setup_logging(verbose=True)

        # Check that logging is configured
        logger = logging.getLogger()
        self.assertEqual(logger.level, logging.DEBUG)


def create_integration_test() -> None:
    """
    Run a simple integration test.

    This function creates temporary files and tests the complete workflow.
    """
    print("Running integration test...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test source file
        source_file = temp_path / "test_source.bin"
        test_data = b"A" * 1024 * 1024  # 1MB of data

        with open(source_file, "wb") as f:
            f.write(test_data)

        # Define destinations
        dest1 = temp_path / "backup1" / "test_file.bin"
        dest2 = temp_path / "backup2" / "test_file.bin"

        # Set up logging
        setup_logging(verbose=True)

        try:
            # Perform copy operation
            result = copy_with_multiple_destinations(
                source=source_file,
                destinations=[dest1, dest2],
                buffer_size=64 * 1024,  # 64KB buffer
                expected_size=len(test_data),
                hash_algorithm="md5",
            )

            # Verify results
            assert result["success"], "Copy operation failed"
            assert dest1.exists(), "Destination 1 does not exist"
            assert dest2.exists(), "Destination 2 does not exist"
            assert dest1.stat().st_size == len(test_data), "Destination 1 size mismatch"
            assert dest2.stat().st_size == len(test_data), "Destination 2 size mismatch"

            print("✓ Integration test passed!")
            print(f"  - Copied {len(test_data)} bytes")
            print(f"  - Speed: {result['speed_mb_sec']:.1f} MB/sec")
            print(f"  - Hash: {result['hash']}")

        except Exception as e:
            print(f"✗ Integration test failed: {e}")
            raise


def main() -> None:
    """Run all tests."""
    print("=" * 60)
    print("pfndispatchcopy Test Suite")
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
