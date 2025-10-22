#!/usr/bin/env python3
"""
Comprehensive test suite for cvv implementation.

Tests cover:
- Hash calculation
- Core copy engine functionality
- All verification modes
- Error handling
- Multi-destination copying
- Directory copying
"""

import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cvv import (
    CLIProcessor,
    CopyEngine,
    CopyEvent,
    CopyResult,
    DestinationResult,
    EventType,
    HashCalculator,
    VerificationMode,
)


class TestHashCalculator(unittest.TestCase):
    """Test cases for HashCalculator class."""

    def setUp(self) -> None:
        """Set up test directory."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)

    def test_hash_file_generator_md5(self) -> None:
        """Test MD5 hash calculation via generator."""
        test_file = self.test_path / "test.txt"
        test_data = b"Hello, World!"
        test_file.write_bytes(test_data)

        # Consume the generator
        final_hash = ""
        for _, final_hash in HashCalculator.hash_file(test_file, "md5"):
            if final_hash:
                break

        expected = hashlib.md5(test_data).hexdigest()
        self.assertEqual(final_hash, expected)

    def test_hash_file_generator_sha256(self) -> None:
        """Test SHA256 hash calculation via generator."""
        test_file = self.test_path / "test.txt"
        test_data = b"Test data for SHA256"
        test_file.write_bytes(test_data)

        # Consume the generator
        final_hash = ""
        for _, final_hash in HashCalculator.hash_file(test_file, "sha256"):
            if final_hash:
                break

        expected = hashlib.sha256(test_data).hexdigest()
        self.assertEqual(final_hash, expected)

    def test_hash_file_generator_xxh64be(self) -> None:
        """Test xxHash calculation via generator."""
        test_file = self.test_path / "test.txt"
        test_data = b"xxHash test data"
        test_file.write_bytes(test_data)

        # Consume the generator
        final_hash = ""
        bytes_processed = 0
        for bytes_processed, final_hash in HashCalculator.hash_file(
            test_file, "xxh64be"
        ):
            pass  # Process all yields

        self.assertTrue(final_hash)  # Should have a hash
        self.assertEqual(bytes_processed, len(test_data))

    def test_unsupported_algorithm(self) -> None:
        """Test error handling for unsupported hash algorithm."""
        with self.assertRaises(ValueError):
            HashCalculator("unsupported_algorithm")


class TestCopyEngine(unittest.TestCase):
    """Test cases for the core CopyEngine class."""

    def setUp(self) -> None:
        """Set up test directory and files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        # Create source file
        self.source_file = self.test_path / "source.txt"
        self.test_data = b"This is test data for CopyEngine." * 100  # ~3.4 KB
        self.source_file.write_bytes(self.test_data)

    def tearDown(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)

    def test_single_destination_copy(self) -> None:
        """Test copying to a single destination."""
        dest = self.test_path / "dest1.txt"

        engine = CopyEngine(
            source=self.source_file,
            destinations=[dest],
            verification_mode=VerificationMode.TRANSFER,
        )

        # Consume generator and get result
        result = None
        for event in engine.copy():
            if isinstance(event, CopyResult):
                result = event
                break

        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertTrue(dest.exists())
        self.assertEqual(dest.read_bytes(), self.test_data)
        self.assertEqual(len(result.destinations), 1)
        self.assertTrue(result.destinations[0].success)

    def test_multi_destination_copy(self) -> None:
        """Test copying to multiple destinations simultaneously."""
        dest1 = self.test_path / "dest1.txt"
        dest2 = self.test_path / "dest2.txt"
        dest3 = self.test_path / "dest3.txt"

        engine = CopyEngine(
            source=self.source_file,
            destinations=[dest1, dest2, dest3],
            verification_mode=VerificationMode.TRANSFER,
        )

        # Consume generator
        result = None
        for event in engine.copy():
            if isinstance(event, CopyResult):
                result = event

        self.assertTrue(result.success)
        self.assertEqual(len(result.destinations), 3)

        # Verify all destinations
        for dest in [dest1, dest2, dest3]:
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_bytes(), self.test_data)

    def test_copy_events_emitted(self) -> None:
        """Test that proper events are emitted during copy."""
        dest = self.test_path / "dest.txt"

        engine = CopyEngine(
            source=self.source_file,
            destinations=[dest],
            verification_mode=VerificationMode.TRANSFER,
        )

        events = []
        result = None

        for event in engine.copy():
            if isinstance(event, CopyResult):
                result = event
            elif isinstance(event, CopyEvent):
                events.append(event.type)

        # Check that expected events were emitted
        self.assertIn(EventType.COPY_START, events)
        self.assertIn(EventType.COPY_PROGRESS, events)
        self.assertIn(EventType.COPY_COMPLETE, events)
        self.assertIsNotNone(result)

    def test_source_not_found(self) -> None:
        """Test error handling when source file doesn't exist."""
        nonexistent = self.test_path / "nonexistent.txt"
        dest = self.test_path / "dest.txt"

        engine = CopyEngine(
            source=nonexistent,
            destinations=[dest],
        )

        result = None
        for event in engine.copy():
            if isinstance(event, CopyResult):
                result = event

        self.assertIsNotNone(result)
        self.assertFalse(result.success)
        self.assertFalse(result.destinations[0].success)
        # Error message should mention the file or directory issue
        error_msg = result.destinations[0].error.lower()
        self.assertTrue(
            "no such file" in error_msg or "not found" in error_msg,
            f"Expected file not found error, got: {result.destinations[0].error}",
        )


class TestVerificationModes(unittest.TestCase):
    """Test all verification modes."""

    def setUp(self) -> None:
        """Set up test directory and files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        # Create larger source file for better testing
        self.source_file = self.test_path / "source.dat"
        self.test_data = b"X" * (1024 * 100)  # 100 KB
        self.source_file.write_bytes(self.test_data)

    def tearDown(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)

    def test_transfer_mode_no_hashing(self) -> None:
        """Test TRANSFER mode - size check only, no hashing."""
        dest = self.test_path / "dest.txt"

        engine = CopyEngine(
            source=self.source_file,
            destinations=[dest],
            verification_mode=VerificationMode.TRANSFER,
        )

        result = None
        for event in engine.copy():
            if isinstance(event, CopyResult):
                result = event

        self.assertTrue(result.success)
        self.assertIsNone(result.source_hash_inflight)  # No hash in TRANSFER mode
        self.assertIsNone(result.source_hash_post)

    def test_source_mode_hashing(self) -> None:
        """Test SOURCE mode - hash source in-flight and post-copy."""
        dest = self.test_path / "dest.txt"

        engine = CopyEngine(
            source=self.source_file,
            destinations=[dest],
            verification_mode=VerificationMode.SOURCE,
        )

        result = None
        events = []

        for event in engine.copy():
            if isinstance(event, CopyResult):
                result = event
            elif isinstance(event, CopyEvent):
                events.append(event.type)

        self.assertTrue(result.success)
        self.assertIsNotNone(result.source_hash_inflight)
        self.assertIsNotNone(result.source_hash_post)
        self.assertEqual(result.source_hash_inflight, result.source_hash_post)

        # Verify events include verification
        self.assertIn(EventType.VERIFY_START, events)
        self.assertIn(EventType.VERIFY_COMPLETE, events)

    def test_full_mode_hashing(self) -> None:
        """Test FULL mode - hash source and all destinations."""
        dest1 = self.test_path / "dest1.txt"
        dest2 = self.test_path / "dest2.txt"

        engine = CopyEngine(
            source=self.source_file,
            destinations=[dest1, dest2],
            verification_mode=VerificationMode.FULL,
        )

        result = None
        for event in engine.copy():
            if isinstance(event, CopyResult):
                result = event

        self.assertTrue(result.success)
        self.assertIsNotNone(result.source_hash_inflight)
        self.assertIsNotNone(result.source_hash_post)

        # Check destination hashes
        for dest_result in result.destinations:
            self.assertIsNotNone(dest_result.hash_post)
            self.assertEqual(dest_result.hash_post, result.source_hash_inflight)

    def test_full_mode_detects_corruption(self) -> None:
        """Test that FULL mode detects file corruption."""
        dest = self.test_path / "dest.txt"

        # Patch the writer thread to corrupt the file
        original_replace = Path.replace

        def corrupt_on_replace(self, target):
            original_replace(self, target)
            # Corrupt the destination after rename
            with open(target, "ab") as f:
                f.write(b"CORRUPTED")

        with patch.object(Path, "replace", corrupt_on_replace):
            engine = CopyEngine(
                source=self.source_file,
                destinations=[dest],
                verification_mode=VerificationMode.FULL,
            )

            result = None
            for event in engine.copy():
                if isinstance(event, CopyResult):
                    result = event

            # Should detect the corruption
            self.assertFalse(result.success)
            self.assertFalse(result.destinations[0].success)
            self.assertIn("mismatch", result.destinations[0].error.lower())


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""

    def setUp(self) -> None:
        """Set up test directory."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        self.source_file = self.test_path / "source.txt"
        self.source_file.write_bytes(b"test data")

    def tearDown(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)

    def test_insufficient_disk_space(self) -> None:
        """Test error when insufficient disk space."""
        dest = self.test_path / "dest.txt"

        engine = CopyEngine(
            source=self.source_file,
            destinations=[dest],
        )

        # Mock disk_usage to simulate no space
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = Mock(free=0)

            result = None
            for event in engine.copy():
                if isinstance(event, CopyResult):
                    result = event

            self.assertFalse(result.success)
            self.assertIn("space", result.destinations[0].error.lower())

    def test_write_permission_error(self) -> None:
        """Test error when destination is not writable."""
        # Use /dev/full which always returns "disk full" error
        if not Path("/dev/full").exists():
            self.skipTest("/dev/full not available on this system")

        dest = Path("/dev/full")

        engine = CopyEngine(
            source=self.source_file,
            destinations=[dest],
        )

        result = None
        for event in engine.copy():
            if isinstance(event, CopyResult):
                result = event

        self.assertFalse(result.success)

    def test_per_destination_errors(self) -> None:
        """Test that errors are tracked per destination."""
        dest1 = self.test_path / "dest1.txt"  # Good destination
        dest2 = Path("/nonexistent/path/dest2.txt")  # Bad destination

        engine = CopyEngine(
            source=self.source_file,
            destinations=[dest1, dest2],
        )

        result = None
        for event in engine.copy():
            if isinstance(event, CopyResult):
                result = event

        self.assertIsNotNone(result)
        # Overall operation failed because one destination failed
        self.assertFalse(result.success)


class TestCLIProcessor(unittest.TestCase):
    """Test the CLI processor layer."""

    def setUp(self) -> None:
        """Set up test directory and files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        # Create source directory with files
        self.source_dir = self.test_path / "source"
        self.source_dir.mkdir()
        (self.source_dir / "file1.txt").write_text("content1")
        (self.source_dir / "file2.txt").write_text("content2")

        # Create subdirectory
        subdir = self.source_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("content3")

        # Create destination directories
        self.dest1 = self.test_path / "dest1"
        self.dest2 = self.test_path / "dest2"
        self.dest1.mkdir()
        self.dest2.mkdir()

    def tearDown(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)

    def test_single_file_copy(self) -> None:
        """Test copying a single file to multiple destinations."""
        source_file = self.source_dir / "file1.txt"

        processor = CLIProcessor(
            source=source_file,
            destinations=[self.dest1, self.dest2],
            verification_mode=VerificationMode.TRANSFER,
            hash_algorithm="xxh64be",
        )

        success = processor.run()

        self.assertTrue(success)
        self.assertTrue((self.dest1 / "file1.txt").exists())
        self.assertTrue((self.dest2 / "file1.txt").exists())
        self.assertEqual((self.dest1 / "file1.txt").read_text(), "content1")
        self.assertEqual((self.dest2 / "file1.txt").read_text(), "content1")

    def test_directory_copy_preserves_structure(self) -> None:
        """Test that directory copying preserves directory structure."""
        processor = CLIProcessor(
            source=self.source_dir,
            destinations=[self.dest1, self.dest2],
            verification_mode=VerificationMode.TRANSFER,
            hash_algorithm="xxh64be",
        )

        success = processor.run()

        self.assertTrue(success)

        # Check all files exist in both destinations
        for dest in [self.dest1, self.dest2]:
            self.assertTrue((dest / "file1.txt").exists())
            self.assertTrue((dest / "file2.txt").exists())
            self.assertTrue((dest / "subdir" / "file3.txt").exists())

            # Verify content
            self.assertEqual((dest / "file1.txt").read_text(), "content1")
            self.assertEqual((dest / "file2.txt").read_text(), "content2")
            self.assertEqual((dest / "subdir" / "file3.txt").read_text(), "content3")

    def test_discover_files_directory(self) -> None:
        """Test file discovery from directory."""
        processor = CLIProcessor(
            source=self.source_dir,
            destinations=[self.dest1],
            verification_mode=VerificationMode.TRANSFER,
            hash_algorithm="xxh64be",
        )

        files = processor._discover_files()

        self.assertEqual(len(files), 3)  # file1, file2, file3
        self.assertIn(self.source_dir / "file1.txt", files)
        self.assertIn(self.source_dir / "file2.txt", files)
        self.assertIn(self.source_dir / "subdir" / "file3.txt", files)

    def test_discover_files_single_file(self) -> None:
        """Test file discovery from single file."""
        source_file = self.source_dir / "file1.txt"

        processor = CLIProcessor(
            source=source_file,
            destinations=[self.dest1],
            verification_mode=VerificationMode.TRANSFER,
            hash_algorithm="xxh64be",
        )

        files = processor._discover_files()

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0], source_file)


class TestIntegration(unittest.TestCase):
    """Integration tests covering end-to-end workflows."""

    def setUp(self) -> None:
        """Set up test directory."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)

    def test_full_workflow_multiple_files_multiple_destinations(self) -> None:
        """Test complete workflow: multiple files to multiple destinations with full verification."""
        # Create source files
        source_dir = self.test_path / "source"
        source_dir.mkdir()

        files_data = {
            "video1.mp4": b"fake video data 1" * 1000,
            "video2.mp4": b"fake video data 2" * 1000,
            "metadata.json": b'{"key": "value"}',
        }

        for filename, data in files_data.items():
            (source_dir / filename).write_bytes(data)

        # Create destinations
        dest1 = self.test_path / "backup1"
        dest2 = self.test_path / "backup2"
        dest1.mkdir()
        dest2.mkdir()

        # Run processor
        processor = CLIProcessor(
            source=source_dir,
            destinations=[dest1, dest2],
            verification_mode=VerificationMode.FULL,
            hash_algorithm="xxh64be",
        )

        success = processor.run()

        # Verify success
        self.assertTrue(success)

        # Verify all files in both destinations
        for filename, data in files_data.items():
            for dest in [dest1, dest2]:
                dest_file = dest / filename
                self.assertTrue(dest_file.exists(), f"{dest_file} should exist")
                self.assertEqual(
                    dest_file.read_bytes(),
                    data,
                    f"{dest_file} should have correct content",
                )


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
