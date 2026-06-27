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
from unittest.mock import Mock, patch

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cvv import (
    CLIProcessor,
    CopyEngine,
    CopyEvent,
    CopyResult,
    EventType,
    HashCalculator,
    HashFileWriter,
    VerificationMode,
)


class TestHashCalculator(unittest.TestCase):
    """Test cases for HashCalculator class."""

    def setUp(self) -> None:
        """Set up test directory."""
        CopyEngine.reset_shared_state()
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

    def test_hash_file_generator_xxh3_64(self) -> None:
        """Test xxHash3 calculation via generator."""
        test_file = self.test_path / "test.txt"
        test_data = b"xxHash test data"
        test_file.write_bytes(test_data)

        # Consume the generator
        final_hash = ""
        bytes_processed = 0
        for bytes_processed, final_hash in HashCalculator.hash_file(
            test_file, "xxh3_64"
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
        CopyEngine.reset_shared_state()
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
        error_msg = result.destinations[0].error.lower()
        self.assertTrue(
            "no such file" in error_msg
            or "not found" in error_msg
            or "cannot find" in error_msg,
            f"Expected file not found error, got: {result.destinations[0].error}",
        )


class TestVerificationModes(unittest.TestCase):
    """Test all verification modes."""

    def setUp(self) -> None:
        """Set up test directory and files."""
        CopyEngine.reset_shared_state()
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        # Create larger source file for better testing
        self.source_file = self.test_path / "source.dat"
        self.test_data = b"X" * (1024 * 100)  # 100 KB
        self.source_file.write_bytes(self.test_data)

    def tearDown(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)

    def test_transfer_mode_hashing(self) -> None:
        """Test TRANSFER mode - hashes in-flight but no post-copy verification."""
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
        # TRANSFER mode now computes in-flight hash (essentially free)
        self.assertIsNotNone(result.source_hash_inflight)
        # But doesn't do post-copy verification
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
        CopyEngine.reset_shared_state()
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

    def test_per_destination_errors(self) -> None:
        """Test that errors are tracked per destination."""
        dest1 = self.test_path / "dest1.txt"  # Good destination
        # Use an invalid path that can't be created on any OS
        if sys.platform == "win32":
            dest2 = Path("Z:\\__no_such_drive__\\dest2.txt")
        else:
            dest2 = Path("/nonexistent/path/dest2.txt")

        engine = CopyEngine(
            source=self.source_file,
            destinations=[dest1, dest2],
        )

        result = None
        for event in engine.copy():
            if isinstance(event, CopyResult):
                result = event

        self.assertIsNotNone(result)
        self.assertFalse(result.success)


class TestCLIProcessor(unittest.TestCase):
    """Test the CLI processor layer."""

    def setUp(self) -> None:
        """Set up test directory and files."""
        CopyEngine.reset_shared_state()
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
            hash_algorithm="xxh3_64",
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
            hash_algorithm="xxh3_64",
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
            hash_algorithm="xxh3_64",
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
            hash_algorithm="xxh3_64",
        )

        files = processor._discover_files()

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0], source_file)


class TestIntegration(unittest.TestCase):
    """Integration tests covering end-to-end workflows."""

    def setUp(self) -> None:
        """Set up test directory."""
        CopyEngine.reset_shared_state()
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
            hash_algorithm="xxh3_64",
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


class TestHashFileWriter(unittest.TestCase):
    """Test hash file generation in .xxh and .mhl formats."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.entries = [
            (Path("video.mp4"), "ED26553EDB44956D", 1234567),
            (Path("subdir/audio.wav"), "0C27693DBFD5E7DA", 9876543),
        ]

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_write_xxh_format(self) -> None:
        """Test TeraCopy-compatible .xxh file generation."""
        output = self.test_path / "test.xxh"
        HashFileWriter.write_xxh(self.entries, output, "xxh3_64")

        self.assertTrue(output.exists())
        content = output.read_text(encoding="utf-8")

        # Header
        self.assertIn("xxHash3-64", content)
        self.assertIn("cvv", content)

        # Entries: uppercase hash, asterisk prefix, forward slashes
        self.assertIn("ED26553EDB44956D *video.mp4", content)
        self.assertIn("0C27693DBFD5E7DA *subdir/audio.wav", content)

    def test_write_xxh_different_algorithms(self) -> None:
        """Test .xxh header reflects the algorithm used."""
        for algo, expected_name in [
            ("xxh64", "xxHash-64"),
            ("md5", "MD5"),
            ("sha256", "SHA-256"),
        ]:
            output = self.test_path / f"test_{algo}.xxh"
            HashFileWriter.write_xxh(self.entries, output, algo)
            content = output.read_text(encoding="utf-8")
            self.assertIn(expected_name, content)

    def test_write_mhl_format(self) -> None:
        """Test ASC MHL XML file generation."""
        path = HashFileWriter.write_mhl(
            self.entries, self.test_path, "xxh3_64", "test_source"
        )

        self.assertTrue(path.exists())
        self.assertTrue(path.parent.name == "ascmhl")

        import xml.etree.ElementTree as ET

        tree = ET.parse(path)
        root = tree.getroot()

        self.assertEqual(root.tag, "hashlist")
        self.assertEqual(root.attrib["version"], "2.0")

        # Creator info
        creator = root.find("creatorinfo")
        self.assertIsNotNone(creator)
        self.assertEqual(creator.find("creationtool").text, "cvv 0.0.1")

        # Hash entries
        hashes = root.find("hashes")
        hash_elems = hashes.findall("hash")
        self.assertEqual(len(hash_elems), 2)

        # First entry
        self.assertEqual(hash_elems[0].find("path").text, "video.mp4")
        self.assertEqual(hash_elems[0].find("path").attrib["size"], "1234567")
        self.assertEqual(
            hash_elems[0].find("xxh3_64").text, "ed26553edb44956d"
        )

        # Second entry (with subdirectory)
        self.assertEqual(hash_elems[1].find("path").text, "subdir/audio.wav")

    def test_write_mhl_creates_ascmhl_dir(self) -> None:
        """Test that MHL writer creates the ascmhl/ subdirectory."""
        ascmhl_dir = self.test_path / "ascmhl"
        self.assertFalse(ascmhl_dir.exists())

        HashFileWriter.write_mhl(self.entries, self.test_path, "xxh3_64", "src")

        self.assertTrue(ascmhl_dir.exists())
        self.assertTrue(ascmhl_dir.is_dir())


class TestHashFileIntegration(unittest.TestCase):
    """Test hash file generation integrated with CLIProcessor."""

    def setUp(self) -> None:
        CopyEngine.reset_shared_state()
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        self.source_dir = self.test_path / "source"
        self.source_dir.mkdir()
        (self.source_dir / "file1.txt").write_bytes(b"content1")
        (self.source_dir / "file2.txt").write_bytes(b"content2")

        self.dest = self.test_path / "dest"
        self.dest.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_cli_generates_xxh_file(self) -> None:
        """Test that CLIProcessor generates .xxh when requested."""
        processor = CLIProcessor(
            source=self.source_dir,
            destinations=[self.dest],
            verification_mode=VerificationMode.FULL,
            hash_algorithm="xxh3_64",
            hash_file_formats=["xxh"],
        )

        success = processor.run()
        self.assertTrue(success)

        xxh_file = self.dest / "source.xxh3"
        self.assertTrue(xxh_file.exists(), f"{xxh_file} should be generated")

        content = xxh_file.read_text(encoding="utf-8")
        self.assertIn("file1.txt", content)
        self.assertIn("file2.txt", content)

    def test_cli_generates_mhl_file(self) -> None:
        """Test that CLIProcessor generates .mhl when requested."""
        processor = CLIProcessor(
            source=self.source_dir,
            destinations=[self.dest],
            verification_mode=VerificationMode.FULL,
            hash_algorithm="xxh3_64",
            hash_file_formats=["mhl"],
        )

        success = processor.run()
        self.assertTrue(success)

        ascmhl_dir = self.dest / "ascmhl"
        self.assertTrue(ascmhl_dir.exists())
        mhl_files = list(ascmhl_dir.glob("*.mhl"))
        self.assertEqual(len(mhl_files), 1)

    def test_cli_generates_both_formats(self) -> None:
        """Test that CLIProcessor generates both .xxh and .mhl."""
        processor = CLIProcessor(
            source=self.source_dir,
            destinations=[self.dest],
            verification_mode=VerificationMode.FULL,
            hash_algorithm="xxh3_64",
            hash_file_formats=["xxh", "mhl"],
        )

        success = processor.run()
        self.assertTrue(success)

        self.assertTrue((self.dest / "source.xxh3").exists())
        self.assertTrue((self.dest / "ascmhl").exists())

    def test_no_hash_file_without_flag(self) -> None:
        """Test that no hash file is generated when not requested."""
        processor = CLIProcessor(
            source=self.source_dir,
            destinations=[self.dest],
            verification_mode=VerificationMode.FULL,
            hash_algorithm="xxh3_64",
        )

        success = processor.run()
        self.assertTrue(success)

        self.assertFalse((self.dest / "source.xxh3").exists())
        self.assertFalse((self.dest / "ascmhl").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
