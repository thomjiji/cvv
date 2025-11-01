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
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cvv import (
    CLIProcessor,
    CopyEngine,
    CopyEvent,
    CopyResult,
    EventType,
    HashCalculator,
    VerificationMode,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hash_test_dir():
    """Create and cleanup test directory for hash tests."""
    CopyEngine.reset_shared_state()
    test_dir = tempfile.mkdtemp()
    yield Path(test_dir)
    shutil.rmtree(test_dir)


@pytest.fixture
def copy_test_env():
    """Create test environment for copy engine tests."""
    CopyEngine.reset_shared_state()
    test_dir = tempfile.mkdtemp()
    test_path = Path(test_dir)

    # Create source file
    source_file = test_path / "source.txt"
    test_data = b"This is test data for CopyEngine." * 100  # ~3.4 KB
    source_file.write_bytes(test_data)

    yield test_path, source_file, test_data
    shutil.rmtree(test_dir)


@pytest.fixture
def verify_test_env():
    """Create test environment for verification tests."""
    CopyEngine.reset_shared_state()
    test_dir = tempfile.mkdtemp()
    test_path = Path(test_dir)

    # Create larger source file for better testing
    source_file = test_path / "source.dat"
    test_data = b"X" * (1024 * 100)  # 100 KB
    source_file.write_bytes(test_data)

    yield test_path, source_file, test_data
    shutil.rmtree(test_dir)


@pytest.fixture
def error_test_env():
    """Create test environment for error handling tests."""
    CopyEngine.reset_shared_state()
    test_dir = tempfile.mkdtemp()
    test_path = Path(test_dir)

    source_file = test_path / "source.txt"
    source_file.write_bytes(b"test data")

    yield test_path, source_file
    shutil.rmtree(test_dir)


@pytest.fixture
def cli_test_env():
    """Create test environment for CLI processor tests."""
    CopyEngine.reset_shared_state()
    test_dir = tempfile.mkdtemp()
    test_path = Path(test_dir)

    # Create source directory with files
    source_dir = test_path / "source"
    source_dir.mkdir()
    (source_dir / "file1.txt").write_text("content1")
    (source_dir / "file2.txt").write_text("content2")

    # Create subdirectory
    subdir = source_dir / "subdir"
    subdir.mkdir()
    (subdir / "file3.txt").write_text("content3")

    # Create destination directories
    dest1 = test_path / "dest1"
    dest2 = test_path / "dest2"
    dest1.mkdir()
    dest2.mkdir()

    yield test_path, source_dir, dest1, dest2
    shutil.rmtree(test_dir)


@pytest.fixture
def integration_test_dir():
    """Create test directory for integration tests."""
    CopyEngine.reset_shared_state()
    test_dir = tempfile.mkdtemp()
    yield Path(test_dir)
    shutil.rmtree(test_dir)


# ============================================================================
# Hash Calculator Tests (synchronous)
# ============================================================================


def test_hash_file_generator_md5(hash_test_dir) -> None:
    """Test MD5 hash calculation via generator."""
    test_file = hash_test_dir / "test.txt"
    test_data = b"Hello, World!"
    test_file.write_bytes(test_data)

    # Consume the generator
    final_hash = ""
    for _, final_hash in HashCalculator.hash_file(test_file, "md5"):
        if final_hash:
            break

    expected = hashlib.md5(test_data).hexdigest()
    assert final_hash == expected


def test_hash_file_generator_sha256(hash_test_dir) -> None:
    """Test SHA256 hash calculation via generator."""
    test_file = hash_test_dir / "test.txt"
    test_data = b"Test data for SHA256"
    test_file.write_bytes(test_data)

    # Consume the generator
    final_hash = ""
    for _, final_hash in HashCalculator.hash_file(test_file, "sha256"):
        if final_hash:
            break

    expected = hashlib.sha256(test_data).hexdigest()
    assert final_hash == expected


def test_hash_file_generator_xxh64be(hash_test_dir) -> None:
    """Test xxHash calculation via generator."""
    test_file = hash_test_dir / "test.txt"
    test_data = b"xxHash test data"
    test_file.write_bytes(test_data)

    # Consume the generator
    final_hash = ""
    bytes_processed = 0
    for bytes_processed, final_hash in HashCalculator.hash_file(test_file, "xxh64be"):
        pass  # Process all yields

    assert final_hash  # Should have a hash
    assert bytes_processed == len(test_data)


def test_unsupported_algorithm() -> None:
    """Test error handling for unsupported hash algorithm."""
    with pytest.raises(ValueError):
        HashCalculator("unsupported_algorithm")


# ============================================================================
# Copy Engine Tests (async)
# ============================================================================


@pytest.mark.asyncio
async def test_single_destination_copy(copy_test_env) -> None:
    """Test copying to a single destination."""
    test_path, source_file, test_data = copy_test_env
    dest = test_path / "dest1.txt"

    engine = CopyEngine(
        source=source_file,
        destinations=[dest],
        verification_mode=VerificationMode.TRANSFER,
    )

    # Consume generator and get result
    result = None
    async for event in engine.copy():
        if isinstance(event, CopyResult):
            result = event
            break

    assert result is not None
    assert result.success
    assert dest.exists()
    assert dest.read_bytes() == test_data
    assert len(result.destinations) == 1
    assert result.destinations[0].success


@pytest.mark.asyncio
async def test_multi_destination_copy(copy_test_env) -> None:
    """Test copying to multiple destinations simultaneously."""
    test_path, source_file, test_data = copy_test_env
    dest1 = test_path / "dest1.txt"
    dest2 = test_path / "dest2.txt"
    dest3 = test_path / "dest3.txt"

    engine = CopyEngine(
        source=source_file,
        destinations=[dest1, dest2, dest3],
        verification_mode=VerificationMode.TRANSFER,
    )

    # Consume generator
    result = None
    async for event in engine.copy():
        if isinstance(event, CopyResult):
            result = event

    assert result.success
    assert len(result.destinations) == 3

    # Verify all destinations
    for dest in [dest1, dest2, dest3]:
        assert dest.exists()
        assert dest.read_bytes() == test_data


@pytest.mark.asyncio
async def test_copy_events_emitted(copy_test_env) -> None:
    """Test that proper events are emitted during copy."""
    test_path, source_file, test_data = copy_test_env
    dest = test_path / "dest.txt"

    engine = CopyEngine(
        source=source_file,
        destinations=[dest],
        verification_mode=VerificationMode.TRANSFER,
    )

    events = []
    result = None

    async for event in engine.copy():
        if isinstance(event, CopyResult):
            result = event
        elif isinstance(event, CopyEvent):
            events.append(event.type)

    # Check that expected events were emitted
    assert EventType.COPY_START in events
    assert EventType.COPY_PROGRESS in events
    assert EventType.COPY_COMPLETE in events
    assert result is not None


@pytest.mark.asyncio
async def test_source_not_found(copy_test_env) -> None:
    """Test error handling when source file doesn't exist."""
    test_path, source_file, test_data = copy_test_env
    nonexistent = test_path / "nonexistent.txt"
    dest = test_path / "dest.txt"

    engine = CopyEngine(
        source=nonexistent,
        destinations=[dest],
    )

    result = None
    async for event in engine.copy():
        if isinstance(event, CopyResult):
            result = event

    assert result is not None
    assert not result.success
    assert not result.destinations[0].success
    # Error message should mention the file or directory issue
    error_msg = result.destinations[0].error.lower()
    assert (
        "no such file" in error_msg or "not found" in error_msg
    ), f"Expected file not found error, got: {result.destinations[0].error}"


# ============================================================================
# Verification Mode Tests (async)
# ============================================================================


@pytest.mark.asyncio
async def test_transfer_mode_hashing(verify_test_env) -> None:
    """Test TRANSFER mode - hashes in-flight but no post-copy verification."""
    test_path, source_file, test_data = verify_test_env
    dest = test_path / "dest.txt"

    engine = CopyEngine(
        source=source_file,
        destinations=[dest],
        verification_mode=VerificationMode.TRANSFER,
    )

    result = None
    async for event in engine.copy():
        if isinstance(event, CopyResult):
            result = event

    assert result.success
    # TRANSFER mode now computes in-flight hash (essentially free)
    assert result.source_hash_inflight is not None
    # But doesn't do post-copy verification
    assert result.source_hash_post is None


@pytest.mark.asyncio
async def test_source_mode_hashing(verify_test_env) -> None:
    """Test SOURCE mode - hash source in-flight and post-copy."""
    test_path, source_file, test_data = verify_test_env
    dest = test_path / "dest.txt"

    engine = CopyEngine(
        source=source_file,
        destinations=[dest],
        verification_mode=VerificationMode.SOURCE,
    )

    result = None
    events = []

    async for event in engine.copy():
        if isinstance(event, CopyResult):
            result = event
        elif isinstance(event, CopyEvent):
            events.append(event.type)

    assert result.success
    assert result.source_hash_inflight is not None
    assert result.source_hash_post is not None
    assert result.source_hash_inflight == result.source_hash_post

    # Verify events include verification
    assert EventType.VERIFY_START in events
    assert EventType.VERIFY_COMPLETE in events


@pytest.mark.asyncio
async def test_full_mode_hashing(verify_test_env) -> None:
    """Test FULL mode - hash source and all destinations."""
    test_path, source_file, test_data = verify_test_env
    dest1 = test_path / "dest1.txt"
    dest2 = test_path / "dest2.txt"

    engine = CopyEngine(
        source=source_file,
        destinations=[dest1, dest2],
        verification_mode=VerificationMode.FULL,
    )

    result = None
    async for event in engine.copy():
        if isinstance(event, CopyResult):
            result = event

    assert result.success
    assert result.source_hash_inflight is not None
    assert result.source_hash_post is not None

    # Check destination hashes
    for dest_result in result.destinations:
        assert dest_result.hash_post is not None
        assert dest_result.hash_post == result.source_hash_inflight


@pytest.mark.asyncio
async def test_full_mode_detects_corruption(verify_test_env) -> None:
    """Test that FULL mode detects file corruption."""
    test_path, source_file, test_data = verify_test_env
    dest = test_path / "dest.txt"

    # Patch the writer thread to corrupt the file
    original_replace = Path.replace

    def corrupt_on_replace(self, target):
        original_replace(self, target)
        # Corrupt the destination after rename
        with open(target, "ab") as f:
            f.write(b"CORRUPTED")

    with patch.object(Path, "replace", corrupt_on_replace):
        engine = CopyEngine(
            source=source_file,
            destinations=[dest],
            verification_mode=VerificationMode.FULL,
        )

        result = None
        async for event in engine.copy():
            if isinstance(event, CopyResult):
                result = event

        # Should detect the corruption
        assert not result.success
        assert not result.destinations[0].success
        assert "mismatch" in result.destinations[0].error.lower()


# ============================================================================
# Error Handling Tests (async)
# ============================================================================


@pytest.mark.asyncio
async def test_insufficient_disk_space(error_test_env) -> None:
    """Test error when insufficient disk space."""
    test_path, source_file = error_test_env
    dest = test_path / "dest.txt"

    engine = CopyEngine(
        source=source_file,
        destinations=[dest],
    )

    # Mock disk_usage to simulate no space
    with patch("shutil.disk_usage") as mock_usage:
        mock_usage.return_value = Mock(free=0)

        result = None
        async for event in engine.copy():
            if isinstance(event, CopyResult):
                result = event

        assert not result.success
        assert "space" in result.destinations[0].error.lower()


@pytest.mark.asyncio
async def test_write_permission_error(error_test_env) -> None:
    """Test error when destination is not writable."""
    test_path, source_file = error_test_env
    # Use /dev/full which always returns "disk full" error
    if not Path("/dev/full").exists():
        pytest.skip("/dev/full not available on this system")

    dest = Path("/dev/full")

    engine = CopyEngine(
        source=source_file,
        destinations=[dest],
    )

    result = None
    async for event in engine.copy():
        if isinstance(event, CopyResult):
            result = event

    assert not result.success


@pytest.mark.asyncio
async def test_per_destination_errors(error_test_env) -> None:
    """Test that errors are tracked per destination."""
    test_path, source_file = error_test_env
    dest1 = test_path / "dest1.txt"  # Good destination
    dest2 = Path("/nonexistent/path/dest2.txt")  # Bad destination

    engine = CopyEngine(
        source=source_file,
        destinations=[dest1, dest2],
    )

    result = None
    async for event in engine.copy():
        if isinstance(event, CopyResult):
            result = event

    assert result is not None
    # Overall operation failed because one destination failed
    assert not result.success


# ============================================================================
# CLI Processor Tests (async)
# ============================================================================


@pytest.mark.asyncio
async def test_single_file_copy(cli_test_env) -> None:
    """Test copying a single file to multiple destinations."""
    test_path, source_dir, dest1, dest2 = cli_test_env
    source_file = source_dir / "file1.txt"

    processor = CLIProcessor(
        source=source_file,
        destinations=[dest1, dest2],
        verification_mode=VerificationMode.TRANSFER,
        hash_algorithm="xxh64be",
    )

    success = await processor.run()

    assert success
    assert (dest1 / "file1.txt").exists()
    assert (dest2 / "file1.txt").exists()
    assert (dest1 / "file1.txt").read_text() == "content1"
    assert (dest2 / "file1.txt").read_text() == "content1"


@pytest.mark.asyncio
async def test_directory_copy_preserves_structure(cli_test_env) -> None:
    """Test that directory copying preserves directory structure."""
    test_path, source_dir, dest1, dest2 = cli_test_env

    processor = CLIProcessor(
        source=source_dir,
        destinations=[dest1, dest2],
        verification_mode=VerificationMode.TRANSFER,
        hash_algorithm="xxh64be",
    )

    success = await processor.run()

    assert success

    # Check all files exist in both destinations
    for dest in [dest1, dest2]:
        assert (dest / "file1.txt").exists()
        assert (dest / "file2.txt").exists()
        assert (dest / "subdir" / "file3.txt").exists()

        # Verify content
        assert (dest / "file1.txt").read_text() == "content1"
        assert (dest / "file2.txt").read_text() == "content2"
        assert (dest / "subdir" / "file3.txt").read_text() == "content3"


def test_discover_files_directory(cli_test_env) -> None:
    """Test file discovery from directory."""
    test_path, source_dir, dest1, dest2 = cli_test_env

    processor = CLIProcessor(
        source=source_dir,
        destinations=[dest1],
        verification_mode=VerificationMode.TRANSFER,
        hash_algorithm="xxh64be",
    )

    files = processor._discover_files()

    assert len(files) == 3  # file1, file2, file3
    assert source_dir / "file1.txt" in files
    assert source_dir / "file2.txt" in files
    assert source_dir / "subdir" / "file3.txt" in files


def test_discover_files_single_file(cli_test_env) -> None:
    """Test file discovery from single file."""
    test_path, source_dir, dest1, dest2 = cli_test_env
    source_file = source_dir / "file1.txt"

    processor = CLIProcessor(
        source=source_file,
        destinations=[dest1],
        verification_mode=VerificationMode.TRANSFER,
        hash_algorithm="xxh64be",
    )

    files = processor._discover_files()

    assert len(files) == 1
    assert files[0] == source_file


# ============================================================================
# Integration Tests (async)
# ============================================================================


@pytest.mark.asyncio
async def test_full_workflow_multiple_files_multiple_destinations(
    integration_test_dir,
) -> None:
    """Test complete workflow: multiple files to multiple destinations with full verification."""
    # Create source files
    source_dir = integration_test_dir / "source"
    source_dir.mkdir()

    files_data = {
        "video1.mp4": b"fake video data 1" * 1000,
        "video2.mp4": b"fake video data 2" * 1000,
        "metadata.json": b'{"key": "value"}',
    }

    for filename, data in files_data.items():
        (source_dir / filename).write_bytes(data)

    # Create destinations
    dest1 = integration_test_dir / "backup1"
    dest2 = integration_test_dir / "backup2"
    dest1.mkdir()
    dest2.mkdir()

    # Run processor
    processor = CLIProcessor(
        source=source_dir,
        destinations=[dest1, dest2],
        verification_mode=VerificationMode.FULL,
        hash_algorithm="xxh64be",
    )

    success = await processor.run()

    # Verify success
    assert success

    # Verify all files in both destinations
    for filename, data in files_data.items():
        for dest in [dest1, dest2]:
            dest_file = dest / filename
            assert dest_file.exists(), f"{dest_file} should exist"
            assert (
                dest_file.read_bytes() == data
            ), f"{dest_file} should have correct content"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
