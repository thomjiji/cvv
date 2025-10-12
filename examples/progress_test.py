#!/usr/bin/env python3
"""
Progress update test script for pfndispatchcopy.

This script tests the optimized progress update frequency by creating
files of different sizes and monitoring the update behavior.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pfndispatchcopy import PSTaskWrapper, setup_logging


def create_test_file_with_pattern(file_path: Path, size_mb: int) -> None:
    """
    Create a test file with a repeating pattern to make it more realistic.

    Parameters
    ----------
    file_path : Path
        Path where to create the test file
    size_mb : int
        Size of the file in megabytes
    """
    chunk_size = 1024 * 1024  # 1MB chunks
    pattern = b"TESTDATA" * (chunk_size // 8)  # Fill 1MB with pattern

    with open(file_path, "wb") as f:
        for i in range(size_mb):
            # Vary the pattern slightly to make it more realistic
            chunk = pattern[: chunk_size - 8] + f"CHUNK{i:03d}".encode()[:8]
            f.write(chunk)


def test_progress_updates(file_size_mb: int, destinations_count: int = 3) -> None:
    """
    Test progress updates with a specific file size.

    Parameters
    ----------
    file_size_mb : int
        Size of test file in megabytes
    destinations_count : int
        Number of destinations to copy to
    """
    print(f"\n{'=' * 60}")
    print(f"📊 TESTING: {file_size_mb}MB file → {destinations_count} destinations")
    print(f"{'=' * 60}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test file
        source_file = temp_path / f"test_file_{file_size_mb}MB.bin"
        print(f"📁 Creating {file_size_mb}MB test file...")

        start_time = time.time()
        create_test_file_with_pattern(source_file, file_size_mb)
        creation_time = time.time() - start_time

        actual_size = source_file.stat().st_size
        print(f"✅ Created file: {actual_size:,} bytes in {creation_time:.2f}s")

        # Create destinations
        destinations = []
        for i in range(destinations_count):
            dest_file = temp_path / f"dest_{i}" / f"copied_file_{file_size_mb}MB.bin"
            destinations.append(dest_file)

        # Test with PSTaskWrapper
        print(f"\n🚀 Starting copy operation...")
        print(f"📋 Monitoring progress updates (should be less frequent now)")

        wrapper = PSTaskWrapper(verbose=True)

        copy_start_time = time.time()

        result = wrapper.launch_copy(
            source=source_file,
            destinations=destinations,
            buffer_size=8 * 1024 * 1024,  # 8MB buffer for large files
            hash_algorithm="md5"
            if file_size_mb < 200
            else None,  # Skip hash for very large files
        )

        copy_duration = time.time() - copy_start_time

        if result["success"]:
            speed_mb_s = (
                file_size_mb * destinations_count / copy_duration
                if copy_duration > 0
                else 0
            )
            print(f"\n✅ Copy completed successfully!")
            print(f"   Duration: {copy_duration:.2f} seconds")
            print(f"   Speed: {speed_mb_s:.1f} MB/s (total throughput)")
            print(f"   Data transferred: {file_size_mb * destinations_count} MB")

            # Verify file sizes
            all_correct = True
            for dest in destinations:
                if dest.exists() and dest.stat().st_size == actual_size:
                    print(
                        f"   ✅ {dest.parent.name}/{dest.name}: {dest.stat().st_size:,} bytes"
                    )
                else:
                    print(f"   ❌ {dest.parent.name}/{dest.name}: FAILED")
                    all_correct = False

            if all_correct:
                print(f"   🎯 All {destinations_count} destinations verified!")
            else:
                print(f"   ⚠️  Some destinations failed verification")

        else:
            print(f"❌ Copy failed!")
            if "files_failed" in result:
                print(f"   Failed files: {result['files_failed']}")


def test_directory_progress() -> None:
    """Test progress updates with directory copying."""
    print(f"\n{'=' * 60}")
    print(f"📁 TESTING: Directory with multiple files")
    print(f"{'=' * 60}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test directory structure
        source_dir = temp_path / "test_project"

        # Create subdirectories
        for subdir in ["video", "audio", "docs", "assets"]:
            (source_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Create files of varying sizes
        file_configs = [
            ("video/clip1.mov", 50),  # 50MB
            ("video/clip2.mov", 75),  # 75MB
            ("audio/track1.wav", 25),  # 25MB
            ("audio/track2.wav", 30),  # 30MB
            ("docs/script.txt", 1),  # 1MB
            ("assets/images.zip", 20),  # 20MB
        ]

        total_size_mb = 0
        print(f"📂 Creating directory structure:")

        for file_path, size_mb in file_configs:
            full_path = source_dir / file_path
            print(f"   📄 {file_path}: {size_mb}MB")
            create_test_file_with_pattern(full_path, size_mb)
            total_size_mb += size_mb

        print(f"📊 Total directory size: {total_size_mb}MB")

        # Create destinations
        destinations = [
            temp_path / "backup_1",
            temp_path / "backup_2",
        ]

        print(f"\n🚀 Starting directory copy...")
        wrapper = PSTaskWrapper(verbose=True)

        start_time = time.time()

        result = wrapper.launch_copy(
            source=source_dir,
            destinations=destinations,
            buffer_size=4 * 1024 * 1024,  # 4MB buffer
            hash_algorithm="sha1",
        )

        duration = time.time() - start_time

        if result["success"]:
            files_copied = result.get("files_copied", 0)
            total_bytes = result.get("total_bytes", 0)
            speed_mb_s = (
                (total_bytes * len(destinations)) / (1024 * 1024) / duration
                if duration > 0
                else 0
            )

            print(f"\n✅ Directory copy completed!")
            print(f"   Files copied: {files_copied}")
            print(f"   Total bytes: {total_bytes:,}")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Speed: {speed_mb_s:.1f} MB/s")
        else:
            print(f"❌ Directory copy failed!")
            print(f"   Failed files: {result.get('files_failed', 0)}")


def main() -> None:
    """Run all progress update tests."""
    print("🧪 Progress Update Frequency Test Suite")
    print("=" * 60)
    print("Testing optimized progress updates:")
    print("• Updates every 2+ seconds (instead of 0.5s)")
    print("• Updates every 5% progress or 10MB+ (whichever comes first)")
    print("• Better formatted output with commas and percentages")

    # Set up logging to see all progress updates
    setup_logging(verbose=True)

    try:
        # Test different file sizes
        test_sizes = [50, 100, 200]  # MB

        for size_mb in test_sizes:
            test_progress_updates(size_mb, destinations_count=2)
            time.sleep(1)  # Small delay between tests

        # Test directory copying
        test_directory_progress()

        print(f"\n{'=' * 60}")
        print("🎉 All progress update tests completed!")
        print("📋 Observations:")
        print("   • Progress updates should now be much less frequent")
        print("   • Updates include percentage and formatted byte counts")
        print("   • Large files show better progress granularity")
        print("   • Directory copying shows per-file progress")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
