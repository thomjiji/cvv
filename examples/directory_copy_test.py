#!/usr/bin/env python3
"""
Directory copy test script for PSTaskWrapper functionality.

This script tests the directory copying capabilities of the pfndispatchcopy tool,
including nested directory structures and multiple destination handling.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pfndispatchcopy import PSTaskWrapper, setup_logging


def create_test_directory_structure(base_path: Path) -> Path:
    """
    Create a complex test directory structure.

    Parameters
    ----------
    base_path : Path
        Base path where to create the test structure

    Returns
    -------
    Path
        Path to the created test directory
    """
    test_dir = base_path / "test_project"

    # Create directory structure
    dirs_to_create = [
        "video/raw_footage",
        "video/edited",
        "audio/original",
        "audio/processed",
        "docs/scripts",
        "docs/notes",
        "assets/images",
        "assets/graphics",
    ]

    for dir_path in dirs_to_create:
        (test_dir / dir_path).mkdir(parents=True, exist_ok=True)

    # Create test files with varying sizes
    test_files = [
        ("video/raw_footage/clip001.mov", "Raw video footage clip 001" * 1000),
        ("video/raw_footage/clip002.mov", "Raw video footage clip 002" * 800),
        ("video/edited/final_cut_v1.mp4", "Final edited video version 1" * 1200),
        ("video/edited/final_cut_v2.mp4", "Final edited video version 2" * 1100),
        ("audio/original/interview.wav", "Original interview audio" * 2000),
        ("audio/processed/interview_clean.wav", "Processed clean audio" * 1500),
        ("docs/scripts/shooting_script.txt", "Shooting script content" * 300),
        ("docs/notes/director_notes.md", "Director notes and comments" * 400),
        ("assets/images/logo.png", "PNG image data" * 500),
        ("assets/graphics/title_card.svg", "SVG graphics data" * 200),
        ("project_info.json", '{"project": "test", "version": "1.0"}'),
        ("README.md", "# Test Project\n\nThis is a test project structure."),
    ]

    for file_path, content in test_files:
        full_path = test_dir / file_path
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

    return test_dir


def verify_directory_structure(source_dir: Path, dest_dir: Path) -> bool:
    """
    Verify that directory structure was copied correctly.

    Parameters
    ----------
    source_dir : Path
        Source directory path
    dest_dir : Path
        Destination directory path

    Returns
    -------
    bool
        True if structure matches
    """
    if not dest_dir.exists():
        print(f"❌ Destination directory does not exist: {dest_dir}")
        return False

    # Get all files in source
    source_files = []
    for item in source_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(source_dir)
            source_files.append(rel_path)

    # Check all files exist in destination
    all_match = True
    for rel_path in source_files:
        source_file = source_dir / rel_path
        dest_file = dest_dir / rel_path

        if not dest_file.exists():
            print(f"❌ Missing file: {dest_file}")
            all_match = False
            continue

        # Check file sizes match
        if source_file.stat().st_size != dest_file.stat().st_size:
            print(f"❌ Size mismatch: {source_file} vs {dest_file}")
            all_match = False
            continue

        # Check file content matches
        try:
            with open(source_file, "r", encoding="utf-8") as sf, open(
                dest_file, "r", encoding="utf-8"
            ) as df:
                if sf.read() != df.read():
                    print(f"❌ Content mismatch: {source_file} vs {dest_file}")
                    all_match = False
                    continue
        except UnicodeDecodeError:
            # For binary files, just check size (already done above)
            pass

        print(f"✅ Verified: {rel_path}")

    return all_match


def test_single_directory_copy():
    """Test copying a directory to a single destination."""
    print("\n" + "=" * 60)
    print("📁 TEST: Single Directory Copy")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test structure
        print("🏗️  Creating test directory structure...")
        source_dir = create_test_directory_structure(temp_path)

        # Create destination
        dest_dir = temp_path / "backup_1" / "test_project"

        # Copy using PSTaskWrapper
        wrapper = PSTaskWrapper(verbose=True)

        print(f"\n📋 Copying {source_dir} to {dest_dir}")
        result = wrapper.launch_copy(
            source=source_dir,
            destinations=[dest_dir],
            buffer_size=64 * 1024,  # 64KB for testing
            hash_algorithm="md5",
        )

        # Verify results
        if result["success"]:
            print(f"✅ Copy completed: {result['files_copied']} files")

            # Verify structure
            if verify_directory_structure(source_dir, dest_dir):
                print("✅ Directory structure verification passed")
                return True
            else:
                print("❌ Directory structure verification failed")
                return False
        else:
            print(f"❌ Copy failed: {result.get('files_failed', 0)} failures")
            return False


def test_multiple_directory_copy():
    """Test copying a directory to multiple destinations."""
    print("\n" + "=" * 60)
    print("📁 TEST: Multiple Directory Copy")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test structure
        print("🏗️  Creating test directory structure...")
        source_dir = create_test_directory_structure(temp_path)

        # Create multiple destinations
        destinations = [
            temp_path / "backup_1" / "test_project",
            temp_path / "backup_2" / "test_project",
            temp_path / "backup_3" / "test_project",
        ]

        # Copy using PSTaskWrapper
        wrapper = PSTaskWrapper(verbose=True)

        print(f"\n📋 Copying {source_dir} to {len(destinations)} destinations")
        result = wrapper.launch_copy(
            source=source_dir,
            destinations=destinations,
            buffer_size=128 * 1024,  # 128KB buffer
            hash_algorithm="sha1",
        )

        # Verify results
        if result["success"]:
            print(
                f"✅ Copy completed: {result['files_copied']} files to {len(destinations)} destinations"
            )

            # Verify all destinations
            all_verified = True
            for dest_dir in destinations:
                print(f"\n🔍 Verifying {dest_dir.name}...")
                if verify_directory_structure(source_dir, dest_dir):
                    print(f"✅ {dest_dir.name} verification passed")
                else:
                    print(f"❌ {dest_dir.name} verification failed")
                    all_verified = False

            return all_verified
        else:
            print(f"❌ Copy failed: {result.get('files_failed', 0)} failures")
            return False


def test_mixed_file_directory_copy():
    """Test copying both individual files and directories."""
    print("\n" + "=" * 60)
    print("📁 TEST: Mixed File and Directory Copy")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test 1: Single file copy
        print("\n🎬 Testing single file copy...")
        test_file = temp_path / "single_test_file.txt"
        test_file.write_text("This is a single test file content")

        file_destinations = [temp_path / "file_dest1.txt", temp_path / "file_dest2.txt"]

        wrapper = PSTaskWrapper(verbose=True)
        file_result = wrapper.launch_copy(
            source=test_file, destinations=file_destinations, hash_algorithm="md5"
        )

        if file_result["success"]:
            print("✅ Single file copy successful")
            # Verify file copies
            for dest in file_destinations:
                if dest.exists() and dest.read_text() == test_file.read_text():
                    print(f"✅ {dest.name} verified")
                else:
                    print(f"❌ {dest.name} verification failed")
        else:
            print("❌ Single file copy failed")
            return False

        # Test 2: Directory copy
        print("\n📁 Testing directory copy...")
        source_dir = create_test_directory_structure(temp_path)

        dir_destinations = [
            temp_path / "mixed_backup1" / "test_project",
            temp_path / "mixed_backup2" / "test_project",
        ]

        dir_result = wrapper.launch_copy(
            source=source_dir,
            destinations=dir_destinations,
            buffer_size=256 * 1024,
            hash_algorithm="sha256",
        )

        if dir_result["success"]:
            print(f"✅ Directory copy successful: {dir_result['files_copied']} files")
            return True
        else:
            print(
                f"❌ Directory copy failed: {dir_result.get('files_failed', 0)} failures"
            )
            return False


def test_performance_comparison():
    """Test performance with different buffer sizes and destinations."""
    print("\n" + "=" * 60)
    print("📊 TEST: Performance Comparison")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create larger test structure
        print("🏗️  Creating large test directory structure...")
        source_dir = create_test_directory_structure(temp_path)

        # Add more files for performance testing
        large_files_dir = source_dir / "large_files"
        large_files_dir.mkdir(exist_ok=True)

        for i in range(5):
            large_file = large_files_dir / f"large_file_{i}.bin"
            large_file.write_text("Large file content " * 10000)  # ~200KB each

        # Test different configurations
        configs = [
            {"buffer_size": 64 * 1024, "label": "64KB buffer"},
            {"buffer_size": 512 * 1024, "label": "512KB buffer"},
            {"buffer_size": 2 * 1024 * 1024, "label": "2MB buffer"},
        ]

        dest_counts = [2, 4]

        for dest_count in dest_counts:
            print(f"\n🎯 Testing with {dest_count} destinations:")

            for config in configs:
                destinations = [
                    temp_path
                    / f"perf_{config['buffer_size']}_{dest_count}_dest_{i}"
                    / "test_project"
                    for i in range(dest_count)
                ]

                wrapper = PSTaskWrapper(
                    verbose=False
                )  # Less verbose for performance test

                import time

                start_time = time.time()

                result = wrapper.launch_copy(
                    source=source_dir,
                    destinations=destinations,
                    buffer_size=config["buffer_size"],
                    hash_algorithm="md5",
                )

                duration = time.time() - start_time

                if result["success"]:
                    files_count = result["files_copied"]
                    total_bytes = result["total_bytes"]
                    speed_mb_s = (
                        (total_bytes * dest_count / (1024 * 1024)) / duration
                        if duration > 0
                        else 0
                    )

                    print(
                        f"  {config['label']:>12}: {duration:.2f}s, {speed_mb_s:.1f} MB/s, {files_count} files"
                    )
                else:
                    print(f"  {config['label']:>12}: FAILED")

        return True


def main():
    """Run all directory copy tests."""
    print("🧪 PSTaskWrapper Directory Copy Tests")
    print("=" * 60)

    # Set up logging
    setup_logging(verbose=True)

    test_results = []

    try:
        # Run tests
        test_results.append(("Single Directory Copy", test_single_directory_copy()))
        test_results.append(("Multiple Directory Copy", test_multiple_directory_copy()))
        test_results.append(
            ("Mixed File/Directory Copy", test_mixed_file_directory_copy())
        )
        test_results.append(("Performance Comparison", test_performance_comparison()))

        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)

        passed = 0
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:30} {status}")
            if result:
                passed += 1

        print(f"\nOverall: {passed}/{len(test_results)} tests passed")

        if passed == len(test_results):
            print("🎉 All tests passed!")
            return 0
        else:
            print("⚠️  Some tests failed")
            return 1

    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n💥 Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
