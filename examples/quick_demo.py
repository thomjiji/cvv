#!/usr/bin/env python3
"""
Quick demonstration of pfndispatchcopy functionality.

This script creates sample files and directories and demonstrates the core features
of the pfndispatchcopy tool including both file and directory copying capabilities.
"""

import logging
import sys
import tempfile
import time
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import our pfndispatchcopy functions
from pfndispatchcopy import (
    copy_with_multiple_destinations,
    PSTaskWrapper,
    setup_logging,
)


def create_demo_file(
    file_path: Path, content: str = "Demo content", size_kb: int = 100
) -> None:
    """
    Create a demo file with specified content and size.

    Parameters
    ----------
    file_path : Path
        Where to create the file
    content : str
        Base content to repeat
    size_kb : int
        Approximate size in KB
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate how many times to repeat content to reach target size
    target_bytes = size_kb * 1024
    content_bytes = content.encode("utf-8")
    repeats = max(1, target_bytes // len(content_bytes))

    with open(file_path, "w", encoding="utf-8") as f:
        for i in range(repeats):
            f.write(f"{content} - Line {i + 1}\n")

    print(f"📁 Created demo file: {file_path} ({file_path.stat().st_size:,} bytes)")


def demo_basic_copy() -> None:
    """Demonstrate basic copying functionality."""
    print("\n" + "=" * 50)
    print("🚀 DEMO: Basic Multi-Destination Copy")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create source file
        source = temp_path / "camera_footage.mov"
        create_demo_file(source, "Camera footage data", size_kb=500)

        # Define destinations
        backup1 = temp_path / "backup_drive_1" / "footage.mov"
        backup2 = temp_path / "backup_drive_2" / "footage.mov"

        print(f"\n📹 Source: {source.name}")
        print(f"💾 Destination 1: {backup1}")
        print(f"💾 Destination 2: {backup2}")

        # Perform copy
        start_time = time.time()
        result = copy_with_multiple_destinations(
            source=source,
            destinations=[backup1, backup2],
            buffer_size=64 * 1024,  # 64KB buffer
            hash_algorithm="md5",
        )
        duration = time.time() - start_time

        # Show results
        if result["success"]:
            print(f"✅ Copy completed in {duration:.2f} seconds")
            print(f"⚡ Speed: {result['speed_mb_sec']:.1f} MB/sec")
            print(f"🔐 MD5 Hash: {result['hash'][:16]}...")

            # Verify files exist and are correct size
            original_size = source.stat().st_size
            for dest in [backup1, backup2]:
                if dest.exists() and dest.stat().st_size == original_size:
                    print(f"✅ {dest.name}: {dest.stat().st_size:,} bytes - OK")
                else:
                    print(f"❌ {dest.name}: Verification failed")
        else:
            print("❌ Copy failed!")


def demo_performance_comparison() -> None:
    """Demonstrate performance with different buffer sizes."""
    print("\n" + "=" * 50)
    print("⚡ DEMO: Performance Comparison")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create larger file for performance testing
        source = temp_path / "large_file.bin"
        create_demo_file(source, "Performance test data", size_kb=2000)  # 2MB

        buffer_sizes = [
            (4 * 1024, "4KB"),
            (64 * 1024, "64KB"),
            (1024 * 1024, "1MB"),
            (8 * 1024 * 1024, "8MB"),
        ]

        print(f"📊 Testing with {source.stat().st_size:,} byte file")

        for buffer_size, label in buffer_sizes:
            dest = temp_path / f"perf_test_{label.replace(' ', '_')}.bin"

            start_time = time.time()
            result = copy_with_multiple_destinations(
                source=source, destinations=[dest], buffer_size=buffer_size
            )

            if result["success"]:
                print(f"  {label:>6} buffer: {result['speed_mb_sec']:>6.1f} MB/sec")
            else:
                print(f"  {label:>6} buffer: Failed")


def demo_integrity_verification() -> None:
    """Demonstrate file integrity verification."""
    print("\n" + "=" * 50)
    print("🔒 DEMO: Integrity Verification")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create source file
        source = temp_path / "important_document.pdf"
        create_demo_file(source, "Critical business document", size_kb=150)

        # Test with different hash algorithms
        hash_algorithms = ["md5", "sha1", "sha256"]

        print(f"🔍 Testing hash algorithms on {source.stat().st_size:,} byte file")

        for hash_algo in hash_algorithms:
            dest = temp_path / f"verified_copy_{hash_algo}.pdf"

            result = copy_with_multiple_destinations(
                source=source,
                destinations=[dest],
                buffer_size=32 * 1024,
                hash_algorithm=hash_algo,
            )

            if result["success"]:
                print(
                    f"  {hash_algo.upper():>6}: {result['hash'][:16]}... "
                    f"({result['speed_mb_sec']:.1f} MB/sec)"
                )
            else:
                print(f"  {hash_algo.upper():>6}: Failed")


def demo_directory_copying() -> None:
    """Demonstrate directory structure copying capabilities."""
    print("\n" + "=" * 50)
    print("📁 DEMO: Directory Structure Copying")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test directory structure
        source_dir = temp_path / "project_source"
        dirs_to_create = [
            "video/raw",
            "video/edited",
            "audio/original",
            "docs/scripts",
            "assets/images",
        ]

        for dir_path in dirs_to_create:
            (source_dir / dir_path).mkdir(parents=True, exist_ok=True)

        # Create test files
        test_files = [
            ("video/raw/clip001.mov", "Raw video clip 001 data" * 200),
            ("video/edited/final.mp4", "Final edited video" * 300),
            ("audio/original/sound.wav", "Audio track data" * 250),
            ("docs/scripts/script.txt", "Video script content" * 100),
            ("assets/images/logo.png", "PNG image data" * 150),
            ("project_info.json", '{"name": "Test Project", "version": "1.0"}'),
        ]

        for file_path, content in test_files:
            full_path = source_dir / file_path
            full_path.write_text(content)

        print(f"📂 Created directory structure with {len(test_files)} files")

        # Create multiple destination directories
        destinations = [
            temp_path / "backup_drive_1",
            temp_path / "backup_drive_2",
            temp_path / "archive_storage",
        ]

        print(f"🎯 Copying directory to {len(destinations)} destinations...")

        # Use PSTaskWrapper for directory copying
        task_wrapper = PSTaskWrapper(verbose=False)  # Less verbose for demo

        try:
            result = task_wrapper.launch_copy(
                source=source_dir,
                destinations=destinations,
                buffer_size=128 * 1024,  # 128KB buffer
                hash_algorithm="md5",
            )

            if result["success"]:
                print(f"✅ Directory copy completed!")
                print(f"   Files copied: {result['files_copied']}")
                print(f"   Total bytes: {result['total_bytes']:,}")

                # Verify one destination
                dest_to_check = destinations[0]
                if dest_to_check.exists():
                    # Count files in destination
                    dest_files = list(dest_to_check.rglob("*"))
                    file_count = len([f for f in dest_files if f.is_file()])
                    print(f"   Verified {file_count} files in {dest_to_check.name}")

                    # Show directory structure
                    print(f"\n📋 Directory structure preserved:")
                    for file_path in sorted([f for f in dest_files if f.is_file()]):
                        rel_path = file_path.relative_to(dest_to_check)
                        print(f"   📄 {rel_path}")
                else:
                    print("❌ Destination verification failed")
            else:
                print(
                    f"❌ Directory copy failed: {result.get('files_failed', 0)} failures"
                )

        except Exception as e:
            print(f"❌ Directory copy error: {e}")


def demo_error_handling() -> None:
    """Demonstrate error handling capabilities."""
    print("\n" + "=" * 50)
    print("⚠️  DEMO: Error Handling")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test 1: Non-existent source file
        print("🔍 Test 1: Non-existent source file")
        try:
            copy_with_multiple_destinations(
                source=temp_path / "does_not_exist.txt",
                destinations=[temp_path / "dest.txt"],
                buffer_size=1024,
            )
            print("❌ Should have failed!")
        except FileNotFoundError:
            print("✅ Correctly handled missing source file")

        # Test 2: File size mismatch
        print("\n🔍 Test 2: File size verification")
        source = temp_path / "size_test.txt"
        create_demo_file(source, "Size test", size_kb=10)

        try:
            copy_with_multiple_destinations(
                source=source,
                destinations=[temp_path / "dest.txt"],
                buffer_size=1024,
                expected_size=999999,  # Wrong size
            )
            print("❌ Should have failed!")
        except ValueError:
            print("✅ Correctly handled file size mismatch")


def main() -> None:
    """Run all demonstrations."""
    print("🎬 pfndispatchcopy - Professional File Copy Tool Demo")
    print("=" * 60)

    # Setup logging for demos
    setup_logging(verbose=True)

    try:
        # Run all demos
        demo_basic_copy()
        demo_performance_comparison()
        demo_integrity_verification()
        demo_directory_copying()
        demo_error_handling()

        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("📚 Key features demonstrated:")
        print("   • Single file to multiple destinations")
        print("   • Performance optimization with different buffer sizes")
        print("   • Hash verification with multiple algorithms")
        print("   • Directory structure copying with preserved hierarchy")
        print("   • Error handling and validation")
        print("")
        print("📚 See README.md for detailed usage instructions")
        print("🧪 Run 'python3 example_usage.py' for more comprehensive examples")

    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
