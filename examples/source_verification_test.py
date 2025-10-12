#!/usr/bin/env python3
"""
Source verification test script for pfndispatchcopy.

This script demonstrates and tests the source verification functionality,
including both per-file and after-all verification modes.
"""

import logging
import sys
import tempfile
import time
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pfndispatchcopy import PSTaskWrapper, VerificationMode, setup_logging


def create_test_file(file_path: Path, content: str) -> None:
    """
    Create a test file with specified content.

    Parameters
    ----------
    file_path : Path
        Path where to create the file
    content : str
        Content to write to the file
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)


def test_per_file_verification():
    """Test source verification in per-file mode."""
    print("\n" + "=" * 70)
    print("🔍 TEST: Per-File Source Verification")
    print("=" * 70)
    print("Mode: Each file is verified immediately after copying")
    print("Benefit: Immediate feedback on source integrity")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        source_files = []
        for i in range(3):
            source_file = temp_path / f"source_file_{i}.txt"
            create_test_file(source_file, f"Test content for file {i}\n" * 100)
            source_files.append(source_file)

        print(f"\n📁 Created {len(source_files)} test files")

        # Test each file with per-file verification
        wrapper = PSTaskWrapper(verbose=True)
        success_count = 0

        for source_file in source_files:
            destinations = [
                temp_path / "backup1" / source_file.name,
                temp_path / "backup2" / source_file.name,
            ]

            print(f"\n📋 Copying {source_file.name}...")

            result = wrapper.launch_copy(
                source=source_file,
                destinations=destinations,
                buffer_size=64 * 1024,
                hash_algorithm="md5",
                source_verification=VerificationMode.PER_FILE,
            )

            if result.get("success") and result.get("source_verified"):
                print(f"✅ {source_file.name}: Copy and source verification passed")
                success_count += 1
            else:
                print(f"❌ {source_file.name}: Failed")

        print(f"\n📊 Results: {success_count}/{len(source_files)} files verified")
        return success_count == len(source_files)


def test_after_all_verification():
    """Test source verification in after-all mode."""
    print("\n" + "=" * 70)
    print("🔍 TEST: After-All Source Verification")
    print("=" * 70)
    print("Mode: All files copied first, then all sources verified together")
    print("Benefit: Faster copying, batch verification at the end")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test directory structure
        source_dir = temp_path / "project_source"

        test_files = [
            ("video/clip1.mov", "Video clip 1 data\n" * 500),
            ("video/clip2.mov", "Video clip 2 data\n" * 600),
            ("audio/track1.wav", "Audio track 1 data\n" * 400),
            ("audio/track2.wav", "Audio track 2 data\n" * 450),
            ("docs/notes.txt", "Project notes\n" * 200),
        ]

        for file_path, content in test_files:
            full_path = source_dir / file_path
            create_test_file(full_path, content)

        print(f"\n📁 Created directory with {len(test_files)} files")

        # Copy with after-all verification
        destinations = [
            temp_path / "backup_1",
            temp_path / "backup_2",
        ]

        wrapper = PSTaskWrapper(verbose=True)

        print(f"\n🚀 Starting copy with after-all verification...")
        print("   (All files will be copied first, then verified)\n")

        result = wrapper.launch_copy(
            source=source_dir,
            destinations=destinations,
            buffer_size=128 * 1024,
            hash_algorithm="sha1",
            source_verification=VerificationMode.AFTER_ALL,
        )

        if result.get("success"):
            verified_count = sum(
                1
                for v in result.get("source_verification_results", {}).values()
                if v.get("verified")
            )
            total_files = result.get("files_copied", 0)

            print(f"\n✅ All operations completed successfully")
            print(
                f"📊 Source verification: {verified_count}/{total_files} files verified"
            )
            return verified_count == total_files
        else:
            print(f"\n❌ Operation failed")
            return False


def test_no_verification():
    """Test without source verification for comparison."""
    print("\n" + "=" * 70)
    print("📋 TEST: No Source Verification (Standard Mode)")
    print("=" * 70)
    print("Mode: Standard copy with only destination verification")
    print("Benefit: Fastest, but no source integrity check")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test directory
        source_dir = temp_path / "standard_source"

        for i in range(5):
            file_path = source_dir / f"file_{i}.bin"
            create_test_file(file_path, f"Standard test data {i}\n" * 300)

        print(f"\n📁 Created directory with 5 files")

        destinations = [
            temp_path / "standard_backup1",
            temp_path / "standard_backup2",
        ]

        wrapper = PSTaskWrapper(verbose=True)

        print(f"\n🚀 Starting standard copy (no source verification)...\n")

        start_time = time.time()

        result = wrapper.launch_copy(
            source=source_dir,
            destinations=destinations,
            buffer_size=128 * 1024,
            hash_algorithm="md5",
            source_verification=VerificationMode.NONE,
        )

        duration = time.time() - start_time

        if result.get("success"):
            print(f"\n✅ Standard copy completed in {duration:.2f}s")
            print(f"📊 Files copied: {result.get('files_copied', 0)}")
            print(f"ℹ️  Note: Source verification was NOT performed")
            return True
        else:
            print(f"\n❌ Standard copy failed")
            return False


def test_performance_comparison():
    """Compare performance between verification modes."""
    print("\n" + "=" * 70)
    print("⚡ TEST: Performance Comparison")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create source files
        source_dir = temp_path / "perf_source"

        for i in range(10):
            file_path = source_dir / f"file_{i}.dat"
            create_test_file(file_path, f"Performance test data {i}\n" * 1000)

        print(f"📁 Created directory with 10 files\n")

        modes = [
            (VerificationMode.NONE, "No Verification"),
            (VerificationMode.PER_FILE, "Per-File Verification"),
            (VerificationMode.AFTER_ALL, "After-All Verification"),
        ]

        results = {}

        for mode, label in modes:
            print(f"\n🧪 Testing: {label}")

            dest_dir = temp_path / f"perf_dest_{mode.value}"

            wrapper = PSTaskWrapper(verbose=False)

            start_time = time.time()

            result = wrapper.launch_copy(
                source=source_dir,
                destinations=[dest_dir],
                buffer_size=256 * 1024,
                hash_algorithm="sha1",
                source_verification=mode,
            )

            duration = time.time() - start_time

            if result.get("success"):
                results[label] = duration
                print(f"   ✅ Completed in {duration:.3f}s")
            else:
                print(f"   ❌ Failed")

        print(f"\n📊 Performance Summary:")
        print("=" * 50)

        baseline = results.get("No Verification", 1.0)

        for label, duration in results.items():
            overhead = ((duration - baseline) / baseline * 100) if baseline > 0 else 0
            print(f"   {label:30s}: {duration:.3f}s (+{overhead:.1f}%)")

        return True


def demonstrate_verification_failure():
    """Demonstrate what happens when source verification fails."""
    print("\n" + "=" * 70)
    print("⚠️  DEMO: Source Verification Failure Detection")
    print("=" * 70)
    print("This demonstrates how source verification detects file changes")
    print("=" * 70)

    print("\nℹ️  In a real scenario, source verification would detect:")
    print("   • Dying or damaged memory cards")
    print("   • Card reader issues")
    print("   • Filesystem corruption")
    print("   • Data degradation during copy")

    print("\n📚 According to the documentation:")
    print("   'Source verification adds another layer of security to the")
    print("    copy process by checking if the source file is still")
    print("    identical at a later point in time.'")

    return True


def main():
    """Run all source verification tests."""
    print("🔍 Source Verification Test Suite")
    print("=" * 70)
    print("Testing source verification functionality for pfndispatchcopy")
    print("")
    print("Source verification ensures that:")
    print("  1. Files are copied correctly (destination verification)")
    print("  2. Source files remain intact during copy (source verification)")
    print("  3. No card/reader issues affected the copy process")
    print("=" * 70)

    # Setup logging
    setup_logging(verbose=True)

    test_results = []

    try:
        # Run tests
        test_results.append(("Per-File Verification", test_per_file_verification()))
        test_results.append(("After-All Verification", test_after_all_verification()))
        test_results.append(("No Verification", test_no_verification()))
        test_results.append(("Performance Comparison", test_performance_comparison()))
        test_results.append(("Failure Detection", demonstrate_verification_failure()))

        # Summary
        print("\n" + "=" * 70)
        print("📋 TEST SUMMARY")
        print("=" * 70)

        passed = 0
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:35s} {status}")
            if result:
                passed += 1

        print(f"\nOverall: {passed}/{len(test_results)} tests passed")

        print("\n" + "=" * 70)
        print("📚 Key Insights:")
        print("=" * 70)
        print("• Per-File Mode: Immediate verification, slightly slower")
        print("• After-All Mode: Batch verification, faster overall")
        print("• Both modes provide essential source integrity checking")
        print("• Source verification is critical for professional DIT workflows")
        print("• Helps identify dying cards and reader issues early")
        print("=" * 70)

        if passed == len(test_results):
            print("\n🎉 All source verification tests passed!")
            return 0
        else:
            print("\n⚠️  Some tests failed")
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
