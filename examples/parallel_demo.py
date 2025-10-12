#!/usr/bin/env python3
"""
Parallel vs Sequential Copy Performance Demo.

This script demonstrates and compares the performance between the original
sequential copy implementation and the new truly parallel copy implementation.
"""

import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pfndispatchcopy import (
    copy_with_multiple_destinations_parallel,
    setup_logging,
)


def create_test_file(file_path: Path, size_mb: int = 10) -> None:
    """
    Create a test file with specified size.

    Parameters
    ----------
    file_path : Path
        Path where to create the test file
    size_mb : int
        Size of the file in megabytes
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_size = 1024 * 1024  # 1MB chunks
    with open(file_path, "wb") as f:
        for i in range(size_mb):
            # Create varied data to make the test more realistic
            data = bytes([(i * 37 + j) % 256 for j in range(chunk_size)])
            f.write(data)


def sequential_copy_simulation(
    source: Path,
    destinations: list[Path],
    buffer_size: int,
) -> Dict[str, Any]:
    """
    Simulate the old sequential copy approach for comparison.

    Parameters
    ----------
    source : Path
        Source file path
    destinations : list[Path]
        List of destination paths
    buffer_size : int
        Buffer size for copying

    Returns
    -------
    Dict[str, Any]
        Copy results with timing information
    """
    start_time = time.time()
    results = {"success": True, "destinations": {}}

    try:
        # Create temporary files
        temp_files = {}
        file_handles = {}

        for dest in destinations:
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp_file = dest.parent / f".{dest.name}.tmp"
            temp_files[dest] = temp_file
            file_handles[dest] = open(temp_file, "wb")

        # Sequential write approach (original implementation)
        with open(source, "rb") as src:
            while chunk := src.read(buffer_size):
                # Write to each destination sequentially
                for dest, handle in file_handles.items():
                    handle.write(chunk)

        # Close handles
        for handle in file_handles.values():
            handle.close()

        # Move files to final destinations
        for dest in destinations:
            temp_files[dest].rename(dest)
            results["destinations"][str(dest)] = "success"

        duration = time.time() - start_time
        results["duration"] = duration

        file_size = source.stat().st_size
        speed_mb_sec = (file_size / (1024 * 1024)) / duration if duration > 0 else 0
        results["speed_mb_sec"] = speed_mb_sec

        return results

    except Exception as e:
        # Clean up on error
        for handle in file_handles.values():
            if not handle.closed:
                handle.close()
        for temp_file in temp_files.values():
            if temp_file.exists():
                temp_file.unlink()
        raise IOError(f"Sequential copy failed: {e}")


def run_performance_comparison(file_size_mb: int, num_destinations: int) -> None:
    """
    Run performance comparison between sequential and parallel approaches.

    Parameters
    ----------
    file_size_mb : int
        Size of test file in megabytes
    num_destinations : int
        Number of destination copies to create
    """
    print(f"\n{'=' * 80}")
    print(
        f"🏁 PERFORMANCE COMPARISON: {file_size_mb}MB file → {num_destinations} destinations"
    )
    print(f"{'=' * 80}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test file
        source_file = temp_path / "source" / "test_file.bin"
        print(f"📁 Creating {file_size_mb}MB test file...")
        create_test_file(source_file, file_size_mb)

        # Create destination paths
        seq_destinations = []
        par_destinations = []

        for i in range(num_destinations):
            seq_dest = temp_path / f"sequential_dest_{i}" / "test_file.bin"
            par_dest = temp_path / f"parallel_dest_{i}" / "test_file.bin"
            seq_destinations.append(seq_dest)
            par_destinations.append(par_dest)

        buffer_size = 8 * 1024 * 1024  # 8MB buffer

        # Test 1: Sequential approach
        print(f"\n🐌 Testing Sequential Copy...")
        start_time = time.time()
        try:
            seq_results = sequential_copy_simulation(
                source_file, seq_destinations, buffer_size
            )
            seq_duration = seq_results["duration"]
            seq_speed = seq_results["speed_mb_sec"]
            print(f"   ✅ Sequential: {seq_duration:.3f}s, {seq_speed:.1f} MB/sec")
        except Exception as e:
            print(f"   ❌ Sequential failed: {e}")
            return

        # Test 2: Parallel approach
        print(f"\n🚀 Testing Parallel Copy...")
        try:
            par_results = copy_with_multiple_destinations_parallel(
                source=source_file,
                destinations=par_destinations,
                buffer_size=buffer_size,
                hash_algorithm="md5",
            )
            par_duration = par_results["duration"]
            par_speed = par_results["speed_mb_sec"]
            print(f"   ✅ Parallel: {par_duration:.3f}s, {par_speed:.1f} MB/sec")
        except Exception as e:
            print(f"   ❌ Parallel failed: {e}")
            return

        # Performance comparison
        print(f"\n📊 RESULTS COMPARISON:")
        print(f"   Sequential: {seq_duration:.3f}s ({seq_speed:.1f} MB/sec)")
        print(f"   Parallel:   {par_duration:.3f}s ({par_speed:.1f} MB/sec)")

        if par_duration < seq_duration:
            speedup = seq_duration / par_duration
            improvement = ((seq_duration - par_duration) / seq_duration) * 100
            print(
                f"   🎯 Parallel is {speedup:.2f}x faster ({improvement:.1f}% improvement)"
            )
        else:
            slowdown = par_duration / seq_duration
            regression = ((par_duration - seq_duration) / seq_duration) * 100
            print(
                f"   ⚠️  Parallel is {slowdown:.2f}x slower ({regression:.1f}% regression)"
            )

        # Verify file integrity
        print(f"\n🔍 Verifying file integrity...")
        source_size = source_file.stat().st_size

        all_destinations = seq_destinations + par_destinations
        for dest in all_destinations:
            if dest.exists() and dest.stat().st_size == source_size:
                print(f"   ✅ {dest.name}: OK")
            else:
                print(f"   ❌ {dest.name}: FAILED")


def test_thread_scaling() -> None:
    """Test how performance scales with number of destinations."""
    print(f"\n{'=' * 80}")
    print(f"📈 THREAD SCALING TEST")
    print(f"{'=' * 80}")

    file_size_mb = 20
    destination_counts = [1, 2, 4, 6, 8]

    for num_dest in destination_counts:
        print(f"\n🎯 Testing with {num_dest} destination(s)...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test file
            source_file = temp_path / "source.bin"
            create_test_file(source_file, file_size_mb)

            # Create destinations
            destinations = []
            for i in range(num_dest):
                dest = temp_path / f"dest_{i}" / "file.bin"
                destinations.append(dest)

            try:
                # Test parallel copy
                start_time = time.time()
                results = copy_with_multiple_destinations_parallel(
                    source=source_file,
                    destinations=destinations,
                    buffer_size=8 * 1024 * 1024,
                )

                duration = results["duration"]
                speed = results["speed_mb_sec"]

                print(
                    f"   {num_dest} destinations: {duration:.3f}s, {speed:.1f} MB/sec"
                )

            except Exception as e:
                print(f"   ❌ Failed with {num_dest} destinations: {e}")


def test_buffer_size_impact() -> None:
    """Test how buffer size affects parallel performance."""
    print(f"\n{'=' * 80}")
    print(f"🔧 BUFFER SIZE IMPACT TEST")
    print(f"{'=' * 80}")

    file_size_mb = 50
    num_destinations = 4
    buffer_sizes = [
        (64 * 1024, "64KB"),
        (512 * 1024, "512KB"),
        (2 * 1024 * 1024, "2MB"),
        (8 * 1024 * 1024, "8MB"),
        (16 * 1024 * 1024, "16MB"),
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test file once
        source_file = temp_path / "source.bin"
        print(f"📁 Creating {file_size_mb}MB test file...")
        create_test_file(source_file, file_size_mb)

        print(
            f"🧪 Testing different buffer sizes with {num_destinations} destinations..."
        )

        for buffer_size, label in buffer_sizes:
            # Create fresh destinations for each test
            destinations = []
            for i in range(num_destinations):
                dest = temp_path / f"buffer_{label}_dest_{i}" / "file.bin"
                destinations.append(dest)

            try:
                results = copy_with_multiple_destinations_parallel(
                    source=source_file,
                    destinations=destinations,
                    buffer_size=buffer_size,
                )

                duration = results["duration"]
                speed = results["speed_mb_sec"]

                print(f"   {label:>8} buffer: {duration:.3f}s, {speed:.1f} MB/sec")

            except Exception as e:
                print(f"   ❌ {label} buffer failed: {e}")


def main() -> None:
    """Run all parallel copy demonstrations."""
    print("🚀 Parallel Copy Performance Analysis")
    print("=" * 80)

    # Setup logging to suppress INFO messages for cleaner output
    setup_logging(verbose=False)
    logging.getLogger().setLevel(logging.WARNING)

    try:
        # Test 1: Direct performance comparison
        print("\n🎯 Test 1: Sequential vs Parallel Comparison")
        run_performance_comparison(file_size_mb=30, num_destinations=3)
        run_performance_comparison(file_size_mb=50, num_destinations=5)

        # Test 2: Thread scaling
        print("\n🎯 Test 2: Thread Scaling Analysis")
        test_thread_scaling()

        # Test 3: Buffer size impact
        print("\n🎯 Test 3: Buffer Size Impact")
        test_buffer_size_impact()

        print(f"\n{'=' * 80}")
        print("🎉 All parallel copy tests completed!")
        print("📊 Key insights:")
        print(
            "   • Parallel copy should show better performance with multiple destinations"
        )
        print("   • Performance may vary based on storage speed and CPU cores")
        print(
            "   • Optimal buffer size depends on file size and storage characteristics"
        )
        print("   • Thread overhead may impact small files")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
