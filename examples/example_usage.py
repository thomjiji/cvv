#!/usr/bin/env python3
"""
Example usage script for pfndispatchcopy tool.

This script demonstrates how to use the pfndispatchcopy tool with various options
and provides examples similar to professional DIT workflows.
"""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def setup_logging() -> None:
    """Configure logging for the example script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def create_sample_file(file_path: Path, size_mb: int = 10) -> None:
    """
    Create a sample file for testing.

    Parameters
    ----------
    file_path : Path
        Path where to create the sample file
    size_mb : int
        Size of the file in megabytes
    """
    logging.info(f"Creating sample file: {file_path} ({size_mb}MB)")

    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file with random data
    with open(file_path, "wb") as f:
        chunk_size = 1024 * 1024  # 1MB chunks
        for _ in range(size_mb):
            # Write 1MB of zeros (for simplicity)
            f.write(b"\x00" * chunk_size)

    logging.info(f"Created sample file: {file_path.stat().st_size} bytes")


def run_pfndispatchcopy(args: List[str]) -> int:
    """
    Run pfndispatchcopy with given arguments.

    Parameters
    ----------
    args : List[str]
        Command line arguments for pfndispatchcopy

    Returns
    -------
    int
        Exit code from pfndispatchcopy
    """
    cmd = [
        "python3",
        str(Path(__file__).parent.parent / "src" / "pfndispatchcopy.py"),
    ] + args
    logging.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        logging.info(f"Exit code: {result.returncode}")
        return result.returncode

    except Exception as e:
        logging.error(f"Failed to run pfndispatchcopy: {e}")
        return 1


def example_basic_copy() -> None:
    """Example 1: Basic file copy to multiple destinations."""
    logging.info("=" * 60)
    logging.info("EXAMPLE 1: Basic copy to multiple destinations")
    logging.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample source file
        source_file = temp_path / "source" / "sample_video.mov"
        create_sample_file(source_file, size_mb=5)

        # Define destinations
        dest1 = temp_path / "backup1" / "sample_video.mov"
        dest2 = temp_path / "backup2" / "sample_video.mov"

        # Run pfndispatchcopy
        args = [
            "-v",  # Verbose output
            str(source_file),
            str(dest1),
            str(dest2),
        ]

        exit_code = run_pfndispatchcopy(args)

        if exit_code == 0:
            logging.info("✓ Basic copy completed successfully")
            logging.info(f"Source size: {source_file.stat().st_size}")
            logging.info(f"Dest1 size: {dest1.stat().st_size}")
            logging.info(f"Dest2 size: {dest2.stat().st_size}")
        else:
            logging.error("✗ Basic copy failed")


def example_with_hash_verification() -> None:
    """Example 2: Copy with hash verification."""
    logging.info("=" * 60)
    logging.info("EXAMPLE 2: Copy with XXH64BE hash verification")
    logging.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample source file
        source_file = temp_path / "source" / "A013C0002_251011_VW5X15.MOV"
        create_sample_file(source_file, size_mb=3)

        # Define destinations (simulating the log example structure)
        dest1 = (
            temp_path / "Pegasus32_R8" / "25082-project" / "A013C0002_251011_VW5X15.MOV"
        )
        dest2 = (
            temp_path
            / "GSJ-promise_r6_80TB"
            / "25082-project"
            / "A013C0002_251011_VW5X15.MOV"
        )

        file_size = source_file.stat().st_size

        # Run pfndispatchcopy with hash verification
        args = [
            "-v",  # Verbose output
            "-t",
            "xxh64be",  # Hash algorithm
            "-k",  # Keep source file
            "-f_size",
            str(file_size),  # Expected file size
            "-b",
            "8388608",  # 8MB buffer size
            str(source_file),
            str(dest1),
            str(dest2),
        ]

        exit_code = run_pfndispatchcopy(args)

        if exit_code == 0:
            logging.info("✓ Copy with hash verification completed successfully")
        else:
            logging.error("✗ Copy with hash verification failed")


def example_with_noflush() -> None:
    """Example 3: Copy with selective flush control."""
    logging.info("=" * 60)
    logging.info("EXAMPLE 3: Copy with selective flush control")
    logging.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample source file
        source_file = temp_path / "DJI" / "A013_VW5X15" / "A013C0002_251011_VW5X15.MOV"
        create_sample_file(source_file, size_mb=2)

        # Define destinations
        dest1 = temp_path / "fast_ssd" / "project" / "A013C0002_251011_VW5X15.MOV"
        dest2 = temp_path / "slow_hdd" / "project" / "A013C0002_251011_VW5X15.MOV"

        # Run pfndispatchcopy with noflush for fast SSD
        args = [
            "-v",  # Verbose output
            "-noflush_dest",
            str(dest1),  # Skip flush for fast SSD
            "-t",
            "md5",  # Use MD5 hash (no xxhash dependency needed)
            "-b",
            "4194304",  # 4MB buffer size
            str(source_file),
            str(dest1),
            str(dest2),
        ]

        exit_code = run_pfndispatchcopy(args)

        if exit_code == 0:
            logging.info("✓ Copy with selective flush control completed successfully")
        else:
            logging.error("✗ Copy with selective flush control failed")


def example_large_file_simulation() -> None:
    """Example 4: Simulate copying a large file (similar to the log example)."""
    logging.info("=" * 60)
    logging.info("EXAMPLE 4: Large file copy simulation")
    logging.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a larger sample file (50MB to simulate behavior without taking too long)
        source_file = temp_path / "DJI" / "A013_VW5X15" / "A013C0002_251011_VW5X15.MOV"
        create_sample_file(source_file, size_mb=50)

        # Define destinations similar to the log
        dest1 = (
            temp_path
            / "Pegasus32_R8"
            / "25082-祁又一「两条咸鱼」电影"
            / "素材"
            / "251011_Day10"
            / "Cam#A"
            / "A013"
            / "A013_VW5X15"
            / "A013C0002_251011_VW5X15.MOV"
        )

        dest2 = (
            temp_path
            / "GSJ-promise_r6_80TB"
            / "25082-祁又一「两条咸鱼」电影"
            / "素材"
            / "251011_Day10"
            / "Cam#A"
            / "A013"
            / "A013_VW5X15"
            / "A013C0002_251011_VW5X15.MOV"
        )

        file_size = source_file.stat().st_size

        # Run pfndispatchcopy exactly like in the log
        args = [
            "-v",  # Verbose output
            "-noflush_dest",
            str(dest1),  # No flush for first destination
            "-noflush_dest",
            str(dest2),  # No flush for second destination
            "-t",
            "sha256",  # Use SHA256 (no xxhash dependency)
            "-k",  # Keep source file
            "-f_size",
            str(file_size),  # Expected file size
            "-b",
            "8388608",  # 8MB buffer size (same as log)
            str(source_file),
            str(dest1),
            str(dest2),
        ]

        exit_code = run_pfndispatchcopy(args)

        if exit_code == 0:
            logging.info("✓ Large file copy simulation completed successfully")
        else:
            logging.error("✗ Large file copy simulation failed")


def example_error_handling() -> None:
    """Example 5: Demonstrate error handling."""
    logging.info("=" * 60)
    logging.info("EXAMPLE 5: Error handling demonstration")
    logging.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Try to copy non-existent file
        logging.info("Testing non-existent source file...")
        non_existent = temp_path / "does_not_exist.mov"
        dest = temp_path / "destination.mov"

        args = ["-v", str(non_existent), str(dest)]

        exit_code = run_pfndispatchcopy(args)

        if exit_code != 0:
            logging.info("✓ Correctly handled non-existent source file")
        else:
            logging.error("✗ Should have failed for non-existent source")

        # Test file size mismatch
        logging.info("Testing file size mismatch...")
        source_file = temp_path / "source.mov"
        create_sample_file(source_file, size_mb=1)

        args = [
            "-v",
            "-f_size",
            "999999999",  # Wrong expected size
            str(source_file),
            str(dest),
        ]

        exit_code = run_pfndispatchcopy(args)

        if exit_code != 0:
            logging.info("✓ Correctly handled file size mismatch")
        else:
            logging.error("✗ Should have failed for file size mismatch")


def main() -> None:
    """Run all example scenarios."""
    setup_logging()

    logging.info("Starting pfndispatchcopy examples")
    logging.info("Make sure pfndispatchcopy.py is in the current directory")

    try:
        # Check if pfndispatchcopy.py exists
        pfndispatchcopy_path = (
            Path(__file__).parent.parent / "src" / "pfndispatchcopy.py"
        )
        if not pfndispatchcopy_path.exists():
            logging.error(f"pfndispatchcopy.py not found at {pfndispatchcopy_path}!")
            return

        # Run examples
        example_basic_copy()
        example_with_hash_verification()
        example_with_noflush()
        example_large_file_simulation()
        example_error_handling()

        logging.info("All examples completed!")

    except KeyboardInterrupt:
        logging.info("Examples interrupted by user")
    except Exception as e:
        logging.error(f"Error running examples: {e}")


if __name__ == "__main__":
    main()
