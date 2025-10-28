#!/usr/bin/env python3
"""
Command-line entry point for cvv.

This script provides a convenient command-line interface to the cvv
tool without requiring users to specify the full path to the Python module.
"""

import sys
from pathlib import Path

# Add src directory to Python path (go up to project root first)
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Import and run the main function
from cvv import main

if __name__ == "__main__":
    sys.exit(main())
