"""
offload-ai: Professional file copying tools for DIT workflows.

This package implements professional-grade file copying utilities inspired by
Offload Manager, providing reliable file transfer with integrity verification,
progress monitoring, and multi-destination support.
"""

from .pfndispatchcopy import (
    HashCalculator,
    ProgressTracker,
    copy_with_multiple_destinations,
    main as pfndispatchcopy_main,
    parse_arguments,
    setup_logging,
)

__version__ = "1.0.0"
__author__ = "offload-ai project"
__description__ = "Professional file copying tools for DIT workflows"

__all__ = [
    "HashCalculator",
    "ProgressTracker",
    "copy_with_multiple_destinations",
    "pfndispatchcopy_main",
    "parse_arguments",
    "setup_logging",
]
