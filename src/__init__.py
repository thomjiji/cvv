"""
cvv: Professional file copying tools for DIT workflows.

This package implements professional-grade file copying utilities inspired by
Offload Manager, providing reliable file transfer with integrity verification,
progress monitoring, and multi-destination support.
"""

from .cvv import (
    CLIProcessor,
    CopyEngine,
    CopyEvent,
    CopyResult,
    DestinationResult,
    EventType,
    HashCalculator,
    VerificationMode,
    main,
)

__version__ = "1.0.0"
__author__ = "cvv project"
__description__ = "Professional file copying tools for DIT workflows"

__all__ = [
    "CLIProcessor",
    "CopyEngine",
    "CopyEvent",
    "CopyResult",
    "DestinationResult",
    "EventType",
    "HashCalculator",
    "VerificationMode",
    "main",
]
