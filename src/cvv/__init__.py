"""cvv: Professional file copying tools for DIT workflows."""

from .cli import CLIProcessor, main
from .engine import CopyEngine, HashCalculator
from .hash_file import HashFileWriter
from .models import (
    CopyEvent,
    CopyResult,
    DestinationResult,
    EventType,
    VerificationMode,
)

__version__ = "0.0.1"
__all__ = [
    "CLIProcessor",
    "CopyEngine",
    "CopyEvent",
    "CopyResult",
    "DestinationResult",
    "EventType",
    "HashCalculator",
    "HashFileWriter",
    "VerificationMode",
    "main",
]
