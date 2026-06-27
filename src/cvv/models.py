"""Data models and constants for cvv."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

BUFFER_SIZE = 8 * 1024 * 1024  # 8MB
QUEUE_SIZE = 10  # Max chunks buffered per destination

HASH_FILE_EXTENSIONS = {
    "xxh3_64": ".xxh3",
    "xxh64": ".xxh",
    "md5": ".md5",
    "sha1": ".sha1",
    "sha256": ".sha256",
}


class VerificationMode(Enum):
    """Verification strategy for copy operations."""

    TRANSFER = "transfer"
    SOURCE = "source"
    FULL = "full"


class EventType(Enum):
    """Events emitted during copy operations."""

    COPY_START = "copy_start"
    COPY_PROGRESS = "copy_progress"
    COPY_COMPLETE = "copy_complete"
    VERIFY_START = "verify_start"
    VERIFY_PROGRESS = "verify_progress"
    VERIFY_COMPLETE = "verify_complete"


@dataclass
class CopyEvent:
    """Event emitted during copy/verification operations."""

    type: EventType
    bytes_processed: int = 0
    total_bytes: int = 0
    message: str = ""


@dataclass
class DestinationResult:
    """Result for a single destination."""

    path: Path
    success: bool
    bytes_written: int = 0
    hash_post: str | None = None
    error: str | None = None


@dataclass
class CopyResult:
    """Complete result of a multi-destination copy operation."""

    source_path: Path
    source_size: int
    destinations: list[DestinationResult] = field(default_factory=list)
    source_hash_inflight: str | None = None
    source_hash_post: str | None = None
    duration: float = 0.0
    verification_mode: VerificationMode = VerificationMode.TRANSFER

    @property
    def success(self) -> bool:
        return all(d.success for d in self.destinations)

    @property
    def speed_mb_sec(self) -> float:
        if self.duration > 0:
            return (self.source_size / (1024 * 1024)) / self.duration
        return 0.0
