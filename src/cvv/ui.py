"""
UI helpers for cvv.

This module keeps renderer and throughput-visualization logic separate from the
copy engine so the CLI layer can choose between text and TUI presentation.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_TUI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via fallback behavior
    Console = Group = Layout = Live = Panel = Table = Text = None
    RICH_TUI_AVAILABLE = False


def format_bytes(num_bytes: int) -> str:
    """Format bytes into a human-friendly string."""
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            precision = 0 if unit == "B" else 1
            return f"{value:.{precision}f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def format_rate(mb_per_sec: float) -> str:
    """Format MB/s for display."""
    return f"{mb_per_sec:,.1f} MB/s"


def truncate_path(path: Path | str, max_chars: int = 52) -> str:
    """Truncate long paths while keeping the filename visible."""
    text = str(path)
    if len(text) <= max_chars:
        return text
    keep = max_chars - 3
    head = keep // 2
    tail = keep - head
    return f"{text[:head]}...{text[-tail:]}"


@dataclass
class DestinationInspection:
    """Planning details for one destination path."""

    path: Path
    skip_copy: bool = False
    existing_size: int | None = None
    needs_delete_existing: bool = False
    tmp_path: Path | None = None
    tmp_size: int | None = None
    needs_delete_tmp: bool = False


@dataclass
class FileWorkPlan:
    """Execution plan for one source file."""

    source_path: Path
    source_size: int
    destination_inspections: list[DestinationInspection]
    copy_bytes_total: int
    verify_bytes_total: int
    preflight_messages: list[str] = field(default_factory=list)

    @property
    def destinations(self) -> list[Path]:
        return [inspection.path for inspection in self.destination_inspections]

    @property
    def destinations_to_copy(self) -> list[Path]:
        return [
            inspection.path
            for inspection in self.destination_inspections
            if not inspection.skip_copy
        ]

    @property
    def skipped_destinations(self) -> int:
        return sum(1 for inspection in self.destination_inspections if inspection.skip_copy)

    @property
    def total_work_bytes(self) -> int:
        return self.copy_bytes_total + self.verify_bytes_total


@dataclass
class SessionProgressState:
    """Mutable state consumed by renderers."""

    total_files: int = 0
    completed_files: int = 0
    current_file_index: int = 0
    current_file_name: str = ""
    current_source_path: str = ""
    current_phase: str = "idle"
    current_phase_bytes: int = 0
    current_phase_total: int = 0
    current_file_work_done: int = 0
    current_file_work_total: int = 0
    session_work_done: int = 0
    session_work_total: int = 0
    copy_rate_mb_sec: float = 0.0
    verify_rate_mb_sec: float = 0.0
    copy_history: list[float] = field(default_factory=list)
    verify_history: list[float] = field(default_factory=list)
    verification_mode: str = "full"
    hash_algorithm: str = "xxh64be"
    current_destination_count: int = 0
    last_message: str = ""
    fallback_message: str = ""
    started_at: float = field(default_factory=time.monotonic)

    @property
    def session_percent(self) -> float:
        if self.session_work_total <= 0:
            return 0.0
        return min(100.0, (self.session_work_done / self.session_work_total) * 100)

    @property
    def current_phase_percent(self) -> float:
        if self.current_phase_total <= 0:
            return 0.0
        return min(100.0, (self.current_phase_bytes / self.current_phase_total) * 100)

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)


class ThroughputTracker:
    """Track rolling copy and verify throughput histories."""

    def __init__(self, window_seconds: int = 60, clock=time.monotonic):
        self.window_seconds = window_seconds
        self._clock = clock
        self._copy_history: deque[float] = deque(maxlen=window_seconds)
        self._verify_history: deque[float] = deque(maxlen=window_seconds)
        self._current_rates = {"copy": 0.0, "verify": 0.0}
        self._last_samples: dict[str, tuple[int, float] | None] = {
            "copy": None,
            "verify": None,
        }
        self._active_phase: str | None = None
        self._bucket_second = int(self._clock())

    def set_phase(self, phase: str) -> None:
        now = self._clock()
        self._advance(now)
        self._active_phase = phase
        self._last_samples[phase] = None
        other = "verify" if phase == "copy" else "copy"
        self._current_rates[other] = 0.0
        self._last_samples[other] = None

    def finish_phase(self, phase: str) -> None:
        now = self._clock()
        self._advance(now)
        self._current_rates[phase] = 0.0
        self._last_samples[phase] = None
        if self._active_phase == phase:
            self._active_phase = None

    def record(self, phase: str, bytes_processed: int) -> None:
        now = self._clock()
        self._advance(now)

        last_sample = self._last_samples[phase]
        if last_sample is not None:
            last_bytes, last_time = last_sample
            delta_bytes = max(0, bytes_processed - last_bytes)
            delta_time = max(now - last_time, 1e-6)
            self._current_rates[phase] = delta_bytes / delta_time / (1024 * 1024)

        self._last_samples[phase] = (bytes_processed, now)
        self._active_phase = phase

    def snapshot(self) -> tuple[list[float], list[float], float, float]:
        now = self._clock()
        self._advance(now)
        copy_history = list(self._copy_history)
        verify_history = list(self._verify_history)

        while len(copy_history) < self.window_seconds:
            copy_history.insert(0, 0.0)
        while len(verify_history) < self.window_seconds:
            verify_history.insert(0, 0.0)

        return (
            copy_history,
            verify_history,
            self._current_rates["copy"],
            self._current_rates["verify"],
        )

    def _advance(self, now: float) -> None:
        current_second = int(now)
        while self._bucket_second < current_second:
            copy_rate = self._current_rates["copy"] if self._active_phase == "copy" else 0.0
            verify_rate = (
                self._current_rates["verify"] if self._active_phase == "verify" else 0.0
            )
            self._copy_history.append(copy_rate)
            self._verify_history.append(verify_rate)
            self._bucket_second += 1


def summary_lines(result, hash_algorithm: str) -> list[str]:
    """Build human-readable result summary lines."""
    lines: list[str] = []
    if result.success:
        lines.append(
            f"✓ Success ({result.speed_mb_sec:.2f} MB/s, "
            f"{result.source_size / (1024 * 1024):.2f} MB)"
        )

        if result.verification_mode.value == "transfer":
            if result.source_hash_inflight:
                lines.append(
                    f"  Source hash ({hash_algorithm}): {result.source_hash_inflight}"
                )
        elif result.verification_mode.value == "source":
            if result.source_hash_inflight:
                lines.append(f"  Source hash (in-flight):  {result.source_hash_inflight}")
            if result.source_hash_post:
                match_indicator = (
                    "✓"
                    if result.source_hash_post == result.source_hash_inflight
                    else "✗"
                )
                lines.append(
                    f"  Source hash (post-copy):  {result.source_hash_post} "
                    f"[{match_indicator}]"
                )
        elif result.verification_mode.value == "full":
            if result.source_hash_inflight:
                lines.append(f"  Source hash (in-flight):  {result.source_hash_inflight}")
            if result.source_hash_post:
                match_indicator = (
                    "✓"
                    if result.source_hash_post == result.source_hash_inflight
                    else "✗"
                )
                lines.append(
                    f"  Source hash (post-copy):  {result.source_hash_post} "
                    f"[{match_indicator}]"
                )
            for dest_result in result.destinations:
                if dest_result.hash_post:
                    match_indicator = (
                        "✓"
                        if dest_result.hash_post == result.source_hash_inflight
                        else "✗"
                    )
                    lines.append(
                        f"  {dest_result.path.name}: {dest_result.hash_post} "
                        f"[{match_indicator}]"
                    )
    else:
        lines.append("✗ Failed")
        for dest_result in result.destinations:
            if not dest_result.success:
                lines.append(f"  ✗ {dest_result.path.name}: {dest_result.error}")
    return lines


class TextRenderer:
    """Current line-oriented terminal renderer."""

    def __init__(self, hash_algorithm: str):
        self.hash_algorithm = hash_algorithm
        self._progress_line_open = False

    def start_session(self, state: SessionProgressState) -> None:
        if state.fallback_message:
            print(state.fallback_message)

    def start_file(self, state: SessionProgressState) -> None:
        print(f"\nFile {state.current_file_index}/{state.total_files}: {state.current_file_name}")

    def note(self, message: str) -> None:
        print(message)

    def update(self, state: SessionProgressState, event_type: str) -> None:
        if event_type == "copy_progress":
            self._progress_line_open = True
            sys_line = (
                f"\rCopying: {state.current_phase_percent:.1f}% "
                f"({state.current_phase_bytes / (1024 * 1024):.1f}/"
                f"{state.current_phase_total / (1024 * 1024):.1f} MB)"
            )
            print(sys_line.ljust(80), end="", flush=True)
        elif event_type == "verify_start":
            if self._progress_line_open:
                print()
                self._progress_line_open = False
        elif event_type == "verify_progress":
            self._progress_line_open = True
            sys_line = (
                f"\rVerifying: {state.current_phase_percent:.1f}% "
                f"({state.current_phase_bytes / (1024 * 1024):.1f}/"
                f"{state.current_phase_total / (1024 * 1024):.1f} MB)"
            )
            print(sys_line.ljust(80), end="", flush=True)

    def finish_file(self, result) -> None:
        if self._progress_line_open:
            print()
            self._progress_line_open = False
        for line in summary_lines(result, self.hash_algorithm):
            print(line)

    def finish_session(self, results: list) -> None:
        print("\n" + "=" * 60)
        total = len(results)
        success = sum(1 for result in results if result.success)
        failed = total - success
        if failed == 0:
            print(f"All {total} operation(s) completed successfully")
        else:
            print(f"{failed} of {total} operation(s) failed")


class RichTuiRenderer:
    """Rich-powered real-time throughput dashboard."""

    def __init__(self, hash_algorithm: str):
        if not RICH_TUI_AVAILABLE:
            raise RuntimeError("Rich TUI renderer is unavailable")
        self.hash_algorithm = hash_algorithm
        self.console = Console()
        self.live: Live | None = None

    def start_session(self, state: SessionProgressState) -> None:
        self.live = Live(
            self._build_layout(state),
            console=self.console,
            refresh_per_second=4,
            transient=True,
            screen=True,
        )
        self.live.start()
        self._refresh(state)

    def start_file(self, state: SessionProgressState) -> None:
        self._refresh(state)

    def note(self, message: str) -> None:
        # The message itself is stored in the state by the processor.
        return

    def update(self, state: SessionProgressState, event_type: str) -> None:
        self._refresh(state)

    def finish_file(self, result) -> None:
        return

    def finish_session(self, results: list) -> None:
        if self.live:
            self.live.stop()
        for result in results:
            for line in summary_lines(result, self.hash_algorithm):
                self.console.print(line)
        self.console.print("\n" + "=" * 60)
        total = len(results)
        success = sum(1 for result in results if result.success)
        failed = total - success
        if failed == 0:
            self.console.print(f"All {total} operation(s) completed successfully")
        else:
            self.console.print(f"{failed} of {total} operation(s) failed")

    def _refresh(self, state: SessionProgressState) -> None:
        if self.live:
            self.live.update(self._build_layout(state), refresh=True)

    def _build_layout(self, state: SessionProgressState):
        layout = Layout()
        layout.split_column(
            Layout(self._build_summary_panel(state), size=8),
            Layout(self._build_chart_panel(state), ratio=1),
            Layout(self._build_footer_panel(state), size=9),
        )
        return layout

    def _build_summary_panel(self, state: SessionProgressState):
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_row(
            f"[bold]File[/bold] {state.current_file_index}/{state.total_files}",
            f"[bold]Mode[/bold] {state.verification_mode}",
            f"[bold]Hash[/bold] {state.hash_algorithm}",
            f"[bold]Elapsed[/bold] {state.elapsed_seconds:,.1f}s",
        )
        table.add_row(
            f"[bold]Current[/bold] {truncate_path(state.current_file_name, 38)}",
            f"[bold]Destinations[/bold] {state.current_destination_count}",
            f"[bold]Phase[/bold] {state.current_phase}",
            f"[bold]Session[/bold] {state.session_percent:.1f}%",
        )
        table.add_row(
            f"[bold]Source[/bold] {truncate_path(state.current_source_path, 38)}",
            f"[bold]Done[/bold] {format_bytes(state.current_phase_bytes)}",
            f"[bold]Target[/bold] {format_bytes(state.current_phase_total)}",
            f"[bold]Work[/bold] {format_bytes(state.session_work_done)} / "
            f"{format_bytes(state.session_work_total)}",
        )
        return Panel(table, title="cvv Throughput Dashboard", border_style="bright_blue")

    def _build_chart_panel(self, state: SessionProgressState):
        chart = self._build_rate_chart(state.copy_history, state.verify_history)
        legend = Text()
        legend.append("copy", style="bold #a4c8ff")
        legend.append("  ")
        legend.append("verify", style="bold #8be28b")
        legend.append(
            f"  window=60s  copy={format_rate(state.copy_rate_mb_sec)}"
            f"  verify={format_rate(state.verify_rate_mb_sec)}",
            style="grey70",
        )
        return Panel(
            Group(chart, legend),
            title="Realtime Throughput (MB/s)",
            border_style="grey42",
        )

    def _build_footer_panel(self, state: SessionProgressState):
        table = Table.grid(expand=True)
        table.add_column(justify="left", ratio=2)
        table.add_column(justify="left")
        table.add_row(
            "[bold]Phase progress[/bold]",
            f"{state.current_phase_percent:.1f}% "
            f"({format_bytes(state.current_phase_bytes)} / "
            f"{format_bytes(state.current_phase_total)})",
        )
        table.add_row(
            "[bold]Current file work[/bold]",
            f"{format_bytes(state.current_file_work_done)} / "
            f"{format_bytes(state.current_file_work_total)}",
        )
        table.add_row("[bold]Status[/bold]", state.last_message or "Waiting for progress events")
        return Panel(table, title="Status", border_style="grey42")

    def _build_rate_chart(
        self,
        copy_history: list[float],
        verify_history: list[float],
        width: int = 60,
        height: int = 10,
    ) -> Text:
        copy_values = (copy_history or [0.0])[-width:]
        verify_values = (verify_history or [0.0])[-width:]

        if len(copy_values) < width:
            copy_values = [0.0] * (width - len(copy_values)) + copy_values
        if len(verify_values) < width:
            verify_values = [0.0] * (width - len(verify_values)) + verify_values

        max_value = max(max(copy_values), max(verify_values), 1.0)
        rows: list[list[tuple[str, str]]] = [
            [(" ", "default") for _ in range(width)] for _ in range(height)
        ]

        self._draw_series(rows, copy_values, max_value, "#a4c8ff")
        self._draw_series(rows, verify_values, max_value, "#8be28b")

        chart_text = Text()
        for row_index, row in enumerate(rows):
            value = max_value * (height - 1 - row_index) / max(height - 1, 1)
            label = f"{value:>6.1f} "
            chart_text.append(label, style="grey62")
            chart_text.append("│", style="grey46")
            for cell_text, style in row:
                chart_text.append(cell_text, style=style)
            chart_text.append("\n")

        chart_text.append(" " * 7, style="grey62")
        chart_text.append("└" + "─" * width + "\n", style="grey46")
        chart_text.append(" " * 9 + "60s ago".ljust(width - 9), style="grey50")
        chart_text.append("now", style="grey50")
        return chart_text

    def _draw_series(
        self,
        rows: list[list[tuple[str, str]]],
        values: list[float],
        max_value: float,
        style: str,
    ) -> None:
        previous_y: int | None = None
        for x, value in enumerate(values):
            scaled = 0 if max_value <= 0 else value / max_value
            y = len(rows) - 1 - int(round(scaled * (len(rows) - 1)))
            if previous_y is None:
                self._merge_cell(rows, x, y, "•", style)
                previous_y = y
                continue

            for fill_x in range(min(x - 1, x), x + 1):
                self._merge_cell(rows, fill_x, previous_y, "─", style)
            if y != previous_y:
                for fill_y in range(min(y, previous_y), max(y, previous_y) + 1):
                    self._merge_cell(rows, x, fill_y, "│", style)
            self._merge_cell(rows, x, y, "•", style)
            previous_y = y

    def _merge_cell(
        self,
        rows: list[list[tuple[str, str]]],
        x: int,
        y: int,
        char: str,
        style: str,
    ) -> None:
        existing_char, existing_style = rows[y][x]
        if existing_char.strip() and existing_style != style:
            rows[y][x] = ("◆", "bold white")
            return
        rows[y][x] = (char, style)
