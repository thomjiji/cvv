"""CLI orchestration and presentation layer."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from .engine import CopyEngine
from .hash_file import HashFileWriter
from .models import (
    HASH_FILE_EXTENSIONS,
    CopyEvent,
    CopyResult,
    DestinationResult,
    EventType,
    VerificationMode,
)


class CLIProcessor:
    """CLI orchestration and presentation layer."""

    def __init__(
        self,
        source: Path,
        destinations: list[Path],
        verification_mode: VerificationMode,
        hash_algorithm: str,
        hash_file_formats: list[str] | None = None,
        hash_file_dest: str = "dest",
        verify_deferred: bool = False,
    ):
        self.source = source
        self.destinations = destinations
        self.verification_mode = verification_mode
        self.hash_algorithm = hash_algorithm
        self.hash_file_formats = hash_file_formats or []
        self.hash_file_dest = hash_file_dest
        self.verify_deferred = verify_deferred
        self.console = Console()

    def run(self) -> bool:
        """Execute copy jobs for all source files."""
        source_files = self._discover_files()
        if not source_files:
            self.console.print("No files to copy")
            return True

        total_bytes = sum(f.stat().st_size for f in source_files)
        copy_mode = VerificationMode.TRANSFER if self.verify_deferred else self.verification_mode
        results: list[CopyResult] = []
        engines: list[CopyEngine] = []
        bytes_completed = 0

        # Phase 1: Copy all files
        with self._make_progress() as progress:
            overall_task = progress.add_task(
                f"[bold]Copying [0/{len(source_files)}]",
                total=total_bytes,
            )

            for i, source_file in enumerate(source_files, 1):
                if CopyEngine._shared_abort_event.is_set():
                    break

                file_size = source_file.stat().st_size
                dest_paths = self._calculate_destinations(source_file)
                destinations_to_copy = self._check_duplicates_and_cleanup(
                    source_file, dest_paths
                )

                if not destinations_to_copy:
                    self.console.print(
                        f"[dim]- {source_file.name} (already exists)[/dim]"
                    )
                    result = CopyResult(
                        source_path=source_file,
                        source_size=file_size,
                        verification_mode=self.verification_mode,
                    )
                    for dest in dest_paths:
                        result.destinations.append(
                            DestinationResult(path=dest, success=True)
                        )
                    results.append(result)
                    engines.append(CopyEngine(
                        source=source_file,
                        destinations=dest_paths,
                        verification_mode=self.verification_mode,
                        hash_algorithm=self.hash_algorithm,
                    ))
                    bytes_completed += file_size
                    progress.update(
                        overall_task,
                        completed=bytes_completed,
                        description=f"[bold]Copying [{i}/{len(source_files)}]",
                    )
                    continue

                engine = CopyEngine(
                    source=source_file,
                    destinations=destinations_to_copy,
                    verification_mode=copy_mode,
                    hash_algorithm=self.hash_algorithm,
                )

                file_task = progress.add_task(
                    f"Copying {source_file.name}", total=file_size
                )

                result = None
                for event in engine.copy():
                    if isinstance(event, CopyResult):
                        result = event
                        break
                    elif event.type == EventType.COPY_PROGRESS:
                        progress.update(
                            file_task, completed=event.bytes_processed
                        )
                        progress.update(
                            overall_task,
                            completed=bytes_completed + event.bytes_processed,
                        )
                    elif event.type == EventType.COPY_COMPLETE:
                        bytes_completed += file_size
                        progress.update(
                            overall_task, completed=bytes_completed
                        )
                    elif event.type == EventType.VERIFY_START:
                        progress.reset(
                            file_task,
                            total=event.total_bytes,
                            description=f"Verifying {source_file.name}",
                        )
                    elif event.type == EventType.VERIFY_PROGRESS:
                        progress.update(
                            file_task, completed=event.bytes_processed
                        )

                progress.remove_task(file_task)
                progress.update(
                    overall_task,
                    description=f"[bold]Copying [{i}/{len(source_files)}]",
                )

                results.append(result)
                engines.append(engine)
                self._print_file_result(result)

                if not result.success:
                    break

        # Phase 2: Deferred verification
        if self.verify_deferred and self.verification_mode != VerificationMode.TRANSFER:
            all_copied = all(r.success for r in results)
            if all_copied and results and not CopyEngine._shared_abort_event.is_set():
                self.console.print(
                    f"\n[bold]All {len(results)} file(s) copied.[/bold] "
                    f"Verification mode: [cyan]{self.verification_mode.value}[/cyan]"
                )
                answer = input("Start verification? [Y/n] ").strip().lower()
                if answer in ("", "y", "yes"):
                    self._run_deferred_verify(results, engines)
                else:
                    self.console.print("[dim]Verification skipped.[/dim]")

        # Generate hash files
        all_success = all(r.success for r in results)
        if self.hash_file_formats and all_success and results:
            self._generate_hash_files(results)

        self._show_final_summary(results)
        return all_success

    def _run_deferred_verify(
        self, results: list[CopyResult], engines: list[CopyEngine]
    ) -> None:
        """Run verification on all copied files."""
        total_bytes = 0
        verify_pairs: list[tuple[CopyEngine, CopyResult]] = []
        for engine, result in zip(engines, results):
            if not result.success:
                continue
            engine.verification_mode = self.verification_mode
            result.verification_mode = self.verification_mode
            if self.verification_mode == VerificationMode.SOURCE:
                total_bytes += result.source_size
            elif self.verification_mode == VerificationMode.FULL:
                total_bytes += result.source_size * (1 + len(result.destinations))
            verify_pairs.append((engine, result))

        with self._make_progress() as progress:
            overall_task = progress.add_task(
                f"[bold]Verifying [0/{len(verify_pairs)}]", total=total_bytes
            )
            bytes_completed = 0

            for i, (engine, result) in enumerate(verify_pairs, 1):
                if CopyEngine._shared_abort_event.is_set():
                    break

                file_task = progress.add_task(
                    f"Verifying {result.source_path.name}",
                    total=result.source_size,
                )

                for event in engine.verify(result):
                    if isinstance(event, CopyResult):
                        break
                    elif event.type == EventType.VERIFY_START:
                        progress.reset(file_task, total=event.total_bytes)
                    elif event.type == EventType.VERIFY_PROGRESS:
                        progress.update(
                            file_task, completed=event.bytes_processed
                        )
                        progress.update(
                            overall_task,
                            completed=bytes_completed + event.bytes_processed,
                        )

                if self.verification_mode == VerificationMode.SOURCE:
                    bytes_completed += result.source_size
                elif self.verification_mode == VerificationMode.FULL:
                    bytes_completed += result.source_size * (1 + len(result.destinations))

                progress.remove_task(file_task)
                progress.update(
                    overall_task,
                    completed=bytes_completed,
                    description=f"[bold]Verifying [{i}/{len(verify_pairs)}]",
                )
                self._print_file_result(result)

    def _make_progress(self) -> Progress:
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            "|",
            DownloadColumn(),
            "|",
            TransferSpeedColumn(),
            "|",
            TimeRemainingColumn(),
            console=self.console,
        )

    def _discover_files(self) -> list[Path]:
        if self.source.is_file():
            return [self.source]
        elif self.source.is_dir():
            return sorted([f for f in self.source.rglob("*") if f.is_file()])
        else:
            raise FileNotFoundError(f"Source not found: {self.source}")

    def _calculate_destinations(self, source_file: Path) -> list[Path]:
        if self.source.is_dir():
            relative_path = source_file.relative_to(self.source)
            return [dest_root / relative_path for dest_root in self.destinations]

        dest_paths = []
        for dest in self.destinations:
            if dest.is_dir():
                dest_paths.append(dest / source_file.name)
            else:
                dest_paths.append(dest)
        return dest_paths

    def _check_duplicates_and_cleanup(
        self, source: Path, destinations: list[Path]
    ) -> list[Path]:
        """Skip already-completed files and clean up incomplete .tmp files."""
        source_size = source.stat().st_size
        destinations_to_copy = []

        for dest in destinations:
            if dest.exists():
                dest_size = dest.stat().st_size
                if dest_size == source_size:
                    self.console.print(
                        f"  [green]OK[/green] {dest.name} already exists [dim](skipping)[/dim]"
                    )
                    continue
                else:
                    self.console.print(
                        f"  [yellow]![/yellow] {dest.name} wrong size "
                        f"({dest_size} vs {source_size}), will overwrite"
                    )
                    dest.unlink()

            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            if tmp_path.exists():
                tmp_size = tmp_path.stat().st_size
                self.console.print(
                    f"  [yellow]![/yellow] Found incomplete {tmp_path.name} "
                    f"({tmp_size / (1024 * 1024):.1f}/{source_size / (1024 * 1024):.1f} MB), restarting"
                )
                tmp_path.unlink()

            destinations_to_copy.append(dest)

        return destinations_to_copy

    def _print_file_result(self, result: CopyResult) -> None:
        name = result.source_path.name
        size = self._format_size(result.source_size)
        speed = f"{result.speed_mb_sec:.1f} MB/s"

        if not result.success:
            errors = [dr.error for dr in result.destinations if dr.error]
            self.console.print(
                f"[red]FAIL[/red] {name}  [red]{errors[0] if errors else 'Failed'}[/red]"
            )
            return

        parts = [f"[green]OK[/green] {name}", f"[dim]{size}[/dim]", f"[dim]{speed}[/dim]"]

        if result.source_hash_inflight:
            parts.append(f"[cyan]{result.source_hash_inflight}[/cyan]")

        if result.source_hash_post:
            ok = result.source_hash_post == result.source_hash_inflight
            parts.append(f"[{'green' if ok else 'red'}]src {'OK' if ok else 'FAIL'}[/]")

        if result.verification_mode == VerificationMode.FULL:
            for dr in result.destinations:
                if dr.hash_post:
                    ok = dr.hash_post == result.source_hash_inflight
                    parts.append(
                        f"[{'green' if ok else 'red'}]dst {'OK' if ok else 'FAIL'}[/]"
                    )

        self.console.print("  ".join(parts))

    def _show_final_summary(self, results: list[CopyResult]) -> None:
        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success
        total_size = sum(r.source_size for r in results)
        total_duration = sum(r.duration for r in results)

        self.console.print()
        if failed == 0:
            avg_speed = (
                f"{(total_size / (1024 ** 2)) / total_duration:.1f} MB/s"
                if total_duration > 0
                else ""
            )
            self.console.print(
                f"[bold green]All {total} file(s) completed[/bold green]  "
                f"[dim]{self._format_size(total_size)}  {avg_speed}[/dim]"
            )
        else:
            self.console.print(
                f"[bold red]{failed} of {total} file(s) failed[/bold red]"
            )

    def _generate_hash_files(self, results: list[CopyResult]) -> None:
        """Write hash files at source and/or destination roots."""
        source_name = self.source.stem if self.source.is_file() else self.source.name
        ext = HASH_FILE_EXTENSIONS.get(self.hash_algorithm, ".xxh")

        dirs_to_write: list[tuple[Path, list[tuple[Path, str, int]]]] = []

        if self.hash_file_dest in ("dest", "both"):
            for dest_root in self.destinations:
                hash_dir = dest_root if self.source.is_dir() or dest_root.is_dir() else dest_root.parent
                entries = self._collect_entries(results, hash_dir)
                if entries:
                    dirs_to_write.append((hash_dir, entries))

        if self.hash_file_dest in ("source", "both"):
            source_dir = self.source if self.source.is_dir() else self.source.parent
            entries = self._collect_source_entries(results, source_dir)
            if entries:
                dirs_to_write.append((source_dir, entries))

        for hash_dir, entries in dirs_to_write:
            if "xxh" in self.hash_file_formats:
                path = HashFileWriter.write_xxh(
                    entries, hash_dir / (source_name + ext), self.hash_algorithm
                )
                self.console.print(f"  [dim]Hash file:[/dim] {path}")
            if "mhl" in self.hash_file_formats:
                path = HashFileWriter.write_mhl(
                    entries, hash_dir, self.hash_algorithm, source_name
                )
                self.console.print(f"  [dim]MHL file:[/dim] {path}")

    def _collect_entries(
        self, results: list[CopyResult], hash_dir: Path
    ) -> list[tuple[Path, str, int]]:
        """Collect (relative_path, hash, size) for a destination hash_dir."""
        entries: list[tuple[Path, str, int]] = []
        for result in results:
            hash_hex = result.source_hash_inflight
            if not hash_hex:
                continue
            for dr in result.destinations:
                try:
                    rel = dr.path.relative_to(hash_dir)
                except ValueError:
                    continue
                entries.append((rel, dr.hash_post or hash_hex, result.source_size))
                break
        return entries

    def _collect_source_entries(
        self, results: list[CopyResult], source_dir: Path
    ) -> list[tuple[Path, str, int]]:
        """Collect (relative_path, hash, size) relative to source directory."""
        entries: list[tuple[Path, str, int]] = []
        for result in results:
            hash_hex = result.source_hash_inflight
            if not hash_hex:
                continue
            try:
                rel = result.source_path.relative_to(source_dir)
            except ValueError:
                continue
            entries.append((rel, hash_hex, result.source_size))
        return entries

    @staticmethod
    def _format_size(n: int) -> str:
        if n >= 1024**3:
            return f"{n / 1024 ** 3:.1f} GB"
        if n >= 1024**2:
            return f"{n / 1024 ** 2:.1f} MB"
        if n >= 1024:
            return f"{n / 1024:.1f} KB"
        return f"{n} B"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Professional file copying tool with integrity verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cvv -m transfer /source /dest1 /dest2         # Fast copy with size check only
  cvv -m full /source_dir /dest_dir             # Copy entire directory with full integrity verification
  cvv source.mp4 /dest1 /dest2                  # Copy single file is also ok, default use full verification mode
        """,
    )

    parser.add_argument(
        "source",
        type=Path,
        help="Source file or directory to copy",
    )

    parser.add_argument(
        "destinations",
        type=Path,
        nargs="+",
        help="One or more destination paths",
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="full",
        choices=["transfer", "source", "full"],
        help="Verification mode (default: full)",
    )

    parser.add_argument(
        "--hash-algorithm",
        type=str,
        default="xxh3_64",
        choices=["xxh3_64", "xxh64", "md5", "sha1", "sha256"],
        help="Hash algorithm for verification (default: xxh3_64)",
    )

    parser.add_argument(
        "--hash-file",
        type=str,
        nargs="+",
        choices=["xxh", "mhl"],
        help="Generate hash file(s) (xxh: TeraCopy format, mhl: ASC MHL)",
    )

    parser.add_argument(
        "--hash-file-dest",
        type=str,
        default="dest",
        choices=["source", "dest", "both"],
        help="Where to write hash files: source, dest, or both (default: dest)",
    )

    parser.add_argument(
        "--verify-after",
        action="store_true",
        help="Defer verification until all files are copied, then prompt",
    )

    args = parser.parse_args()

    try:
        processor = CLIProcessor(
            source=args.source,
            destinations=args.destinations,
            verification_mode=VerificationMode(args.mode),
            hash_algorithm=args.hash_algorithm,
            hash_file_formats=args.hash_file or [],
            hash_file_dest=args.hash_file_dest,
            verify_deferred=args.verify_after,
        )

        success = processor.run()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
