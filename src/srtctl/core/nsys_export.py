# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finalize single-range Nsight SQLite exports before serving-worker cleanup."""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from srtctl.core.slurm import start_srun_process

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NsysReport:
    """One possible report name and its corresponding SQLite destination."""

    host_path: Path
    container_path: Path
    host_sqlite: Path
    container_sqlite: Path


@dataclass(frozen=True)
class NsysExport:
    """Everything needed to export one profiler rank in its allocation."""

    rank: int
    node: str
    het_group: int | None
    executable: str
    reports: tuple[NsysReport, ...]
    log_path: Path


def _capture_range_end(command: Sequence[str]) -> str:
    value = "stop-shutdown"
    arguments = iter(command)
    for argument in arguments:
        if argument == "--capture-range-end":
            value = next(arguments, "")
        elif argument.startswith("--capture-range-end="):
            value = argument.split("=", 1)[1]
    return value


def _remove_sqlite_export(command: Sequence[str]) -> tuple[list[str], bool]:
    arguments = iter(command)
    filtered: list[str] = []
    requested = False
    for argument in arguments:
        if argument == "--export":
            export = next(arguments, None)
            if export is None:
                raise ValueError("--export requires a format")
            if export == "sqlite":
                requested = True
            else:
                filtered.extend([argument, export])
        elif argument.startswith("--export="):
            if argument == "--export=sqlite":
                requested = True
            else:
                filtered.append(argument)
        else:
            filtered.append(argument)
    return filtered, requested


def prepare_sqlite_export(
    command: list[str],
    log_dir: Path,
    *,
    rank_nodes: Sequence[str],
    rank_het_groups: Sequence[int | None] | None = None,
    container_log_dir: Path = Path("/logs"),
) -> tuple[list[str], list[NsysExport]]:
    """Remove built-in export and describe supervised post-benchmark exports.

    Nsight's ``repeat:1:async`` writes a complete range report while leaving the
    profiling session attached to a long-lived serving process. Built-in export
    and ``--after-report-ready`` are session-finalization hooks, so waiting for
    either before stopping the worker deadlocks. Instead, the orchestrator runs
    a foreground exporter in a separate Slurm step after the benchmark.

    A UUID directory makes every report rank- and invocation-specific. Both
    ordinary and ``.1`` report spellings are admitted because an executable
    wrapper may translate ``stop`` to ``repeat:1:async`` after this function has
    prepared the command; the fresh directory guarantees that only this
    invocation can match.
    """
    filtered, sqlite_requested = _remove_sqlite_export(command)
    if not sqlite_requested:
        return command, []
    if any(argument.startswith("--after-report-ready") for argument in command):
        raise ValueError("Managed SQLite export is incompatible with a custom report-ready callback")
    if _capture_range_end(command) not in {"stop", "stop-shutdown", "repeat:1:async"}:
        raise ValueError("Managed SQLite export requires a single completed capture range per rank")
    if not rank_nodes:
        raise ValueError("Managed SQLite export requires at least one profiler rank")
    groups = tuple(rank_het_groups or (None,) * len(rank_nodes))
    if len(groups) != len(rank_nodes):
        raise ValueError("rank_het_groups must have one entry per profiler rank")

    output_index = filtered.index("-o") + 1
    original_output = Path(filtered[output_index])
    if len(rank_nodes) > 1 and "%q{SLURM_PROCID}" not in str(original_output):
        raise ValueError("MPI SQLite exports require distinct SLURM_PROCID output paths for all ranks")
    relative_dir = original_output.parent.relative_to(container_log_dir) / f"capture-{uuid.uuid4().hex}"
    host_dir = log_dir / relative_dir
    host_dir.mkdir(parents=True)
    container_dir = container_log_dir / relative_dir
    output = container_dir / original_output.name
    filtered[output_index] = str(output)

    exports: list[NsysExport] = []
    for rank, (node, het_group) in enumerate(zip(rank_nodes, groups, strict=True)):
        container_prefix = Path(str(output).replace("%q{SLURM_PROCID}", str(rank)))
        host_prefix = host_dir / container_prefix.name
        reports = tuple(
            NsysReport(
                host_path=Path(f"{host_prefix}{suffix}.nsys-rep"),
                container_path=Path(f"{container_prefix}{suffix}.nsys-rep"),
                host_sqlite=Path(f"{host_prefix}{suffix}.sqlite"),
                container_sqlite=Path(f"{container_prefix}{suffix}.sqlite"),
            )
            for suffix in (".1", "")
        )
        exports.append(
            NsysExport(
                rank=rank,
                node=node,
                het_group=het_group,
                executable=command[0],
                reports=reports,
                log_path=host_dir / f"export-rank-{rank}.log",
            )
        )
    return filtered, exports


def _wait_for_report(spec: NsysExport, cancellation: Event, poll_seconds: float) -> NsysReport:
    next_log = 0.0
    while not cancellation.is_set():
        present = [report for report in spec.reports if report.host_path.is_file()]
        if len(present) > 1:
            raise RuntimeError(f"More than one Nsight report exists for rank {spec.rank}")
        if present:
            return present[0]
        now = time.monotonic()
        if now >= next_log:
            logger.info(
                "Waiting for rank %d Nsight range report: %s",
                spec.rank,
                ", ".join(str(report.host_path) for report in spec.reports),
            )
            next_log = now + 60
        cancellation.wait(poll_seconds)
    raise InterruptedError("Nsight export interrupted by job cancellation")


def _wait_process(process, cancellation: Event, poll_seconds: float) -> int:
    while (result := process.poll()) is None:
        if cancellation.wait(poll_seconds):
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise InterruptedError("Nsight export interrupted by job cancellation")
    return result


def _validate_sqlite(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Nsight exporter produced no SQLite data: {path}")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
        result = database.execute("PRAGMA quick_check").fetchone()
        table_count = database.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'").fetchone()
    if result != ("ok",) or table_count is None or table_count[0] == 0:
        raise RuntimeError(f"Nsight exporter produced an invalid SQLite database: {path}")


def _export_one(
    spec: NsysExport,
    *,
    container_image: str,
    container_mounts: dict[Path, Path],
    srun_options: dict[str, str],
    cancellation: Event,
    poll_seconds: float,
) -> Path:
    report = _wait_for_report(spec, cancellation, poll_seconds)
    attempt = 0
    while not cancellation.is_set():
        attempt += 1
        nonce = uuid.uuid4().hex
        temporary_host = report.host_sqlite.with_name(f"{report.host_sqlite.stem}.exporting-{nonce}.sqlite")
        temporary_container = report.container_sqlite.with_name(
            f"{report.container_sqlite.stem}.exporting-{nonce}.sqlite"
        )
        process = start_srun_process(
            command=[
                spec.executable,
                "export",
                "--type=sqlite",
                "--force-overwrite=true",
                f"--output={temporary_container}",
                str(report.container_path),
            ],
            nodelist=[spec.node],
            output=str(spec.log_path),
            container_image=container_image,
            container_mounts=container_mounts,
            srun_options=srun_options,
            het_group=spec.het_group,
        )
        result = _wait_process(process, cancellation, poll_seconds)
        if result == 0:
            _validate_sqlite(temporary_host)
            os.replace(temporary_host, report.host_sqlite)
            logger.info("Nsight SQLite export complete for rank %d: %s", spec.rank, report.host_sqlite)
            return report.host_sqlite
        temporary_host.unlink(missing_ok=True)
        logger.warning(
            "Nsight SQLite export attempt %d failed for rank %d (exit %d); retrying",
            attempt,
            spec.rank,
            result,
        )
        cancellation.wait(poll_seconds)
    raise InterruptedError("Nsight export interrupted by job cancellation")


def finalize_sqlite_exports(
    exports: Sequence[NsysExport],
    *,
    container_image: str,
    container_mounts: dict[Path, Path],
    srun_options: dict[str, str] | None = None,
    cancel_event: Event | None = None,
    poll_seconds: float = 0.5,
) -> list[Path]:
    """Export all ranks concurrently and return only after durable SQLite data.

    Production deliberately has no independent wall-clock deadline. The Slurm
    allocation and explicit cancellation own the lifetime.
    """
    if not exports:
        return []
    cancellation = cancel_event if cancel_event is not None else Event()
    options = dict(srun_options or {})
    with ThreadPoolExecutor(max_workers=len(exports), thread_name_prefix="nsys-export") as pool:
        futures = [
            pool.submit(
                _export_one,
                spec,
                container_image=container_image,
                container_mounts=container_mounts,
                srun_options=options,
                cancellation=cancellation,
                poll_seconds=poll_seconds,
            )
            for spec in exports
        ]
        try:
            return [future.result() for future in as_completed(futures)]
        except BaseException:
            cancellation.set()
            for future in futures:
                future.cancel()
            raise
