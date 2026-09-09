# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the actual callback with a delayed exporter, including MPI isolation."""

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Timer

import pytest

from srtctl.core.nsys_export import prepare_sqlite_export, wait_for_sqlite_exports


@pytest.fixture
def exporter(tmp_path: Path) -> Path:
    executable = tmp_path / "nsys"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sqlite3, sys, time\n"
        "assert sys.argv[1] == 'export'\n"
        "output = pathlib.Path(next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--output=')))\n"
        "with sqlite3.connect(output) as db:\n"
        "    db.execute('CREATE TABLE kernels (name TEXT)')\n"
        "    db.commit()\n"
        "    output.with_suffix('.started').touch()\n"
        "    while not pathlib.Path(os.environ['EXPORT_RELEASE']).exists(): time.sleep(0.01)\n"
        "    db.execute(\"INSERT INTO kernels VALUES ('last kernel')\")\n"
        "sys.exit(int(os.environ.get('EXPORT_EXIT', '0')))\n"
    )
    executable.chmod(0o700)
    return executable


def prepare(tmp_path: Path, exporter: Path, ranks: int = 1) -> tuple[list[str], list[Path]]:
    return prepare_sqlite_export(
        [str(exporter), "profile", "--export=sqlite", "-o", str(tmp_path / "profiles" / "worker_rank%q{SLURM_PROCID}")],
        tmp_path,
        ranks=ranks,
        container_log_dir=tmp_path,
    )


def callback(command: list[str], release: Path, *, rank: int = 0, exit_code: int = 0) -> subprocess.Popen:
    report = command[command.index("-o") + 1].replace("%q{SLURM_PROCID}", str(rank)) + ".1.nsys-rep"
    Path(report).touch()
    hook = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--after-report-ready="))
    return subprocess.Popen(
        [hook],
        env={
            **os.environ,
            "NSYS_REPORT_PATH": report,
            "EXPORT_RELEASE": str(release),
            "EXPORT_EXIT": str(exit_code),
        },
    )


def test_valid_partial_sqlite_does_not_release_cleanup(tmp_path: Path, exporter: Path) -> None:
    command, statuses = prepare(tmp_path, exporter)
    release = tmp_path / "release"
    process = callback(command, release)
    try:
        deadline = time.monotonic() + 5
        while not list(statuses[0].parent.glob("*.started")):
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        partial = next(statuses[0].parent.glob("*.exporting.sqlite"))
        with sqlite3.connect(partial) as db:
            assert db.execute("PRAGMA quick_check").fetchone() == ("ok",)
            assert db.execute("SELECT COUNT(*) FROM kernels").fetchone() == (0,)
        with pytest.raises(RuntimeError, match="did not finish"):
            wait_for_sqlite_exports(statuses, timeout_seconds=0.05, poll_seconds=0.01)
        assert not statuses[0].exists()
        release.touch()
        assert process.wait(timeout=5) == 0
        wait_for_sqlite_exports(statuses, timeout_seconds=0)
        with sqlite3.connect(next(statuses[0].parent.glob("*.1.sqlite"))) as db:
            assert db.execute("SELECT name FROM kernels").fetchone() == ("last kernel",)
    finally:
        release.touch()
        process.wait(timeout=5)


def test_concurrent_runs_and_missing_mpi_rank_cannot_false_pass(tmp_path: Path, exporter: Path) -> None:
    first, first_statuses = prepare(tmp_path, exporter, ranks=2)
    second, second_statuses = prepare(tmp_path, exporter, ranks=2)
    assert set(first_statuses).isdisjoint(second_statuses)
    assert first[first.index("-o") + 1] != second[second.index("-o") + 1]
    release = tmp_path / "release"
    release.touch()
    for rank in range(2):
        assert callback(first, release, rank=rank).wait(timeout=5) == 0
    wait_for_sqlite_exports(first_statuses, timeout_seconds=0)
    with pytest.raises(RuntimeError, match="did not finish"):
        wait_for_sqlite_exports(second_statuses, timeout_seconds=0)
    assert callback(second, release, rank=0).wait(timeout=5) == 0
    with pytest.raises(RuntimeError, match="rank-1.status"):
        wait_for_sqlite_exports(second_statuses, timeout_seconds=0)
    assert callback(second, release, rank=1).wait(timeout=5) == 0
    wait_for_sqlite_exports(second_statuses, timeout_seconds=0)


def test_failed_export_is_reported_without_publishing_sqlite(tmp_path: Path, exporter: Path) -> None:
    command, statuses = prepare(tmp_path, exporter)
    release = tmp_path / "release"
    release.touch()
    assert callback(command, release, exit_code=7).wait(timeout=5) == 7
    with pytest.raises(RuntimeError, match="exit 7"):
        wait_for_sqlite_exports(statuses, timeout_seconds=0)
    assert not list(statuses[0].parent.glob("*.1.sqlite"))


def test_only_requested_sqlite_export_is_managed(tmp_path: Path, exporter: Path) -> None:
    original = [str(exporter), "profile", "--export=text", "-o", "/logs/worker"]
    assert prepare_sqlite_export(original, tmp_path) == (original, [])
    command, _ = prepare(tmp_path, exporter)
    assert "--export=sqlite" not in command
    assert len([arg for arg in command if arg.startswith("--after-report-ready=")]) == 1


@pytest.mark.parametrize("exports", [["--export=text", "--export=sqlite"], ["--export", "sqlite", "--export", "text"]])
def test_other_export_formats_are_preserved(tmp_path: Path, exports: list[str]) -> None:
    command, statuses = prepare_sqlite_export(["nsys", "profile", *exports, "-o", "/logs/worker"], tmp_path)
    assert statuses
    assert "--export=text" in command or command[command.index("--export") + 1] == "text"


def test_job_cancellation_interrupts_export_wait(tmp_path: Path) -> None:
    cancellation = Event()
    timer = Timer(0.05, cancellation.set)
    timer.start()
    try:
        wait_for_sqlite_exports([tmp_path / "pending.status"], cancel_event=cancellation, timeout_seconds=2)
    finally:
        timer.join()
    assert cancellation.is_set()


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
@pytest.mark.parametrize("outcome", ["success", "export_failure", "benchmark_failure", "eval_only", "cancelled"])
def test_sweep_preserves_export_before_worker_cleanup(
    tmp_path: Path, exporter: Path, monkeypatch: pytest.MonkeyPatch, backend: str, outcome: str
) -> None:
    """Drive real worker command construction and sweep cleanup, replacing only infrastructure."""
    import yaml

    from srtctl.cli.do_sweep import ProcessRegistry, SweepOrchestrator
    from srtctl.cli.mixins import worker_stage
    from srtctl.mock import MockOptions, run_mock_sweep

    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "export-lifecycle",
                "model": {"path": str(tmp_path), "container": "nvcr.io/fake:latest", "precision": "fp8"},
                "resources": {"gpu_type": "h100", "gpus_per_node": 2, "agg_nodes": 1, "agg_workers": 1},
                "backend": {"type": backend},
                "frontend": {
                    "type": "vllm" if backend == "vllm" else "trtllm_serve",
                    "enable_multiple_frontends": False,
                },
                "benchmark": {"type": "custom", "command": "true"},
                "profiling": {
                    "type": "nsys",
                    "aggregated": {"start_step": 1, "stop_step": 3},
                    "extra_nsys_args": ["--export=sqlite"],
                },
                "observability": {"tachometer": {"enabled": False}},
            }
        )
    )
    commands = []
    children = []
    release = tmp_path / "release"

    def local_paths(command, log_dir, **kwargs):
        # Stand in for the container's /logs mount; keep real hook generation.
        command = [str(exporter), *[arg.replace("/logs/", f"{log_dir}/") for arg in command[1:]]]
        prepared, statuses = prepare_sqlite_export(command, log_dir, container_log_dir=log_dir, **kwargs)
        commands.append((prepared, statuses))
        return prepared, statuses

    def benchmark(self, registry, stop_event, reporter):
        assert commands and self.nsys_export_statuses  # Real worker launch registered its producers.
        if outcome == "benchmark_failure":
            return 1
        if outcome == "cancelled":
            Timer(0.05, stop_event.set).start()
            return 0
        for command, statuses in commands:
            for rank in range(len(statuses)):
                children.append(
                    callback(command, release, rank=rank, exit_code=7 if outcome == "export_failure" else 0)
                )
        Timer(0.1, release.touch).start()
        return 0

    original_cleanup = ProcessRegistry.cleanup

    def cleanup(self):
        if outcome == "success":
            statuses = [status for _, entries in commands for status in entries]
            assert statuses and all(status.is_file() for status in statuses)
            assert all(status.read_text().strip() == "0" for status in statuses)
        original_cleanup(self)

    monkeypatch.setattr(worker_stage, "prepare_sqlite_export", local_paths)
    monkeypatch.setattr(SweepOrchestrator, "run_benchmark", benchmark)
    monkeypatch.setattr(ProcessRegistry, "cleanup", cleanup)
    monkeypatch.setattr(SweepOrchestrator, "run_postprocess", lambda *args, **kwargs: None)
    monkeypatch.setattr(SweepOrchestrator, "_run_post_eval", lambda *args: 0)
    monkeypatch.setenv("EVAL_ONLY", "true" if outcome == "eval_only" else "false")
    monkeypatch.setenv("RUN_EVAL", "false")
    try:
        result = run_mock_sweep(
            config_path=config, output_dir=tmp_path / "output", job_id="42049", options=MockOptions(child_duration_s=60)
        )
        assert commands
        assert result == (1 if outcome in {"export_failure", "benchmark_failure", "cancelled"} else 0)
    finally:
        release.touch()
        for child in children:
            child.wait(timeout=5)


@pytest.mark.parametrize("range_end", ["none", "repeat", "repeat:2:async"])
def test_multiple_or_unfinished_ranges_cannot_acknowledge_completion(tmp_path: Path, range_end: str) -> None:
    with pytest.raises(ValueError, match="single completed capture range"):
        prepare_sqlite_export(
            ["nsys", "profile", "--export=sqlite", "--capture-range-end", range_end, "-o", "/logs/worker"], tmp_path
        )
