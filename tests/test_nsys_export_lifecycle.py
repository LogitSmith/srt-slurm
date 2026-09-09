# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise supervised Nsight export, including the real sweep handoff."""

import sqlite3
import subprocess
import sys
from pathlib import Path
from threading import Event, Timer

import pytest

from srtctl.core.nsys_export import NsysExport, finalize_sqlite_exports, prepare_sqlite_export


@pytest.fixture
def exporter(tmp_path: Path) -> Path:
    executable = tmp_path / "nsys"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sqlite3, sys, time\n"
        "assert sys.argv[1] == 'export'\n"
        "output = pathlib.Path(next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--output=')))\n"
        "if pathlib.Path(str(output) + '.fail').exists(): sys.exit(7)\n"
        "with sqlite3.connect(output) as db:\n"
        "    db.execute('CREATE TABLE kernels (name TEXT)')\n"
        "    db.commit()\n"
        "    time.sleep(0.05)\n"
        "    db.execute(\"INSERT INTO kernels VALUES ('last kernel')\")\n"
    )
    executable.chmod(0o700)
    return executable


def prepare(tmp_path: Path, exporter: Path, ranks: int = 1) -> tuple[list[str], list[NsysExport]]:
    return prepare_sqlite_export(
        [
            str(exporter),
            "profile",
            "--capture-range-end=repeat:1:async",
            "--export=sqlite",
            "-o",
            str(tmp_path / "profiles" / "worker_rank%q{SLURM_PROCID}"),
        ],
        tmp_path,
        rank_nodes=[f"node-{rank}" for rank in range(ranks)],
        container_log_dir=tmp_path,
    )


def emit_reports(exports: list[NsysExport]) -> None:
    for spec in exports:
        spec.reports[0].host_path.touch()


def local_srun(command: list[str], **_kwargs) -> subprocess.Popen:
    return subprocess.Popen(command)


def test_exporter_exit_is_the_publication_boundary(
    tmp_path: Path, exporter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _command, exports = prepare(tmp_path, exporter, ranks=2)
    emit_reports(exports)
    monkeypatch.setattr("srtctl.core.nsys_export.start_srun_process", local_srun)

    outputs = finalize_sqlite_exports(
        exports,
        container_image="unused",
        container_mounts={},
        poll_seconds=0.01,
    )

    assert set(outputs) == {spec.reports[0].host_sqlite for spec in exports}
    for output in outputs:
        with sqlite3.connect(output) as database:
            assert database.execute("PRAGMA quick_check").fetchone() == ("ok",)
            assert database.execute("SELECT name FROM kernels").fetchone() == ("last kernel",)
    assert not list(tmp_path.rglob("*.exporting-*"))


def test_concurrent_invocations_and_missing_rank_cannot_false_pass(
    tmp_path: Path, exporter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _first_command, first = prepare(tmp_path, exporter, ranks=2)
    _second_command, second = prepare(tmp_path, exporter, ranks=2)
    assert {report.host_path for spec in first for report in spec.reports}.isdisjoint(
        {report.host_path for spec in second for report in spec.reports}
    )
    emit_reports(first)
    second[0].reports[0].host_path.touch()
    cancellation = Event()
    Timer(0.1, cancellation.set).start()
    monkeypatch.setattr("srtctl.core.nsys_export.start_srun_process", local_srun)

    finalize_sqlite_exports(first, container_image="unused", container_mounts={}, poll_seconds=0.01)
    with pytest.raises(InterruptedError, match="cancellation"):
        finalize_sqlite_exports(
            second,
            container_image="unused",
            container_mounts={},
            cancel_event=cancellation,
            poll_seconds=0.01,
        )


def test_failed_export_is_not_published(tmp_path: Path, exporter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _command, exports = prepare(tmp_path, exporter)
    emit_reports(exports)

    def failing_srun(command: list[str], **_kwargs) -> subprocess.Popen:
        output = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output=")))
        Path(f"{output}.fail").touch()
        return subprocess.Popen(command)

    monkeypatch.setattr("srtctl.core.nsys_export.start_srun_process", failing_srun)
    cancellation = Event()
    Timer(0.1, cancellation.set).start()
    with pytest.raises(InterruptedError, match="cancellation"):
        finalize_sqlite_exports(
            exports,
            container_image="unused",
            container_mounts={},
            cancel_event=cancellation,
            poll_seconds=0.01,
        )
    assert not exports[0].reports[0].host_sqlite.exists()


def test_incomplete_report_export_is_retried(tmp_path: Path, exporter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _command, exports = prepare(tmp_path, exporter)
    emit_reports(exports)
    attempts = 0

    def transient_srun(command: list[str], **_kwargs) -> subprocess.Popen:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])
        return subprocess.Popen(command)

    monkeypatch.setattr("srtctl.core.nsys_export.start_srun_process", transient_srun)
    outputs = finalize_sqlite_exports(
        exports,
        container_image="unused",
        container_mounts={},
        poll_seconds=0.01,
    )
    assert attempts == 2
    assert outputs == [exports[0].reports[0].host_sqlite]


def test_only_requested_sqlite_export_is_managed(tmp_path: Path) -> None:
    original = ["nsys", "profile", "--export=text", "-o", "/logs/worker"]
    assert prepare_sqlite_export(original, tmp_path, rank_nodes=["node-0"]) == (original, [])
    command, exports = prepare_sqlite_export(
        ["nsys", "profile", "--export=sqlite", "-o", "/logs/worker"],
        tmp_path,
        rank_nodes=["node-0"],
    )
    assert exports
    assert "--export=sqlite" not in command
    assert not any(argument.startswith("--after-report-ready") for argument in command)


@pytest.mark.parametrize("exports", [["--export=text", "--export=sqlite"], ["--export", "sqlite", "--export", "text"]])
def test_other_export_formats_are_preserved(tmp_path: Path, exports: list[str]) -> None:
    command, specs = prepare_sqlite_export(
        ["nsys", "profile", *exports, "-o", "/logs/worker"], tmp_path, rank_nodes=["node-0"]
    )
    assert specs
    assert "--export=text" in command or command[command.index("--export") + 1] == "text"


def test_ambiguous_report_names_fail_closed(tmp_path: Path, exporter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _command, exports = prepare(tmp_path, exporter)
    for report in exports[0].reports:
        report.host_path.touch()
    monkeypatch.setattr("srtctl.core.nsys_export.start_srun_process", local_srun)
    with pytest.raises(RuntimeError, match="More than one"):
        finalize_sqlite_exports(exports, container_image="unused", container_mounts={}, poll_seconds=0.01)


@pytest.mark.parametrize("range_end", ["none", "repeat", "repeat:2:async"])
def test_multiple_or_unfinished_ranges_are_rejected(tmp_path: Path, range_end: str) -> None:
    with pytest.raises(ValueError, match="single completed capture range"):
        prepare_sqlite_export(
            [
                "nsys",
                "profile",
                "--export=sqlite",
                "--capture-range-end",
                range_end,
                "-o",
                "/logs/worker",
            ],
            tmp_path,
            rank_nodes=["node-0"],
        )


def test_custom_report_callback_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="custom report-ready callback"):
        prepare_sqlite_export(
            [
                "nsys",
                "profile",
                "--export=sqlite",
                "--after-report-ready=/tmp/hook",
                "-o",
                "/logs/worker",
            ],
            tmp_path,
            rank_nodes=["node-0"],
        )


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
@pytest.mark.parametrize("outcome", ["success", "export_failure", "benchmark_failure", "eval_only", "cancelled"])
def test_sweep_preserves_export_before_worker_cleanup(
    tmp_path: Path,
    exporter: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    outcome: str,
) -> None:
    """Drive real worker construction and sweep cleanup, replacing only infrastructure."""
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
    prepared: list[NsysExport] = []

    def local_paths(command, log_dir, **kwargs):
        command = [str(exporter), *[argument.replace("/logs/", f"{log_dir}/") for argument in command[1:]]]
        result, specs = prepare_sqlite_export(command, log_dir, container_log_dir=log_dir, **kwargs)
        prepared.extend(specs)
        return result, specs

    def benchmark(self, registry, stop_event, reporter):
        assert prepared and self.nsys_exports
        if outcome == "benchmark_failure":
            return 1
        if outcome == "cancelled":
            stop_event.set()
            return 0
        if outcome == "export_failure":
            Timer(0.1, stop_event.set).start()
        emit_reports(prepared)
        return 0

    def export_srun(command: list[str], **_kwargs) -> subprocess.Popen:
        if outcome == "export_failure":
            output = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output=")))
            Path(f"{output}.fail").touch()
        return subprocess.Popen(command)

    original_cleanup = ProcessRegistry.cleanup

    def cleanup(self):
        if outcome == "success":
            assert prepared and all(spec.reports[0].host_sqlite.is_file() for spec in prepared)
        original_cleanup(self)

    monkeypatch.setattr(worker_stage, "prepare_sqlite_export", local_paths)
    monkeypatch.setattr("srtctl.core.nsys_export.start_srun_process", export_srun)
    monkeypatch.setattr(SweepOrchestrator, "run_benchmark", benchmark)
    monkeypatch.setattr(ProcessRegistry, "cleanup", cleanup)
    monkeypatch.setattr(SweepOrchestrator, "run_postprocess", lambda *args, **kwargs: None)
    monkeypatch.setattr(SweepOrchestrator, "_run_post_eval", lambda *args: 0)
    monkeypatch.setenv("EVAL_ONLY", "true" if outcome == "eval_only" else "false")
    monkeypatch.setenv("RUN_EVAL", "false")
    monkeypatch.setattr(
        "srtctl.cli.do_sweep.finalize_sqlite_exports",
        lambda specs, **kwargs: finalize_sqlite_exports(
            specs,
            container_image=kwargs["container_image"],
            container_mounts=kwargs["container_mounts"],
            cancel_event=kwargs["cancel_event"],
            poll_seconds=0.01,
        ),
    )
    result = run_mock_sweep(
        config_path=config,
        output_dir=tmp_path / "output",
        job_id="42049",
        options=MockOptions(child_duration_s=60),
    )
    assert prepared
    assert result == (1 if outcome in {"export_failure", "benchmark_failure", "cancelled"} else 0)
