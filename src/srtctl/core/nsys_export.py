# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Producer acknowledgements for single-range Nsight SQLite exports."""

import logging
import shlex
import time
import uuid
from pathlib import Path
from threading import Event

logger = logging.getLogger(__name__)


def prepare_sqlite_export(
    command: list[str], log_dir: Path, *, ranks: int = 1, container_log_dir: Path = Path("/logs")
) -> tuple[list[str], list[Path]]:
    """Replace inline SQLite export with a report-ready callback before launch.

    Each invocation gets its own output directory and one acknowledgement per
    srun rank. Neither another job's processes nor a previous export can satisfy
    these acknowledgements. The callback runs inside the worker container.
    """
    arguments = iter(command)
    filtered: list[str] = []
    sqlite_requested = False
    for argument in arguments:
        if argument == "--export":
            export = next(arguments, None)
            if export is None:
                raise ValueError("--export requires a format")
            if export == "sqlite":
                sqlite_requested = True
            else:
                filtered.extend([argument, export])
        elif argument.startswith("--export="):
            if argument == "--export=sqlite":
                sqlite_requested = True
            else:
                filtered.append(argument)
        else:
            filtered.append(argument)
    if not sqlite_requested:
        return command, []
    if any(arg.startswith("--after-report-ready") for arg in command):
        raise ValueError("Managed SQLite export needs the Nsight report-ready callback; remove the custom callback")
    range_end = "stop-shutdown"
    arguments = iter(command)
    for argument in arguments:
        if argument == "--capture-range-end":
            range_end = next(arguments, "")
        elif argument.startswith("--capture-range-end="):
            range_end = argument.split("=", 1)[1]
    if range_end not in {"stop", "stop-shutdown", "repeat:1:async"}:
        raise ValueError("Managed SQLite export requires a single completed capture range per rank")

    output_index = filtered.index("-o") + 1
    original_output = Path(filtered[output_index])
    if ranks < 1 or (ranks > 1 and "%q{SLURM_PROCID}" not in str(original_output)):
        raise ValueError("MPI SQLite exports require distinct SLURM_PROCID output paths for all ranks")
    relative_dir = original_output.parent.relative_to(container_log_dir) / f"capture-{uuid.uuid4().hex}"
    directory = log_dir / relative_dir
    directory.mkdir(parents=True)
    container_dir = container_log_dir / relative_dir
    output = container_dir / original_output.name
    filtered[output_index] = str(output)

    # Match Nsight's exact report path rather than relying on inherited Slurm
    # environment or searching global profiler processes. Async single-range
    # capture appends '.1'; ordinary single-range capture does not.
    cases = []
    statuses = []
    for rank in range(ranks):
        prefix = str(output).replace("%q{SLURM_PROCID}", str(rank))
        status = container_dir / f"rank-{rank}.status"
        cases.append(
            f"  {shlex.quote(prefix + '.nsys-rep')}|{shlex.quote(prefix + '.1.nsys-rep')}) "
            f"status={shlex.quote(str(status))} ;;"
        )
        statuses.append(directory / status.name)
    script = directory / "export.sh"
    script.write_text(
        "#!/bin/sh\nset -eu\n"
        'case "${NSYS_REPORT_PATH:?missing Nsight report path}" in\n'
        + "\n".join(cases)
        + '\n  *) echo "Unexpected Nsight report: $NSYS_REPORT_PATH" >&2; exit 1 ;;\nesac\n'
        'output="${NSYS_REPORT_PATH%.nsys-rep}.sqlite"\n'
        'temporary="${output}.exporting.sqlite"\n'
        "rc=0\n"
        + shlex.quote(command[0])
        + ' export --type=sqlite --force-overwrite=true --output="$temporary" "$NSYS_REPORT_PATH" || rc=$?\n'
        'if [ "$rc" -eq 0 ]; then mv -- "$temporary" "$output" || rc=$?; fi\n'
        'printf "%s\\n" "$rc" > "${status}.tmp"\n'
        'mv -- "${status}.tmp" "$status"\n'
        'exit "$rc"\n'
    )
    script.chmod(0o700)
    filtered[output_index - 1 : output_index - 1] = [f"--after-report-ready={container_dir / script.name}"]
    return filtered, statuses


def wait_for_sqlite_exports(
    statuses: list[Path],
    *,
    cancel_event: Event | None = None,
    timeout_seconds: float | None = None,
    poll_seconds: float = 0.5,
) -> None:
    """Wait for producer exit, bounded by cancellation and the Slurm allocation.

    A valid export can take many minutes. An optional timeout is useful for
    tests, but elapsed time is neither completion nor failure in production.
    """
    pending = set(statuses)
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    cancellation = cancel_event if cancel_event is not None else Event()
    next_log = 0.0
    while pending:
        if cancellation.is_set():
            logger.info("Nsight export wait interrupted by job cancellation")
            return
        for status in tuple(pending):
            try:
                result = status.read_text().strip()
            except FileNotFoundError:
                continue
            if result != "0":
                raise RuntimeError(f"Nsight SQLite export failed (exit {result}): {status}")
            pending.remove(status)
        if not pending:
            logger.info("Nsight SQLite export complete for all %d expected rank(s)", len(statuses))
            return
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise RuntimeError(
                "Nsight SQLite export did not finish before worker cleanup: " + ", ".join(map(str, sorted(pending)))
            )
        if now >= next_log:
            logger.info("Waiting for Nsight SQLite export acknowledgements: %s", ", ".join(map(str, sorted(pending))))
            next_log = now + 60
        cancellation.wait(poll_seconds)
