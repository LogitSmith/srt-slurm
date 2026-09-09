# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for Nsight export finalization before worker cleanup."""

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from srtctl.cli.do_sweep import wait_for_nsys_sqlite_exports


def _write_complete_export(path: Path) -> None:
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER)")
        database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1)")


def test_waits_for_slow_nsys_sqlite_export(tmp_path: Path) -> None:
    """Do not clean workers up while an asynchronous export is incomplete."""
    profile_dir = tmp_path / "profiles" / "agg"
    profile_dir.mkdir(parents=True)
    (profile_dir / "worker.1.nsys-rep").write_bytes(b"completed report")
    sqlite_path = profile_dir / "worker.1.sqlite"
    sqlite_path.write_bytes(b"\0" * 4096)

    delay_seconds = 0.25

    def finish_export() -> None:
        time.sleep(delay_seconds)
        _write_complete_export(sqlite_path)

    producer = threading.Thread(target=finish_export)
    producer.start()
    started = time.monotonic()
    wait_for_nsys_sqlite_exports(tmp_path, timeout_seconds=2, poll_seconds=0.01)
    elapsed = time.monotonic() - started
    producer.join()

    assert elapsed >= delay_seconds
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as database:
        assert database.execute("PRAGMA quick_check").fetchone() == ("ok",)


def test_rejects_nsys_sqlite_export_that_never_finishes(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profiles" / "agg"
    profile_dir.mkdir(parents=True)
    (profile_dir / "worker.1.nsys-rep").write_bytes(b"completed report")
    (profile_dir / "worker.1.sqlite").write_bytes(b"\0" * 4096)

    with pytest.raises(RuntimeError, match="did not finish"):
        wait_for_nsys_sqlite_exports(
            tmp_path,
            timeout_seconds=0.05,
            poll_seconds=0.01,
        )
