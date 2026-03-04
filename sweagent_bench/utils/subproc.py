"""Subprocess utilities with log streaming."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def run(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    timeout_s: int | None = 1800,
) -> int:
    run_env: dict[str, str] | None = None
    if env is not None:
        run_env = {**os.environ, **env}

    if stdout_path:
        stdout_path = Path(stdout_path)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path:
        stderr_path = Path(stderr_path)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

    stdout_file: Any = None
    stderr_file: Any = None
    try:
        if stdout_path:
            stdout_file = open(stdout_path, "w", encoding="utf-8")
        if stderr_path:
            stderr_file = open(stderr_path, "w", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=run_env,
            stdout=stdout_file if stdout_file else subprocess.DEVNULL,
            stderr=stderr_file if stderr_file else subprocess.DEVNULL,
            text=True,
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return 124
        return proc.returncode
    finally:
        if stdout_file:
            stdout_file.close()
        if stderr_file:
            stderr_file.close()
