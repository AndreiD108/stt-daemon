"""What both service managers share: the command runner and the installed executable."""

import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

type Runner = Callable[[Sequence[str]], int]
"""Runs a command with the terminal as its output; returns its exit code."""


class ServiceError(Exception):
    pass


def run_command(cmd: Sequence[str]) -> int:
    try:
        return subprocess.run(list(cmd), check=False).returncode
    except FileNotFoundError:
        raise ServiceError(f"{cmd[0]} is not installed") from None


def daemon_executable() -> Path:
    """This install's `stt-daemon` script: beside the interpreter (uv tool, venv), else on PATH."""
    beside = Path(sys.executable).parent / "stt-daemon"
    if beside.is_file():
        return beside.absolute()
    if found := shutil.which("stt-daemon"):
        return Path(found).absolute()
    raise ServiceError("cannot find the stt-daemon executable; install it with uv tool install")
