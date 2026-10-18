"""The systemd user unit and its limits drop-in."""

import contextlib
import os
import re
from collections.abc import Mapping
from pathlib import Path

from ..config import atomic_write
from .common import Runner, ServiceError

UNIT = "stt-daemon.service"
_MARK = "# Written by `stt-daemon service install`. Change it with a drop-in:"

LIMITS = """\
[Service]
LogRateLimitIntervalSec=30s
LogRateLimitBurst=200
MemoryHigh=1G
MemoryMax=2G
"""


def default_unit_dir(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    base = env.get("XDG_CONFIG_HOME", "")
    root = Path(base) if os.path.isabs(base) else Path.home() / ".config"
    return root / "systemd" / "user"


def render_unit(executable: Path) -> str:
    # No start limit: at login the mic can be missing for a while; the daemon exits and retries.
    return f"""\
{_MARK}
# systemctl --user edit {UNIT}

[Unit]
Description=stt-daemon: local speech-to-text over a Unix socket
After=pipewire.service pipewire-pulse.service pulseaudio.service
StartLimitIntervalSec=0

[Service]
ExecStart={exec_arg(str(executable))} run
Restart=always
RestartSec=1

[Install]
WantedBy=default.target
"""


def exec_arg(arg: str) -> str:
    """One `ExecStart` word: specifiers and variables escaped, quoted when needed."""
    if "\n" in arg:
        raise ServiceError(f"the path has a line break in it: {arg!r}")
    arg = arg.replace("%", "%%").replace("$", "$$")
    if re.fullmatch(r"[\w@%$+=:,./-]+", arg):
        return arg
    return '"' + arg.replace("\\", "\\\\").replace('"', '\\"') + '"'


class Systemd:
    def __init__(self, run: Runner, unit_dir: Path | None = None) -> None:
        """With a `unit_dir`, only the files are written or removed; systemctl isn't called."""
        self._run = run
        self._manage = unit_dir is None
        self.unit_dir = default_unit_dir() if unit_dir is None else unit_dir

    @property
    def unit_file(self) -> Path:
        return self.unit_dir / UNIT

    @property
    def limits_file(self) -> Path:
        return self.unit_dir / f"{UNIT}.d" / "limits.conf"

    def install(self, executable: Path) -> list[Path]:
        """Writes both files; then reloads systemd, enables the unit and (re)starts it."""
        atomic_write(self.unit_file, render_unit(executable), mode=0o644)
        atomic_write(self.limits_file, LIMITS, mode=0o644)
        if self._manage:
            self._systemctl("daemon-reload")
            self._systemctl("enable", UNIT)
            self._systemctl("restart", UNIT)
        return [self.unit_file, self.limits_file]

    def remove(self) -> list[Path]:
        """Stops and disables the unit, deletes both files; other drop-ins stay."""
        try:
            text = self.unit_file.read_text()
        except FileNotFoundError:
            return []
        if _MARK not in text:
            raise ServiceError(
                f"{self.unit_file} was not written by `stt-daemon service install`; "
                "remove it by hand"
            )
        if self._manage:
            self._systemctl("disable", "--now", UNIT)
        removed = [p for p in (self.unit_file, self.limits_file) if p.exists()]
        for path in removed:
            path.unlink()
        with contextlib.suppress(OSError):  # other drop-ins keep the directory
            self.limits_file.parent.rmdir()
        if self._manage:
            self._systemctl("daemon-reload")
        return removed

    def status(self) -> int:
        return self._run(["systemctl", "--user", "status", UNIT])

    def logs(self, *, follow: bool, lines: int) -> int:
        cmd = ["journalctl", "--user", "-u", UNIT, "-n", str(lines), "--no-pager"]
        return self._run([*cmd, "-f"] if follow else cmd)

    def _systemctl(self, *args: str) -> None:
        cmd = ["systemctl", "--user", *args]
        if self._run(cmd) != 0:
            raise ServiceError(f"`{' '.join(cmd)}` failed")
