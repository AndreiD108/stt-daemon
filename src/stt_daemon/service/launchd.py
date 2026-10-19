"""The macOS launchd agent."""

import os
import plistlib
from collections.abc import Mapping
from pathlib import Path

from .common import Runner, ServiceError

LABEL = "io.github.andreid108.stt-daemon"
_NOT_FOUND = 113
"""`launchctl print`'s exit code for a service that isn't loaded."""


def render_plist(executable: Path, log_file: Path, env: Mapping[str, str]) -> bytes:
    plist: dict[str, object] = {
        "Label": LABEL,
        "ProgramArguments": [str(executable), "run"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(log_file),
        "StandardErrorPath": str(log_file),
    }
    # The socket lives under $TMPDIR; clients started from a shell must see the same one.
    # launchd's own PATH has no Homebrew or user dirs, where ffmpeg and audiotee live.
    kept = {name: env[name] for name in ("TMPDIR", "PATH") if env.get(name)}
    if kept:
        plist["EnvironmentVariables"] = kept
    return plistlib.dumps(plist)


class Launchd:
    def __init__(
        self, run: Runner, agents_dir: Path | None = None, env: Mapping[str, str] | None = None
    ) -> None:
        """With an `agents_dir`, only the file is written or removed; launchctl isn't called."""
        self._run = run
        self._manage = agents_dir is None
        self._env = os.environ if env is None else env
        home = Path.home()
        self.agents_dir = home / "Library" / "LaunchAgents" if agents_dir is None else agents_dir
        self.log_file = home / "Library" / "Logs" / "stt-daemon.log"

    @property
    def plist_file(self) -> Path:
        return self.agents_dir / f"{LABEL}.plist"

    @property
    def _domain(self) -> str:
        return f"gui/{os.getuid()}"

    def install(self, executable: Path) -> list[Path]:
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.plist_file.write_bytes(render_plist(executable, self.log_file, self._env))
        if self._manage:
            self._run(["launchctl", "bootout", f"{self._domain}/{LABEL}"])
            if self._run(["launchctl", "bootstrap", self._domain, str(self.plist_file)]) != 0:
                raise ServiceError(f"`launchctl bootstrap {self._domain}` failed")
        return [self.plist_file]

    def remove(self) -> list[Path]:
        if not self.plist_file.exists():
            return []
        target = f"{self._domain}/{LABEL}"
        if self._manage and self._loaded() and self._run(["launchctl", "bootout", target]) != 0:
            raise ServiceError(f"`launchctl bootout {target}` failed")
        self.plist_file.unlink()
        return [self.plist_file]

    def _loaded(self) -> bool:
        code = self._run(["launchctl", "print", f"{self._domain}/{LABEL}"])
        if code not in (0, _NOT_FOUND):
            raise ServiceError(f"`launchctl print {self._domain}/{LABEL}` failed ({code})")
        return code == 0

    def status(self) -> int:
        return self._run(["launchctl", "print", f"{self._domain}/{LABEL}"])

    def logs(self, *, follow: bool, lines: int) -> int:
        cmd = ["tail", "-n", str(lines), str(self.log_file)]
        return self._run([*cmd[:1], "-f", *cmd[1:]] if follow else cmd)
