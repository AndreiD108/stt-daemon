"""System audio on macOS: `audiotee` (Core Audio process taps, macOS 14.2 or later) on everything
the speakers play, or on one process's output.

Without the System Audio Recording permission `audiotee` runs but writes nothing, so a stall's
reason names the setting. `audiotee` logs JSON lines to stderr; an exit's reason takes the last
error from them.
"""

import json
from collections.abc import Sequence
from typing import Any, cast

from .capture import SAMPLE_RATE
from .system_capture import BUFFER_MS, RESTART_BACKOFF_S, STALL_S, AppTarget, Command, SystemCapture

CHUNK_S = 0.1
PERMISSION_HINT = (
    "if it stays silent, allow System Audio Recording for the app that runs the daemon "
    "(stt-daemon under launchd, else the terminal) in System Settings > Privacy & Security > "
    "Screen & System Audio Recording"
)


def audiotee_command(pid: str | None = None, sample_rate: int = SAMPLE_RATE) -> list[str]:
    """s16 mono on stdout (any `--sample-rate` converts to s16); with `pid`, only that process's
    output."""
    command = ["audiotee", "--sample-rate", str(sample_rate), "--chunk-duration", str(CHUNK_S)]
    if pid is not None:
        command += ["--include-processes", pid]
    return command


class AudioteeCapture(SystemCapture):
    def __init__(
        self,
        command: Sequence[str] | Command = audiotee_command,
        *,
        target: AppTarget | None = None,
        sample_rate: int = SAMPLE_RATE,
        stall_s: float = STALL_S,
        backoff_s: Sequence[float] = RESTART_BACKOFF_S,
        buffer_ms: int = BUFFER_MS,
    ) -> None:
        super().__init__(
            command,
            target=target,
            sample_rate=sample_rate,
            stall_s=stall_s,
            backoff_s=backoff_s,
            buffer_ms=buffer_ms,
        )

    def stall_reason(self) -> str:
        return f"{super().stall_reason()}; {PERMISSION_HINT}"

    def exit_reason(self, code: int) -> str:
        reason = super().exit_reason(code)
        error = last_error(self.stderr_tail())
        return f"{reason}: {error}" if error else reason


def last_error(stderr: str) -> str | None:
    """The last `"message_type": "error"` line's message and context."""
    for line in reversed(stderr.splitlines()):
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if not isinstance(message, dict):
            continue
        message = cast(dict[str, Any], message)
        data = message.get("data")
        if message.get("message_type") != "error" or not isinstance(data, dict):
            continue
        data = cast(dict[str, Any], data)
        text = str(data.get("message") or "error")
        context = data.get("context")
        if isinstance(context, dict) and context:
            pairs = cast(dict[str, Any], context).items()
            text += " (" + ", ".join(f"{key}: {value}" for key, value in pairs) + ")"
        return text
    return None
