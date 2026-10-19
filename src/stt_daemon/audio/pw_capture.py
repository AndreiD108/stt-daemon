"""System audio on Linux: `pw-record` on the default output's monitor, or on one application's
stream."""

from collections.abc import Sequence

from .capture import SAMPLE_RATE
from .system_capture import RESTART_BACKOFF_S, STALL_S, AppTarget, Command, SystemCapture


def pw_record_command(target: str | None = None, sample_rate: int = SAMPLE_RATE) -> list[str]:
    """s16 mono on stdout; without `target`, the default output's monitor (it follows the user's
    output switches). A `target` serial is never swapped for another node when it goes."""
    command = ["pw-record", "--raw", "--rate", str(sample_rate), "--channels", "1"]
    command += ["--format", "s16", "--latency", "20ms"]
    if target is None:
        command += ["-P", "{ stream.capture.sink = true }"]
    else:
        command += ["--target", target]
        command += ["-P", "{ node.dont-fallback = true, node.dont-reconnect = true }"]
    return [*command, "-"]


class PwRecordCapture(SystemCapture):
    def __init__(
        self,
        command: Sequence[str] | Command = pw_record_command,
        *,
        target: AppTarget | None = None,
        sample_rate: int = SAMPLE_RATE,
        stall_s: float = STALL_S,
        backoff_s: Sequence[float] = RESTART_BACKOFF_S,
    ) -> None:
        super().__init__(
            command, target=target, sample_rate=sample_rate, stall_s=stall_s, backoff_s=backoff_s
        )
