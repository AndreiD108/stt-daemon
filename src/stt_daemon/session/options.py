"""What a session is started with, and what it refuses."""

from dataclasses import dataclass

from ..backends.base import OpenOptions


class MarkerRefused(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SessionOptions:
    open: OpenOptions
    """`language` already resolved: segments carry the detected language only for "auto"."""
    word_timestamps: bool = False


def start_point(requested_at: float, pre_start_ms: int, previous_end: float | None) -> float:
    """Where a session started by command takes its audio from: `pre_start_ms` before the
    command, but not before the previous session's audio ended (the ring still holds it)."""
    since = requested_at - pre_start_ms / 1000
    return since if previous_end is None else max(since, previous_end)
