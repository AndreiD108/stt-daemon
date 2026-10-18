"""What a session reports, in the order it happens; the server turns these into protocol events."""

from dataclasses import dataclass
from typing import Literal

from .transcript import Segment

type EndReason = Literal["stopped", "cancelled", "evicted", "client_gone", "error"]


@dataclass(frozen=True)
class SessionStarted:
    session_id: str
    backend: str
    language: str | None
    started_at: int
    """Epoch ms (UTC) of the session's first sample."""
    previous_text: str | None


@dataclass(frozen=True)
class PartialText:
    text: str


@dataclass(frozen=True)
class SessionError:
    code: str
    message: str
    fatal: bool
    """`SessionEnded` with reason "error" follows."""


@dataclass(frozen=True)
class SessionEnded:
    reason: EndReason
    final_text: str
    duration_ms: int
    segment_count: int


type SessionEvent = SessionStarted | PartialText | Segment | SessionError | SessionEnded
