"""What a session reports, in the order it happens; the server turns these into protocol events."""

from dataclasses import dataclass, field
from typing import Literal

from .transcript import Segment

type EndReason = Literal["stopped", "cancelled", "evicted", "client_gone", "error"]
type SessionKind = Literal["mic", "system", "transcribe"]

KIND_RANK: dict[SessionKind, int] = {"mic": 0, "system": 1, "transcribe": 2}
"""Lower ranks first: a dictation suspends the highest-ranked session in its way."""


@dataclass(frozen=True)
class SessionStarted:
    session_id: str
    backend: str
    language: str | None
    started_at: int
    """Epoch ms (UTC) of the session's first sample."""
    previous_text: str | None
    kind: SessionKind = "mic"
    tracks: tuple[str, ...] | None = None
    """Track names, for a session with named tracks (`system`)."""


@dataclass(frozen=True)
class PartialText:
    text: str
    track: str | None = None


@dataclass(frozen=True)
class Progress:
    audio_seconds: float
    """Input audio taken in so far (`transcribe` sessions)."""


@dataclass(frozen=True)
class SessionError:
    code: str
    message: str
    fatal: bool
    """`SessionEnded` with reason "error" follows."""
    track: str | None = None


@dataclass(frozen=True)
class EvictedBy:
    client_id: str
    pid: int | None
    client_label: str | None


@dataclass(frozen=True)
class PreemptedBy:
    kind: SessionKind
    pid: int | None
    client_label: str | None


@dataclass(frozen=True)
class SessionSuspended:
    by: PreemptedBy


@dataclass(frozen=True)
class SessionResumed:
    suspended_seconds: float


@dataclass(frozen=True)
class SessionEnded:
    reason: EndReason
    final_text: str
    duration_ms: int
    segment_count: int
    evicted_by: EvictedBy | None = None
    """Set by the slot, with reason "evicted"."""
    plain_text: str = field(default="", compare=False)
    """`final_text` without markers."""


type SessionEvent = (
    SessionStarted
    | PartialText
    | Progress
    | Segment
    | SessionError
    | SessionSuspended
    | SessionResumed
    | SessionEnded
)
