"""The contract between a session and a transcription backend.

One interface serves streaming backends (audio sent as it comes, text back per commit) and
after-the-fact ones (audio buffered, transcribed at each commit). The session always sends audio
and asks for commits; when to commit follows `Capabilities.commit_policy`, and the session also
commits at stop, rollover and eviction.

Times in events are seconds from the start of the audio the session sent on that connection.
Mapping them to session time is the session's job.

Every `commit()` gets exactly one `Committed` with its `commit_id`, in commit order, unless the
connection ends first. A `Committed` with `commit_id` None was started by the backend or the
provider; when the provider cut at a point it doesn't report, `audio_end_s` stays at the previous
boundary (it claims no audio) and the next commit's text starts after the provider's cut.
A connection always ends with one `Closed`; when it ends for any reason other than `close()`, a
`Failed` comes before it. After the end, `send_audio` and `commit` do nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar, Self

from ..audio.frames import Samples
from ..config import DEFAULT_MAX_SESSIONS, Config
from ..credentials import Credentials

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Word:
    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Partial:
    connection_id: int
    text: str


@dataclass(frozen=True)
class Committed:
    connection_id: int
    commit_id: int | None
    text: str
    language: str | None
    words: tuple[Word, ...] | None
    """None when the backend has no word times; empty when there were no words."""
    audio_end_s: float
    """Covers all audio sent before this point, back to the previous `Committed`."""


@dataclass(frozen=True)
class Failed:
    connection_id: int
    code: str
    message: str
    retryable: bool
    """A new connection may work."""
    fatal: bool = True
    """The connection is over; `Closed` follows."""


@dataclass(frozen=True)
class Closed:
    connection_id: int


type BackendEvent = Partial | Committed | Failed | Closed
type EventSink = Callable[[BackendEvent], None]


@dataclass(frozen=True)
class CommitPolicy:
    interval_s: float | None = None
    """Commit once this much wall time has passed since the last commit, if speech was sent."""
    check_every_s: float | None = None
    """How often to check `interval_s`; it can cut mid-speech."""
    at_pauses: bool = False
    max_chunk_s: float | None = None
    """Commit once this much audio was sent since the last commit, pause or not."""


@dataclass(frozen=True)
class Capabilities:
    languages: tuple[str, ...]
    """Codes accepted in `OpenOptions.language`; "auto" when the backend detects the language."""
    word_timestamps: bool
    partials: bool
    hints: frozenset[str]
    """Of "keyterms", "previous_text"."""
    max_connection_s: float | None
    commit_policy: CommitPolicy


@dataclass(frozen=True)
class OpenOptions:
    language: str | None = None
    """None: the backend's default from config."""
    keyterms: tuple[str, ...] = ()
    """The client's; the backend adds its config defaults first."""
    previous_text: str | None = None
    partials: bool = False


class BackendUnavailable(Exception):
    """The backend is not set up on this machine (no key, no model)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class OpenFailed(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class Connection(ABC):
    def __init__(self, connection_id: int) -> None:
        self.connection_id = connection_id

    @abstractmethod
    async def send_audio(self, samples: Samples) -> None:
        """16 kHz mono float32 samples, continuing the audio sent before."""

    @abstractmethod
    async def commit(self) -> int:
        """Ask for the text of all audio sent since the last commit; returns its commit_id."""

    @abstractmethod
    async def close(self) -> None:
        """End now; commits still pending are dropped."""


class Backend(ABC):
    name: ClassVar[str]
    takes_key: ClassVar[bool] = False
    """Set up with `stt-daemon key set <name>`."""
    capabilities: Capabilities

    @property
    def default_language(self) -> str | None:
        """The language from config used when a session names none; None: no such setting."""
        return None

    @property
    def max_sessions(self) -> int:
        """How many sessions may use this backend at once; config `backends.<name>.max_sessions`."""
        return DEFAULT_MAX_SESSIONS

    @classmethod
    @abstractmethod
    def create(cls, config: Config, credentials: Credentials) -> Self:
        """Raises BackendUnavailable when this machine lacks what the backend needs."""

    @abstractmethod
    async def open(self, options: OpenOptions, connection_id: int, sink: EventSink) -> Connection:
        """Raises OpenFailed. Events for the connection go to `sink`, tagged with its id."""


def merge_keyterms(defaults: Iterable[str], extra: Iterable[str]) -> list[str]:
    """Config defaults first, then the client's; blanks and case-insensitive repeats dropped."""
    seen: set[str] = set()
    terms: list[str] = []
    for raw in (*defaults, *extra):
        term = " ".join(raw.split())
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            terms.append(term)
    return terms


def trim_previous_text(text: str | None, limit: int) -> str | None:
    """The last `limit` characters, starting at a word boundary when the cut lands mid-word."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text or None
    tail = text[-limit:]
    if not text[-limit - 1].isspace():
        space = tail.find(" ")
        if 0 <= space < len(tail) - 1:
            tail = tail[space + 1 :]
    return tail.strip() or None
