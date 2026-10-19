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
    speaker: str | None = None
    """Who said it, as the backend names speakers; None without diarization."""


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
    speakers: tuple[str | None, ...] | None = None
    """With `words` None: each whitespace-split word's speaker, kept when times are estimated."""


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
class PlaybackSpan:
    """What the speakers played around one stretch of a commit's audio that was captured without
    a break (pauses the session cut out end a stretch)."""

    offset: int
    """The stretch's first sample in the commit's audio."""
    samples: int
    playback: Samples
    """From `lead` samples before the stretch's capture time to `lookahead` samples after its
    end, on the mic's clock; silence where the playback capture had nothing."""
    lead: int
    lookahead: int


PLAYBACK_LEAD_S = 0.8
"""Playback a span has before its stretch: the echo's longest delay plus its reverb tail."""
PLAYBACK_LOOKAHEAD_S = 0.1
"""And after it: playback frames are stamped on arrival, maybe later than the mic's."""


@dataclass(frozen=True)
class Reference:
    """The playback behind a commit's mic audio, for `Capabilities.playback_reference`."""

    spans: tuple[PlaybackSpan, ...]


type PauseSteps = tuple[tuple[float, float], ...]

PAUSE_STEPS: PauseSteps = ((0, 3.5), (8, 1.0), (14, 0.8), (19, 0.6), (23, 0.3), (27, 0.15))
"""Long pauses early so a chunk keeps its sentence when the speaker stops to think, shorter ones
as the chunk grows toward the cap."""


@dataclass(frozen=True)
class CommitPolicy:
    pause_steps: PauseSteps = ()
    """(sent_s, pause_s) by rising `sent_s`: once `sent_s` of audio was sent since the last
    commit, a pause of `pause_s` commits, if speech was sent. Silence the session cut isn't
    sent, so it doesn't count."""
    max_chunk_s: float | None = None
    """Commit once this much audio was sent since the last commit, pause or not."""

    def pause_for(self, sent_s: float) -> float | None:
        """The pause that commits after `sent_s`; None: pauses don't commit yet."""
        pause = None
        for since_s, pause_s in self.pause_steps:
            if sent_s < since_s:
                break
            pause = pause_s
        return pause


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
    paced_replay: bool = True
    """Audio replayed onto a new connection goes at a bounded rate; off for local backends."""
    diarize: bool = False
    """Takes `OpenOptions.diarize` and names each word's speaker."""
    playback_reference: bool = False
    """Takes a `Reference` with each commit of mic audio."""


@dataclass(frozen=True)
class OpenOptions:
    language: str | None = None
    """None: the backend's default from config."""
    keyterms: tuple[str, ...] = ()
    """The client's; the backend adds its config defaults first."""
    previous_text: str | None = None
    partials: bool = False
    dictation: bool = False
    """A mic session: on a shared local model its decodes go before other sessions'."""
    mic: bool = False
    """The audio comes from the mic (a mic session, or a system session's mic track)."""
    diarize: bool = False
    num_speakers: int | None = None
    """With `diarize`: at most this many speakers."""


class BackendUnavailable(Exception):
    """The backend is not set up on this machine (no key, no model), or its plugin failed."""

    def __init__(self, code: str, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail
        """A plugin failure's traceback, for the daemon's log."""


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
    async def commit(self, reference: Reference | None = None) -> int:
        """Ask for the text of all audio sent since the last commit; returns its commit_id.
        `reference`: with `Capabilities.playback_reference`, what the speakers played then."""

    @abstractmethod
    async def close(self) -> None:
        """End now; commits still pending are dropped."""


class Backend(ABC):
    name: str
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


@dataclass(frozen=True)
class FileAudio:
    data: bytes
    """The bytes as the client sent them: a container file, or headerless s16le."""
    pcm: tuple[int, int] | None
    """(rate, channels) when `data` is headerless s16le."""


class FileBackend(Backend):
    """Takes whole files after the input ends (`transcribe` sessions only), not live audio.

    Answers come back as `Committed` events with `connection_id` 0, times from the file's start;
    the last one covers the end of the audio.
    """

    default_for: ClassVar[str | None] = None
    """A `transcribe` with no backend uses this one when that backend is the default."""

    async def open(self, options: OpenOptions, connection_id: int, sink: EventSink) -> Connection:
        raise OpenFailed(
            "unsupported", f"{self.name} takes transcribe sessions only", retryable=False
        )

    @abstractmethod
    async def transcribe_file(self, audio: FileAudio, options: OpenOptions) -> list[Committed]:
        """Raises OpenFailed when the backend refuses or fails."""


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
