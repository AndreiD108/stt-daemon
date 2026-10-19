"""What a backend plugin implements: one `Backend` subclass in a file in
`~/.config/stt-daemon/backends/`. See docs/plugins.md."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


@dataclass(frozen=True)
class Capabilities:
    languages: tuple[str, ...] = ("auto",)
    """Codes a client may send as `language`; "auto": the backend detects it."""
    word_times: bool = False
    """`Result.words` carry times; without them the daemon estimates times from the audio."""
    diarize: bool = False
    """Takes `Options.diarize` and names each word's speaker."""
    keyterms: bool = False
    previous_text: bool = False


@dataclass(frozen=True)
class Options:
    language: str | None = None
    """None: detect it."""
    keyterms: tuple[str, ...] = ()
    previous_text: str | None = None
    diarize: bool = False
    num_speakers: int | None = None
    """With `diarize`: at most this many speakers."""


@dataclass(frozen=True)
class Word:
    text: str
    start: float | None = None
    """Seconds from the start of the audio passed to `transcribe`."""
    end: float | None = None
    speaker: str | None = None


@dataclass(frozen=True)
class Result:
    text: str = ""
    """Empty: the words joined by spaces."""
    words: Sequence[Word] = ()


class Backend:
    name: ClassVar[str]
    """The backend's name in the protocol and the CLI: lowercase letters, digits and `-`."""
    capabilities: ClassVar[Capabilities] = Capabilities()
    needs_key: ClassVar[bool] = False
    """Without a key (`stt-daemon key set <name>`), the backend is not set up."""

    def setup(self, key: str | None, config: Mapping[str, Any]) -> None:
        """At load and at each reload. `key` from credentials.toml `[<name>]`, `config` from
        config.toml `[backends.<name>]`. An exception here leaves the backend failed."""

    async def transcribe(self, pcm: bytes, options: Options) -> Result:
        """16 kHz mono s16le audio, one chunk between pauses. Runs on the daemon's event loop:
        put blocking work in `asyncio.to_thread`."""
        raise NotImplementedError

    async def install(self, data_dir: Path) -> None:
        """`stt-daemon backend install <name>`; `data_dir` exists. Override to offer it."""
        raise NotImplementedError

    def available(self) -> bool:
        """False: the backend is not installed (status `not installed`)."""
        return True
