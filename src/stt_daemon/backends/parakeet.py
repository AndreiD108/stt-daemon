"""NVIDIA Parakeet TDT 0.6B v3 on this machine, through sherpa-onnx.

An after-the-fact backend: audio is buffered and each commit decodes what came since the last one,
in a worker thread. Beam search; a session's keyterms go in as hotwords on each decode.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

from ..audio.frames import Samples
from ..config import Config, ParakeetConfig, data_dir
from ..credentials import Credentials
from .base import (
    PAUSE_STEPS,
    SAMPLE_RATE,
    Backend,
    BackendUnavailable,
    Capabilities,
    CommitPolicy,
    EventSink,
    OpenFailed,
    OpenOptions,
    Word,
    merge_keyterms,
)
from .chunked import Chunk, ChunkedConnection, Transcribed, Turns

log = logging.getLogger(__name__)

MODEL_NAME = "parakeet-tdt-0.6b-v3-int8"
ENCODER, DECODER, JOINER, TOKENS, BPE_VOCAB = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
    "bpe.vocab",
)
MODEL_FILES = (ENCODER, DECODER, JOINER, TOKENS, BPE_VOCAB)
CAPABILITIES = Capabilities(
    languages=("auto",),
    word_timestamps=True,
    partials=False,
    hints=frozenset({"keyterms"}),
    max_connection_s=None,
    commit_policy=CommitPolicy(pause_steps=PAUSE_STEPS, max_chunk_s=30.0),
    paced_replay=False,
)


def model_dir(env: Mapping[str, str] | None = None) -> Path:
    return data_dir(env) / "models" / MODEL_NAME


@dataclass(frozen=True)
class Decoded:
    text: str
    tokens: Sequence[str]
    timestamps: Sequence[float]
    """Each token's start, seconds from the decoded audio's start."""
    durations: Sequence[float]


@dataclass(frozen=True)
class RecognizerKey:
    model: Path
    num_threads: int
    hotwords_score: float


type Loader = Callable[[RecognizerKey], Any]


def load_recognizer(key: RecognizerKey) -> Any:
    import sherpa_onnx

    m = key.model
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(m / ENCODER),
        decoder=str(m / DECODER),
        joiner=str(m / JOINER),
        tokens=str(m / TOKENS),
        num_threads=key.num_threads,
        decoding_method="modified_beam_search",
        model_type="nemo_transducer",
        modeling_unit="bpe",
        bpe_vocab=str(m / BPE_VOCAB),
        hotwords_score=key.hotwords_score,
    )


def _decode(recognizer: Any, audio: Samples, hotwords: str | None) -> Decoded:
    stream = recognizer.create_stream(hotwords) if hotwords else recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, audio)
    recognizer.decode_stream(stream)
    result = stream.result
    return Decoded(
        result.text, list(result.tokens), list(result.timestamps), list(result.durations)
    )


class Engine:
    """The daemon's one Parakeet recognizer (~700 MB), shared by every connection on it.

    `acquire` loads it in a thread unless it is loaded; decodes take turns, dictation's first;
    once no connection uses it, it is dropped after the idle time. Runs on the event loop."""

    def __init__(self, loader: Loader = load_recognizer) -> None:
        self._loader = loader
        self._key: RecognizerKey | None = None
        self._recognizer: asyncio.Future[Any] | None = None
        self._decoding = Turns()
        self._users = 0
        self._unload_after_s: float | None = None
        self._unload_timer: asyncio.TimerHandle | None = None
        self._unload_when_idle = False

    @property
    def users(self) -> int:
        return self._users

    @property
    def loaded(self) -> bool:
        r = self._recognizer
        return r is not None and r.done() and not r.cancelled() and r.exception() is None

    def acquire(self, key: RecognizerKey, unload_after_s: float | None) -> asyncio.Future[Any]:
        """One more user. The future gives the recognizer once loaded; shield it when awaiting.
        A different key (config changed) takes effect once no one uses the loaded one."""
        replace_idle = self._key != key and self._users == 0
        self._users += 1
        self._unload_after_s = unload_after_s
        self._unload_when_idle = False
        self._cancel_timer()
        if self._recognizer is None or replace_idle:
            self._key = key
            loading = asyncio.get_running_loop().run_in_executor(None, self._load, key)
            loading.add_done_callback(self._load_done)
            self._recognizer = loading
        return self._recognizer

    def release(self) -> None:
        self._users -= 1
        if self._users:
            return
        if self._unload_when_idle:
            self.unload()
        elif self._unload_after_s is not None:
            self._unload_timer = asyncio.get_running_loop().call_later(
                self._unload_after_s, self.unload
            )

    def unload_when_idle(self) -> None:
        """Now, or when the last user releases it (the model was removed)."""
        if self._users:
            self._unload_when_idle = True
        else:
            self.unload()

    def unload(self) -> None:
        if self._users:
            return
        self._cancel_timer()
        if self.loaded:
            log.info("unloading Parakeet")
        self._recognizer, self._key = None, None

    async def decode(
        self, recognizer: Any, audio: Samples, hotwords: str | None, *, first: bool = False
    ) -> Decoded:
        """`first`: a dictation's decode, which goes before every waiting decode of others."""
        # A decode counts as a user until its thread is done, even if its caller is cancelled.
        self._users += 1
        try:
            await self._decoding.acquire(first=first)
        except BaseException:
            self.release()
            raise
        try:
            work = asyncio.ensure_future(asyncio.to_thread(_decode, recognizer, audio, hotwords))
        except BaseException:
            self._decoding.release()
            self.release()
            raise

        def done(_: asyncio.Future[Decoded]) -> None:
            self._decoding.release()
            self.release()

        work.add_done_callback(done)
        return await asyncio.shield(work)

    def _load(self, key: RecognizerKey) -> Any:
        log.info("loading Parakeet from %s", key.model)
        return self._loader(key)

    def _load_done(self, loading: asyncio.Future[Any]) -> None:
        failed = loading.cancelled() or loading.exception() is not None
        if failed and loading is self._recognizer:
            self._recognizer, self._key = None, None

    def _cancel_timer(self) -> None:
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None


ENGINE = Engine()


def group_words(decoded: Decoded, offset_s: float) -> tuple[Word, ...]:
    """Tokens starting with a space begin a word; times move by `offset_s`."""
    words: list[Word] = []
    text, start, end = "", 0.0, 0.0
    for token, at, duration in zip(
        decoded.tokens, decoded.timestamps, decoded.durations, strict=True
    ):
        if token.startswith(" ") and text.strip():
            words.append(Word(text.strip(), offset_s + start, offset_s + end))
            text = ""
        if not text.strip():
            start = at
        text += token
        end = at + duration
    if text.strip():
        words.append(Word(text.strip(), offset_s + start, offset_s + end))
    return tuple(words)


def hotwords(terms: Sequence[str]) -> str | None:
    """sherpa's per-stream form: phrases split by "/", words by spaces."""
    phrases = [" ".join(t.replace("/", " ").split()) for t in terms]
    return "/".join(p for p in phrases if p) or None


def missing_file(model: Path) -> Path | None:
    return next((model / f for f in MODEL_FILES if not (model / f).is_file()), None)


def _not_installed(missing: Path) -> str:
    return (
        f"the Parakeet model is not installed ({missing} is missing); "
        "install it with `stt-daemon backend install parakeet`"
    )


class ParakeetBackend(Backend):
    name = "parakeet"

    def __init__(self, config: ParakeetConfig, model: Path, engine: Engine = ENGINE) -> None:
        self._config = config
        self._model = model
        self._engine = engine
        policy = CommitPolicy(pause_steps=PAUSE_STEPS, max_chunk_s=config.max_chunk_seconds)
        self.capabilities = replace(CAPABILITIES, commit_policy=policy)

    def __repr__(self) -> str:
        return f"ParakeetBackend(model={str(self._model)!r})"

    @property
    def max_sessions(self) -> int:
        return self._config.max_sessions

    @classmethod
    def create(cls, config: Config, credentials: Credentials) -> Self:
        model = model_dir()
        if (missing := missing_file(model)) is not None:
            raise BackendUnavailable("no_model", _not_installed(missing))
        return cls(config.backends.parakeet, model)

    async def open(
        self, options: OpenOptions, connection_id: int, sink: EventSink
    ) -> ParakeetConnection:
        """Returns at once; the model loads meanwhile and the first decode waits for it."""
        if options.language not in (None, "auto"):
            raise OpenFailed(
                "unsupported_language",
                f"Parakeet detects the language itself; it can't take {options.language!r}",
                retryable=False,
            )
        if (missing := missing_file(self._model)) is not None:
            raise OpenFailed("model_error", _not_installed(missing), retryable=False)
        terms = hotwords(merge_keyterms(self._config.keyterms, options.keyterms))
        key = RecognizerKey(self._model, self._config.num_threads, self._config.hotwords_score)
        unload_after_s = self._config.unload_after_minutes * 60 or None
        loading = self._engine.acquire(key, unload_after_s)
        model = self._model
        dictation = options.dictation

        async def ready() -> None:
            try:
                await asyncio.shield(loading)
            except Exception as exc:
                log.error("could not load the Parakeet model from %s: %s", model, exc)
                raise OpenFailed(
                    "model_error",
                    f"could not load the Parakeet model from {model}: {exc}; "
                    f"delete {model} and run `stt-daemon backend install parakeet`",
                    retryable=False,
                ) from None

        async def decode(audio: Samples) -> Decoded:
            recognizer = await asyncio.shield(loading)
            return await self._engine.decode(recognizer, audio, terms, first=dictation)

        return ParakeetConnection(
            connection_id, sink, decode, ready=ready, on_end=self._engine.release
        )


class ParakeetConnection(ChunkedConnection):
    def __init__(
        self,
        connection_id: int,
        sink: EventSink,
        decode: Callable[[Samples], Awaitable[Decoded]],
        *,
        ready: Callable[[], Awaitable[None]] | None = None,
        on_end: Callable[[], None] | None = None,
    ) -> None:
        """`ready` raises OpenFailed when the model can't load; `on_end` is called once."""

        async def transcribe(chunk: Chunk) -> Transcribed:
            decoded = await decode(chunk.audio)
            text = decoded.text.strip()
            return Transcribed(text, group_words(decoded, chunk.start_s) if text else ())

        super().__init__(
            connection_id, sink, transcribe, label="Parakeet", ready=ready, on_end=on_end
        )
