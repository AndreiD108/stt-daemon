"""NVIDIA Parakeet TDT 0.6B v3 on this machine, through sherpa-onnx.

An after-the-fact backend: audio is buffered and each commit decodes what came since the last one,
in a worker thread. Beam search; a session's keyterms go in as hotwords on each decode.
"""

from __future__ import annotations

import asyncio
import collections
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import numpy as np

from ..audio.frames import Samples
from ..config import Config, ParakeetConfig, data_dir
from ..credentials import Credentials
from .base import (
    SAMPLE_RATE,
    Backend,
    BackendUnavailable,
    Capabilities,
    Closed,
    CommitPolicy,
    Committed,
    Connection,
    EventSink,
    Failed,
    OpenFailed,
    OpenOptions,
    Word,
    merge_keyterms,
)

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
# Shorter audio has no feature frames, and the encoder raises on it.
_MIN_DECODE_SAMPLES = SAMPLE_RATE // 10

CAPABILITIES = Capabilities(
    languages=("auto",),
    word_timestamps=True,
    partials=False,
    hints=frozenset({"keyterms"}),
    max_connection_s=None,
    commit_policy=CommitPolicy(at_pauses=True, max_chunk_s=30.0),
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


class Turns:
    """One holder at a time; waiters marked `first` get the next turn before the others."""

    def __init__(self) -> None:
        self._busy = False
        self._first: collections.deque[asyncio.Future[None]] = collections.deque()
        self._rest: collections.deque[asyncio.Future[None]] = collections.deque()

    async def acquire(self, *, first: bool) -> None:
        if not self._busy:
            self._busy = True
            return
        turn: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        line = self._first if first else self._rest
        line.append(turn)
        try:
            await turn
        except asyncio.CancelledError:
            if turn.done() and not turn.cancelled():
                self.release()  # given the turn as the caller was cancelled: pass it on
            elif turn in line:
                line.remove(turn)
            raise

    def release(self) -> None:
        for line in (self._first, self._rest):
            while line:
                turn = line.popleft()
                if not turn.done():
                    turn.set_result(None)
                    return
        self._busy = False


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
        policy = CommitPolicy(at_pauses=True, max_chunk_s=config.max_chunk_seconds)
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


@dataclass(frozen=True)
class _Pending:
    commit_id: int
    audio: Samples
    start_s: float
    end_s: float


class ParakeetConnection(Connection):
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
        super().__init__(connection_id)
        self._sink = sink
        self._decode = decode
        self._ready = ready
        self._on_end = on_end
        self._chunks: list[Samples] = []
        self._sent_samples = 0
        self._committed_samples = 0
        self._commits = 0
        self._pending: collections.deque[_Pending] = collections.deque()
        self._has_pending = asyncio.Event()
        self._ended = False
        self._worker = asyncio.create_task(self._decode_commits())

    async def send_audio(self, samples: Samples) -> None:
        if self._ended:
            return
        self._chunks.append(np.asarray(samples, np.float32).copy())
        self._sent_samples += len(samples)

    async def commit(self) -> int:
        self._commits += 1
        if self._ended:
            return self._commits
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        self._chunks = []
        start, end = self._committed_samples, self._sent_samples
        self._committed_samples = end
        self._pending.append(_Pending(self._commits, audio, start / SAMPLE_RATE, end / SAMPLE_RATE))
        self._has_pending.set()
        return self._commits

    async def close(self) -> None:
        self._end(None)

    async def _decode_commits(self) -> None:
        if self._ready is not None:
            try:
                await self._ready()
            except OpenFailed as exc:
                self._end(Failed(self.connection_id, exc.code, exc.message, exc.retryable))
                return
        while True:
            await self._has_pending.wait()
            self._has_pending.clear()
            while self._pending:
                pending = self._pending[0]
                try:
                    event = await self._transcribe(pending)
                except Exception as exc:
                    log.exception(
                        "Parakeet failed to decode %.1f s", len(pending.audio) / SAMPLE_RATE
                    )
                    self._end(
                        Failed(
                            self.connection_id,
                            "backend_error",
                            f"decoding failed: {exc}",
                            retryable=False,
                        )
                    )
                    return
                self._pending.popleft()
                self._sink(event)

    async def _transcribe(self, pending: _Pending) -> Committed:
        if len(pending.audio) < _MIN_DECODE_SAMPLES:
            text, words = "", ()
        else:
            decoded = await self._decode(pending.audio)
            text = decoded.text.strip()
            words = group_words(decoded, pending.start_s) if text else ()
        return Committed(self.connection_id, pending.commit_id, text, None, words, pending.end_s)

    def _end(self, failure: Failed | None) -> None:
        if self._ended:
            return
        self._ended = True
        self._pending.clear()
        self._chunks = []
        if asyncio.current_task() is not self._worker:
            self._worker.cancel()
        if self._on_end is not None:
            self._on_end()
        if failure is not None:
            self._sink(failure)
        self._sink(Closed(self.connection_id))
