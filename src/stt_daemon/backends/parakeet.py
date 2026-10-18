"""NVIDIA Parakeet TDT 0.6B v3 on this machine, through sherpa-onnx.

An after-the-fact backend: audio is buffered and each commit decodes what came since the last one,
in a worker thread. Greedy decoding without keyterms; with them, beam search with hotwords.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
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
    commit_policy=CommitPolicy(interval_s=30.0, check_every_s=5.0, at_pauses=True),
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
    beam: bool
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
        num_threads=min(4, os.cpu_count() or 1),
        decoding_method="modified_beam_search" if key.beam else "greedy_search",
        model_type="nemo_transducer",
        modeling_unit="bpe",
        bpe_vocab=str(m / BPE_VOCAB),
        hotwords_score=key.hotwords_score,
    )


class Engine:
    """The daemon's one loaded recognizer (~700 MB); loading another kind replaces it.

    Blocking; call from worker threads. Loads and decodes take turns."""

    def __init__(self, loader: Loader = load_recognizer) -> None:
        self._loader = loader
        self._lock = threading.Lock()
        self._key: RecognizerKey | None = None
        self._recognizer: Any = None

    def load(self, key: RecognizerKey) -> None:
        with self._lock:
            self._load(key)

    def decode(self, key: RecognizerKey, audio: Samples, hotwords: str | None) -> Decoded:
        with self._lock:
            recognizer = self._load(key)
            stream = recognizer.create_stream(hotwords) if hotwords else recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            recognizer.decode_stream(stream)
            result = stream.result
            return Decoded(
                result.text, list(result.tokens), list(result.timestamps), list(result.durations)
            )

    def _load(self, key: RecognizerKey) -> Any:
        if self._key != key:
            self._recognizer, self._key = None, None
            log.info("loading Parakeet (%s)", "beam search" if key.beam else "greedy")
            self._recognizer = self._loader(key)
            self._key = key
        return self._recognizer


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


class ParakeetBackend(Backend):
    name = "parakeet"
    capabilities = CAPABILITIES

    def __init__(self, config: ParakeetConfig, model: Path, engine: Engine = ENGINE) -> None:
        self._config = config
        self._model = model
        self._engine = engine

    def __repr__(self) -> str:
        return f"ParakeetBackend(model={str(self._model)!r})"

    @classmethod
    def create(cls, config: Config, credentials: Credentials) -> Self:
        model = model_dir()
        if missing := [f for f in MODEL_FILES if not (model / f).is_file()]:
            raise BackendUnavailable(
                "no_model",
                f"the Parakeet model is not installed ({model / missing[0]} is missing); "
                "install it with `stt-daemon backend install parakeet`",
            )
        return cls(config.backends.parakeet, model)

    async def open(
        self, options: OpenOptions, connection_id: int, sink: EventSink
    ) -> ParakeetConnection:
        if options.language not in (None, "auto"):
            raise OpenFailed(
                "unsupported_language",
                f"Parakeet detects the language itself; it can't take {options.language!r}",
                retryable=False,
            )
        terms = hotwords(merge_keyterms(self._config.keyterms, options.keyterms))
        key = RecognizerKey(self._model, terms is not None, self._config.hotwords_score)
        try:
            await asyncio.to_thread(self._engine.load, key)
        except Exception as exc:
            log.exception("could not load the Parakeet model from %s", self._model)
            raise OpenFailed(
                "model_error",
                f"could not load the Parakeet model from {self._model}: {exc}; "
                f"delete {self._model} and run `stt-daemon backend install parakeet`",
                retryable=False,
            ) from None

        async def decode(audio: Samples) -> Decoded:
            return await asyncio.to_thread(self._engine.decode, key, audio, terms)

        return ParakeetConnection(connection_id, sink, decode)


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
    ) -> None:
        super().__init__(connection_id)
        self._sink = sink
        self._decode = decode
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
        if failure is not None:
            self._sink(failure)
        self._sink(Closed(self.connection_id))
