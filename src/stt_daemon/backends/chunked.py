"""The chunk-per-commit connection: audio is buffered, and each commit transcribes what came since
the last one, in commit order. Used by after-the-fact backends (Parakeet, plugins)."""

from __future__ import annotations

import asyncio
import collections
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np

from ..audio.frames import Samples
from .base import (
    SAMPLE_RATE,
    Closed,
    Committed,
    Connection,
    EventSink,
    Failed,
    OpenFailed,
    Reference,
    Word,
)

log = logging.getLogger(__name__)

MIN_CHUNK_SAMPLES = SAMPLE_RATE // 10
"""Shorter chunks get empty text without a call; Parakeet's encoder raises on them."""


@dataclass(frozen=True)
class Chunk:
    audio: Samples
    start_s: float
    """Seconds from the connection's first audio."""
    end_s: float
    reference: Reference | None = None
    """What the speakers played meanwhile, for a backend with `playback_reference`."""


@dataclass(frozen=True)
class Transcribed:
    text: str
    words: tuple[Word, ...] | None
    """Connection seconds; None when the backend gave no word times."""
    speakers: tuple[str | None, ...] | None = None
    """With `words` None: each word's speaker, as in `Committed.speakers`."""


type Transcribe = Callable[[Chunk], Awaitable[Transcribed]]


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


class ChunkedConnection(Connection):
    def __init__(
        self,
        connection_id: int,
        sink: EventSink,
        transcribe: Transcribe,
        *,
        label: str,
        ready: Callable[[], Awaitable[None]] | None = None,
        on_end: Callable[[], None] | None = None,
    ) -> None:
        """`transcribe` raises OpenFailed for a failure with its own code, anything else for
        `backend_error`; `label` names the backend in logs. `ready` raises OpenFailed when the
        backend can't start; `on_end` is called once."""
        super().__init__(connection_id)
        self._sink = sink
        self._transcribe = transcribe
        self._label = label
        self._ready = ready
        self._on_end = on_end
        self._chunks: list[Samples] = []
        self._sent_samples = 0
        self._committed_samples = 0
        self._commits = 0
        self._pending: collections.deque[tuple[int, Chunk]] = collections.deque()
        self._has_pending = asyncio.Event()
        self._ended = False
        self._worker = asyncio.create_task(self._work())

    async def send_audio(self, samples: Samples) -> None:
        if self._ended:
            return
        self._chunks.append(np.asarray(samples, np.float32).copy())
        self._sent_samples += len(samples)

    async def commit(self, reference: Reference | None = None) -> int:
        self._commits += 1
        if self._ended:
            return self._commits
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        self._chunks = []
        start, end = self._committed_samples, self._sent_samples
        self._committed_samples = end
        chunk = Chunk(audio, start / SAMPLE_RATE, end / SAMPLE_RATE, reference)
        self._pending.append((self._commits, chunk))
        self._has_pending.set()
        return self._commits

    async def close(self) -> None:
        self._end(None)

    async def _work(self) -> None:
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
                commit_id, chunk = self._pending[0]
                try:
                    result = await self._chunk_text(chunk)
                except OpenFailed as exc:
                    log.error("%s failed on %.1f s: %s", self._label, _seconds(chunk), exc.message)
                    self._end(Failed(self.connection_id, exc.code, exc.message, retryable=False))
                    return
                except Exception as exc:
                    log.exception("%s failed on %.1f s", self._label, _seconds(chunk))
                    message = f"{self._label} failed: {exc}"
                    self._end(Failed(self.connection_id, "backend_error", message, retryable=False))
                    return
                self._pending.popleft()
                event = Committed(
                    self.connection_id,
                    commit_id,
                    result.text,
                    None,
                    result.words,
                    chunk.end_s,
                    result.speakers,
                )
                self._sink(event)

    async def _chunk_text(self, chunk: Chunk) -> Transcribed:
        if len(chunk.audio) < MIN_CHUNK_SAMPLES:
            return Transcribed("", ())
        return await self._transcribe(chunk)

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


def _seconds(chunk: Chunk) -> float:
    return len(chunk.audio) / SAMPLE_RATE
