"""A scripted backend for tests: replies per commit, partials, delays, failures, time limit."""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from ..audio.frames import Samples
from ..config import Config
from ..credentials import Credentials
from .base import (
    SAMPLE_RATE,
    Backend,
    Capabilities,
    Closed,
    CommitPolicy,
    Committed,
    Connection,
    EventSink,
    Failed,
    OpenFailed,
    OpenOptions,
    Partial,
    Word,
)

FAKE_CAPABILITIES = Capabilities(
    languages=("auto", "en"),
    word_timestamps=True,
    partials=True,
    hints=frozenset({"keyterms", "previous_text"}),
    max_connection_s=None,
    commit_policy=CommitPolicy(interval_s=25.0, check_every_s=5.0),
)


@dataclass(frozen=True)
class FakeReply:
    """The answer to one commit that had audio. Replies are used in commit order, across
    connections; once they run out, commits get empty text."""

    text: str = ""
    words: Sequence[tuple[str, float, float]] | None = None
    """Connection-relative (text, start_s, end_s); None spreads the text's words over the
    commit's audio."""
    language: str | None = None
    delay_s: float = 0.0
    partials: tuple[str, ...] = ()
    """Sent one per `send_audio` call while this reply is the next one, with partials on."""
    fail: tuple[str, bool] | None = None
    """(code, retryable): the connection fails instead of answering."""


class FakeBackend(Backend):
    name = "fake"

    def __init__(
        self,
        replies: Iterable[FakeReply | str] = (),
        *,
        capabilities: Capabilities = FAKE_CAPABILITIES,
        open_failures: Iterable[OpenFailed] = (),
        open_delay_s: float = 0.0,
    ) -> None:
        self.capabilities = capabilities
        self.replies: collections.deque[FakeReply] = collections.deque(
            r if isinstance(r, FakeReply) else FakeReply(text=r) for r in replies
        )
        self.open_failures = collections.deque(open_failures)
        """Raised by successive `open` calls before any succeeds."""
        self.open_delay_s = open_delay_s
        self.open_calls = 0
        self.connections: list[FakeConnection] = []
        self._partial_index = 0

    @classmethod
    def create(cls, config: Config, credentials: Credentials) -> Self:
        return cls()

    async def open(
        self, options: OpenOptions, connection_id: int, sink: EventSink
    ) -> FakeConnection:
        self.open_calls += 1
        if self.open_delay_s:
            await asyncio.sleep(self.open_delay_s)
        if self.open_failures:
            raise self.open_failures.popleft()
        conn = FakeConnection(self, options, connection_id, sink)
        self.connections.append(conn)
        return conn

    def fail(self, code: str = "network", retryable: bool = True) -> None:
        """Drop the newest live connection, as a network or server failure would."""
        live = [c for c in self.connections if not c.ended]
        if live:
            live[-1].fail(code, retryable)

    def take_reply(self) -> FakeReply:
        self._partial_index = 0
        return self.replies.popleft() if self.replies else FakeReply()

    def take_partial(self) -> str | None:
        if not self.replies or self._partial_index >= len(self.replies[0].partials):
            return None
        self._partial_index += 1
        return self.replies[0].partials[self._partial_index - 1]


@dataclass(frozen=True)
class _PendingCommit:
    commit_id: int
    reply: FakeReply | None
    start_s: float
    end_s: float


class FakeConnection(Connection):
    def __init__(
        self, backend: FakeBackend, options: OpenOptions, connection_id: int, sink: EventSink
    ) -> None:
        super().__init__(connection_id)
        self.options = options
        self.chunks: list[Samples] = []
        self.commit_ids: list[int] = []
        self.ended = False
        self._backend = backend
        self._sink = sink
        self._sent_samples = 0
        self._committed_samples = 0
        self._pending: asyncio.Queue[_PendingCommit] = asyncio.Queue()
        self._worker = asyncio.create_task(self._answer_commits())
        limit = backend.capabilities.max_connection_s
        self._limit_timer = (
            asyncio.get_running_loop().call_later(
                limit, self.fail, "session_limit", True, "connection time limit reached"
            )
            if limit is not None
            else None
        )

    @property
    def audio(self) -> Samples:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)

    @property
    def sent_s(self) -> float:
        return self._sent_samples / SAMPLE_RATE

    async def send_audio(self, samples: Samples) -> None:
        if self.ended:
            return
        self.chunks.append(samples.copy())
        self._sent_samples += len(samples)
        if self.options.partials and (text := self._backend.take_partial()) is not None:
            self._sink(Partial(self.connection_id, text))

    async def commit(self) -> int:
        commit_id = len(self.commit_ids) + 1
        self.commit_ids.append(commit_id)
        if self.ended:
            return commit_id
        start, end = self._committed_samples, self._sent_samples
        self._committed_samples = end
        reply = self._backend.take_reply() if end > start else None
        self._pending.put_nowait(
            _PendingCommit(commit_id, reply, start / SAMPLE_RATE, end / SAMPLE_RATE)
        )
        return commit_id

    async def close(self) -> None:
        self._end(None)

    def fail(self, code: str, retryable: bool, message: str = "fake failure") -> None:
        self._end(Failed(self.connection_id, code, message, retryable))

    async def _answer_commits(self) -> None:
        while True:
            pending = await self._pending.get()
            reply = pending.reply or FakeReply()
            if reply.delay_s:
                await asyncio.sleep(reply.delay_s)
            if reply.fail is not None:
                self.fail(*reply.fail)
                return
            self._sink(
                Committed(
                    self.connection_id,
                    pending.commit_id,
                    reply.text,
                    reply.language,
                    _words(reply, pending.start_s, pending.end_s),
                    pending.end_s,
                )
            )

    def _end(self, failure: Failed | None) -> None:
        if self.ended:
            return
        self.ended = True
        if self._limit_timer is not None:
            self._limit_timer.cancel()
        if self._worker is not asyncio.current_task():
            self._worker.cancel()
        if failure is not None:
            self._sink(failure)
        self._sink(Closed(self.connection_id))


def _words(reply: FakeReply, start_s: float, end_s: float) -> tuple[Word, ...]:
    if reply.words is not None:
        return tuple(Word(t, s, e) for t, s, e in reply.words)
    texts = reply.text.split()
    if not texts:
        return ()
    step = (end_s - start_s) / len(texts)
    return tuple(Word(t, start_s + i * step, start_s + (i + 1) * step) for i, t in enumerate(texts))
