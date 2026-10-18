"""The session's backend stream across connections: rollover, reconnect and replay.

Audio and commits go into a log in order, and one task sends them to the current connection.
A commit's answer drops the log up to that commit before the session hears of it, so the log
holds exactly what no connection has answered yet. When a connection ends, the next one gets the
whole log first, paced, then live audio; audio that comes meanwhile waits in the log. A
connection rolls over before the backend's time limit: at a pause near the limit, or mid-speech
when closer still.
"""

import asyncio
import itertools
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

import numpy as np

from ..backends.base import (
    Backend,
    BackendEvent,
    Closed,
    Committed,
    Connection,
    Failed,
    OpenFailed,
    OpenOptions,
    Partial,
    trim_previous_text,
)
from .pause_gate import Outgoing
from .timeline import Timeline

log = logging.getLogger(__name__)

SOFT_ROLLOVER_MARGIN_S = 120.0
"""Roll over at the next pause once a connection is this close to the backend's limit."""
HARD_ROLLOVER_MARGIN_S = 60.0
"""Roll over even mid-speech this close to the limit."""
ROLLOVER_ACK_WAIT_S = 3.0
"""How long the old connection gets to answer its last commit; after that the audio is replayed."""
RECONNECT_COOLDOWN_S = 2.0
"""Least time between two connection opens after a failure."""
MAX_ATTEMPTS = 3
"""Reconnects in a row without an answer before the session fails."""
REPLAY_RATE = 2.0
"""Log audio goes to a new connection at up to this many times real time."""
SEND_TIMEOUT_S = 5.0
OPEN_TIMEOUT_S = 15.0
CONTEXT_CHARS = 200


@dataclass(eq=False)
class _Point:
    final: bool


type _Item = Outgoing | _Point


class _Fatal(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Link:
    def __init__(
        self,
        *,
        backend: Backend,
        options: OpenOptions,
        timeline: Timeline,
        chunk_samples: int,
        on_committed: Callable[[Committed, bool], None],
        on_partial: Callable[[str], None],
        on_error: Callable[[str, str], None],
        on_fatal: Callable[[str, str], None],
    ) -> None:
        """`on_committed(event, final)`: `final` when it answers the commit queued with `final`.
        `on_error` gets non-fatal problems; `on_fatal` is called once, when the link gives up."""
        self._backend = backend
        self._options = options
        self._timeline = timeline
        self._chunk = chunk_samples
        self._on_committed = on_committed
        self._on_partial = on_partial
        self._on_error = on_error
        self._on_fatal = on_fatal
        limit = backend.capabilities.max_connection_s
        self._rollover_s = None
        if limit is not None:
            self._rollover_s = (limit - SOFT_ROLLOVER_MARGIN_S, limit - HARD_ROLLOVER_MARGIN_S)
        self._log: list[_Item] = []
        self._sent = 0
        """Log items sent on the current connection; always a prefix of the log."""
        self._acked_samples = 0
        """Samples of the current connection answered and dropped from the log."""
        self._conn: Connection | None = None
        self._accept_id = 0
        """The connection whose events count; still the old one while it closes."""
        self._next_id = 1
        self._opened_at = 0.0
        self._lost: Failed | None = None
        self._roll_due = False
        self._catch_up: tuple[float, int] | None = None
        """(start, samples sent) while the log is replayed onto a new connection."""
        self._failures = 0
        self._context = options.previous_text or ""
        self._uncertain: list[Committed] = []
        self._alive = True
        self._wake = asyncio.Event()
        self._progress = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def open(self) -> None:
        """Open the first connection and start sending. Raises `OpenFailed`."""
        await self._open()
        self._task = asyncio.create_task(self._run())
        self._task.add_done_callback(self._run_done)

    def queue(self, out: Outgoing) -> None:
        if out.frame.samples.size:
            self._log.append(out)
            self._wake.set()

    def commit(self, *, final: bool = False) -> None:
        """Ask for the text of the audio queued since the last commit. With none queued, `final`
        marks the last unanswered commit instead, and nothing is sent."""
        if not self._log:
            return
        last = self._log[-1]
        if isinstance(last, _Point):
            last.final = last.final or final
        else:
            self._log.append(_Point(final))
            self._wake.set()

    def tick(self, *, speaking: bool) -> None:
        """Called as audio comes in; rolls the connection over when it is due."""
        if self._rollover_s is None or self._conn is None or self._catch_up is not None:
            return
        soft_s, hard_s = self._rollover_s
        age = time.monotonic() - self._opened_at
        if not self._roll_due and (age >= hard_s or (age >= soft_s and not speaking)):
            self._roll_due = True
            self._wake.set()

    async def settle(self) -> None:
        """Wait until every queued commit is answered, or the link has failed."""
        while self._alive and (
            self._lost is not None or any(isinstance(item, _Point) for item in self._log)
        ):
            self._progress.clear()
            await self._progress.wait()

    async def close(self) -> None:
        self._alive = False
        self._progress.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.wait([self._task])
        await self._close_conn()
        self._publish_uncertain()

    async def _run(self) -> None:
        try:
            while True:
                self._wake.clear()
                if self._lost is not None:
                    await self._recover()
                elif self._conn is None:
                    if self._log:
                        await self._reconnect(cooldown=True)
                    else:
                        await self._wake.wait()
                elif self._roll_due:
                    await self._roll()
                elif (batch := self._ready()) is not None:
                    await self._send(*batch)
                else:
                    # Same step as seeing nothing left to send: no live audio can slip past replay.
                    self._catch_up = None
                    await self._wake.wait()
        except _Fatal as exc:
            self._alive = False
            self._publish_uncertain()
            self._on_fatal(exc.code, exc.message)
            self._progress.set()
            await self._close_conn()

    def _run_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() or (error := task.exception()) is None:
            return
        log.error("backend link failed", exc_info=error)
        self._alive = False
        self._progress.set()
        self._on_fatal("internal", f"the daemon failed while talking to the backend: {error}")

    def _ready(self) -> tuple[list[Outgoing], _Point | None] | None:
        """Unsent audio up to a chunk or to the next commit, when there is that much."""
        audio: list[Outgoing] = []
        samples = 0
        for item in itertools.islice(self._log, self._sent, None):
            if isinstance(item, _Point):
                return audio, item
            audio.append(item)
            samples += item.frame.samples.size
            if samples >= self._chunk:
                return audio, None
        return None

    async def _send(self, audio: list[Outgoing], point: _Point | None) -> None:
        conn = self._conn
        assert conn is not None
        if audio:
            for out in audio:
                start, size = out.frame.start, out.frame.samples.size
                self._timeline.sent(conn.connection_id, start, size, cut=out.cut)
            self._sent += len(audio)
            samples = np.concatenate([out.frame.samples for out in audio])
            await self._bounded(conn, conn.send_audio(samples))
            await self._pace(samples.size)
        if point is not None:
            self._sent += 1  # before the await: a backend may answer inside commit()
            await self._bounded(conn, conn.commit())

    async def _bounded(self, conn: Connection, op: Awaitable[object]) -> None:
        try:
            async with asyncio.timeout(SEND_TIMEOUT_S):
                await op
        except TimeoutError:
            if self._lost is None:
                message = f"the backend took no data for {SEND_TIMEOUT_S:g} s"
                self._lost = Failed(conn.connection_id, "network", message, retryable=True)

    async def _pace(self, samples: int) -> None:
        if self._catch_up is None:
            return
        started, sent = self._catch_up
        sent += samples
        self._catch_up = (started, sent)
        ahead = started + sent / self._timeline.sample_rate / REPLAY_RATE - time.monotonic()
        if ahead > 0:
            await asyncio.sleep(ahead)

    async def _recover(self) -> None:
        failure = self._lost
        assert failure is not None
        await self._close_conn()
        self._lost = None
        self._progress.set()
        if not failure.retryable:
            raise _Fatal(failure.code, failure.message)
        if not self._log:
            log.info("backend connection ended (%s); reconnecting at the next audio", failure.code)
            return
        log.warning("backend connection failed (%s); replaying onto a new one", failure.message)
        self._count_failure(failure.code, failure.message)
        self._on_error(failure.code, f"{failure.message}; reconnecting")
        await self._reconnect(cooldown=True)

    def _count_failure(self, code: str, message: str) -> None:
        self._failures += 1
        if self._failures > MAX_ATTEMPTS:
            raise _Fatal(code, f"gave up after {MAX_ATTEMPTS} reconnects: {message}")

    async def _reconnect(self, *, cooldown: bool) -> None:
        while True:
            if cooldown and (wait := self._opened_at + RECONNECT_COOLDOWN_S - time.monotonic()) > 0:
                await asyncio.sleep(wait)
            try:
                await self._open()
                return
            except OpenFailed as exc:
                if not exc.retryable:
                    raise _Fatal(exc.code, exc.message) from None
                log.warning("backend reconnect failed: %s", exc.message)
                self._count_failure(exc.code, exc.message)
                cooldown = True

    async def _open(self) -> None:
        conn_id = self._next_id
        self._next_id += 1
        self._opened_at = time.monotonic()
        self._accept_id = conn_id
        options = self._options
        if "previous_text" in self._backend.capabilities.hints:
            options = replace(options, previous_text=self._context or None)
        try:
            async with asyncio.timeout(OPEN_TIMEOUT_S):
                conn = await self._backend.open(options, conn_id, self._on_event)
        except TimeoutError:
            message = f"no connection after {OPEN_TIMEOUT_S:g} s"
            raise OpenFailed("network", message, retryable=True) from None
        self._conn = conn
        self._uncertain.clear()
        self._sent = 0
        self._acked_samples = 0
        self._catch_up = (time.monotonic(), 0) if self._log else None

    async def _roll(self) -> None:
        try:
            await self._swap()
        finally:
            self._roll_due = False

    async def _swap(self) -> None:
        self.commit()
        target = next((item for item in reversed(self._log) if isinstance(item, _Point)), None)
        if target is not None:
            while self._lost is None and target in self._log[self._sent :]:
                batch = self._ready()
                assert batch is not None
                await self._send(*batch)
            try:
                async with asyncio.timeout(ROLLOVER_ACK_WAIT_S):
                    while self._lost is None and target in self._log:
                        self._progress.clear()
                        await self._progress.wait()
            except TimeoutError:
                log.warning("no answer to the rollover commit; replaying its audio")
        if self._lost is not None:
            return
        await self._close_conn()
        await self._reconnect(cooldown=False)

    async def _close_conn(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            # Shielded: a cancelled caller must not leave the connection half closed.
            await asyncio.shield(conn.close())
        self._accept_id = 0

    def _on_event(self, event: BackendEvent) -> None:
        if not self._alive or event.connection_id != self._accept_id:
            return
        match event:
            case Partial():
                self._on_partial(event.text)
            case Committed():
                self._acknowledge(event)
            case Failed() if not event.fatal:
                self._on_error(event.code, event.message)
            case Failed():
                self._connection_lost(event)
            case Closed():
                self._connection_lost(
                    Failed(event.connection_id, "network", "the backend closed", retryable=True)
                )

    def _connection_lost(self, failure: Failed) -> None:
        if self._conn is None or self._conn.connection_id != failure.connection_id:
            return  # closed by us
        if self._lost is None:
            self._lost = failure
        self._wake.set()
        self._progress.set()

    def _acknowledge(self, event: Committed) -> None:
        if event.commit_id is None and event.audio_end_s <= (
            self._acked_samples / self._timeline.sample_rate
        ):
            # No safe audio cut: keep the text private until a known boundary covers it.
            # Reconnection discards it and replays all of that audio instead.
            self._uncertain.append(event)
            return
        final = False
        if event.commit_id is not None:
            self._failures = 0
            point = next((i for i, item in enumerate(self._log) if isinstance(item, _Point)), None)
            if point is not None and point < self._sent:
                answered = self._log[point]
                assert isinstance(answered, _Point)
                final = answered.final
                self._drop(point + 1)
        else:
            self._drop_through(round(event.audio_end_s * self._timeline.sample_rate))
        self._publish_uncertain()
        if event.text.strip():
            self._context = trim_previous_text(f"{self._context} {event.text}", CONTEXT_CHARS) or ""
        self._progress.set()
        self._on_committed(event, final)

    def _publish_uncertain(self) -> None:
        pending, self._uncertain = self._uncertain, []
        for event in pending:
            if event.text.strip():
                self._context = (
                    trim_previous_text(f"{self._context} {event.text}", CONTEXT_CHARS) or ""
                )
            self._on_committed(event, False)

    def _drop(self, count: int) -> None:
        for item in self._log[:count]:
            if isinstance(item, Outgoing):
                self._acked_samples += item.frame.samples.size
        del self._log[:count]
        self._sent -= count

    def _drop_through(self, offset: int) -> None:
        """Drop sent audio a provider commit covered, up to the first unanswered commit."""
        count = 0
        acked = self._acked_samples
        for item in self._log[: self._sent]:
            if isinstance(item, _Point) or acked + item.frame.samples.size > offset:
                break
            acked += item.frame.samples.size
            count += 1
        self._drop(count)
