"""The session's backend stream across connections: rollover, reconnect and replay.

Audio and commits go into a log in order, and one task sends them to the current connection.
A commit's answer drops the log up to that commit before the session hears of it, so the log
holds exactly what no connection has answered yet. When a connection ends, the next one gets the
whole log first, paced, then live audio; audio that comes meanwhile waits in the log. A
connection rolls over before the backend's time limit: at a pause near the limit, or mid-speech
when closer still. A suspended link has no connection; its log keeps growing, as int16. An answer
with text but no word times gets times estimated from the speech in the audio it covers.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace

import numpy as np

from ..backends.base import (
    PLAYBACK_LOOKAHEAD_S,
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
from .link_log import Audio, LinkLog, Point, unpacked
from .pause_gate import Outgoing, SpeechLog
from .reference import PlaybackHistory, reference
from .timeline import Timeline
from .word_times import with_times

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
        speech: SpeechLog | None = None,
        playback: PlaybackHistory | None = None,
    ) -> None:
        """`on_committed(event, final)`: `final` when it answers the commit queued with `final`.
        `on_error` gets non-fatal problems; `on_fatal` is called once, when the link gives up.
        `speech`: where the queued audio's speech is, for answers without word times.
        `playback`: what the speakers played, sent with each commit as its `Reference`."""
        self._backend = backend
        self._options = options
        self._timeline = timeline
        self._chunk = chunk_samples
        self._on_committed = on_committed
        self._on_partial = on_partial
        self._on_error = on_error
        self._on_fatal = on_fatal
        self._speech = speech
        self._playback = playback
        limit = backend.capabilities.max_connection_s
        self._rollover_s = None
        if limit is not None:
            self._rollover_s = (limit - SOFT_ROLLOVER_MARGIN_S, limit - HARD_ROLLOVER_MARGIN_S)
        self._log = LinkLog(timeline.sample_rate)
        self._conn: Connection | None = None
        self._accept_id = 0
        """The connection whose events count; still the old one while it closes."""
        self._next_id = 1
        self._opened_at = 0.0
        self._lost: Failed | None = None
        self._roll_due = False
        self._catch_up: tuple[float, int] | None = None
        """(start, samples sent) while the log is replayed onto a new connection."""
        self._replay_samples = 0
        self._failures = 0
        self._context = options.previous_text or ""
        self._uncertain: list[Committed] = []
        self._alive = True
        self._suspended = False
        self._reopen = False
        """Open a connection even with nothing to send (after a resume)."""
        self._opened: asyncio.Future[None] | None = None
        self._closed_for_suspend = asyncio.Event()
        self._closed_for_suspend.set()
        self.sent_samples = 0
        """Audio sent on every connection so far; grows during replay too."""
        self._wake = asyncio.Event()
        self._progress = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def open(self) -> None:
        """Open the first connection and start sending. Raises `OpenFailed`."""
        await self._open()
        self._start()

    def _start(self) -> None:
        self._task = asyncio.create_task(self._run())
        self._task.add_done_callback(self._run_done)

    def queue(self, out: Outgoing) -> None:
        if self._log.add(out, pack=self._suspended):
            self._wake.set()

    @property
    def suspended(self) -> bool:
        return self._suspended

    @property
    def alive(self) -> bool:
        return self._alive

    async def suspend(self) -> None:
        """Close the connection and send nothing until `resume`; the log keeps growing."""
        if not self._alive or self._suspended:
            return
        self._suspended = True
        self._progress.set()
        self._closed_for_suspend.clear()
        try:
            task, self._task = self._task, None
            if task is not None:
                task.cancel()
                await asyncio.wait([task])
            self._lost, self._roll_due, self._catch_up, self._reopen = None, False, None, False
            await self._close_conn()
        finally:
            self._closed_for_suspend.set()

    async def resume(self) -> bool:
        """Open a new connection, which gets the log first. True once it is open; False when
        the link failed, closed or was suspended again first."""
        # The old connection must be gone first: the provider may count it, and closing it
        # resets which connection's events are accepted.
        await self._closed_for_suspend.wait()
        if not self._alive or not self._suspended:
            return False
        self._suspended = False
        self._progress.set()
        self._reopen = True
        self._opened = opened = asyncio.get_running_loop().create_future()
        self._start()
        assert self._task is not None
        await asyncio.wait([opened, self._task], return_when=asyncio.FIRST_COMPLETED)
        return opened.done()

    def fail(self, code: str, message: str) -> None:
        """Give up for a reason found outside the link."""
        if not self._alive:
            return
        self._alive = False
        self._publish_uncertain()
        self._on_fatal(code, message)
        self._progress.set()

    @property
    def backlog_s(self) -> float:
        """Audio queued that no commit answer has covered yet."""
        return self._log.samples / self._timeline.sample_rate

    async def wait_backlog(self, max_s: float, stall_s: float) -> None:
        """Wait until the backlog is at most `max_s`, or the link has failed. Raises
        TimeoutError when the backend answers nothing for `stall_s` meanwhile."""
        while self._alive and self.backlog_s > max_s:
            self._progress.clear()
            if self._suspended:
                await self._progress.wait()
                continue
            async with asyncio.timeout(stall_s):
                await self._progress.wait()

    def commit(self, *, final: bool = False) -> None:
        """Ask for the text of the audio queued since the last commit. With none queued, `final`
        marks the last unanswered commit instead, and nothing is sent."""
        if self._log.commit(final):
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
        while self._alive and (self._lost is not None or self._log.points):
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
                    if self._log or self._reopen:
                        await self._reconnect(cooldown=not self._reopen)
                    else:
                        await self._wake.wait()
                elif self._roll_due:
                    await self._roll()
                elif (batch := self._log.ready(self._chunk)) is not None:
                    await self._send(*batch)
                else:
                    # Same step as seeing nothing left to send: no live audio can slip past replay.
                    self._catch_up = None
                    await self._wake.wait()
        except _Fatal as exc:
            self.fail(exc.code, exc.message)
            await self._close_conn()

    def _run_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() or (error := task.exception()) is None:
            return
        log.error("backend link failed", exc_info=error)
        self.fail("internal", f"the daemon failed while talking to the backend: {error}")

    async def _send(self, audio: list[Outgoing], point: Point | None) -> None:
        conn = self._conn
        assert conn is not None
        if audio:
            for out in audio:
                start, size = out.frame.start, out.frame.samples.size
                self._timeline.sent(conn.connection_id, start, size, cut=out.cut)
            self._log.sent += len(audio)
            samples = np.concatenate([out.frame.samples for out in audio])
            await self._bounded(conn, conn.send_audio(samples))
            self.sent_samples += samples.size
            await self._pace(samples.size)
        if point is not None:
            if self._playback is not None and point.reference is None:
                if chunk := self._log.chunk_before(self._log.sent):
                    await self._playback.cover(chunk[-1].frame.end + PLAYBACK_LOOKAHEAD_S)
                # An answer during the wait drops log items before this one.
                chunk = self._log.chunk_before(self._log.sent)
                point.reference = reference(chunk, self._playback)
            self._log.sent += 1  # before the await: a backend may answer inside commit()
            await self._bounded(conn, conn.commit(point.reference))

    async def _bounded(self, conn: Connection, op: Awaitable[object]) -> None:
        try:
            async with asyncio.timeout(SEND_TIMEOUT_S):
                await op
        except TimeoutError:
            if self._lost is None:
                message = f"the backend took no data for {SEND_TIMEOUT_S:g} s"
                self._lost = Failed(conn.connection_id, "network", message, retryable=True)

    async def _pace(self, samples: int) -> None:
        if self._catch_up is None or not self._backend.capabilities.paced_replay:
            return
        started, sent = self._catch_up
        if sent >= self._replay_samples:
            return  # past the log as it was at the open: newer audio isn't held back
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
        self._reopen = False
        if self._opened is not None and not self._opened.done():
            self._opened.set_result(None)
        self._uncertain.clear()
        self._log.restart()
        self._catch_up = (time.monotonic(), 0) if self._log else None
        self._replay_samples = self._log.samples

    async def _roll(self) -> None:
        try:
            await self._swap()
        finally:
            self._roll_due = False

    async def _swap(self) -> None:
        self.commit()
        target = self._log.last_point()
        if target is not None:
            while self._lost is None and self._log.unsent(target):
                batch = self._log.ready(self._chunk)
                assert batch is not None
                await self._send(*batch)
            try:
                async with asyncio.timeout(ROLLOVER_ACK_WAIT_S):
                    while self._lost is None and target in self._log.items:
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
            self._log.acked_samples / self._timeline.sample_rate
        ):
            # No safe audio cut: keep the text private until a known boundary covers it.
            # Reconnection discards it and replays all of that audio instead.
            self._uncertain.append(event)
            return
        final = False
        offset = self._log.acked_samples
        answered: list[Audio] = []
        if event.commit_id is not None:
            self._failures = 0
            if (point := self._log.answer_point()) is not None:
                final, answered = point
        else:
            answered = self._log.answer_through(
                round(event.audio_end_s * self._timeline.sample_rate)
            )
        if self._speech is not None and answered:
            frames = [out.frame for out in unpacked(answered)]
            event = with_times(event, frames, self._speech.between, offset)
            self._speech.forget_before(frames[-1].end)
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

    @property
    def queued_since(self) -> float | None:
        """Start of the oldest audio in the log, which may still be sent or replayed."""
        return self._log.queued_since
