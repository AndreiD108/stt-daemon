"""One session: capture → VAD → pause gate → backend → segments, until stop, cancel or failure.

Everything runs on the event loop. One task reads the capture attachment; another commits on the
backend's cadence. Audio and commits go to a `Link`, which sends them in order and survives the
backend's connections ending.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..audio.capture import CaptureSource
from ..audio.frames import Attachment, Frame
from ..audio.vad import WINDOW, SpeechEnded, StreamingVAD
from ..backends.base import Backend, CommitPolicy, Committed, OpenOptions
from ..config import Config
from .events import EndReason, PartialText, SessionEnded, SessionError, SessionEvent, SessionStarted
from .markers import MarkerQueue, TimedWord, place
from .pause_gate import Outgoing, PauseGate
from .recovery import Link
from .timeline import Timeline
from .transcript import Transcript

log = logging.getLogger(__name__)

_END_OF_AUDIO = Frame(np.zeros(0, dtype=np.float32), 0.0, 16000)

type State = Literal["new", "running", "stopping", "ending", "ended"]


class MarkerRefused(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SessionOptions:
    open: OpenOptions
    """`language` already resolved: segments carry the detected language only for "auto"."""
    word_timestamps: bool = False


def start_point(requested_at: float, pre_start_ms: int, previous_end: float | None) -> float:
    """Where a session started by command takes its audio from: `pre_start_ms` before the
    command, but not before the previous session's audio ended (the ring still holds it)."""
    since = requested_at - pre_start_ms / 1000
    return since if previous_end is None else max(since, previous_end)


class Session:
    def __init__(
        self,
        *,
        config: Config,
        capture: CaptureSource,
        backend: Backend,
        options: SessionOptions,
        emit: Callable[[SessionEvent], None],
    ) -> None:
        self.session_id = uuid.uuid4().hex
        self.started: SessionStarted | None = None
        self.audio_end = 0.0
        """Monotonic end of the last captured audio the session took in."""
        self._config = config
        self._capture = capture
        self._backend = backend
        self._options = options
        self._emit = emit
        self._state: State = "new"
        rate = capture.sample_rate
        self._vad = StreamingVAD(
            sample_rate=rate,
            threshold=config.vad.threshold,
            min_speech_ms=config.vad.min_speech_ms,
            min_silence_ms=config.vad.min_silence_ms,
        )
        pad_s = config.vad.pause_keep_ms / 1000
        self._gate = PauseGate(
            hold_s=self._vad.start_delay_s + pad_s + 2 * WINDOW / rate, pad_s=pad_s
        )
        self._markers = MarkerQueue()
        self._transcript = Transcript()
        self._timeline = Timeline(0.0, rate)
        self._link: Link | None = None
        self._attachment: Attachment | None = None
        self._boundary_s: dict[int, float] = {}
        self._speech_since_commit = False
        self._last_commit_at = 0.0
        self._last_speech_at: float | None = None
        self._audio_task: asyncio.Task[None] | None = None
        self._commit_task: asyncio.Task[None] | None = None
        self._ending_task: asyncio.Task[None] | None = None
        self._stop_failure: SessionError | None = None
        self._ended: asyncio.Future[SessionEnded] = asyncio.get_running_loop().create_future()

    @property
    def state(self) -> State:
        return self._state

    @property
    def last_speech_at(self) -> float | None:
        """Monotonic time speech was last heard; the latest audio while speech goes on."""
        return self.audio_end if self._vad.speech_id is not None else self._last_speech_at

    async def start(
        self, *, since: float | None = None, attachment: Attachment | None = None
    ) -> SessionStarted:
        """Take audio from `since` (monotonic; see `start_point`), or from `attachment`, made
        earlier by a caller that also took a capture hold; the session releases that hold.

        Raises `CaptureUnavailable` or `OpenFailed`; nothing is emitted then.
        """
        if self._state != "new":
            raise RuntimeError("a session starts once")
        if attachment is None:
            await self._capture.hold()
            try:
                attachment = self._capture.attach(since)
            except BaseException:
                await self._capture.release()
                raise
        self._attachment = attachment
        origin = attachment.started_at
        self._timeline = Timeline(origin, self._capture.sample_rate)
        self.audio_end = origin
        self._link = Link(
            backend=self._backend,
            options=self._options.open,
            timeline=self._timeline,
            chunk_samples=self._capture.sample_rate * self._config.audio.chunk_ms // 1000,
            on_committed=self._on_committed,
            on_partial=self._on_partial,
            on_error=self._on_error,
            on_fatal=self._on_fatal,
        )
        try:
            await self._link.open()
        except BaseException:
            attachment.detach()
            await self._capture.release()
            raise
        started_at = round((time.time() - (time.monotonic() - origin)) * 1000)
        self.started = SessionStarted(
            self.session_id,
            self._backend.name,
            self._options.open.language,
            started_at,
            self._options.open.previous_text,
        )
        self._state = "running"
        self._last_commit_at = time.monotonic()
        self._audio_task = asyncio.create_task(self._audio_loop())
        self._audio_task.add_done_callback(self._audio_loop_done)
        policy = self._backend.capabilities.commit_policy
        if policy.interval_s is not None:
            self._commit_task = asyncio.create_task(self._commit_loop(policy))
        self._emit(self.started)
        return self.started

    def add_marker(
        self, marker: str, at: int | None = None, *, arrived: float | None = None
    ) -> int:
        """Accept a marker; `at` is epoch ms, no later than `arrived` (monotonic, when the
        command came; default now), which is also the default. Returns its session ms."""
        if self._state != "running" or self.started is None:
            if self._state == "new":
                raise MarkerRefused("no_session", "the session hasn't started; wait for it")
            raise MarkerRefused(
                "session_stopping", "the session is ending and takes no more markers"
            )
        arrived_ms = self._timeline.to_ms(time.monotonic() if arrived is None else arrived)
        at_ms = arrived_ms if at is None else min(at - self.started.started_at, arrived_ms)
        if at_ms < 0:
            raise MarkerRefused(
                "out_of_range", "the marker's time is before the session's first audio"
            )
        self._markers.add(marker, at_ms)
        return at_ms

    async def stop(self, reason: Literal["stopped", "evicted"] = "stopped") -> SessionEnded:
        """Send what is left, wait a bounded time for its text, end."""
        if self._state == "running":
            self._state = "stopping"
            self._ending_task = asyncio.create_task(self._stop_and_finish(reason))
        return await self.wait_ended()

    async def cancel(
        self, reason: Literal["cancelled", "client_gone"] = "cancelled"
    ) -> SessionEnded:
        """Drop what isn't committed and end; segments already sent stay in `final_text`."""
        if self._state == "running":
            self._state = "ending"
            self._markers.clear()
            self._ending_task = asyncio.create_task(self._end(reason))
        return await self.wait_ended()

    async def wait_ended(self) -> SessionEnded:
        if self._state == "new":
            raise RuntimeError("the session hasn't started")
        return await asyncio.shield(self._ended)

    async def _audio_loop(self) -> None:
        assert self._attachment is not None
        queue = self._attachment.queue
        while (frame := await queue.get()) is not _END_OF_AUDIO:
            await self._take(frame)

    def _audio_loop_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() or (error := task.exception()) is None:
            return
        log.error("session %s: audio loop failed", self.session_id, exc_info=error)
        self._fail("internal", f"the daemon failed while sending audio: {error}")

    async def _take(self, frame: Frame) -> None:
        # With an empty ring, the first live frame began before the session's first sample.
        if frame.start < self._timeline.origin:
            part = frame.since(self._timeline.origin)
            if part is None:
                return
            frame = part
        self.audio_end = frame.end
        events = self._vad.push(frame)
        for event in events:
            self._last_speech_at = event.at
        for part in self._gate.push(frame, events):
            self._queue(part)
        policy = self._backend.capabilities.commit_policy
        paused = any(isinstance(e, SpeechEnded) for e in events)
        if policy.at_pauses and paused and self._speech_since_commit:
            self._commit()
        assert self._link is not None
        self._link.tick(speaking=self._vad.speech_id is not None)

    async def _commit_loop(self, policy: CommitPolicy) -> None:
        assert policy.interval_s is not None
        while True:
            await asyncio.sleep(policy.check_every_s or policy.interval_s)
            due = time.monotonic() - self._last_commit_at >= policy.interval_s
            if due and self._speech_since_commit:
                self._commit()

    def _queue(self, out: Outgoing) -> None:
        assert self._link is not None
        self._link.queue(out)
        if self._gate.speaking:
            self._speech_since_commit = True

    def _commit(self, *, final: bool = False) -> None:
        assert self._link is not None
        self._link.commit(final=final)
        self._speech_since_commit = False
        self._last_commit_at = time.monotonic()

    def _on_partial(self, text: str) -> None:
        if self._options.open.partials and self._state in ("running", "stopping"):
            self._emit(PartialText(text))

    def _on_error(self, code: str, message: str) -> None:
        if self._state in ("running", "stopping"):
            self._emit(SessionError(code, message, fatal=False))

    def _on_fatal(self, code: str, message: str) -> None:
        """The link gave up: the backend failed for good, or reconnecting didn't work."""
        if self._state == "running":
            self._fail(code, message)
        elif self._state == "stopping" and self._stop_failure is None:
            self._stop_failure = SessionError(code, message, fatal=True)

    def _on_committed(self, event: Committed, final: bool) -> None:
        if self._state not in ("running", "stopping"):
            return  # cancelled or failed: uncommitted text is dropped
        tl, conn = self._timeline, event.connection_id
        boundary = self._boundary_s.get(conn, 0.0)
        end_s = max(event.audio_end_s, boundary)
        self._boundary_s[conn] = end_s
        start_ms = tl.to_ms(tl.to_capture(conn, boundary, "start"))
        end_ms = max(tl.to_ms(tl.to_capture(conn, end_s, "end")), start_ms)
        words: tuple[TimedWord, ...] | None = None
        if event.words is not None:
            words = tuple(self._timed(conn, w.text, w.start_s, w.end_s) for w in event.words)
        markers = self._markers.take(None if final else end_ms)
        text, placed = place(event.text, words, (start_ms, end_ms), markers)
        segment = self._transcript.add(
            text,
            plain=event.text,
            language=event.language if self._options.open.language == "auto" else None,
            span=(start_ms, end_ms),
            words=words if self._options.word_timestamps else None,
            markers=tuple(placed),
        )
        if segment is not None:
            self._emit(segment)

    def _timed(self, conn: int, text: str, start_s: float, end_s: float) -> TimedWord:
        tl = self._timeline
        start = tl.to_ms(tl.to_capture(conn, start_s, "start"))
        end = tl.to_ms(tl.to_capture(conn, end_s, "end"))
        return TimedWord(text, start, max(end, start))

    def _fail(self, code: str, message: str) -> None:
        if self._state != "running":
            return
        self._state = "ending"
        self._ending_task = asyncio.create_task(
            self._end("error", SessionError(code, message, fatal=True))
        )

    async def _stop_and_finish(self, reason: Literal["stopped", "evicted"]) -> None:
        try:
            async with asyncio.timeout(self._config.session.stop_wait_ms / 1000):
                await self._stop_intake(drain=True)
                for out in self._gate.flush():
                    self._queue(out)
                self._commit(final=True)
                assert self._link is not None
                await self._link.settle()
        except TimeoutError:
            log.warning("session %s: stopped before the backend's last text came", self.session_id)
            await self._stop_intake(drain=False)
        except Exception as exc:
            log.exception("session %s: stopping failed", self.session_id)
            self._stop_failure = self._stop_failure or SessionError(
                "internal", f"the daemon failed while stopping: {exc}", fatal=True
            )
            await self._stop_intake(drain=False)
        if self._stop_failure is not None:
            await self._finish("error", self._stop_failure)
        else:
            await self._finish(reason)

    async def _end(self, reason: EndReason, error: SessionError | None = None) -> None:
        await self._stop_intake(drain=False)
        await self._finish(reason, error)

    async def _stop_intake(self, *, drain: bool) -> None:
        """Stop taking audio; with `drain`, first take what capture delivered before now."""
        assert self._attachment is not None
        assert self._audio_task is not None
        self._attachment.detach()
        if drain:
            asyncio.get_running_loop().call_soon(self._attachment.queue.put_nowait, _END_OF_AUDIO)
        elif self._audio_task is not asyncio.current_task():
            self._audio_task.cancel()
        await asyncio.wait([self._audio_task])
        if self._commit_task is not None:
            self._commit_task.cancel()

    async def _finish(self, reason: EndReason, error: SessionError | None = None) -> None:
        if self._link is not None:
            await self._link.close()
        if self._markers:
            at = self._timeline.to_ms(self.audio_end)
            text, placed = place("", (), (at, at), self._markers.take())
            segment = self._transcript.add(
                text,
                plain="",
                language=None,
                span=(at, at),
                words=() if self._options.word_timestamps else None,
                markers=tuple(placed),
            )
            if segment is not None:
                self._emit(segment)
        if error is not None:
            self._emit(error)
        await self._capture.release()
        ended = SessionEnded(
            reason,
            self._transcript.final_text,
            self._timeline.to_ms(self.audio_end),
            len(self._transcript.segments),
            plain_text=self._transcript.plain_text,
        )
        self._state = "ended"
        self._ended.set_result(ended)
        self._emit(ended)
