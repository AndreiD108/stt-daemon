"""One session: capture → VAD → pause gate → backend → segments, until stop, cancel or failure.

Everything runs on the event loop. One task reads the capture attachment; another commits on the
backend's cadence. Audio and commits go to a `Link`, which sends them in order and survives the
backend's connections ending.

A `transcribe` session reads a `ByteStreamCapture` on audio time: it commits at pauses and at a
chunk cap instead of on a wall-clock cadence, and its `stop` means the end of input.

A non-`mic` session can be suspended for a dictation: its link drops its connection and keeps
the audio; on resume a new connection gets all of it.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import replace
from typing import Any, Literal

import numpy as np

from ..audio.capture import CaptureSource
from ..audio.frames import Attachment, Frame, epoch_ms
from ..audio.stream_capture import ByteStreamCapture, DecodeFailed
from ..audio.vad import WINDOW, SpeechEnded, StreamingVAD
from ..backends.base import Backend, CommitPolicy, Committed, OpenFailed
from ..config import Config
from .events import (
    EndReason,
    PartialText,
    PreemptedBy,
    Progress,
    SessionEnded,
    SessionError,
    SessionEvent,
    SessionKind,
    SessionResumed,
    SessionStarted,
    SessionSuspended,
)
from .moving import while_moving
from .options import MarkerRefused, SessionOptions
from .pause_gate import Outgoing, PauseGate
from .recovery import Link
from .segments import Segments
from .timeline import Timeline

log = logging.getLogger(__name__)

_END_OF_AUDIO = Frame(np.zeros(0, dtype=np.float32), 0.0, 16000)

type State = Literal["new", "running", "stopping", "ending", "ended"]

PROGRESS_EVERY_S = 1.0
MIN_BACKLOG_S = 60.0
"""A `transcribe` session takes no more input while this much audio waits for the backend
(or twice its chunk cap, if more)."""
STALL_S = 60.0
"""A `transcribe` session at that limit fails when its backend answers nothing for this long."""


class Session:
    def __init__(
        self,
        *,
        config: Config,
        capture: CaptureSource,
        backend: Backend,
        options: SessionOptions,
        emit: Callable[[SessionEvent], None],
        kind: SessionKind = "mic",
    ) -> None:
        self.session_id = uuid.uuid4().hex
        self.kind: SessionKind = kind
        self.started: SessionStarted | None = None
        self.audio_end = 0.0
        """Monotonic end of the last captured audio the session took in."""
        self._config = config
        self._capture = capture
        self._backend = backend
        if kind == "mic":
            options = replace(options, open=replace(options.open, dictation=True))
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
        self._segments = Segments(
            auto_language=options.open.language == "auto",
            word_timestamps=options.word_timestamps,
        )
        self._markers = self._segments.markers
        self._timeline = Timeline(0.0, rate)
        self._link: Link | None = None
        self._attachment: Attachment | None = None
        self.make_room: Callable[[], Coroutine[Any, Any, bool]] | None = None
        """Suspends another session on the backend when the provider refuses one more
        connection (`concurrency_limit`); True when it did, and the open is tried once more."""
        self._suspended_at: float | None = None
        self._was_suspended = False
        self._speech_since_commit = False
        self._samples_since_commit = 0
        self._last_commit_at = 0.0
        self._audio_task: asyncio.Task[None] | None = None
        self._commit_task: asyncio.Task[None] | None = None
        self._ending_task: asyncio.Task[None] | None = None
        self._stop_failure: SessionError | None = None
        self._finishing = False
        self._end_put: asyncio.Future[None] | None = None
        self._ended: asyncio.Future[SessionEnded] = asyncio.get_running_loop().create_future()
        policy = backend.capabilities.commit_policy
        self._at_pauses = policy.at_pauses
        self._chunk_cap_s = policy.max_chunk_s
        self._stream: ByteStreamCapture | None = None
        self._last_progress = 0.0
        if kind == "transcribe":
            assert isinstance(capture, ByteStreamCapture)
            self._stream = capture
            capture.on_failed = lambda message: self._fail("invalid_request", message)
            self._at_pauses = True
            self._chunk_cap_s = policy.max_chunk_s or policy.interval_s
        self._max_backlog_s = max(MIN_BACKLOG_S, 2 * (self._chunk_cap_s or 0))

    @property
    def state(self) -> State:
        return self._state

    @property
    def backend_name(self) -> str:
        return self._backend.name

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
        self._segments.timeline = self._timeline
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
            try:
                await self._link.open()
            except OpenFailed as exc:
                if exc.code != "concurrency_limit" or self.make_room is None:
                    raise
                if not await self.make_room():
                    raise
                await self._link.open()
        except BaseException:
            attachment.detach()
            await self._capture.release()
            raise
        started_at = epoch_ms(origin)
        self.started = SessionStarted(
            self.session_id,
            self._backend.name,
            self._options.open.language,
            started_at,
            self._options.open.previous_text,
            self.kind,
        )
        self._state = "running"
        self._last_commit_at = time.monotonic()
        self._audio_task = asyncio.create_task(self._audio_loop())
        self._audio_task.add_done_callback(self._audio_loop_done)
        policy = self._backend.capabilities.commit_policy
        if policy.interval_s is not None and self._stream is None:
            self._commit_task = asyncio.create_task(self._commit_loop(policy))
        self._emit(self.started)
        return self.started

    def add_marker(
        self, marker: str, at: int | None = None, *, arrived: float | None = None
    ) -> int:
        """Accept a marker; `at` is epoch ms, no later than `arrived` (monotonic, when the
        command came; default now), which is also the default. Returns its session ms.

        On audio time (`transcribe`), `at` is `started_at` plus audio ms, and the default is the
        end of the audio taken in so far."""
        if self._state != "running" or self.started is None:
            if self._state == "new":
                raise MarkerRefused("no_session", "the session hasn't started; wait for it")
            raise MarkerRefused(
                "session_stopping", "the session is ending and takes no more markers"
            )
        if self._stream is not None:
            arrived_ms = self._timeline.to_ms(self.audio_end)
            at_ms = arrived_ms if at is None else at - self.started.started_at
        else:
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
        """Drop what isn't committed and end; segments already sent stay in `final_text`.
        A stopping `transcribe` session is cut short too: its stop can take minutes."""
        if self._state == "running":
            self._state = "ending"
            self._markers.clear()
            self._ending_task = asyncio.create_task(self._end(reason))
        elif self._state == "stopping" and self._stream is not None and not self._finishing:
            assert self._ending_task is not None
            self._state = "ending"
            self._markers.clear()
            self._ending_task = asyncio.create_task(self._abandon(self._ending_task, reason))
        return await self.wait_ended()

    async def wait_ended(self) -> SessionEnded:
        if self._state == "new":
            raise RuntimeError("the session hasn't started")
        return await asyncio.shield(self._ended)

    @property
    def suspendable(self) -> bool:
        link = self._link
        return (
            self.kind != "mic"
            and self._state in ("running", "stopping")
            and not self._finishing
            and link is not None
            and link.alive
            and not link.suspended
        )

    @property
    def suspended(self) -> bool:
        return self._link is not None and self._link.suspended

    async def suspend(self, by: PreemptedBy) -> bool:
        """Drop the backend connection and keep the audio until `resume`. False when the
        session is past suspending (ending, or its link failed)."""
        if not self.suspendable:
            return False
        assert self._link is not None
        await self._link.suspend()
        if not self._link.suspended or self._finishing:
            return False
        self._suspended_at = time.monotonic()
        self._was_suspended = True
        self._emit(SessionSuspended(by))
        return True

    async def resume(self) -> bool:
        """Reconnect; the new connection gets the audio kept meanwhile. False when the session
        ended or failed first."""
        link = self._link
        if link is None or not link.suspended or not await link.resume():
            return False
        if self._state in ("running", "stopping") and self._suspended_at is not None:
            self._emit(SessionResumed(round(time.monotonic() - self._suspended_at, 3)))
        return True

    def give_up(self, code: str, message: str) -> None:
        """End with a fatal error found outside the session (its backend is gone)."""
        if self._link is not None:
            self._link.fail(code, message)

    async def _audio_loop(self) -> None:
        assert self._attachment is not None
        queue = self._attachment.queue
        while (frame := await queue.get()) is not _END_OF_AUDIO:
            await self._take(frame)
            if self._stream is not None:
                assert self._link is not None
                self._report_progress(final=False)
                try:
                    await self._link.wait_backlog(self._max_backlog_s, STALL_S)
                except TimeoutError:
                    # Else the input waits on this session for good, and the client's commands
                    # wait behind its audio.
                    message = f"the backend answered nothing for {STALL_S:g} s"
                    self._on_fatal("backend_error", message)
                    return

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
        for part in self._gate.push(frame, events):
            self._queue(part)
        paused = any(isinstance(e, SpeechEnded) for e in events)
        if self._at_pauses and paused and self._speech_since_commit:
            self._commit()
        cap = self._chunk_cap_s
        if cap is not None and self._samples_since_commit >= cap * self._capture.sample_rate:
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

    def _report_progress(self, *, final: bool) -> None:
        now = time.monotonic()
        if final or now - self._last_progress >= PROGRESS_EVERY_S:
            self._last_progress = now
            self._emit(Progress(round(self.audio_end - self._timeline.origin, 3)))

    def _queue(self, out: Outgoing) -> None:
        assert self._link is not None
        self._link.queue(out)
        self._samples_since_commit += out.frame.samples.size
        if self._gate.speaking:
            self._speech_since_commit = True

    def _commit(self, *, final: bool = False) -> None:
        assert self._link is not None
        self._link.commit(final=final)
        self._samples_since_commit = 0
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
        if (segment := self._segments.committed(event, final)) is not None:
            self._emit(segment)

    def _fail(self, code: str, message: str) -> None:
        if self._state != "running":
            return
        self._state = "ending"
        self._ending_task = asyncio.create_task(
            self._end("error", SessionError(code, message, fatal=True))
        )

    async def _stop_and_finish(self, reason: Literal["stopped", "evicted"]) -> None:
        wait_s = self._config.session.stop_wait_ms / 1000
        try:
            if self._stream is not None:
                await while_moving(self._drain_input(), wait_s, self._link_mark)
            elif self._was_suspended:
                await while_moving(self._send_rest(), wait_s, self._link_mark)
            else:
                async with asyncio.timeout(wait_s):
                    await self._send_rest()
        except DecodeFailed as exc:
            await self._stop_intake(drain=False)
            await self._finish("error", SessionError("invalid_request", exc.message, True))
            return
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

    def _link_mark(self) -> tuple[object, ...] | None:
        """Changes as audio is taken, sent or answered; None while suspended."""
        link = self._link
        assert link is not None
        if link.suspended:
            return None
        return (self.audio_end, link.backlog_s, link.sent_samples)

    async def _drain_input(self) -> None:
        assert self._stream is not None
        await self._stream.finish()
        await self._send_rest()

    async def _send_rest(self) -> None:
        await self._stop_intake(drain=True)
        if self._stream is not None:
            self._report_progress(final=True)
        for out in self._gate.flush():
            self._queue(out)
        self._commit(final=True)
        assert self._link is not None
        await self._link.settle()

    async def _end(self, reason: EndReason, error: SessionError | None = None) -> None:
        await self._stop_intake(drain=False)
        await self._finish(reason, error)

    async def _abandon(self, stopping: asyncio.Task[None], reason: EndReason) -> None:
        if not self._finishing:
            stopping.cancel()
        await asyncio.wait([stopping])
        if not self._finishing:
            await self._end(reason)

    async def _stop_intake(self, *, drain: bool) -> None:
        """Stop taking audio; with `drain`, first take what capture delivered before now."""
        assert self._attachment is not None
        assert self._audio_task is not None
        self._attachment.detach()
        if drain:
            # A task, not put_nowait: a stream's queue is bounded. It runs after the frames
            # capture already scheduled.
            self._end_put = asyncio.ensure_future(self._attachment.queue.put(_END_OF_AUDIO))
        elif self._audio_task is not asyncio.current_task():
            self._audio_task.cancel()
            if self._end_put is not None:
                self._end_put.cancel()
        await asyncio.wait([self._audio_task])
        if self._commit_task is not None:
            self._commit_task.cancel()

    async def _finish(self, reason: EndReason, error: SessionError | None = None) -> None:
        self._finishing = True
        if self._link is not None:
            await self._link.close()
        if (segment := self._segments.leftover(self.audio_end)) is not None:
            self._emit(segment)
        if error is not None:
            self._emit(error)
        await self._capture.release()
        transcript = self._segments.transcript
        ended = SessionEnded(
            reason,
            transcript.final_text,
            self._timeline.to_ms(self.audio_end),
            len(transcript.segments),
            plain_text=transcript.plain_text,
        )
        self._state = "ended"
        self._ended.set_result(ended)
        self._emit(ended)
