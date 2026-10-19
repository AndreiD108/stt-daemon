"""One session: its tracks (capture → VAD → pause gate → backend), until stop, cancel or failure.

Everything runs on the event loop. Each track reads its capture attachment and sends audio and
commits to its own `Link`, which survives the backend's connections ending; segments of all
tracks go into one transcript.

A `transcribe` session reads a `ByteStreamCapture`, and its `stop` means the end of input.

A `system` session has a `system` track (a `SystemCapture`) and, optionally, a `mic` track on
the shared mic or its echo-cancelled copy. A track whose capture stops goes silent and is
reported; only a backend failure, on either track, ends the session.

A non-`mic` session can be suspended for a dictation: its links drop their connections and keep
the audio; on resume new connections get all of it.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import replace
from typing import Any, Literal

from ..audio.aec_pipewire import EchoCancelledMic
from ..audio.capture import CaptureSource, CaptureUnavailable
from ..audio.frames import Attachment, epoch_ms
from ..audio.stream_capture import ByteStreamCapture, DecodeFailed
from ..audio.system_capture import SystemCapture
from ..backends.base import Backend, Committed, OpenFailed
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
from .markers import MarkerQueue
from .moving import while_moving
from .options import MarkerRefused, SessionOptions
from .segments import Segments
from .tracks import Track
from .transcript import Transcript

log = logging.getLogger(__name__)

type State = Literal["new", "running", "stopping", "ending", "ended"]

PROGRESS_EVERY_S = 1.0
MIN_BACKLOG_S = 60.0
"""A `transcribe` session takes no more input while this much audio waits for the backend
(or twice its chunk cap, if more)."""
STALL_S = 60.0
"""A `transcribe` session at that limit fails when its backend answers nothing for this long."""
FIRST_FRAME_WAIT_S = 2.0
"""A `system` session starts once its system track has a frame, or after this long without."""


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
        mic: CaptureSource | None = None,
        session_id: str | None = None,
    ) -> None:
        """`capture` is the first track's source; `mic` adds a `mic` track to a `system`
        session."""
        self.session_id = session_id or uuid.uuid4().hex
        self.kind: SessionKind = kind
        self.started: SessionStarted | None = None
        self._config = config
        self._backend = backend
        if kind == "mic":
            options = replace(options, open=replace(options.open, dictation=True))
        self._options = options
        self._emit = emit
        self._state: State = "new"
        self.make_room: Callable[[], Coroutine[Any, Any, bool]] | None = None
        """Suspends another session on the backend when the provider refuses one more
        connection (`concurrency_limit`); True when it did, and the open is tried once more."""
        self._suspended_at: float | None = None
        self._was_suspended = False
        self._ending_task: asyncio.Task[None] | None = None
        self._stop_failure: SessionError | None = None
        self._start_failure: OpenFailed | None = None
        """A link that gave up before `session_started`: the start fails with it."""
        self._finishing = False
        self._ended: asyncio.Future[SessionEnded] = asyncio.get_running_loop().create_future()
        self._stream: ByteStreamCapture | None = None
        self._last_progress = 0.0
        if kind == "transcribe":
            assert isinstance(capture, ByteStreamCapture)
            self._stream = capture
            capture.on_failed = lambda message: self._fail("invalid_request", message)
        chunk_cap_s = backend.capabilities.commit_policy.max_chunk_s
        self._max_backlog_s = max(MIN_BACKLOG_S, 2 * (chunk_cap_s or 0))
        sources: list[tuple[str | None, CaptureSource]] = [
            ("system" if kind == "system" else None, capture)
        ]
        if mic is not None:
            sources.append(("mic", mic))
        markers, transcript = MarkerQueue(), Transcript()
        self._markers = markers
        self._tracks: list[Track] = []
        for name, source in sources:
            segments = Segments(
                auto_language=options.open.language == "auto",
                word_timestamps=options.word_timestamps,
                track=name,
                markers=markers,
                transcript=transcript,
            )
            self._tracks.append(
                Track(
                    name=name,
                    owner=self,
                    config=config,
                    capture=source,
                    backend=backend,
                    segments=segments,
                )
            )
            if isinstance(source, SystemCapture | EchoCancelledMic):
                source.on_lost = self._capture_lost(name)
                source.on_restored = self._capture_restored(name)
            if isinstance(source, EchoCancelledMic):
                source.on_unavailable = self._aec_unavailable(name)
        for track in self._tracks:
            track.segments.peers = [t.segments for t in self._tracks if t is not track]
        self._first = self._tracks[0]

    @property
    def state(self) -> State:
        return self._state

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def audio_end(self) -> float:
        """Monotonic end of the last captured audio the session took in."""
        return max(track.audio_end for track in self._tracks)

    async def start(
        self, *, since: float | None = None, attachment: Attachment | None = None
    ) -> SessionStarted:
        """Take audio from `since` (monotonic; see `start_point`), or from `attachment`, made
        earlier by a caller that also took a capture hold; the session releases that hold.

        Raises `CaptureUnavailable`, `ToolMissing` or `OpenFailed`; nothing is emitted then.
        """
        if self._state != "new":
            raise RuntimeError("a session starts once")
        first, rest = self._first, self._tracks[1:]
        # The mic track is held before the system capture starts: the mic is open first.
        held: list[Track] = []
        try:
            for track in rest:
                await _hold(track.capture)
                held.append(track)
            if attachment is None:
                await first.capture.hold()
                try:
                    attachment = first.capture.attach(since)
                except BaseException:
                    await first.capture.release()
                    raise
        except BaseException:
            for track in held:
                await track.capture.release()
            raise
        origin = attachment.started_at
        first.begin(attachment, origin, self._options.open)
        begun = [first]
        try:
            for track in rest:
                track.begin(_attach_from(track.capture, origin), origin, self._options.open)
                begun.append(track)
            for track in self._tracks:
                await track.open(self.make_room)
            if isinstance(first.capture, SystemCapture):
                await first.capture.ready(FIRST_FRAME_WAIT_S)
            if self._start_failure is not None:
                raise self._start_failure
        except BaseException:
            for track in begun:
                assert track.link is not None
                await track.link.close()
                track.detach()
                await track.capture.release()
            for track in rest:
                if track not in begun:
                    await track.capture.release()
            raise
        self.started = SessionStarted(
            self.session_id,
            self._backend.name,
            self._options.open.language,
            epoch_ms(origin),
            self._options.open.previous_text,
            self.kind,
            tuple(t.name for t in self._tracks if t.name is not None) or None,
        )
        self._state = "running"
        for track in self._tracks:
            track.run()
        self._emit(self.started)
        for track in self._tracks:
            capture = track.capture
            if isinstance(capture, EchoCancelledMic) and capture.unavailable_reason:
                self._aec_unavailable(track.name)(capture.unavailable_reason)
            if isinstance(capture, SystemCapture | EchoCancelledMic) and capture.lost_reason:
                self._capture_lost(track.name)(capture.lost_reason)
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
        timeline = self._first.timeline
        if self._stream is not None:
            arrived_ms = timeline.to_ms(self.audio_end)
            at_ms = arrived_ms if at is None else at - self.started.started_at
        else:
            arrived_ms = timeline.to_ms(time.monotonic() if arrived is None else arrived)
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
        A stopping non-`mic` session is cut short too: its stop can take minutes."""
        if self._state == "running":
            self._state = "ending"
            self._markers.clear()
            self._ending_task = asyncio.create_task(self._end(reason))
        elif self._state == "stopping" and self.kind != "mic" and not self._finishing:
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
        return (
            self.kind != "mic"
            and self._state in ("running", "stopping")
            and not self._finishing
            and all(
                t.link is not None and t.link.alive and not t.link.suspended for t in self._tracks
            )
        )

    @property
    def suspended(self) -> bool:
        return any(t.link is not None and t.link.suspended for t in self._tracks)

    async def suspend(self, by: PreemptedBy) -> bool:
        """Drop the backend connections and keep the audio until `resume`. False when the
        session is past suspending (ending, or a link failed)."""
        if not self.suspendable:
            return False
        links = [t.link for t in self._tracks if t.link is not None]
        await asyncio.gather(*(link.suspend() for link in links))
        if not all(link.suspended for link in links) or self._finishing:
            return False
        self._suspended_at = time.monotonic()
        self._was_suspended = True
        self._emit(SessionSuspended(by))
        return True

    async def resume(self) -> bool:
        """Reconnect; the new connections get the audio kept meanwhile. False when the session
        ended or failed first."""
        links = [t.link for t in self._tracks if t.link is not None and t.link.suspended]
        if not links or not all(await asyncio.gather(*(link.resume() for link in links))):
            return False
        if self._state in ("running", "stopping") and self._suspended_at is not None:
            self._emit(SessionResumed(round(time.monotonic() - self._suspended_at, 3)))
        return True

    def give_up(self, code: str, message: str) -> None:
        """End with a fatal error found outside the session (its backend is gone)."""
        if self._first.link is not None:
            self._first.link.fail(code, message)

    def track_committed(self, track: Track, event: Committed, final: bool) -> None:
        if self._state not in ("running", "stopping"):
            return  # cancelled or failed: uncommitted text is dropped
        if (segment := track.segments.committed(event, final)) is not None:
            self._emit(segment)

    def track_partial(self, track: Track, text: str) -> None:
        if self._options.open.partials and self._state in ("running", "stopping"):
            self._emit(PartialText(text, track.name))

    def track_error(self, track: Track, code: str, message: str) -> None:
        if self._state in ("running", "stopping"):
            self._emit(SessionError(code, message, fatal=False, track=track.name))

    def track_fatal(self, track: Track, code: str, message: str) -> None:
        """A link gave up: the backend failed for good, or reconnecting didn't work."""
        if self._state == "running":
            self._fail(code, message, track.name)
        elif self._state == "new" and self._start_failure is None:
            self._start_failure = OpenFailed(code, message, retryable=False)
        elif self._state == "stopping" and self._stop_failure is None:
            self._stop_failure = SessionError(code, message, fatal=True, track=track.name)

    def track_failed(self, track: Track, error: BaseException) -> None:
        log.error("session %s: audio loop failed", self.session_id, exc_info=error)
        self._fail("internal", f"the daemon failed while sending audio: {error}", track.name)

    async def after_frame(self, track: Track) -> bool:
        if self._stream is None:
            return True
        assert track.link is not None
        self._report_progress(final=False)
        try:
            await track.link.wait_backlog(self._max_backlog_s, STALL_S)
        except TimeoutError:
            # Else the input waits on this session for good, and the client's commands wait
            # behind its audio.
            message = f"the backend answered nothing for {STALL_S:g} s"
            self.track_fatal(track, "backend_error", message)
            return False
        return True

    def _capture_lost(self, name: str | None) -> Callable[[str], None]:
        def lost(reason: str) -> None:
            if self._state in ("running", "stopping"):
                message = f"the {name} track gets no audio ({reason}); retrying"
                self._emit(SessionError("capture_lost", message, fatal=False, track=name))

        return lost

    def _capture_restored(self, name: str | None) -> Callable[[], None]:
        def restored() -> None:
            if self._state in ("running", "stopping"):
                message = f"the {name} track has audio again"
                self._emit(SessionError("capture_restored", message, fatal=False, track=name))

        return restored

    def _aec_unavailable(self, name: str | None) -> Callable[[str], None]:
        def unavailable(message: str) -> None:
            if self._state in ("running", "stopping"):
                self._emit(SessionError("aec_unavailable", message, fatal=False, track=name))

        return unavailable

    def _report_progress(self, *, final: bool) -> None:
        now = time.monotonic()
        if final or now - self._last_progress >= PROGRESS_EVERY_S:
            self._last_progress = now
            seconds = self.audio_end - self._first.timeline.origin
            self._emit(Progress(round(seconds, 3)))

    def _fail(self, code: str, message: str, track: str | None = None) -> None:
        if self._state != "running":
            return
        self._state = "ending"
        self._ending_task = asyncio.create_task(
            self._end("error", SessionError(code, message, fatal=True, track=track))
        )

    async def _stop_and_finish(self, reason: Literal["stopped", "evicted"]) -> None:
        wait_s = self._config.session.stop_wait_ms / 1000
        try:
            if self._stream is not None:
                await while_moving(self._drain_input(), wait_s, self._mark)
            elif self._was_suspended:
                await while_moving(self._send_rest(), wait_s, self._mark)
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

    def _mark(self) -> tuple[object, ...] | None:
        """Changes as audio is taken, sent or answered; None while suspended."""
        marks = [track.mark() for track in self._tracks]
        return None if None in marks else tuple(marks)

    async def _drain_input(self) -> None:
        assert self._stream is not None
        await self._stream.finish()
        await self._send_rest()

    async def _send_rest(self) -> None:
        await self._stop_intake(drain=True)
        if self._stream is not None:
            self._report_progress(final=True)
        await asyncio.gather(*(track.flush() for track in self._tracks))

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
        await asyncio.gather(*(track.stop_intake(drain=drain) for track in self._tracks))

    async def _finish(self, reason: EndReason, error: SessionError | None = None) -> None:
        self._finishing = True
        await asyncio.gather(*(t.link.close() for t in self._tracks if t.link is not None))
        audio_end = self.audio_end
        if (segment := self._first.segments.leftover(audio_end)) is not None:
            self._emit(segment)
        if error is not None:
            self._emit(error)
        for track in self._tracks:
            await track.capture.release()
        transcript = self._first.segments.transcript
        ended = SessionEnded(
            reason,
            transcript.final_text,
            self._first.timeline.to_ms(audio_end),
            len(transcript.segments),
            plain_text=transcript.plain_text,
        )
        self._state = "ended"
        self._ended.set_result(ended)
        self._emit(ended)


async def _hold(capture: CaptureSource) -> None:
    """Hold a shared capture. Raises `CaptureUnavailable` when it can't open."""
    try:
        await capture.hold()
    except CaptureUnavailable:
        raise
    except Exception as exc:
        raise CaptureUnavailable(f"the mic did not open: {exc}") from exc


def _attach_from(capture: CaptureSource, origin: float) -> Attachment:
    """Attach a held capture from `origin`; while it recovers, live frames once it is back."""
    try:
        return capture.attach(origin)
    except CaptureUnavailable:
        return capture.listen()
