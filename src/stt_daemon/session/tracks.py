"""One audio source of a session: capture → VAD → pause gate → link, committing on the backend's
cadence. A session has one track, or two for system audio (`system` and `mic`), each with its
own connection and timeline on the session's clock."""

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any, Protocol

import numpy as np

from ..audio.capture import CaptureSource, CaptureUnavailable
from ..audio.frames import Attachment, Frame
from ..audio.speakers import SpeakerMonitor
from ..audio.vad import WINDOW, StreamingVAD
from ..backends.base import PLAYBACK_LEAD_S, Backend, Committed, OpenFailed, OpenOptions
from ..config import Config
from .pause_gate import Outgoing, PauseGate
from .recovery import Link
from .reference import PlaybackHistory
from .segments import Segments
from .timeline import Timeline

log = logging.getLogger(__name__)

END_OF_AUDIO = Frame(np.zeros(0, dtype=np.float32), 0.0, 16000)


class TrackOwner(Protocol):
    """The session a track reports to."""

    def track_committed(self, track: "Track", event: Committed, final: bool) -> None: ...

    def track_partial(self, track: "Track", text: str) -> None: ...

    def track_error(self, track: "Track", code: str, message: str) -> None: ...

    def track_fatal(self, track: "Track", code: str, message: str) -> None: ...

    def track_failed(self, track: "Track", error: BaseException) -> None:
        """Its audio loop raised."""
        ...

    async def after_frame(self, track: "Track") -> bool:
        """Called after each frame; False ends the audio loop."""
        ...


class Track:
    def __init__(
        self,
        *,
        name: str | None,
        owner: TrackOwner,
        config: Config,
        capture: CaptureSource,
        backend: Backend,
        segments: Segments,
        playback: SpeakerMonitor | None = None,
    ) -> None:
        """`playback`: the speakers' capture, sent to the backend with each commit."""
        self.name = name
        self.capture = capture
        self._playback_source = playback
        self._playback: PlaybackHistory | None = None
        self._holding = False
        """`take_playback` took a hold on `_playback_source`."""
        self.segments = segments
        self.audio_end = 0.0
        """Monotonic end of the last captured audio the track took in."""
        self.link: Link | None = None
        self._owner = owner
        self._config = config
        self._backend = backend
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
        self.timeline = Timeline(0.0, rate)
        self._policy = backend.capabilities.commit_policy
        self._attachment: Attachment | None = None
        self._speech_since_commit = False
        self._samples_since_commit = 0
        self._audio_task: asyncio.Task[None] | None = None
        self._end_put: asyncio.Future[None] | None = None

    def begin(self, attachment: Attachment, origin: float, options: OpenOptions) -> None:
        """Take frames from `attachment`; session times count from `origin` (monotonic)."""
        self._attachment = attachment
        self.timeline = Timeline(origin, self.capture.sample_rate)
        self.segments.timeline = self.timeline
        self.audio_end = origin
        self.link = Link(
            backend=self._backend,
            options=options,
            timeline=self.timeline,
            chunk_samples=self.capture.sample_rate * self._config.audio.chunk_ms // 1000,
            on_committed=lambda event, final: self._owner.track_committed(self, event, final),
            on_partial=lambda text: self._owner.track_partial(self, text),
            on_error=lambda code, message: self._owner.track_error(self, code, message),
            on_fatal=lambda code, message: self._owner.track_fatal(self, code, message),
            speech=self._gate.speech,
            playback=self._playback,
        )

    async def take_playback(self, origin: float) -> None:
        """Before `begin`: keep the playback from `origin` on; without it, commits go without
        a reference."""
        source = self._playback_source
        if source is None:
            return
        await source.hold()
        self._holding = True
        try:
            attachment = source.attach(origin - PLAYBACK_LEAD_S)
        except CaptureUnavailable as exc:
            log.warning("no speaker audio for the echo gate: %s", exc)
            await self.release_playback()
            return
        self._playback = PlaybackHistory(attachment, self.capture.sample_rate)

    async def release_playback(self) -> None:
        if self._playback is not None:
            await self._playback.close()
            self._playback = None
        if self._holding and self._playback_source is not None:
            self._holding = False
            await self._playback_source.release()

    async def open(self, make_room: Callable[[], Coroutine[Any, Any, bool]] | None) -> None:
        """Open the link; on `concurrency_limit`, `make_room` may free a provider connection
        for one more try. Raises `OpenFailed`."""
        assert self.link is not None
        try:
            await self.link.open()
        except OpenFailed as exc:
            if exc.code != "concurrency_limit" or make_room is None or not await make_room():
                raise
            await self.link.open()

    def run(self) -> None:
        self._audio_task = asyncio.create_task(self._audio_loop())
        self._audio_task.add_done_callback(self._audio_loop_done)

    def detach(self) -> None:
        if self._attachment is not None:
            self._attachment.detach()

    def mark(self) -> tuple[object, ...] | None:
        """Changes as audio is taken, sent or answered; None while suspended."""
        link = self.link
        assert link is not None
        if link.suspended:
            return None
        return (self.audio_end, link.backlog_s, link.sent_samples)

    async def stop_intake(self, *, drain: bool) -> None:
        """Stop taking audio; with `drain`, first take what capture delivered before now."""
        assert self._attachment is not None
        assert self._audio_task is not None
        if drain:
            await self.capture.settle(time.monotonic())
        self._attachment.detach()
        if drain:
            # A task, not put_nowait: a stream's queue is bounded. It runs after the frames
            # capture already scheduled.
            self._end_put = asyncio.ensure_future(self._attachment.queue.put(END_OF_AUDIO))
        elif self._audio_task is not asyncio.current_task():
            self._audio_task.cancel()
            if self._end_put is not None:
                self._end_put.cancel()
        await asyncio.wait([self._audio_task])

    async def flush(self) -> None:
        """After `stop_intake`: send the held audio, commit, wait for every answer."""
        for out in self._gate.flush():
            self._queue(out)
        self._commit(final=True)
        assert self.link is not None
        await self.link.settle()

    async def _audio_loop(self) -> None:
        assert self._attachment is not None
        queue = self._attachment.queue
        while (frame := await queue.get()) is not END_OF_AUDIO:
            self._take(frame)
            if not await self._owner.after_frame(self):
                return

    def _audio_loop_done(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            self._owner.track_failed(self, error)

    def _take(self, frame: Frame) -> None:
        # With an empty ring, the first live frame began before the session's first sample.
        if frame.start < self.timeline.origin:
            part = frame.since(self.timeline.origin)
            if part is None:
                return
            frame = part
        self.audio_end = frame.end
        events = self._vad.push(frame)
        parts = self._gate.push(frame, events)
        for part in parts:
            self._queue(part)
        # The gate stays open through a pause until the VAD confirms it; that tail isn't speech.
        heard = self._vad.speech_id is not None and self._vad.pause_s < frame.duration
        if parts and heard:
            self._speech_since_commit = True
        sent_s = self._samples_since_commit / self.capture.sample_rate
        pause_s = self._policy.pause_for(sent_s)
        if pause_s is not None and self._speech_since_commit and self._vad.pause_s >= pause_s:
            self._commit()
        cap = self._policy.max_chunk_s
        if cap is not None and sent_s >= cap:
            self._commit()
        assert self.link is not None
        self.link.tick(speaking=self._vad.speech_id is not None)
        if self._playback is not None:
            pending = (self.link.queued_since, self._gate.held_since)
            oldest = min((t for t in pending if t is not None), default=frame.end)
            self._playback.forget_before(oldest - PLAYBACK_LEAD_S)

    def _queue(self, out: Outgoing) -> None:
        assert self.link is not None
        self.link.queue(out)
        self._samples_since_commit += out.frame.samples.size

    def _commit(self, *, final: bool = False) -> None:
        assert self.link is not None
        self.link.commit(final=final)
        self._samples_since_commit = 0
        self._speech_since_commit = False
