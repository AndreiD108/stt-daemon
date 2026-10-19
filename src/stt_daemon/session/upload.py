"""A `transcribe` session on a file backend: the bytes are kept as sent and uploaded at stop.

It answers the slot and the server like a `Session`, but has no capture, VAD or link: no
`partial` or `progress` events, one upload, then the segments and the end.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, Literal

from ..audio.frames import Attachment, epoch_ms
from ..backends.base import SAMPLE_RATE, FileAudio, FileBackend, OpenFailed
from .events import (
    EndReason,
    PreemptedBy,
    SessionEnded,
    SessionError,
    SessionEvent,
    SessionKind,
    SessionStarted,
)
from .options import MarkerRefused, SessionOptions
from .segments import Segments
from .session import State
from .timeline import Timeline

log = logging.getLogger(__name__)

INPUT_LIMIT = 256 * 1024 * 1024
"""Bytes one session keeps; they stay in memory until the upload ends."""


class UploadSession:
    def __init__(
        self,
        *,
        backend: FileBackend,
        options: SessionOptions,
        pcm: tuple[int, int] | None,
        emit: Callable[[SessionEvent], None],
        drain: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """`drain` waits until the client has read the events so far; the answer to a long
        file comes all at once, so the segments go out one by one."""
        self.session_id = uuid.uuid4().hex
        self.kind: SessionKind = "transcribe"
        self.started: SessionStarted | None = None
        self.audio_end = 0.0
        self.make_room: Callable[[], Coroutine[Any, Any, bool]] | None = None
        self._backend = backend
        self._options = options
        self._pcm = pcm
        self._emit = emit
        self._drain = drain
        self._state: State = "new"
        self._chunks: list[bytes] = []
        self._size = 0
        self._segments = Segments(
            auto_language=options.open.language == "auto",
            word_timestamps=options.word_timestamps,
        )
        self._ending_task: asyncio.Task[None] | None = None
        self._finishing = False
        self._ended: asyncio.Future[SessionEnded] = asyncio.get_running_loop().create_future()

    @property
    def state(self) -> State:
        return self._state

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def suspendable(self) -> bool:
        return False

    @property
    def suspended(self) -> bool:
        return False

    async def start(
        self, *, since: float | None = None, attachment: Attachment | None = None
    ) -> SessionStarted:
        if self._state != "new":
            raise RuntimeError("a session starts once")
        origin = time.monotonic()
        self.audio_end = origin
        self._segments.timeline = Timeline(origin, SAMPLE_RATE)
        self.started = SessionStarted(
            self.session_id,
            self._backend.name,
            self._options.open.language,
            epoch_ms(origin),
            self._options.open.previous_text,
            self.kind,
        )
        self._state = "running"
        self._emit(self.started)
        return self.started

    async def write(self, data: bytes) -> None:
        if self._state != "running":
            return
        if self._size + len(data) > INPUT_LIMIT:
            self._fail(
                "invalid_request",
                f"the input is over {INPUT_LIMIT // (1024 * 1024)} MiB, the most "
                f"{self._backend.name} keeps; send compressed audio, or use another backend",
            )
            return
        self._chunks.append(data)
        self._size += len(data)

    def add_marker(
        self, marker: str, at: int | None = None, *, arrived: float | None = None
    ) -> int:
        """`at` is `started_at` plus audio ms, and required: the audio's length is known only
        once the backend answers."""
        if self._state != "running" or self.started is None:
            if self._state == "new":
                raise MarkerRefused("no_session", "the session hasn't started; wait for it")
            raise MarkerRefused(
                "session_stopping", "the session is ending and takes no more markers"
            )
        if at is None:
            raise MarkerRefused(
                "invalid",
                f"{self._backend.name} reads the audio only at stop, so a marker needs at "
                "(started_at plus the audio ms)",
            )
        at_ms = at - self.started.started_at
        if at_ms < 0:
            raise MarkerRefused(
                "out_of_range", "the marker's time is before the session's first audio"
            )
        self._segments.markers.add(marker, at_ms)
        return at_ms

    async def stop(self, reason: Literal["stopped", "evicted"] = "stopped") -> SessionEnded:
        """Upload what came, wait for the answer however long it takes, end."""
        if self._state == "running":
            self._state = "stopping"
            self._ending_task = asyncio.create_task(self._upload(reason))
        return await self.wait_ended()

    async def cancel(
        self, reason: Literal["cancelled", "client_gone"] = "cancelled"
    ) -> SessionEnded:
        """Drop the input; an upload in flight is cut off and its answer never comes."""
        if self._state == "running":
            self._state = "ending"
            self._drop()
            self._finish(reason)
        elif self._state == "stopping" and not self._finishing:
            assert self._ending_task is not None
            self._state = "ending"
            self._drop()
            self._ending_task.cancel()
            await asyncio.wait([self._ending_task])
            if not self._finishing:
                self._finish(reason)
        return await self.wait_ended()

    async def wait_ended(self) -> SessionEnded:
        if self._state == "new":
            raise RuntimeError("the session hasn't started")
        return await asyncio.shield(self._ended)

    async def suspend(self, by: PreemptedBy) -> bool:
        return False

    async def resume(self) -> bool:
        return False

    def give_up(self, code: str, message: str) -> None:
        self._fail(code, message)

    def _drop(self) -> None:
        self._chunks.clear()
        self._segments.markers.clear()

    def _fail(self, code: str, message: str) -> None:
        if self._state == "running":
            self._state = "ending"
            self._chunks.clear()
            self._finish("error", SessionError(code, message, fatal=True))

    async def _upload(self, reason: Literal["stopped", "evicted"]) -> None:
        if not self._size:
            self._finish(
                "error", SessionError("invalid_request", "no audio came before stop", True)
            )
            return
        data = b"".join(self._chunks)
        self._chunks.clear()
        try:
            answers = await self._backend.transcribe_file(
                FileAudio(data, self._pcm), self._options.open
            )
        except OpenFailed as exc:
            self._finish("error", SessionError(exc.code, exc.message, fatal=True))
            return
        except Exception as exc:
            log.exception("session %s: upload failed", self.session_id)
            message = f"the daemon failed while uploading: {exc}"
            self._finish("error", SessionError("internal", message, fatal=True))
            return
        assert self.started is not None
        origin = self._segments.timeline.origin
        length_s = max((a.audio_end_s for a in answers), default=0.0)
        self._segments.timeline.sent(0, origin, round(length_s * SAMPLE_RATE), cut=False)
        self.audio_end = origin + length_s
        for i, answer in enumerate(answers):
            if (segment := self._segments.committed(answer, i == len(answers) - 1)) is not None:
                self._emit(segment)
                if self._drain is not None:
                    await self._drain()
        self._finish(reason)

    def _finish(self, reason: EndReason, error: SessionError | None = None) -> None:
        self._finishing = True
        if (segment := self._segments.leftover(self.audio_end)) is not None:
            self._emit(segment)
        if error is not None:
            self._emit(error)
        transcript = self._segments.transcript
        ended = SessionEnded(
            reason,
            transcript.final_text,
            self._segments.timeline.to_ms(self.audio_end),
            len(transcript.segments),
            plain_text=transcript.plain_text,
        )
        self._state = "ended"
        self._ended.set_result(ended)
        self._emit(ended)
