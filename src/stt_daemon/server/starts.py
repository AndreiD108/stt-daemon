"""Session starts: `start_session`, `transcribe` and `start_system_session` → sessions on the
slot, and their failures as `error` replies. Mixed into `Server`."""

import contextlib
import sys
from collections.abc import Callable, Generator
from dataclasses import dataclass

from ..audio.capture import CaptureSource, CaptureUnavailable
from ..audio.pw_capture import PwRecordCapture
from ..audio.stream_capture import ByteStreamCapture, ToolMissing, ffmpeg_command
from ..backends.base import FileBackend, OpenFailed
from ..service.preflight import check_tool
from ..session.context import ContextStore, scope_of
from ..session.events import SessionEvent
from ..session.session import Session
from ..session.upload import UploadSession
from .backend_set import Setup, resolve
from .connection import Connection
from .messages import Message, MessageId, RequestError, parse_start, parse_system, parse_transcribe
from .slot import AnySession, ClientGone, Slot
from .slot_state import StartRefused
from .speech import SpeechWatch


@dataclass(eq=False)
class Running:
    """A client's session, with what its events need."""

    conn: Connection
    session: AnySession
    session_label: str | None
    context_key: str | None
    start_id: MessageId | None
    end_id: MessageId | None = None
    """The `stop_session` or `cancel_session` that `session_ended` replies to."""
    stream: ByteStreamCapture | UploadSession | None = None
    """A `transcribe` session's input."""


class Starts:
    """The `Server` state the starts use."""

    setup: Setup
    capture: CaptureSource
    _running: dict[str, Running]
    _contexts: ContextStore
    _slot: Slot
    _speech: SpeechWatch
    _system_capture: Callable[[], PwRecordCapture]

    async def _start(self, conn: Connection, message: Message, arrived: float) -> None:
        request = parse_start(message.data)
        setup = self.setup
        since = None
        if request.speech_id is not None:
            speech_start = self._speech.start_of(request.speech_id)
            if speech_start is None:
                raise RequestError(
                    "speech_expired",
                    f"speech {request.speech_id} is unknown or left the audio buffer; "
                    "start without speech_id",
                )
            since = speech_start - setup.config.vad.pause_keep_ms / 1000
        cid = conn.client.client_id
        scope = scope_of(cid, conn.client.label)

        def make_session(emit: Callable[[SessionEvent], None]) -> Session:
            resolved = resolve(setup, request, lambda key: self._contexts.get(scope, key))
            session = Session(
                config=setup.config,
                capture=self.capture,
                backend=resolved.backend,
                options=resolved.options,
                emit=emit,
            )
            self._running[cid] = Running(
                conn, session, request.label, resolved.context_key, message.id
            )
            return session

        with self._starting(cid):
            await self._slot.start(
                conn.client,
                make_session=make_session,
                session_label=request.label,
                since=since,
                requested_at=arrived,
            )

    async def _transcribe(self, conn: Connection, message: Message) -> None:
        request = parse_transcribe(message.data)
        setup = self.setup
        decoder = None if request.pcm == (16000, 1) else ffmpeg_command(request.pcm)
        cid = conn.client.client_id
        scope = scope_of(cid, conn.client.label)

        def make_session(emit: Callable[[SessionEvent], None]) -> AnySession:
            resolved = resolve(
                setup, request.start, lambda key: self._contexts.get(scope, key), kind="transcribe"
            )
            if isinstance(resolved.backend, FileBackend):
                upload = UploadSession(
                    backend=resolved.backend,
                    options=resolved.options,
                    pcm=request.pcm,
                    emit=emit,
                    drain=conn.drain,
                )
                self._running[cid] = Running(
                    conn, upload, request.start.label, None, message.id, stream=upload
                )
                return upload
            if decoder is not None and (problem := check_tool("ffmpeg")) is not None:
                raise RequestError("tool_unavailable", problem, tool="ffmpeg")
            stream = ByteStreamCapture(decoder)
            session = Session(
                config=setup.config,
                capture=stream,
                backend=resolved.backend,
                options=resolved.options,
                emit=emit,
                kind="transcribe",
            )
            self._running[cid] = Running(
                conn, session, request.start.label, resolved.context_key, message.id, stream=stream
            )
            return session

        with self._starting(cid):
            await self._slot.start_input(
                conn.client, make_session=make_session, session_label=request.start.label
            )

    async def _system(self, conn: Connection, message: Message) -> None:
        request = parse_system(message.data)
        if sys.platform != "linux":
            raise RequestError(
                "unsupported", f"this daemon can't capture system audio on {sys.platform}"
            )
        setup = self.setup
        cid = conn.client.client_id
        scope = scope_of(cid, conn.client.label)

        def make_session(emit: Callable[[SessionEvent], None]) -> Session:
            resolved = resolve(
                setup, request.start, lambda key: self._contexts.get(scope, key), kind="system"
            )
            capture = self._system_capture()
            tool = capture.command[0]
            if (problem := check_tool(tool)) is not None:
                raise RequestError("tool_unavailable", problem, tool=tool)
            session = Session(
                config=setup.config,
                capture=capture,
                backend=resolved.backend,
                options=resolved.options,
                emit=emit,
                kind="system",
                mic=self.capture if request.with_mic else None,
            )
            self._running[cid] = Running(
                conn, session, request.start.label, resolved.context_key, message.id
            )
            return session

        with self._starting(cid):
            await self._slot.start_input(
                conn.client, make_session=make_session, session_label=request.start.label
            )

    @contextlib.contextmanager
    def _starting(self, cid: str) -> Generator[None]:
        """Start failures as `error` replies; a start that didn't happen leaves no `_Running`."""
        try:
            yield
        except StartRefused as exc:
            raise _refused(exc) from None
        except ClientGone:
            pass
        except CaptureUnavailable as exc:
            raise RequestError(
                "mic_unavailable", f"{exc}; check the mic, then start again"
            ) from None
        except ToolMissing as exc:
            problem = check_tool(str(exc)) or f"{exc} could not be started"
            raise RequestError("tool_unavailable", problem, tool=str(exc)) from None
        except OpenFailed as exc:
            raise RequestError(exc.code, exc.message) from None
        finally:
            run = self._running.get(cid)
            if run is not None and run.session.started is None:
                del self._running[cid]


def _refused(exc: StartRefused) -> RequestError:
    if exc.full_slot == "mic":
        next_step = "wait for a slot event with state idle or evictable"
    elif exc.full_slot == "backend":
        next_step = f"wait for a slot event with room on {exc.backend}, or use another backend"
    else:
        next_step = "stop it first"
    return RequestError(exc.code, f"{exc.message}; {next_step}", **exc.to_wire())
