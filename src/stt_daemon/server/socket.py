"""The Unix socket server: connections, the hello gate, commands, and who gets which event.

Each connection reads lines in one task and runs its commands in order in another, so a start
that waits for the backend doesn't hide a disconnect.
"""

import asyncio
import contextlib
import itertools
import logging
import os
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .. import __version__
from ..audio.capture import CaptureSource
from ..audio.pw_capture import AppTarget, PwRecordCapture
from ..audio.vad import VadEvent
from ..config import ConfigError
from ..credentials import CredentialsError
from ..session.context import ContextStore, scope_of
from ..session.events import SessionEnded, SessionEvent, SessionStarted
from ..session.options import MarkerRefused
from . import wire
from .backend_set import Setup
from .connection import LINE_LIMIT, Connection, prepare_socket, put_until_gone, readline
from .messages import (
    PROTOCOL,
    Message,
    RequestError,
    parse_audio,
    parse_hello,
    parse_line,
    parse_marker,
)
from .peer import peer_of
from .slot import Slot
from .slot_state import Client, SlotState
from .sources import SourceEvent, SourceScan
from .speech import SpeechWatch
from .starts import Running, Starts, system_capture

log = logging.getLogger(__name__)

COMMAND_QUEUE = 16
"""Commands a connection may have waiting; then reading waits (an `audio` stream slows down)."""
_CLOSE_TIMEOUT_S = 2.0


type _Item = tuple[Message | RequestError, float]


class Server(Starts):
    def __init__(
        self,
        *,
        path: Path,
        capture: CaptureSource,
        setup: Setup,
        reload_setup: Callable[[], Setup],
        system_capture: Callable[[AppTarget | None], PwRecordCapture] = system_capture,
        sources: SourceScan | None = None,
    ) -> None:
        """`reload_setup` re-reads config and credentials; it raises `ConfigError` or
        `CredentialsError`. `system_capture` makes a `system` session's system track, following
        an app's stream or not. Construct on the loop."""
        self.path = path
        self.capture = capture
        self.setup = setup
        self._reload_setup = reload_setup
        self._system_capture = system_capture
        self._sources = sources or SourceScan()
        self._sources.on_event = self._source_event
        self._connections: set[Connection] = set()
        self._welcomed: dict[str, Connection] = {}
        self._handlers: set[asyncio.Task[None]] = set()
        self._running: dict[str, Running] = {}
        self._contexts = ContextStore()
        self._ids = itertools.count(1)
        self._server: asyncio.Server | None = None
        self._closing = False
        config = setup.config
        self._slot = Slot(
            capture=capture,
            pre_start_ms=config.audio.pre_start_ms,
            evictable_after_s=config.slot.evictable_after_s,
            on_change=self._slot_changed,
            on_event=self._session_event,
            capacities=lambda: self.setup.capacities(),
        )
        self._buffer_ms = config.audio.buffer_ms
        self._speech = SpeechWatch(
            capture=capture, vad=config.vad, buffer_ms=self._buffer_ms, on_event=self._vad_event
        )

    async def start(self) -> None:
        """Binds the socket. Raises `SocketInUse` when another daemon listens on it."""
        prepare_socket(self.path)
        umask = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(
                self._handle, path=self.path, limit=LINE_LIMIT
            )
        finally:
            os.umask(umask)
        os.chmod(self.path, 0o600)
        self._speech.start()
        self._sources.start()
        log.info("listening on %s", self.path)

    async def close(self) -> None:
        """Stops the running session as `stop_session` would, then closes every client."""
        self._closing = True  # before any await: no connection gets a worker from now on
        if self._server is not None:
            self._server.close()
        await self._speech.close()
        await self._sources.close()
        workers = [c.worker for c in self._connections if c.worker is not None]
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.wait(workers)
        await self._slot.shutdown()
        for conn in self._connections:
            conn.writer.close()
        if self._handlers:
            _, pending = await asyncio.wait(self._handlers, timeout=_CLOSE_TIMEOUT_S)
            for task in pending:
                task.cancel()
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._handlers.add(task)
        client = Client(f"c{next(self._ids)}", peer=peer_of(writer.get_extra_info("socket")))
        conn = Connection(client, writer)
        self._connections.add(conn)
        try:
            if await self._greet(conn, reader):
                await self._read(conn, reader)
        except ConnectionError:
            pass
        finally:
            await self._disconnect(conn)
            writer.close()
            with contextlib.suppress(Exception):
                async with asyncio.timeout(_CLOSE_TIMEOUT_S):
                    await writer.wait_closed()
            self._connections.discard(conn)
            self._handlers.discard(task)

    async def _greet(self, conn: Connection, reader: asyncio.StreamReader) -> bool:
        line = await readline(conn, reader)
        if not line or self._closing:
            return False
        try:
            message = parse_line(line)
        except RequestError as exc:
            conn.send("error", exc.to_wire())
            return False
        try:
            if message.type != "hello":
                raise RequestError(
                    "protocol", f"the first message must be hello, with protocol: {PROTOCOL}"
                )
            hello = parse_hello(message.data)
            if hello.observe_all and not self.setup.config.server.observers:
                raise RequestError(
                    "observers_disabled",
                    "the config has server.observers = false; connect without observe_all",
                )
        except RequestError as exc:
            conn.send("error", exc.to_wire(), message.id)
            return False
        cid = conn.client.client_id
        conn.client = replace(conn.client, label=hello.label)
        conn.observe_all, conn.vad_events = hello.observe_all, hello.vad_events
        conn.sources = hello.sources
        self._welcomed[cid] = conn
        peer = conn.client.peer
        log.info("client %s connected: label %s, pid %s", cid, hello.label, peer.pid)
        conn.send(
            "welcome",
            {
                "protocol": PROTOCOL,
                "daemon_version": __version__,
                "client_id": cid,
                **self.setup.to_wire(),
                "mic_open": self.capture.keep_open,
                "slot": self._slot.state.to_wire(),
                "sources": [source.to_wire() for source in self._sources.sources()],
            },
            message.id,
        )
        return True

    async def _read(self, conn: Connection, reader: asyncio.StreamReader) -> None:
        queue: asyncio.Queue[_Item] = asyncio.Queue(COMMAND_QUEUE)
        conn.worker = asyncio.create_task(self._work(conn, queue))
        while line := await readline(conn, reader):
            arrived = time.monotonic()
            if not line.strip():
                continue
            try:
                message = parse_line(line)
            except RequestError as exc:
                await put_until_gone(conn, queue, (exc, arrived))
                continue
            if message.type == "ping":
                conn.send("pong", {}, message.id)
            else:
                await put_until_gone(conn, queue, (message, arrived))

    async def _disconnect(self, conn: Connection) -> None:
        conn.closed = True
        cid = conn.client.client_id
        if self._welcomed.pop(cid, None) is None:
            return
        log.info("client %s disconnected", cid)
        await self._slot.client_gone(cid)
        if conn.worker is not None:
            conn.worker.cancel()
            await asyncio.wait([conn.worker])
        self._contexts.forget(scope_of(cid, None))

    async def _work(self, conn: Connection, queue: asyncio.Queue[_Item]) -> None:
        while True:
            item, arrived = await queue.get()
            if isinstance(item, RequestError):
                conn.send("error", item.to_wire())
                continue
            try:
                await self._command(conn, item, arrived)
            except RequestError as exc:
                conn.send("error", exc.to_wire(), item.id)
            except Exception:
                log.exception("client %s: %s failed", conn.client.client_id, item.type)
                error = {"code": "internal", "message": "the daemon failed; see its log"}
                conn.send("error", error, item.id)

    async def _command(self, conn: Connection, message: Message, arrived: float) -> None:
        match message.type:
            case "start_session":
                await self._start(conn, message, arrived)
            case "transcribe":
                await self._transcribe(conn, message)
            case "start_system_session":
                await self._system(conn, message)
            case "audio":
                await self._audio(conn, message)
            case "add_marker":
                self._add_marker(conn, message, arrived)
            case "stop_session" | "cancel_session":
                await self._end(conn, message)
            case "reload":
                self._reload(conn, message)
            case "hello":
                raise RequestError("bad_request", "hello was already sent on this connection")
            case kind:
                raise RequestError("bad_request", f"unknown message type {kind!r}")

    async def _audio(self, conn: Connection, message: Message) -> None:
        data = parse_audio(message.data)
        cid = conn.client.client_id
        session = self._slot.session_of(cid)
        run = self._running.get(cid)
        if session is None or run is None or run.stream is None:
            raise RequestError(
                "no_session", "this client has no transcribe session; send transcribe first"
            )
        if session.state != "running":
            raise RequestError(
                "no_session", "this client's session is already ending; its session_ended follows"
            )
        await run.stream.write(data)

    def _add_marker(self, conn: Connection, message: Message, arrived: float) -> None:
        request = parse_marker(message.data)
        session = self._slot.session_of(conn.client.client_id)
        if session is None:
            raise RequestError("no_session", "this client has no session; start one first")
        try:
            at_ms = session.add_marker(request.marker, request.at, arrived=arrived)
        except MarkerRefused as exc:
            raise RequestError(exc.code, exc.message) from None
        conn.send("marker_accepted", {"marker": request.marker, "at_ms": at_ms}, message.id)

    async def _end(self, conn: Connection, message: Message) -> None:
        cid = conn.client.client_id
        session = self._slot.session_of(cid)
        run = self._running.get(cid)
        if session is None or run is None:
            raise RequestError("no_session", "this client has no session; start one first")
        if session.state != "running":
            raise RequestError(
                "no_session", "this client's session is already ending; its session_ended follows"
            )
        run.end_id = message.id
        if message.type == "cancel_session":
            await session.cancel()
        else:
            await session.stop()

    def _reload(self, conn: Connection, message: Message) -> None:
        try:
            setup = self._reload_setup()
        except (ConfigError, CredentialsError) as exc:
            raise RequestError(
                "config", f"{exc}; fix it and reload again (the daemon keeps its last setup)"
            ) from None
        self.setup = setup
        data = setup.to_wire()
        log.info("reloaded: backends %s, default %s", list(setup.backends), setup.default)
        conn.send("backends", data, message.id)
        for other in self._welcomed.values():
            if other is not conn:
                other.send("backends", data)
        self._slot.capacities_changed()

    def _slot_changed(self, state: SlotState) -> None:
        data = state.to_wire()
        for conn in self._welcomed.values():
            conn.send("slot", data)

    def _session_event(self, client: Client, event: SessionEvent) -> None:
        run = self._running.get(client.client_id)
        if run is None:
            log.error("session event for client %s, which has no session", client.client_id)
            return
        session_id = run.session.session_id
        kind, data = wire.session_event(event, session_id, client, run.session_label)
        reply_to = None
        if isinstance(event, SessionStarted):
            reply_to = run.start_id
            if run.app is not None:
                data |= {"app": run.app.id, "app_name": run.app.name}
            log.info("session %s started for %s on %s", session_id, client.client_id, event.backend)
        elif isinstance(event, SessionEnded):
            reply_to = run.end_id
            del self._running[client.client_id]
            self._remember(client, run, event)
            log.info(
                "session %s ended: %s, %d segments, %d ms",
                session_id,
                event.reason,
                event.segment_count,
                event.duration_ms,
            )
        run.conn.send(kind, data, reply_to)
        for conn in self._welcomed.values():
            if conn.observe_all and conn is not run.conn:
                conn.send(kind, data)

    def _vad_event(self, event: VadEvent) -> None:
        self._slot.heard(event)
        kind, data = wire.vad_event(event, self._buffer_ms)
        holder = self._slot.state.holder
        for conn in self._welcomed.values():
            if holder is None:
                wanted = conn.vad_events
            else:
                is_holder = conn.client.client_id == holder.client_id
                wanted = conn.observe_all or (is_holder and conn.vad_events)
            if wanted:
                conn.send(kind, data)

    def _source_event(self, event: SourceEvent) -> None:
        kind, data = wire.source_event(event)
        for conn in self._welcomed.values():
            if conn.sources:
                conn.send(kind, data)

    def _remember(self, client: Client, run: Running, ended: SessionEnded) -> None:
        if run.context_key is None or ended.reason not in ("stopped", "evicted"):
            return
        if ended.plain_text.strip():
            scope = scope_of(client.client_id, client.label)
            self._contexts.put(scope, run.context_key, ended.plain_text)
