"""One client connection's sending side, line reading, and the socket file."""

import asyncio
import logging
import select
import socket
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .messages import MessageId, encode
from .slot_state import Client

log = logging.getLogger(__name__)

LINE_LIMIT = 2 << 20
"""Fits an `audio` command with a full chunk in base64."""
WRITE_BUFFER_LIMIT = 8 << 20
"""A client that lets this much pile up unread is closed."""
HANGUP_CHECK_S = 0.5
DRAIN_WAIT_S = 30.0
"""After this, sending goes on and `WRITE_BUFFER_LIMIT` decides."""


class SocketInUse(Exception):
    pass


@dataclass(eq=False)
class Connection:
    client: Client
    writer: asyncio.StreamWriter
    observe_all: bool = False
    vad_events: bool = False
    sources: bool = False
    closed: bool = False
    worker: asyncio.Task[None] | None = None

    def send(
        self, kind: str, data: Mapping[str, object], reply_to: MessageId | None = None
    ) -> None:
        transport = self.writer.transport
        if self.closed or transport.is_closing():
            return
        self.writer.write(encode(kind, data, reply_to))
        if transport.get_write_buffer_size() > WRITE_BUFFER_LIMIT:
            log.warning("client %s reads too slowly; closing it", self.client.client_id)
            self.closed = True
            transport.abort()

    async def drain(self) -> None:
        """Waits, at most `DRAIN_WAIT_S`, until the client has read most of what was sent."""
        if self.closed or self.writer.transport.is_closing():
            return
        try:
            async with asyncio.timeout(DRAIN_WAIT_S):
                await self.writer.drain()
        except (ConnectionError, TimeoutError):
            pass

    def hung_up(self) -> bool:
        sock = self.writer.get_extra_info("socket")
        if sock is None or sock.fileno() < 0:
            return True
        poller = select.poll()
        poller.register(sock.fileno(), select.POLLHUP)
        return bool(poller.poll(0))


async def put_until_gone[T](conn: Connection, queue: asyncio.Queue[T], item: T) -> None:
    """Waits for room in `queue`; raises ConnectionError when the client hangs up meanwhile,
    since the reader, waiting here, would not see the end of input."""
    if not queue.full():
        queue.put_nowait(item)
        return
    put = asyncio.ensure_future(queue.put(item))
    try:
        while not put.done():
            await asyncio.wait([put], timeout=HANGUP_CHECK_S)
            if not put.done() and conn.hung_up():
                raise ConnectionError("the client hung up")
    finally:
        put.cancel()


async def readline(conn: Connection, reader: asyncio.StreamReader) -> bytes:
    """The next line; b"" at the end, or after an over-long line (which gets an error)."""
    try:
        return await reader.readline()
    except ValueError:
        message = f"a line is longer than {LINE_LIMIT} bytes; the daemon closes the connection"
        conn.send("error", {"code": "bad_request", "message": message})
        return b""


def prepare_socket(path: Path) -> None:
    """Makes the socket's directory (700) and removes a stale socket nobody listens on."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise SocketInUse(
            f"{path} exists and is not a socket; move it away or set STT_DAEMON_SOCKET"
        )
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(str(path))
    except (ConnectionRefusedError, FileNotFoundError):
        path.unlink(missing_ok=True)
        return
    finally:
        probe.close()
    raise SocketInUse(f"another stt-daemon is already listening on {path}")
