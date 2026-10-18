"""One CLI request to the running daemon: hello, one command, its reply."""

import json
import socket
from pathlib import Path
from typing import Any

from .config import socket_path
from .server.messages import PROTOCOL

type Msg = dict[str, Any]


class NotRunning(Exception):
    pass


class DaemonError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def request(kind: str, *, path: Path | None = None, timeout: float = 10.0) -> Msg:
    """The reply's `data`. Raises `NotRunning`, `DaemonError` (an `error` reply) or `OSError`."""
    path = socket_path() if path is None else path
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        try:
            sock.connect(str(path))
        except (FileNotFoundError, ConnectionRefusedError):
            raise NotRunning(f"no daemon is listening at {path}") from None
        with sock.makefile("rwb") as stream:
            _send(stream, "hello", {"protocol": PROTOCOL, "label": "stt-daemon"})
            return _send(stream, kind, {})
    finally:
        sock.close()


def _send(stream: Any, kind: str, data: Msg) -> Msg:
    stream.write(json.dumps({"type": kind, "id": kind, "data": data}).encode() + b"\n")
    stream.flush()
    while line := stream.readline():
        reply: Msg = json.loads(line)
        if reply.get("reply_to") != kind:
            continue
        data = reply.get("data", {})
        if reply["type"] == "error":
            raise DaemonError(data.get("code", "?"), data.get("message", ""))
        return data
    raise ConnectionError("the daemon closed the connection")
