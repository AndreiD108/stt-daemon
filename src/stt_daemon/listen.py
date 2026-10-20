"""`stt-daemon listen`: start a session, print its events; Enter stops, a key adds a marker."""

import asyncio
import json
import os
import signal
import sys
import termios
import threading
import tty
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .server.messages import PROTOCOL

type Msg = dict[str, Any]

_LINE_LIMIT = 2**20 + 1


@dataclass(frozen=True)
class ListenOptions:
    backend: str | None = None
    words: bool = False
    partials: bool = False


async def terminal_keys(marker_key: str) -> AsyncGenerator[str]:
    """Actions from stdin: "stop" on Enter or end of input, "marker", "cancel" on Ctrl-C.

    A terminal is read key by key; piped input line by line (a line that is the marker key
    adds a marker, any other line stops).
    """
    loop = asyncio.get_running_loop()
    actions: asyncio.Queue[str | Exception] = asyncio.Queue()
    fd = sys.stdin.fileno()
    is_tty = os.isatty(fd)
    saved = termios.tcgetattr(fd) if is_tty else None

    def on_key() -> None:
        data = os.read(fd, 1024).decode(errors="replace")
        if not data:
            loop.remove_reader(fd)
            actions.put_nowait("stop")
        for ch in data:
            if ch in "\r\n":
                actions.put_nowait("stop")
            elif ch == marker_key:
                actions.put_nowait("marker")

    def read_lines() -> None:
        # Files and /dev/null can't be polled, so pipes and files are read on a thread.
        last: str | Exception = "stop"
        try:
            for line in sys.stdin:
                action = "marker" if line.strip() == marker_key else "stop"
                loop.call_soon_threadsafe(actions.put_nowait, action)
        except Exception as exc:
            last = exc
        loop.call_soon_threadsafe(actions.put_nowait, last)

    if is_tty:
        tty.setcbreak(fd)
        loop.add_reader(fd, on_key)
    else:
        threading.Thread(target=read_lines, daemon=True).start()
    loop.add_signal_handler(signal.SIGINT, actions.put_nowait, "cancel")
    try:
        while True:
            action = await actions.get()
            if isinstance(action, Exception):
                raise action
            yield action
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        if saved is not None:
            loop.remove_reader(fd)
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)


async def listen(
    path: Path, options: ListenOptions, keys: AsyncIterator[str], out: TextIO, marker_key: str
) -> int:
    """0 when the session ends stopped or evicted; 1 after any `session_error` or failure;
    130 when cancelled."""
    try:
        reader, writer = await asyncio.open_unix_connection(path, limit=_LINE_LIMIT)
    except (FileNotFoundError, ConnectionRefusedError):
        print(
            f"stt-daemon is not running (nothing listens at {path}); "
            "start it with `stt-daemon service install` or `stt-daemon run`",
            file=out,
        )
        return 1
    client = _Client(reader, writer, out)
    try:
        return await client.run(options, keys, marker_key)
    finally:
        writer.close()


class _Client:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, out: TextIO):
        self._reader = reader
        self._writer = writer
        self._out = out
        self._ids = 0
        self._failed = False

    async def run(self, options: ListenOptions, keys: AsyncIterator[str], marker_key: str) -> int:
        hello = {"protocol": PROTOCOL, "label": "stt-daemon listen"}
        if await self._request("hello", hello) is None:
            return 1
        start: Msg = {"partials": options.partials, "word_timestamps": options.words}
        if options.backend:
            start["backend"] = options.backend
        started = await self._request("start_session", start)
        if started is None:
            return 1
        self._print(started)
        self._line(f"listening: Enter stops, {marker_key!r} adds a marker, Ctrl-C cancels")
        given_up: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        reading = asyncio.create_task(self._read_until_end())
        acting = asyncio.create_task(self._act(keys, given_up))
        try:
            waiting: set[asyncio.Future[int] | asyncio.Task[None]] = {reading, given_up, acting}
            while True:
                done, waiting = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                if reading in done:
                    return reading.result()
                if given_up in done:
                    return given_up.result()
                acting.result()  # input that fails ends the run; the daemon then drops the session
        finally:
            for task in (reading, acting):
                task.cancel()
            await asyncio.gather(reading, acting, return_exceptions=True)

    async def _read_until_end(self) -> int:
        while line := await self._reader.readline():
            msg: Msg = json.loads(line)
            self._print(msg)
            if msg["type"] == "session_ended":
                reason = msg["data"]["reason"]
                code = 130 if reason == "cancelled" else 1 if reason == "error" else 0
                return 1 if self._failed and code == 0 else code
        self._line("the daemon closed the connection")
        return 1

    async def _act(self, keys: AsyncIterator[str], given_up: asyncio.Future[int]) -> None:
        """A second Ctrl-C, or one during a stop, gives up without waiting for the end."""
        markers = 0
        stopping = False
        async for action in keys:
            if action == "cancel":
                if stopping:
                    given_up.set_result(130)
                    return
                stopping = True
                self._send("cancel_session")
            elif stopping:
                continue
            elif action == "stop":
                stopping = True
                self._send("stop_session")
            elif action == "marker":
                markers += 1
                self._send("add_marker", {"marker": f"⟦{markers}⟧"})

    async def _request(self, kind: str, data: Msg) -> Msg | None:
        """The reply; None after printing an error reply."""
        id = self._send(kind, data)
        while line := await self._reader.readline():
            msg: Msg = json.loads(line)
            if msg.get("reply_to") == id:
                if msg["type"] == "error":
                    self._print(msg)
                    return None
                return msg
        raise ConnectionError("the daemon closed the connection")

    def _send(self, kind: str, data: Msg | None = None) -> str:
        self._ids += 1
        id = f"r{self._ids}"
        msg = {"type": kind, "id": id, "data": data or {}}
        self._writer.write(json.dumps(msg).encode() + b"\n")
        return id

    def _line(self, text: str) -> None:
        print(text, file=self._out, flush=True)

    def _print(self, msg: Msg) -> None:
        d: Msg = msg.get("data", {})
        match msg["type"]:
            case "session_started":
                self._line(f"started   {d['backend']}, language {d.get('language') or '-'}")
            case "partial":
                self._line(f"partial   {d['text']}")
            case "segment":
                self._segment(d)
            case "marker_accepted":
                self._line(f"marker    {d['marker']} at {_s(d['at_ms'])}")
            case "session_error":
                self._failed = True
                fatal = " (fatal)" if d.get("fatal") else ""
                self._line(f"error     {d['code']}: {d['message']}{fatal}")
            case "error":
                self._line(f"error     {d.get('code')}: {d.get('message')}")
            case "session_ended":
                count, duration = d.get("segment_count", 0), _s(d.get("duration_ms", 0))
                noun = "segment" if count == 1 else "segments"
                self._line(f"ended     {d['reason']}, {count} {noun}, {duration}")
                self._line(f"text      {d.get('final_text', '')}")
            case _:
                pass

    def _segment(self, d: Msg) -> None:
        span = f"{_s(d['start_ms'])}-{_s(d['end_ms'])}"
        self._line(f"segment   #{d['index']} {span}  {d['text']}")
        if words := d.get("words"):
            self._line("  words   " + " ".join(f"{w['text']}@{_s(w['start_ms'])}" for w in words))
        for m in d.get("markers", []):
            after = f", after word {m['after_word']}" if "after_word" in m else ""
            self._line(f"  marker  {m['marker']} at {_s(m['at_ms'])}{after}")


def _s(ms: float) -> str:
    return f"{ms / 1000:.2f}s"
