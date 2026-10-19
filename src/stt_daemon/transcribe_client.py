"""`stt-daemon transcribe`: stream files to the daemon, one `transcribe` session each."""

import asyncio
import base64
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .server.messages import PROTOCOL

type Msg = dict[str, Any]

CHUNK_BYTES = 256 * 1024
_LINE_LIMIT = 64 << 20


@dataclass(frozen=True)
class TranscribeOptions:
    backend: str | None = None
    language: str | None = None
    json: bool = False


class TranscribeFailed(Exception):
    pass


async def transcribe(
    path: Path, files: Sequence[Path], options: TranscribeOptions, out: TextIO, err: TextIO
) -> int:
    """0 when every file was transcribed; 1 at the first that failed (the rest are skipped)."""
    try:
        reader, writer = await asyncio.open_unix_connection(path, limit=_LINE_LIMIT)
    except (FileNotFoundError, ConnectionRefusedError):
        print(
            f"stt-daemon is not running (nothing listens at {path}); "
            "start it with `stt-daemon service install` or `stt-daemon run`",
            file=err,
        )
        return 1
    client = _Client(reader, writer, err)
    try:
        await client.hello(options)
        for i, file in enumerate(files):
            try:
                result = await client.one(file, options)
            except TranscribeFailed as exc:
                print(f"stt-daemon: {file}: {exc}", file=err)
                return 1
            if options.json:
                print(json.dumps({"file": str(file), **result}, ensure_ascii=False), file=out)
            else:
                if i:
                    print(file=out)
                print(result["text"], file=out)
            out.flush()
        return 0
    except TranscribeFailed as exc:
        print(f"stt-daemon: {exc}", file=err)
        return 1
    finally:
        client.close()


class _Client:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, err: TextIO):
        self._reader = reader
        self._writer = writer
        self._err = err
        self._ids = 0
        self._word_times = False
        self._inbox: asyncio.Queue[Msg | None] = asyncio.Queue()
        self._reading = asyncio.create_task(self._read())

    async def hello(self, options: TranscribeOptions) -> None:
        welcome = await self._request("hello", {"protocol": PROTOCOL, "label": "stt-daemon"})
        name = options.backend or welcome.get("default_backend")
        backends = {b["name"]: b for b in welcome.get("backends", [])}
        self._word_times = bool(backends.get(name, {}).get("word_timestamps"))

    async def one(self, file: Path, options: TranscribeOptions) -> Msg:
        """{text, words, segments} of one file."""
        try:
            stream = file.open("rb")
        except OSError as exc:
            raise TranscribeFailed(f"can't read it: {exc.strerror}") from None
        with stream:
            start: Msg = {"word_timestamps": self._word_times}
            if options.backend:
                start["backend"] = options.backend
            if options.language:
                start["language"] = options.language
            await self._request("transcribe", start)
            ended: asyncio.Future[Msg] = asyncio.get_running_loop().create_future()
            sending = asyncio.create_task(self._send_file(stream, ended))
            collecting = asyncio.create_task(self._collect(ended))
            try:
                await asyncio.wait([sending, collecting], return_when=asyncio.FIRST_COMPLETED)
                if not collecting.done() and (error := sending.exception()) is not None:
                    if isinstance(error, ConnectionError):
                        raise TranscribeFailed("the daemon closed the connection")
                    self._send("cancel_session")
                    if isinstance(error, OSError):
                        raise TranscribeFailed(f"can't read it: {error.strerror or error}")
                    raise error
                return await collecting
            finally:
                for task in (sending, collecting):
                    task.cancel()
                await asyncio.gather(sending, collecting, return_exceptions=True)

    async def _send_file(self, stream: Any, ended: asyncio.Future[Msg]) -> None:
        while not ended.done() and (chunk := await asyncio.to_thread(stream.read, CHUNK_BYTES)):
            self._send("audio", {"data": base64.b64encode(chunk).decode()})
            await self._writer.drain()
        if not ended.done():
            self._send("stop_session")

    async def _collect(self, ended: asyncio.Future[Msg]) -> Msg:
        segments: list[Msg] = []
        fatal: str | None = None
        while (msg := await self._inbox.get()) is not None:
            kind, data = msg["type"], msg.get("data", {})
            if kind == "segment":
                segments.append({k: v for k, v in data.items() if k != "session_id"})
            elif kind == "session_error" and data.get("fatal"):
                fatal = f"{data.get('code')}: {data.get('message')}"
            elif kind == "session_error":
                print(f"stt-daemon: warning: {data.get('message')}", file=self._err)
            elif kind == "error":
                ended.set_result(msg)
                raise TranscribeFailed(f"{data.get('code')}: {data.get('message')}")
            elif kind == "session_ended":
                ended.set_result(msg)
                if data.get("reason") != "stopped":
                    raise TranscribeFailed(fatal or f"the session ended: {data.get('reason')}")
                words = [w for s in segments for w in s.get("words", [])]
                return {
                    "text": data.get("final_text", ""),
                    "words": words if self._word_times else None,
                    "segments": segments,
                }
        raise TranscribeFailed("the daemon closed the connection")

    async def _request(self, kind: str, data: Msg) -> Msg:
        id = self._send(kind, data)
        while (msg := await self._inbox.get()) is not None:
            if msg.get("reply_to") != id:
                continue
            reply: Msg = msg.get("data", {})
            if msg["type"] == "error":
                raise TranscribeFailed(f"{reply.get('code')}: {reply.get('message')}")
            return reply
        raise TranscribeFailed("the daemon closed the connection")

    async def _read(self) -> None:
        try:
            while line := await self._reader.readline():
                self._inbox.put_nowait(json.loads(line))
        except (ConnectionError, ValueError):
            pass
        finally:
            self._inbox.put_nowait(None)

    def _send(self, kind: str, data: Msg | None = None) -> str:
        self._ids += 1
        id = f"r{self._ids}"
        msg = {"type": kind, "id": id, "data": data or {}}
        self._writer.write(json.dumps(msg).encode() + b"\n")
        return id

    def close(self) -> None:
        self._reading.cancel()
        self._writer.close()
