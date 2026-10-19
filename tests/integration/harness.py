"""A daemon in this process on a temp socket (WAV capture, fake backend), and test clients."""

import asyncio
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from stt_daemon.audio.capture import CaptureSource
from stt_daemon.audio.pw_capture import AppTarget, PwRecordCapture
from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.base import Backend, CommitPolicy
from stt_daemon.backends.fake import FAKE_CAPABILITIES, FakeBackend, FakeReply
from stt_daemon.config import Config
from stt_daemon.credentials import Credentials
from stt_daemon.server.backend_set import Setup, load_setup
from stt_daemon.server.socket import Server
from stt_daemon.server.sources import SourceScan
from stt_daemon.server.starts import system_capture as real_system_capture

FIXTURE = Path(__file__).parents[1] / "fixtures" / "two_phrases.wav"
"""TTS speech, 16 kHz: phrases at 1.00-3.74 s and 5.74-7.81 s."""
RATE = 16000

type Msg = dict[str, Any]


def fake(
    replies: list[FakeReply | str] | None = None, interval_s: float | None = None
) -> FakeBackend:
    replies = replies or []
    policy = CommitPolicy(interval_s=interval_s, check_every_s=0.1)
    return FakeBackend(replies, capabilities=replace(FAKE_CAPABILITIES, commit_policy=policy))


def setup_with(config: Config, backends: dict[str, Backend]) -> Setup:
    factories = {name: _returning(b) for name, b in backends.items()}
    return load_setup(config, Credentials(), factories)


def _returning(backend: Backend) -> Callable[[Config, Credentials], Backend]:
    return lambda _config, _credentials: backend


def capture_ms(fixture: np.ndarray, audio: np.ndarray, seconds: float) -> float:
    """Where in the fixture the audio sent at `seconds` was captured, in ms."""
    block = audio[round(seconds * RATE) :][:160]
    for i in np.flatnonzero(fixture == block[0]):
        if np.array_equal(fixture[i : i + block.size], block):
            return i / RATE * 1000
    raise AssertionError(f"audio at {seconds} s not found in the fixture")


class TestClient:
    __test__ = False

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.received: list[Msg] = []
        self.eof = asyncio.Event()
        self._ids = 0
        self._task = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            while line := await self.reader.readline():
                self.received.append(json.loads(line))
        except ConnectionError:
            pass
        finally:
            self.eof.set()

    def send(self, kind: str, data: Msg | None = None, *, id: str | None = None) -> str:
        if id is None:
            self._ids += 1
            id = f"r{self._ids}"
        msg: Msg = {"type": kind, "id": id}
        if data is not None:
            msg["data"] = data
        self.writer.write(json.dumps(msg).encode() + b"\n")
        return id

    async def wait(self, match: Callable[[Msg], bool], timeout: float = 10.0) -> Msg:
        deadline = time.monotonic() + timeout
        while True:
            for msg in self.received:
                if match(msg):
                    return msg
            if self.eof.is_set() or time.monotonic() > deadline:
                types = [m["type"] for m in self.received]
                raise AssertionError(f"no matching message; got {types}")
            await asyncio.sleep(0.01)

    async def event(self, kind: str, timeout: float = 10.0, **data: Any) -> Msg:
        """The first `kind` message whose data has these values."""
        return await self.wait(
            lambda m: m["type"] == kind and all(m["data"].get(k) == v for k, v in data.items()),
            timeout,
        )

    async def request(self, kind: str, data: Msg | None = None, timeout: float = 10.0) -> Msg:
        """Sends a command; returns its reply."""
        id = self.send(kind, data)
        return await self.wait(lambda m: m.get("reply_to") == id, timeout)

    def of_type(self, kind: str) -> list[Msg]:
        return [m for m in self.received if m["type"] == kind]

    async def close(self) -> None:
        self.writer.close()
        await self.eof.wait()


class Harness:
    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="sttd-"))
        self.path = self.dir / "run" / "stt.sock"
        self.server: Server | None = None
        self.capture: CaptureSource | None = None
        self.clients: list[TestClient] = []
        self.reloads: list[Setup | Exception] = []
        """What successive reloads return or raise."""

    async def start(
        self,
        backend: Backend | None = None,
        *,
        capture: CaptureSource | None = None,
        config: Config | None = None,
        backends: dict[str, Backend] | None = None,
        system_capture: Callable[[AppTarget | None], PwRecordCapture] | None = None,
        sources: SourceScan | None = None,
    ) -> Server:
        """Without `sources`, no source scan runs."""
        config = config or Config()
        if backends is None:
            backends = {"fake": backend or fake()}
        self.capture = capture = capture or WavCapture()
        await capture.start()
        self.server = server = Server(
            path=self.path,
            capture=capture,
            setup=setup_with(config, backends),
            reload_setup=self._reload,
            system_capture=system_capture or real_system_capture,
            sources=sources or SourceScan(enabled=False),
        )
        await server.start()
        return server

    def _reload(self) -> Setup:
        outcome = self.reloads.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def raw(self) -> TestClient:
        reader, writer = await asyncio.open_unix_connection(self.path)
        client = TestClient(reader, writer)
        self.clients.append(client)
        return client

    async def connect(self, label: str | None = None, **hello: Any) -> tuple[TestClient, Msg]:
        client = await self.raw()
        welcome = await client.request("hello", {"protocol": 1, "label": label, **hello})
        assert welcome["type"] == "welcome", welcome
        return client, welcome

    async def stop(self) -> None:
        for client in self.clients:
            client.writer.close()
        if self.server is not None:
            await self.server.close()
        if self.capture is not None:
            await self.capture.stop()
        shutil.rmtree(self.dir, ignore_errors=True)
