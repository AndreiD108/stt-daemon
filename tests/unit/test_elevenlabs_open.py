import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fake_elevenlabs import TEST_KEY, FakeElevenLabs

from stt_daemon.backends import elevenlabs
from stt_daemon.backends.base import BackendEvent, BackendUnavailable, OpenFailed, OpenOptions
from stt_daemon.backends.elevenlabs import ElevenLabsBackend
from stt_daemon.config import Config, ElevenLabsConfig
from stt_daemon.credentials import Credentials


class FakeWS:
    def __init__(self, messages: list[str] | None = None) -> None:
        self._items = list(messages or [])
        self._closed = asyncio.Event()

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[str]:
        for item in self._items:
            yield item
        await self._closed.wait()

    async def send(self, data: str) -> None:
        pass

    async def close(self) -> None:
        self._closed.set()


def _started() -> str:
    return json.dumps({"message_type": "session_started", "session_id": "s"})


def _backend() -> ElevenLabsBackend:
    return ElevenLabsBackend(ElevenLabsConfig(endpoint="wss://example.invalid/rt"), TEST_KEY)


async def _open(backend: ElevenLabsBackend) -> tuple[Any, list[BackendEvent]]:
    events: list[BackendEvent] = []
    return await backend.open(OpenOptions(), 1, events.append), events


@pytest.fixture
def quick_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(elevenlabs, "_ATTEMPT_TIMEOUTS_S", (0.05, 0.05, 1.0))


async def test_retries_a_handshake_timeout_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def connect(*_args: object, **_kwargs: object) -> FakeWS:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("timed out during opening handshake")
        return FakeWS([_started()])

    monkeypatch.setattr(elevenlabs.websockets, "connect", connect)
    conn, _ = await _open(_backend())
    assert calls == 2
    await conn.close()


async def test_gives_up_after_the_attempt_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def connect(*_args: object, **_kwargs: object) -> FakeWS:
        nonlocal calls
        calls += 1
        raise OSError("connection refused")

    monkeypatch.setattr(elevenlabs.websockets, "connect", connect)
    with pytest.raises(OpenFailed) as failed:
        await _open(_backend())
    assert (failed.value.code, failed.value.retryable) == ("network", True)
    assert calls == len(elevenlabs._ATTEMPT_TIMEOUTS_S)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.usefixtures("quick_attempts")
async def test_cuts_a_hung_handshake_at_the_attempt_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def connect(*_args: object, **_kwargs: object) -> FakeWS:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.Event().wait()
        return FakeWS([_started()])

    monkeypatch.setattr(elevenlabs.websockets, "connect", connect)
    loop = asyncio.get_running_loop()
    started = loop.time()
    conn, _ = await _open(_backend())
    assert calls == 2
    assert loop.time() - started < 0.5
    await conn.close()


@pytest.mark.usefixtures("quick_attempts")
async def test_retries_when_session_started_never_comes(monkeypatch: pytest.MonkeyPatch) -> None:
    sockets: list[FakeWS] = []

    async def connect(*_args: object, **_kwargs: object) -> FakeWS:
        sockets.append(FakeWS([] if not sockets else [_started()]))
        return sockets[-1]

    monkeypatch.setattr(elevenlabs.websockets, "connect", connect)
    conn, events = await _open(_backend())
    assert len(sockets) == 2
    assert sockets[0]._closed.is_set()  # pyright: ignore[reportPrivateUsage]
    assert events == []
    await conn.close()


async def test_refused_key_fails_at_once_without_retrying(fake_el: FakeElevenLabs) -> None:
    fake_el.status = 401
    backend = ElevenLabsBackend(ElevenLabsConfig(endpoint=fake_el.url), TEST_KEY)
    with pytest.raises(OpenFailed) as failed:
        await _open(backend)
    assert (failed.value.code, failed.value.retryable) == ("auth", False)
    assert len(fake_el.requests) == 1


@pytest.mark.usefixtures("quick_attempts")
async def test_server_errors_on_the_handshake_are_retried(fake_el: FakeElevenLabs) -> None:
    fake_el.status = 503
    backend = ElevenLabsBackend(ElevenLabsConfig(endpoint=fake_el.url), TEST_KEY)
    with pytest.raises(OpenFailed) as failed:
        await _open(backend)
    assert (failed.value.code, failed.value.retryable) == ("backend_error", True)
    assert len(fake_el.requests) == 3


async def test_error_message_before_session_started_fails_the_open(
    fake_el: FakeElevenLabs,
) -> None:
    fake_el.greeting = [{"message_type": "auth_error", "error": "invalid api key"}]
    backend = ElevenLabsBackend(ElevenLabsConfig(endpoint=fake_el.url), TEST_KEY)
    events: list[BackendEvent] = []
    with pytest.raises(OpenFailed) as failed:
        await backend.open(OpenOptions(), 1, events.append)
    assert (failed.value.code, failed.value.retryable) == ("auth", False)
    assert "invalid api key" in failed.value.message
    assert events == []
    assert len(fake_el.requests) == 1


async def test_key_stays_out_of_debug_logs(
    fake_el: FakeElevenLabs, caplog: pytest.LogCaptureFixture
) -> None:
    backend = ElevenLabsBackend(ElevenLabsConfig(endpoint=fake_el.url), TEST_KEY)
    with caplog.at_level(logging.DEBUG):
        conn, _ = await _open(backend)
        await conn.close()
    ours = [r for r in caplog.records if not r.name.startswith("websockets.server")]
    assert any(r.name.startswith("stt_daemon.backends.elevenlabs") for r in ours)
    assert not [r for r in ours if TEST_KEY in r.getMessage()]
    assert TEST_KEY not in repr(backend)


def test_create_needs_a_key() -> None:
    with pytest.raises(BackendUnavailable) as missing:
        ElevenLabsBackend.create(Config(), Credentials())
    assert missing.value.code == "no_key"
    assert "stt-daemon key set elevenlabs" in missing.value.message
    backend = ElevenLabsBackend.create(Config(), Credentials(file_keys={"elevenlabs": TEST_KEY}))
    assert isinstance(backend, ElevenLabsBackend)


async def test_cancelled_open_closes_the_socket_and_stays_silent(
    fake_el: FakeElevenLabs,
) -> None:
    fake_el.greeting = []
    backend = ElevenLabsBackend(ElevenLabsConfig(endpoint=fake_el.url), TEST_KEY)
    events: list[BackendEvent] = []
    opening = asyncio.create_task(backend.open(OpenOptions(), 1, events.append))
    async with asyncio.timeout(1):
        while not fake_el.connections:
            await asyncio.sleep(0.005)
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening
    server_side = fake_el.connections[0]
    async with asyncio.timeout(1):
        await server_side.wait_closed()
    await asyncio.sleep(0.02)
    assert events == []
    assert len(fake_el.connections) == 1
