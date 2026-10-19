"""The shared recognizer: load at open, decodes in turn, idle unload; no real model."""

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stt_daemon import daemon
from stt_daemon.audio.frames import Samples
from stt_daemon.backends import parakeet
from stt_daemon.backends.base import (
    BackendEvent,
    Closed,
    Committed,
    Failed,
    OpenFailed,
    OpenOptions,
)
from stt_daemon.backends.parakeet import (
    MODEL_FILES,
    Decoded,
    Engine,
    ParakeetBackend,
    RecognizerKey,
    model_dir,
)
from stt_daemon.config import Config, ParakeetConfig


def _audio(seconds: float) -> Samples:
    return np.zeros(round(seconds * 16000), np.float32)


async def _until(check: Any, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.005)


class _Recognizer:
    """Decodes any audio to its sample count; records hotwords and how many decode at once."""

    def __init__(self, key: RecognizerKey, decode_s: float = 0.0) -> None:
        self.key = key
        self.decode_s = decode_s
        self.hotwords: list[str | None] = []
        self.running = 0
        self.most_running = 0
        self._count = threading.Lock()

    def create_stream(self, hotwords: str | None = None) -> Any:
        self.hotwords.append(hotwords)

        class Stream:
            def accept_waveform(self, rate: int, audio: Samples) -> None:
                text = f" {len(audio)}"
                self.result = Decoded(text, [text], [0.0], [0.1])

        return Stream()

    def decode_stream(self, stream: Any) -> None:
        with self._count:
            self.running += 1
            self.most_running = max(self.most_running, self.running)
        time.sleep(self.decode_s)
        with self._count:
            self.running -= 1


class _Loader:
    def __init__(self, decode_s: float = 0.0, fail: Exception | None = None) -> None:
        self.keys: list[RecognizerKey] = []
        self.made: list[_Recognizer] = []
        self.gate = threading.Event()
        self.gate.set()
        self.started = threading.Event()
        self.decode_s = decode_s
        self.fail = fail

    def __call__(self, key: RecognizerKey) -> _Recognizer:
        self.keys.append(key)
        self.started.set()
        self.gate.wait(5)
        if self.fail is not None:
            raise self.fail
        self.made.append(_Recognizer(key, self.decode_s))
        return self.made[-1]


def _model(tmp: Path) -> Path:
    model = model_dir({"XDG_DATA_HOME": str(tmp)})
    model.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        (model / name).write_text("x")
    return model


def _backend(tmp: Path, engine: Engine, **config: Any) -> ParakeetBackend:
    return ParakeetBackend(ParakeetConfig(**config), _model(tmp), engine)


def _texts(events: list[BackendEvent]) -> list[str]:
    return [e.text for e in events if isinstance(e, Committed)]


async def test_open_returns_before_the_load_and_the_first_decode_waits_for_it(
    tmp_path: Path,
) -> None:
    loader = _Loader()
    loader.gate.clear()
    events: list[BackendEvent] = []
    conn = await _backend(tmp_path, Engine(loader)).open(OpenOptions(), 1, events.append)
    await asyncio.to_thread(loader.started.wait, 2)
    await conn.send_audio(_audio(1.0))
    await conn.commit()
    await asyncio.sleep(0.05)
    assert events == []
    loader.gate.set()
    await _until(lambda: events)
    assert _texts(events) == ["16000"]
    await conn.close()


async def test_sessions_share_one_recognizer_and_decode_in_turn(tmp_path: Path) -> None:
    loader = _Loader(decode_s=0.05)
    engine = Engine(loader)
    plain = _backend(tmp_path, engine)
    jargon = _backend(tmp_path, engine, keyterms=["Scribbly"])
    events: list[BackendEvent] = []
    a = await plain.open(OpenOptions(), 1, events.append)
    b = await jargon.open(OpenOptions(keyterms=("Talkey",)), 2, events.append)
    for _ in range(3):
        for conn in (a, b):
            await conn.send_audio(_audio(0.5))
            await conn.commit()
    await _until(lambda: len(events) == 6)
    [recognizer] = loader.made
    assert recognizer.most_running == 1
    assert sorted(recognizer.hotwords, key=str) == [None] * 3 + ["Scribbly/Talkey"] * 3
    await a.close()
    await b.close()
    c = await jargon.open(OpenOptions(), 3, events.append)
    await c.send_audio(_audio(0.5))
    await c.commit()
    await _until(lambda: len(_texts(events)) == 7)
    await c.close()
    assert len(loader.keys) == 1


async def test_a_closed_connection_mid_decode_keeps_the_next_decode_waiting(
    tmp_path: Path,
) -> None:
    loader = _Loader(decode_s=0.2)
    engine = Engine(loader)
    backend = _backend(tmp_path, engine)
    events: list[BackendEvent] = []
    first = await backend.open(OpenOptions(), 1, events.append)
    await first.send_audio(_audio(1.0))
    await first.commit()
    await _until(lambda: loader.made and loader.made[0].running)
    await first.close()
    second = await backend.open(OpenOptions(), 2, events.append)
    await second.send_audio(_audio(1.0))
    await second.commit()
    await _until(lambda: any(isinstance(e, Committed) for e in events))
    assert loader.made[0].most_running == 1
    await second.close()


async def test_the_recognizer_unloads_after_the_idle_time_and_a_new_session_stops_that(
    tmp_path: Path,
) -> None:
    loader = _Loader()
    engine = Engine(loader)
    backend = _backend(tmp_path, engine, unload_after_minutes=0.15 / 60)
    conn = await backend.open(OpenOptions(), 1, lambda _: None)
    await _until(lambda: engine.loaded)
    await conn.close()
    await asyncio.sleep(0.05)
    conn = await backend.open(OpenOptions(), 2, lambda _: None)
    await asyncio.sleep(0.3)
    assert engine.loaded
    await conn.close()
    assert engine.loaded
    await _until(lambda: not engine.loaded)
    assert len(loader.keys) == 1
    conn = await backend.open(OpenOptions(), 3, lambda _: None)
    await _until(lambda: engine.loaded)
    await conn.close()
    assert len(loader.keys) == 2


async def test_zero_unload_minutes_keep_the_recognizer(tmp_path: Path) -> None:
    engine = Engine(_Loader())
    conn = await _backend(tmp_path, engine, unload_after_minutes=0).open(
        OpenOptions(), 1, lambda _: None
    )
    await _until(lambda: engine.loaded)
    await conn.close()
    await asyncio.sleep(0.05)
    assert engine.loaded


async def test_unload_when_idle_waits_for_the_last_connection(tmp_path: Path) -> None:
    engine = Engine(_Loader())
    backend = _backend(tmp_path, engine)
    a = await backend.open(OpenOptions(), 1, lambda _: None)
    b = await backend.open(OpenOptions(), 2, lambda _: None)
    await _until(lambda: engine.loaded)
    engine.unload_when_idle()
    await a.close()
    assert engine.loaded
    await b.close()
    assert not engine.loaded
    assert engine.users == 0


async def test_a_load_failure_ends_the_connection_with_model_error_and_the_next_open_retries(
    tmp_path: Path,
) -> None:
    loader = _Loader(fail=RuntimeError("protobuf parsing failed"))
    engine = Engine(loader)
    backend = _backend(tmp_path, engine)
    events: list[BackendEvent] = []
    conn = await backend.open(OpenOptions(), 1, events.append)
    await conn.commit()
    await _until(lambda: Closed(1) in events)
    [failed, closed] = events
    assert isinstance(failed, Failed)
    assert (failed.code, failed.retryable) == ("model_error", False)
    assert "protobuf parsing failed" in failed.message
    assert "backend install parakeet" in failed.message
    assert closed == Closed(1)
    assert engine.users == 0
    loader.fail = None
    events.clear()
    conn = await backend.open(OpenOptions(), 2, events.append)
    await conn.send_audio(_audio(0.5))
    await conn.commit()
    await _until(lambda: events)
    assert _texts(events) == ["8000"]
    await conn.close()


async def test_open_refuses_a_set_language_and_a_deleted_model(tmp_path: Path) -> None:
    engine = Engine(_Loader())
    backend = _backend(tmp_path, engine)
    with pytest.raises(OpenFailed) as caught:
        await backend.open(OpenOptions(language="de"), 1, lambda _: None)
    assert caught.value.code == "unsupported_language"
    await (await backend.open(OpenOptions(language="auto"), 1, lambda _: None)).close()
    (model_dir({"XDG_DATA_HOME": str(tmp_path)}) / "encoder.int8.onnx").unlink()
    with pytest.raises(OpenFailed) as caught:
        await backend.open(OpenOptions(), 2, lambda _: None)
    assert caught.value.code == "model_error"
    assert engine.users == 0


async def test_recognizer_settings_come_from_config(tmp_path: Path) -> None:
    loader = _Loader()
    engine = Engine(loader)
    backend = _backend(tmp_path, engine, num_threads=3, hotwords_score=2.0, max_chunk_seconds=12)
    assert backend.capabilities.commit_policy.max_chunk_s == 12
    assert backend.capabilities.commit_policy.at_pauses
    await (await backend.open(OpenOptions(), 1, lambda _: None)).close()
    await asyncio.to_thread(loader.started.wait, 2)
    [key] = loader.keys
    assert (key.num_threads, key.hotwords_score) == (3, 2.0)


async def test_a_reload_without_the_model_unloads_the_recognizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = Engine(_Loader())
    monkeypatch.setattr(parakeet, "ENGINE", engine)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    backend = _backend(tmp_path, engine, unload_after_minutes=0)
    await (await backend.open(OpenOptions(), 1, lambda _: None)).close()
    await _until(lambda: engine.loaded)
    assert "parakeet" in daemon.reload(Config(), None, None).backends
    assert engine.loaded
    for name in MODEL_FILES:
        (model_dir() / name).unlink()
    assert "parakeet" not in daemon.reload(Config(), None, None).backends
    assert not engine.loaded


async def test_new_recognizer_settings_wait_until_no_session_uses_the_old_one(
    tmp_path: Path,
) -> None:
    loader = _Loader()
    engine = Engine(loader)
    old = await _backend(tmp_path, engine, num_threads=2).open(OpenOptions(), 1, lambda _: None)
    await _until(lambda: engine.loaded)
    new_backend = _backend(tmp_path, engine, num_threads=3)
    events: list[BackendEvent] = []
    new = await new_backend.open(OpenOptions(), 2, events.append)
    await new.send_audio(_audio(0.5))
    await new.commit()
    await _until(lambda: events)
    assert [k.num_threads for k in loader.keys] == [2]
    await old.close()
    await new.close()
    await (await new_backend.open(OpenOptions(), 3, lambda _: None)).close()
    await _until(lambda: len(loader.made) == 2)
    assert [k.num_threads for k in loader.keys] == [2, 3]


async def test_closing_one_session_during_the_load_leaves_it_to_the_other(
    tmp_path: Path,
) -> None:
    loader = _Loader()
    loader.gate.clear()
    engine = Engine(loader)
    backend = _backend(tmp_path, engine)
    events: list[BackendEvent] = []
    a = await backend.open(OpenOptions(), 1, lambda _: None)
    b = await backend.open(OpenOptions(), 2, events.append)
    await b.send_audio(_audio(0.5))
    await b.commit()
    await asyncio.to_thread(loader.started.wait, 2)
    await a.close()
    loader.gate.set()
    await _until(lambda: events)
    assert _texts(events) == ["8000"]
    assert len(loader.keys) == 1
    engine.unload_when_idle()
    await b.close()
    assert not engine.loaded


async def test_a_decode_left_running_by_a_closed_session_keeps_the_old_settings(
    tmp_path: Path,
) -> None:
    loader = _Loader(decode_s=0.3)
    engine = Engine(loader)
    old = await _backend(tmp_path, engine, num_threads=2).open(OpenOptions(), 1, lambda _: None)
    await old.send_audio(_audio(1.0))
    await old.commit()
    await _until(lambda: loader.made and loader.made[0].running)
    await old.close()
    new_backend = _backend(tmp_path, engine, num_threads=3)
    new = await new_backend.open(OpenOptions(), 2, lambda _: None)
    assert [k.num_threads for k in loader.keys] == [2]
    await new.close()
    await _until(lambda: engine.users == 0)
    await (await new_backend.open(OpenOptions(), 3, lambda _: None)).close()
    await _until(lambda: len(loader.made) == 2)
    assert [k.num_threads for k in loader.keys] == [2, 3]


async def test_a_dictation_decode_goes_before_waiting_decodes_of_others() -> None:
    engine = Engine()
    recognizer = _Recognizer(RecognizerKey(Path("m"), 1, 1.5), decode_s=0.1)
    done: list[str] = []

    async def decode(name: str, *, first: bool) -> None:
        await engine.decode(recognizer, _audio(0.2), None, first=first)
        done.append(name)

    running = asyncio.create_task(decode("file", first=False))
    await _until(lambda: recognizer.running == 1)
    waiting = [asyncio.create_task(decode(f"file-{i}", first=False)) for i in (1, 2)]
    await asyncio.sleep(0.01)
    dictation = asyncio.create_task(decode("mic", first=True))
    await asyncio.gather(running, *waiting, dictation)
    assert done == ["file", "mic", "file-1", "file-2"]
    assert recognizer.most_running == 1


async def test_a_cancelled_waiter_passes_its_turn_on() -> None:
    turns = parakeet.Turns()
    await turns.acquire(first=False)
    gone = asyncio.create_task(turns.acquire(first=True))
    after = asyncio.create_task(turns.acquire(first=False))
    await asyncio.sleep(0)
    gone.cancel()
    await asyncio.wait([gone])
    turns.release()
    await asyncio.wait_for(after, 1.0)
    turns.release()
    await asyncio.wait_for(turns.acquire(first=False), 1.0)
