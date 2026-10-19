"""The example plugin (tests/fixtures/plugins) against a local stand-in for its HTTP API."""

import asyncio
import base64
import io
import json
import shutil
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest
from fake_elevenlabs_batch import FakeBatch, serve_batch
from harness import FIXTURE, Harness, Msg, TestClient

from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends import BACKENDS
from stt_daemon.backends.plugins import load_plugins
from stt_daemon.cli import main
from stt_daemon.config import AudioConfig, Config
from stt_daemon.credentials import Credentials, load_credentials
from stt_daemon.server.backend_set import Setup, load_setup

PLUGINS = Path(__file__).parents[1] / "fixtures" / "plugins"
PHRASES = [
    ((1000, 3740), "The quick brown fox jumps over the lazy dog."),
    ((5740, 7810), "Please send the report before noon tomorrow."),
]
EDGE_MS = 150


@pytest.fixture
def api() -> Iterator[FakeBatch]:
    with serve_batch() as fake:
        fake.default = (200, json.dumps({"text": ""}).encode())
        yield fake


def _config(api: FakeBatch) -> Config:
    config = Config(audio=AudioConfig(echo_cancel="off"))
    config.backends.plugins["openai"] = {"endpoint": api.url, "model": "test-model"}
    return config


def _setup(config: Config, credentials: Credentials) -> Setup:
    return load_setup(config, credentials, {}, load_plugins(PLUGINS, BACKENDS))


def _answer(api: FakeBatch, *texts: str) -> None:
    for text in texts:
        api.reply(200, {"text": text})


def _text_segments(client: TestClient) -> list[Msg]:
    return [m["data"] for m in client.of_type("segment") if m["data"]["text"]]


async def test_a_dictation_on_the_plugin_gets_estimated_word_times_and_a_marker(
    daemon: Harness, api: FakeBatch
) -> None:
    _answer(api, *(text for _, text in PHRASES))
    capture = WavCapture(FIXTURE)
    setup = _setup(_config(api), Credentials(file_keys={"openai": "sk-test"}))
    await daemon.start(capture=capture, setup=setup)
    client, welcome = await daemon.connect("Scribbly")
    assert welcome["data"]["default_backend"] == "openai"
    started = await client.request("start_session", {"word_timestamps": True})
    assert started["type"] == "session_started", started
    at = started["data"]["started_at"] + 6500  # during "report before"
    await asyncio.sleep(max(0.0, at / 1000 - time.time()))
    marked = await client.request("add_marker", {"marker": "⟦A⟧", "at": at})
    assert marked["type"] == "marker_accepted", marked
    await asyncio.to_thread(capture.wait_idle, 25)
    ended = await client.request("stop_session")
    assert ended["type"] == "session_ended", ended

    segments = _text_segments(client)
    assert [s["text"].replace(" ⟦A⟧", "") for s in segments] == [text for _, text in PHRASES]
    for segment, ((start_ms, end_ms), _) in zip(segments, PHRASES, strict=True):
        words = segment["words"]
        assert abs(words[0]["start_ms"] - start_ms) <= EDGE_MS, words[0]
        assert abs(words[-1]["end_ms"] - end_ms) <= EDGE_MS, words[-1]
    [marker] = segments[1]["markers"]
    assert 0 < marker["after_word"] < len(segments[1]["words"]) - 1
    assert "⟦A⟧" in ended["data"]["final_text"].split("Please", 1)[1]

    upload = api.uploads[0]
    assert upload.headers["authorization"] == "Bearer sk-test"
    assert upload.value("model") == "test-model"
    with wave.open(io.BytesIO(upload.file)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)


async def test_a_transcribe_session_on_the_plugin(daemon: Harness, api: FakeBatch) -> None:
    _answer(api, "one two", "three four")
    setup = _setup(_config(api), Credentials(file_keys={"openai": "sk-test"}))
    await daemon.start(setup=setup)
    client, _ = await daemon.connect("Talkey")
    started = await client.request("transcribe", {"pcm": {"rate": 16000, "channels": 1}})
    assert started["type"] == "session_started", started
    with wave.open(str(FIXTURE)) as wav:
        pcm = wav.readframes(wav.getnframes())
    for at in range(0, len(pcm), 48 * 1024):
        client.send("audio", {"data": base64.b64encode(pcm[at : at + 48 * 1024]).decode()})
    ended = await client.request("stop_session")
    assert ended["data"]["final_text"] == "one two three four"


async def test_an_api_error_ends_the_session_with_backend_error(
    daemon: Harness, api: FakeBatch
) -> None:
    api.reply(401, {"error": "bad key"})
    capture = WavCapture(FIXTURE)
    setup = _setup(_config(api), Credentials(file_keys={"openai": "sk-test"}))
    await daemon.start(capture=capture, setup=setup)
    client, _ = await daemon.connect("Scribbly")
    assert (await client.request("start_session", {}))["type"] == "session_started"
    error = await client.event("session_error", timeout=15)
    assert error["data"]["code"] == "backend_error"
    assert "HTTP 401" in error["data"]["message"]
    assert "sk-test" not in error["data"]["message"]
    await client.event("session_ended")


async def test_key_set_then_reload_makes_the_plugin_ready(
    daemon: Harness,
    api: FakeBatch,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    shutil.copytree(PLUGINS, tmp_path / "stt-daemon" / "backends")
    monkeypatch.setenv("STT_DAEMON_SOCKET", str(daemon.path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    config = _config(api)
    await daemon.start(
        setup=_setup(config, load_credentials()),
        reload_setup=lambda: _setup(config, load_credentials()),
    )
    client, welcome = await daemon.connect("Talkey")
    assert welcome["data"]["backends"] == []
    status = await client.request("backends_status")
    openai = next(b for b in status["data"]["backends"] if b["name"] == "openai")
    assert openai["status"] == "no key"

    monkeypatch.setattr("sys.stdin", io.StringIO("sk-new\n"))
    assert await asyncio.to_thread(main, ["key", "set", "openai"]) == 0
    event = await client.event("backends")
    assert [b["name"] for b in event["data"]["backends"]] == ["openai"]
    status = await client.request("backends_status")
    openai = next(b for b in status["data"]["backends"] if b["name"] == "openai")
    assert openai == {"name": "openai", "status": "ready", "default": True}


async def test_install_and_set_default_by_plugin_name_reach_the_daemon(
    daemon: Harness,
    api: FakeBatch,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    plugin_dir = tmp_path / "stt-daemon" / "backends"
    shutil.copytree(PLUGINS, plugin_dir)
    monkeypatch.setenv("STT_DAEMON_SOCKET", str(daemon.path))
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    config = _config(api)

    def current() -> Setup:
        return load_setup(config, load_credentials(), {}, load_plugins(plugin_dir, BACKENDS))

    await daemon.start(setup=current(), reload_setup=current)
    client, welcome = await daemon.connect("Talkey")
    assert welcome["data"]["default_backend"] == "openai"

    assert await asyncio.to_thread(main, ["backend", "install", "local-model"]) == 0
    event = await client.event("backends")
    assert [b["name"] for b in event["data"]["backends"]] == ["local-model", "openai"]

    assert await asyncio.to_thread(main, ["backend", "set-default", "local-model"]) == 0
    event = await client.wait(
        lambda m: m["type"] == "backends" and m["data"]["default_backend"] == "local-model"
    )
    started = await client.request("start_session", {})
    assert started["data"]["backend"] == "local-model"
