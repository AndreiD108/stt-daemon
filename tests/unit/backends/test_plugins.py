import asyncio
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stt_daemon import plugin
from stt_daemon.backends import BACKENDS
from stt_daemon.backends.base import BackendEvent, Committed, Failed, OpenOptions
from stt_daemon.backends.chunked import Chunk
from stt_daemon.backends.plugins import (
    Limits,
    PluginBackend,
    Plugins,
    load_plugins,
    to_transcribed,
)
from stt_daemon.config import Config
from stt_daemon.credentials import Credentials
from stt_daemon.server.backend_set import Setup, load_setup, resolve
from stt_daemon.server.messages import RequestError, parse_start
from unit.paths import PLUGIN_FIXTURES as FIXTURES
from unit.paths import REPO

RATE = 16000


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    """The fixture plugins and a file that does not parse."""
    target = tmp_path / "backends"
    shutil.copytree(FIXTURES, target, ignore=shutil.ignore_patterns("__pycache__"))
    (target / "syntax_error.py").write_text("class Broken(\n")
    return target


def _setup(plugins: Plugins, keys: Mapping[str, str] | None = None) -> Setup:
    return load_setup(Config(), Credentials(file_keys=dict(keys or {})), {}, plugins)


def test_good_plugins_load_and_each_broken_file_fails_alone(plugin_dir: Path) -> None:
    plugins = load_plugins(plugin_dir, BACKENDS)
    assert list(plugins.classes) == ["local-model", "openai", "raising-setup"]
    failed = {file: p.message for file, p in plugins.failed.items()}
    assert set(failed) == {"builtin_name.py", "syntax_error.py", "two_classes.py"}
    assert failed["builtin_name.py"].startswith("'parakeet' is a built-in backend's name")
    assert failed["two_classes.py"].startswith("defines 2 Backend subclasses (First, Second)")
    assert failed["syntax_error.py"].startswith("SyntaxError: ")
    assert plugins.failed["syntax_error.py"].detail is not None


def test_status_shows_ready_no_key_not_installed_and_failed(plugin_dir: Path) -> None:
    setup = _setup(load_plugins(plugin_dir, BACKENDS))
    status = setup.status_wire()
    backends = {b["name"]: b["status"] for b in status["backends"]}
    assert backends == {
        "local-model": "not installed",
        "openai": "no key",
        "raising-setup": "failed: RuntimeError: no device found",
    }
    files = {f["file"]: f["status"] for f in status["failed_files"]}
    assert files["two_classes.py"].startswith("failed: defines 2")
    assert setup.default is None


def test_a_key_makes_the_plugin_ready_and_the_default(plugin_dir: Path) -> None:
    setup = _setup(load_plugins(plugin_dir, BACKENDS), {"openai": "sk-1"})
    assert "openai" in setup.backends
    assert setup.default == "openai"
    entry = next(b for b in setup.backends_wire() if b["name"] == "openai")
    assert entry["word_timestamps"] is True
    assert entry["hints"] == ["previous_text"]
    assert entry["kinds"] == ["mic", "system", "transcribe"]


def test_a_failed_plugin_is_an_unknown_backend_naming_why(plugin_dir: Path) -> None:
    setup = _setup(load_plugins(plugin_dir, BACKENDS), {"openai": "sk-1"})
    with pytest.raises(RequestError) as caught:
        resolve(setup, parse_start({"backend": "raising-setup"}), lambda _: None)
    assert caught.value.code == "unknown_backend"
    assert "RuntimeError: no device found" in caught.value.message


def test_failures_are_logged_once_each_with_the_traceback(
    plugin_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    setup = _setup(load_plugins(plugin_dir, BACKENDS))
    with caplog.at_level(logging.ERROR):
        setup.log_failures()
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 4  # raising-setup, builtin_name, syntax_error, two_classes
    raising = next(line for line in lines if "raising-setup" in line)
    assert "Traceback" in raising
    assert "second line" in raising


def test_two_files_with_one_name_keep_the_first(tmp_path: Path) -> None:
    source = (FIXTURES / "local_model.py").read_text()
    (tmp_path / "a.py").write_text(source)
    (tmp_path / "b.py").write_text(source)
    plugins = load_plugins(tmp_path, BACKENDS)
    assert list(plugins.classes) == ["local-model"]
    assert "also the name of the plugin in a.py" in plugins.failed["b.py"].message


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("x = 1\n", "defines no stt_daemon.plugin.Backend subclass"),
        ("class P(Backend):\n    name = 'Bad Name'\n", "P.name must be lowercase"),
        ("class P(Backend):\n    pass\n", "P.name must be lowercase"),
        ("class P(Backend):\n    name = 'p'\n", "P does not implement transcribe"),
        ("import sys\nsys.exit(3)\n", "SystemExit: 3"),
    ],
)
def test_a_malformed_plugin_fails_with_its_reason(tmp_path: Path, body: str, expected: str) -> None:
    (tmp_path / "p.py").write_text("from stt_daemon.plugin import Backend\n" + body)
    plugins = load_plugins(tmp_path, BACKENDS)
    assert not plugins.classes
    assert plugins.failed["p.py"].message.startswith(expected)


def test_a_missing_directory_has_no_plugins(tmp_path: Path) -> None:
    plugins = load_plugins(tmp_path / "nowhere", BACKENDS)
    assert not plugins.classes
    assert not plugins.failed


def test_plugin_config_reaches_setup_and_sets_limits(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    class Configured(plugin.Backend):
        name = "configured"

        def setup(self, key: str | None, config: Mapping[str, Any]) -> None:
            seen.append({"key": key, **config})

        async def transcribe(self, pcm: bytes, options: plugin.Options) -> plugin.Result:
            return plugin.Result()

    config = Config()
    config.backends.plugins["configured"] = {"max_sessions": 3, "max_chunk_seconds": 12, "x": 1}
    plugins = Plugins({"configured": Configured}, {})
    setup = load_setup(config, Credentials(file_keys={"configured": "k"}), {}, plugins)
    backend = setup.backends["configured"]
    assert seen == [{"key": "k", "max_sessions": 3, "max_chunk_seconds": 12, "x": 1}]
    assert backend.max_sessions == 3
    assert backend.capabilities.commit_policy.max_chunk_s == 12.0

    config.backends.plugins["configured"] = {"max_sessions": 0}
    setup = load_setup(config, Credentials(), {}, plugins)
    message = setup.unavailable["configured"].message
    assert message == "backends.configured.max_sessions must be 1 or more"


class Recorder(plugin.Backend):
    """Answers each chunk with the words it is given; records calls and the options."""

    name = "recorder"
    capabilities = plugin.Capabilities(languages=("auto", "en"), keyterms=True)

    def __init__(self, result: plugin.Result | None = None, delay_s: float = 0.0) -> None:
        self.result = result or plugin.Result("hello there")
        self.delay_s = delay_s
        self.calls: list[tuple[int, plugin.Options]] = []
        self.running = 0
        self.most_at_once = 0

    async def transcribe(self, pcm: bytes, options: plugin.Options) -> plugin.Result:
        self.calls.append((len(pcm), options))
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        try:
            await asyncio.sleep(self.delay_s)
        finally:
            self.running -= 1
        return self.result


async def _until(events: list[BackendEvent], count: int) -> None:
    for _ in range(200):
        if len(events) >= count:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"got {events}")


async def test_each_commit_sends_its_chunk_and_word_times_move_to_connection_time() -> None:
    words = [plugin.Word("one", 0.1, 0.4, "speaker_0"), plugin.Word("two", 0.5, 0.9)]
    recorder = Recorder(plugin.Result(words=words))
    backend = PluginBackend(recorder, Limits(2, 30.0))
    events: list[BackendEvent] = []
    conn = await backend.open(OpenOptions(language="en"), 7, events.append)
    for _ in range(2):
        await conn.send_audio(np.full(RATE, 0.25, np.float32))
        await conn.commit()
    await _until(events, 2)
    first, second = events
    assert isinstance(first, Committed)
    assert isinstance(second, Committed)
    assert (second.connection_id, second.commit_id, second.text) == (7, 2, "one two")
    assert second.words is not None
    assert [(w.text, w.start_s, w.end_s, w.speaker) for w in second.words] == [
        ("one", pytest.approx(1.1), pytest.approx(1.4), "speaker_0"),
        ("two", pytest.approx(1.5), pytest.approx(1.9), None),
    ]
    assert recorder.calls[0][0] == RATE * 2  # s16le bytes
    assert recorder.calls[0][1].language == "en"
    await conn.close()


async def test_audio_reaches_the_plugin_as_s16le() -> None:
    recorder = Recorder()
    got: list[bytes] = []

    async def keep(pcm: bytes, options: plugin.Options) -> plugin.Result:
        got.append(pcm)
        return plugin.Result("x")

    recorder.transcribe = keep  # type: ignore[method-assign]
    backend = PluginBackend(recorder, Limits(2, 30.0))
    events: list[BackendEvent] = []
    conn = await backend.open(OpenOptions(), 1, events.append)
    samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0] * 400, np.float32)
    await conn.send_audio(samples)
    await conn.commit()
    await _until(events, 1)
    pcm = np.frombuffer(got[0], dtype="<i2")
    assert list(pcm[:5]) == [0, 16384, -16384, 32767, -32768]
    await conn.close()


async def test_a_chunk_under_a_tenth_of_a_second_gets_empty_text_without_a_call() -> None:
    recorder = Recorder()
    backend = PluginBackend(recorder, Limits(2, 30.0))
    events: list[BackendEvent] = []
    conn = await backend.open(OpenOptions(), 1, events.append)
    await conn.send_audio(np.zeros(RATE // 20, np.float32))
    await conn.commit()
    await _until(events, 1)
    assert isinstance(events[0], Committed)
    assert events[0].text == ""
    assert not recorder.calls
    await conn.close()


async def test_calls_take_turns_across_connections() -> None:
    recorder = Recorder(delay_s=0.05)
    backend = PluginBackend(recorder, Limits(2, 30.0))
    events: list[BackendEvent] = []
    conns = [await backend.open(OpenOptions(), i, events.append) for i in (1, 2)]
    for conn in conns:
        await conn.send_audio(np.zeros(RATE, np.float32))
        await conn.commit()
    await _until(events, 2)
    assert recorder.most_at_once == 1
    for conn in conns:
        await conn.close()


async def test_a_slow_call_ends_the_connection_with_backend_error() -> None:
    backend = PluginBackend(Recorder(delay_s=5.0), Limits(2, 30.0), timeout_s=0.05)
    events: list[BackendEvent] = []
    conn = await backend.open(OpenOptions(), 3, events.append)
    await conn.send_audio(np.zeros(RATE, np.float32))
    await conn.commit()
    await _until(events, 2)
    failed = events[0]
    assert isinstance(failed, Failed)
    assert (failed.code, failed.retryable) == ("backend_error", False)
    assert failed.message == "recorder gave no answer within 0.05 s"


async def test_a_raising_call_ends_the_connection_with_its_text() -> None:
    recorder = Recorder()

    async def fail(pcm: bytes, options: plugin.Options) -> plugin.Result:
        raise ValueError("HTTP 500: overloaded")

    recorder.transcribe = fail  # type: ignore[method-assign]
    backend = PluginBackend(recorder, Limits(2, 30.0))
    events: list[BackendEvent] = []
    conn = await backend.open(OpenOptions(), 3, events.append)
    await conn.send_audio(np.zeros(RATE, np.float32))
    await conn.commit()
    await _until(events, 2)
    failed = events[0]
    assert isinstance(failed, Failed)
    assert failed.code == "backend_error"
    assert failed.message == "recorder failed: HTTP 500: overloaded"


async def test_a_result_of_the_wrong_type_is_a_backend_error() -> None:
    recorder = Recorder()

    async def wrong(pcm: bytes, options: plugin.Options) -> plugin.Result:
        return "text"  # type: ignore[return-value]

    recorder.transcribe = wrong  # type: ignore[method-assign]
    backend = PluginBackend(recorder, Limits(2, 30.0))
    events: list[BackendEvent] = []
    conn = await backend.open(OpenOptions(), 3, events.append)
    await conn.send_audio(np.zeros(RATE, np.float32))
    await conn.commit()
    await _until(events, 2)
    assert isinstance(events[0], Failed)
    assert "transcribe returned str, not a Result" in events[0].message


def _chunk(start_s: float, end_s: float) -> Chunk:
    return Chunk(np.zeros(round((end_s - start_s) * RATE), np.float32), start_s, end_s)


def test_words_without_times_leave_times_to_the_estimate() -> None:
    result = plugin.Result(words=[plugin.Word("no"), plugin.Word("times", 0.1, 0.2)])
    transcribed = to_transcribed(result, _chunk(2.0, 3.0))
    assert (transcribed.text, transcribed.words) == ("no times", None)


def test_text_alone_leaves_times_to_the_estimate_and_nothing_is_no_words() -> None:
    assert to_transcribed(plugin.Result("  hi  "), _chunk(0, 1)).words is None
    empty = to_transcribed(plugin.Result(), _chunk(0, 1))
    assert (empty.text, empty.words) == ("", ())


def test_word_times_are_kept_inside_the_chunk() -> None:
    result = plugin.Result("a b", [plugin.Word("a", -0.5, 0.2), plugin.Word("b", 0.8, 9.0)])
    words = to_transcribed(result, _chunk(4.0, 5.0)).words
    assert words is not None
    assert [(w.start_s, w.end_s) for w in words] == [
        (4.0, pytest.approx(4.2)),
        (pytest.approx(4.8), 5.0),
    ]


def _resolved_options(backend: PluginBackend, data: dict[str, Any]) -> OpenOptions:
    setup = load_setup(Config(), Credentials(), {"recorder": lambda _c, _k: backend})
    return resolve(setup, parse_start(data), lambda _: "stored text").options.open


async def test_the_plugin_gets_only_the_options_its_capabilities_allow() -> None:
    recorder = Recorder()
    backend = PluginBackend(recorder, Limits(2, 30.0))
    options = _resolved_options(
        backend, {"keyterms": ["Scribbly"], "previous_text": "before", "language": "auto"}
    )
    events: list[BackendEvent] = []
    conn = await backend.open(options, 1, events.append)
    await conn.send_audio(np.zeros(RATE, np.float32))
    await conn.commit()
    await _until(events, 1)
    got = recorder.calls[0][1]
    assert got == plugin.Options(language=None, keyterms=("Scribbly",), previous_text=None)
    await conn.close()


def test_options_the_plugin_lacks_are_refused() -> None:
    backend = PluginBackend(Recorder(), Limits(2, 30.0))
    with pytest.raises(RequestError) as caught:
        _resolved_options(backend, {"diarize": True})
    assert (caught.value.code, caught.value.to_wire()["field"]) == ("unsupported_option", "diarize")
    with pytest.raises(RequestError) as caught:
        _resolved_options(backend, {"language": "ro"})
    assert caught.value.code == "unsupported_language"
    with pytest.raises(RequestError) as caught:
        _resolved_options(backend, {"partials": True})
    assert caught.value.code == "unsupported"


def test_the_docs_example_is_the_fixture_plugin() -> None:
    doc = (REPO / "docs" / "plugins.md").read_text()
    example = (FIXTURES / "openai_transcriptions.py").read_text()
    assert f"```python\n{example}```" in doc


async def test_a_call_its_caller_stopped_waiting_for_still_holds_the_turn() -> None:
    release = asyncio.Event()
    running: list[str] = []

    class Threaded(plugin.Backend):
        name = "threaded"

        async def transcribe(self, pcm: bytes, options: plugin.Options) -> plugin.Result:
            running.append("in")
            try:
                await asyncio.shield(release.wait())  # like asyncio.to_thread: not stopped
            finally:
                running.remove("in")
            return plugin.Result("x")

    backend = PluginBackend(Threaded(), Limits(2, 30.0), timeout_s=0.05)
    events: list[BackendEvent] = []
    first = await backend.open(OpenOptions(), 1, events.append)
    await first.send_audio(np.zeros(RATE, np.float32))
    await first.commit()
    await _until(events, 2)  # timed out: Failed, Closed
    second = await backend.open(OpenOptions(), 2, events.append)
    await second.send_audio(np.zeros(RATE, np.float32))
    await second.commit()
    await asyncio.sleep(0.03)
    assert len(running) == 1  # the second call waits for the first to end
    release.set()
    await _until(events, 3)
    assert isinstance(events[2], Committed)
    await second.close()


async def test_the_key_never_reaches_messages_or_logs(caplog: pytest.LogCaptureFixture) -> None:
    class Leaky(plugin.Backend):
        name = "leaky"
        needs_key = True

        def setup(self, key: str | None, config: Mapping[str, Any]) -> None:
            self.key = key
            if config.get("fail"):
                raise RuntimeError(f"bad key {key}")

        async def transcribe(self, pcm: bytes, options: plugin.Options) -> plugin.Result:
            raise RuntimeError(f"HTTP 401 for {self.key}")

    plugins = Plugins({"leaky": Leaky}, {})
    credentials = Credentials(file_keys={"leaky": "sk-secret-9"})
    config = Config()
    config.backends.plugins["leaky"] = {"fail": True}
    with caplog.at_level(logging.DEBUG):
        failed = load_setup(config, credentials, {}, plugins)
        failed.log_failures()
        backend = load_setup(Config(), credentials, {}, plugins).backends["leaky"]
        events: list[BackendEvent] = []
        conn = await backend.open(OpenOptions(), 1, events.append)
        await conn.send_audio(np.zeros(RATE, np.float32))
        await conn.commit()
        await _until(events, 2)
    assert failed.status_wire()["backends"][0]["status"] == "failed: RuntimeError: bad key [key]"
    assert isinstance(events[0], Failed)
    assert events[0].message == "leaky failed: HTTP 401 for [key]"
    assert "sk-secret-9" not in caplog.text
    assert "[key]" in caplog.text


def test_untimed_words_keep_their_speakers_for_the_estimate() -> None:
    words = [plugin.Word("hi there", speaker="speaker_0"), plugin.Word("yes", speaker="speaker_1")]
    transcribed = to_transcribed(plugin.Result(words=words), _chunk(0, 1))
    assert transcribed.words is None
    assert transcribed.speakers == ("speaker_0", "speaker_0", "speaker_1")
    other_text = to_transcribed(plugin.Result("hello", words), _chunk(0, 1))
    assert other_text.speakers is None


def test_loading_writes_nothing_into_the_plugin_directory(plugin_dir: Path) -> None:
    before = sorted(p.name for p in plugin_dir.rglob("*"))
    load_plugins(plugin_dir, BACKENDS)
    assert sorted(p.name for p in plugin_dir.rglob("*")) == before
