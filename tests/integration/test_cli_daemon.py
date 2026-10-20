"""The CLI's clients against a daemon: `key set` reloading it, `listen` printing a session."""

import asyncio
import io
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from harness import FIXTURE, Harness, fake, setup_with

from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.fake import FakeReply
from stt_daemon.cli import main
from stt_daemon.config import Config
from stt_daemon.listen import ListenOptions, listen, terminal_keys


async def test_key_set_reloads_the_daemon_and_clients_get_backends(
    daemon: Harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await daemon.start(backends={})
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("STT_DAEMON_SOCKET", str(daemon.path))
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-abc123\n"))
    daemon.reloads.append(setup_with(Config(), {"fake": fake()}))
    client, _ = await daemon.connect("Talkey")
    assert await asyncio.to_thread(main, ["key", "set", "elevenlabs"]) == 0
    event = await client.event("backends")
    assert event["data"]["default_backend"] == "fake"
    assert "reply_to" not in event
    assert "the daemon reloaded" in capsys.readouterr().out


async def _keys_at(*steps: tuple[float, str]) -> AsyncIterator[str]:
    for delay, action in steps:
        await asyncio.sleep(delay)
        yield action
    await asyncio.Event().wait()


async def test_listen_prints_segments_and_exits_0(daemon: Harness) -> None:
    await daemon.start(fake([FakeReply("hello there")]), capture=WavCapture(FIXTURE))
    out = io.StringIO()
    keys = _keys_at((2.0, "marker"), (2.0, "stop"))
    code = await listen(daemon.path, ListenOptions(words=True), keys, out, "m")
    text = out.getvalue()
    assert code == 0, text
    assert "marker    ⟦1⟧" in text
    segment = next(line for line in text.splitlines() if line.startswith("segment"))
    assert "hello" in segment
    assert "⟦1⟧" in segment
    assert "  words   hello@" in text
    assert "ended     stopped, 1 segment," in text


async def test_listen_exits_1_after_a_session_error(daemon: Harness) -> None:
    backend = fake([FakeReply("lost", fail=("auth", False))])
    await daemon.start(backend, capture=WavCapture(FIXTURE))
    out = io.StringIO()
    code = await listen(daemon.path, ListenOptions(), _keys_at((4.0, "stop")), out, "m")
    assert code == 1, out.getvalue()
    assert "error     auth" in out.getvalue()


async def test_listen_without_a_daemon_says_how_to_start_one(tmp_path: Path) -> None:
    out = io.StringIO()
    code = await listen(tmp_path / "none.sock", ListenOptions(), _keys_at(), out, "m")
    assert code == 1
    assert "stt-daemon service install" in out.getvalue()


class _FailingStdin:
    """Not a terminal; reading it fails after one marker line."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def fileno(self) -> int:
        return self._fd

    def __iter__(self) -> Iterator[str]:
        yield "m\n"
        time.sleep(0.5)
        raise OSError("stdin went away")


async def test_listen_that_loses_its_input_ends_and_frees_the_slot(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await daemon.start(capture=WavCapture(FIXTURE))
    out = io.StringIO()
    with Path("/dev/null").open() as null:
        monkeypatch.setattr("sys.stdin", _FailingStdin(null.fileno()))
        with pytest.raises(OSError, match="stdin went away"):
            await asyncio.wait_for(
                listen(daemon.path, ListenOptions(), terminal_keys("m"), out, "m"), 5
            )
    assert "marker    ⟦1⟧" in out.getvalue()
    other, _ = await daemon.connect("Talkey")
    started = await other.request("start_session")
    assert started["type"] == "session_started", started


@pytest.mark.parametrize(
    ("text", "actions"), [(None, ["stop"]), ("", ["stop"]), ("m\n\n", ["marker", "stop"])]
)
async def test_terminal_keys_read_files_and_dev_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str | None, actions: list[str]
) -> None:
    source = Path("/dev/null")
    if text is not None:
        source = tmp_path / "keys.txt"
        source.write_text(text)
    with source.open() as stdin:
        monkeypatch.setattr("sys.stdin", stdin)
        keys = terminal_keys("m")
        got = [await anext(keys) for _ in actions]
        await keys.aclose()
    assert got == actions
