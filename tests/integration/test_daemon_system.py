"""System sessions over the socket: a stand-in `pw-record` or `audiotee` (on PATH, picked as on
macOS) for the system track, WAV for the mic."""

import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from harness import FIXTURE, Harness, Msg, TestClient

from stt_daemon.audio.pw_capture import PwRecordCapture
from stt_daemon.audio.system_capture import AppTarget, SystemCapture
from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.base import CommitPolicy
from stt_daemon.backends.fake import FAKE_CAPABILITIES, FakeBackend, FakeReply
from stt_daemon.server.starts import system_capture

FAKE_PW_RECORD = os.path.join(os.path.dirname(__file__), "..", "fake_pw_record.py")
FAKE_AUDIOTEE = Path(__file__).parents[1] / "fake_audiotee.py"
MIC_DELAY_S = 0.5
"""The mic plays the fixture this long after the system track, so their phrases alternate."""
PAUSES = replace(FAKE_CAPABILITIES, commit_policy=CommitPolicy(at_pauses=True))


def at_pauses(replies: list[FakeReply | str] | None = None) -> FakeBackend:
    return FakeBackend(replies or [], capabilities=PAUSES)


def audiotee_on_path(bin_dir: Path, monkeypatch: pytest.MonkeyPatch, *options: str) -> None:
    """An `audiotee` on PATH: the stand-in, with `options` after the daemon's flags."""
    bin_dir.mkdir(exist_ok=True)
    shim = bin_dir / "audiotee"
    extra = " ".join(f"'{o}'" for o in options)
    shim.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{FAKE_AUDIOTEE}' \"$@\" {extra}\n")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture(params=["linux", "darwin"])
def platform(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    """The system capture a platform picks; on darwin the daemon's own `audiotee` command."""
    if request.param == "darwin":
        audiotee_on_path(tmp_path / "bin", monkeypatch, "--wav", str(FIXTURE))
    return request.param


class Captures:
    """Makes system captures playing the fixture; keeps them for the test to check."""

    def __init__(self, platform: str = "linux") -> None:
        self.platform = platform
        self.made: list[SystemCapture] = []

    def __call__(self, target: AppTarget | None) -> SystemCapture:
        if self.platform == "darwin":
            capture = system_capture(target, platform="darwin")
        else:
            capture = PwRecordCapture([sys.executable, FAKE_PW_RECORD, "--wav", str(FIXTURE)])
        self.made.append(capture)
        return capture


def segments(client: TestClient) -> list[Msg]:
    return [m["data"] for m in client.of_type("segment")]


async def start_system(client: TestClient, **data: object) -> Msg:
    started = await client.request("start_system_session", {"word_timestamps": True, **data})
    assert started["type"] == "session_started", started
    return started


async def test_both_tracks_commit_and_final_text_interleaves_them(
    daemon: Harness, platform: str
) -> None:
    mic = WavCapture()
    await daemon.start(
        at_pauses([f"text {i}" for i in range(4)]), capture=mic, system_capture=Captures(platform)
    )
    scribbly, _ = await daemon.connect("Scribbly")
    started = await start_system(scribbly, with_mic=True)
    assert started["data"]["kind"] == "system"
    assert started["data"]["tracks"] == ["system", "mic"]
    await asyncio.sleep(MIC_DELAY_S)
    mic.play(FIXTURE)
    await scribbly.wait(lambda _m: len(segments(scribbly)) == 4, timeout=15)
    ended = await scribbly.request("stop_session")
    assert ended["data"]["reason"] == "stopped", ended
    by_start = sorted(segments(scribbly), key=lambda s: s["start_ms"])
    assert [s["track"] for s in by_start] == ["system", "mic", "system", "mic"]
    system_starts = [s["words"][0]["start_ms"] for s in by_start if s["track"] == "system"]
    mic_starts = [s["words"][0]["start_ms"] for s in by_start if s["track"] == "mic"]
    for system_ms, mic_ms in zip(system_starts, mic_starts, strict=True):
        assert 300 <= mic_ms - system_ms <= 900  # the mic's delay, plus the time play() took
    lines = [f"[{s['track']}] {s['text']}" for s in by_start]
    assert ended["data"]["final_text"] == "\n".join(lines)


async def test_dictation_runs_alongside_a_system_session(daemon: Harness, platform: str) -> None:
    await daemon.start(at_pauses(), system_capture=Captures(platform))
    scribbly, _ = await daemon.connect("Scribbly")
    await start_system(scribbly)
    talkey, _ = await daemon.connect("Talkey")
    assert (await talkey.request("start_session"))["type"] == "session_started"
    slot = await talkey.wait(
        lambda m: m["type"] == "slot" and len(m["data"]["backends"]["fake"]["sessions"]) == 2
    )
    assert slot["data"]["state"] == "busy"
    kinds = [(s["kind"], s["suspended"]) for s in slot["data"]["backends"]["fake"]["sessions"]]
    assert sorted(kinds) == [("mic", False), ("system", False)]
    assert (await talkey.request("stop_session"))["data"]["reason"] == "stopped"
    assert (await scribbly.request("stop_session"))["data"]["reason"] == "stopped"
    assert scribbly.of_type("session_suspended") == []


async def system_run(daemon: Harness, *, dictation: bool, platform: str) -> list[Msg]:
    """The fixture's two phrases on the system track of a backend with one slot; with
    `dictation`, a mic session holds the slot from the first phrase's commit for a second."""
    backend = at_pauses(["one two", "three four"])
    backend.capacity = 1
    await daemon.start(backend, system_capture=Captures(platform))
    scribbly, _ = await daemon.connect("Scribbly")
    await start_system(scribbly)
    await scribbly.event("segment", index=0)
    if dictation:
        talkey, _ = await daemon.connect("Talkey")
        assert (await talkey.request("start_session"))["type"] == "session_started"
        suspended = await scribbly.event("session_suspended")
        assert suspended["data"]["by"]["kind"] == "mic"
        await asyncio.sleep(1.0)  # into the second phrase (5.74 s)
        assert (await talkey.request("cancel_session"))["data"]["reason"] == "cancelled"
        await scribbly.event("session_resumed")
    await scribbly.event("segment", index=1, timeout=15)
    assert (await scribbly.request("stop_session"))["data"]["reason"] == "stopped"
    return segments(scribbly)


async def test_a_suspended_system_session_keeps_its_audio_and_resumes(platform: str) -> None:
    plain_daemon, suspended_daemon = Harness(), Harness()
    try:
        plain = await system_run(plain_daemon, dictation=False, platform=platform)
        suspended = await system_run(suspended_daemon, dictation=True, platform=platform)
    finally:
        await plain_daemon.stop()
        await suspended_daemon.stop()
    assert [s["text"] for s in plain] == ["one two", "three four"]
    assert [s["text"] for s in suspended] == ["one two", "three four"]
    for a, b in zip(plain, suspended, strict=True):
        for x, y in zip(a["words"], b["words"], strict=True):
            assert abs(x["start_ms"] - y["start_ms"]) <= 100, (a, b)


async def test_a_fatal_error_on_one_track_ends_the_session_with_both_tracks_text(
    daemon: Harness, platform: str
) -> None:
    mic = WavCapture()
    backend = at_pauses(["system one", "mic one"])
    await daemon.start(backend, capture=mic, system_capture=Captures(platform))
    scribbly, _ = await daemon.connect("Scribbly")
    await start_system(scribbly, with_mic=True)
    await asyncio.sleep(MIC_DELAY_S)
    mic.play(FIXTURE)
    await scribbly.wait(lambda _m: len(segments(scribbly)) == 2, timeout=10)
    backend.fail("auth", retryable=False)
    error = await scribbly.event("session_error", fatal=True)
    assert error["data"]["code"] == "auth"
    assert error["data"]["track"] in ("system", "mic")
    ended = await scribbly.event("session_ended")
    assert ended["data"]["reason"] == "error"
    lines = ended["data"]["final_text"].splitlines()
    assert sorted(lines) == ["[mic] mic one", "[system] system one"]


async def test_a_missing_tool_is_refused_with_its_name(daemon: Harness) -> None:
    await daemon.start(system_capture=lambda _: PwRecordCapture(["/nonexistent/pw-record"]))
    scribbly, _ = await daemon.connect("Scribbly")
    refused = await scribbly.request("start_system_session")
    assert refused["data"]["code"] == "tool_unavailable"
    assert refused["data"]["tool"] == "/nonexistent/pw-record"
    _, welcome = await daemon.connect("Watch")
    assert welcome["data"]["slot"]["backends"]["fake"]["sessions"] == []


async def test_a_missing_audiotee_is_refused_with_how_to_get_it(
    daemon: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    await daemon.start(system_capture=Captures("darwin"))
    scribbly, _ = await daemon.connect("Scribbly")
    refused = await scribbly.request("start_system_session")
    assert refused["data"]["code"] == "tool_unavailable"
    assert refused["data"]["tool"] == "audiotee"
    assert "github.com/makeusabrew/audiotee" in refused["data"]["message"]


async def test_audiotee_includes_the_apps_process(
    daemon: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "audiotee.log"
    audiotee_on_path(tmp_path / "bin", monkeypatch, "--log", str(log))
    captures = Captures("darwin")
    await daemon.start(system_capture=lambda _target: captures(StaticTarget("4242")))
    scribbly, _ = await daemon.connect("Scribbly")
    await start_system(scribbly)
    assert log.read_text().split() == ["start", "4242"]
    assert captures.made[0].command[-2:] == ["--include-processes", "4242"]


class StaticTarget:
    def __init__(self, serial: str) -> None:
        self.serial = serial

    def current(self) -> str | None:
        return self.serial

    async def appeared(self) -> str:
        return self.serial


async def test_system_sessions_are_unsupported_off_linux_and_macos(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await daemon.start(system_capture=Captures())
    scribbly, _ = await daemon.connect("Scribbly")
    monkeypatch.setattr(sys, "platform", "win32")
    refused = await scribbly.request("start_system_session")
    assert refused["data"]["code"] == "unsupported"


async def test_a_client_gone_ends_the_session_and_its_capture(
    daemon: Harness, platform: str
) -> None:
    captures = Captures(platform)
    await daemon.start(system_capture=captures)
    watch, _ = await daemon.connect("Watch", observe_all=True)
    scribbly, _ = await daemon.connect("Scribbly")
    await start_system(scribbly)
    scribbly.writer.transport.abort()
    await watch.event("session_ended", timeout=3, reason="client_gone")
    [capture] = captures.made
    assert not capture.is_open  # released: its process and restarts end
