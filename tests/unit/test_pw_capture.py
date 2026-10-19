"""PwRecordCapture with a stand-in pw-record: frames, loss and restore, backoff, stalls, release."""

import asyncio
import itertools
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
import pytest

from stt_daemon.audio import pw_capture
from stt_daemon.audio.frames import Frame
from stt_daemon.audio.pw_capture import PwRecordCapture, pw_record_command
from stt_daemon.audio.stream_capture import ToolMissing

FAKE = Path(__file__).parents[1] / "fake_pw_record.py"
REAL_SLEEP = asyncio.sleep


def fake(*args: str) -> list[str]:
    return [sys.executable, str(FAKE), *args]


def script(code: str) -> list[str]:
    return [sys.executable, "-c", f"import sys, time\n{code}"]


async def until(check: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        await REAL_SLEEP(0.01)


class Recorder:
    def __init__(self, capture: PwRecordCapture) -> None:
        self.events: list[str] = []
        capture.on_lost = lambda reason: self.events.append(f"lost: {reason}")
        capture.on_restored = lambda: self.events.append("restored")


def record_backoff(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The restart waits `PwRecordCapture` asks for; each takes 10 ms."""
    delays: list[float] = []

    def sleep(delay: float) -> Awaitable[None]:
        delays.append(delay)
        return REAL_SLEEP(0.01)

    monkeypatch.setattr(pw_capture.asyncio, "sleep", sleep)
    return delays


def test_the_command_records_the_default_outputs_monitor_or_a_target() -> None:
    default = pw_record_command()
    assert default[0] == "pw-record"
    assert default[default.index("-P") + 1] == "{ stream.capture.sink = true }"
    assert default[-1] == "-"
    targeted = pw_record_command("alsa_output.usb")
    assert targeted[targeted.index("--target") + 1] == "alsa_output.usb"
    assert "-P" not in targeted


async def test_frames_carry_the_samples_in_order_and_never_overlap() -> None:
    pcm = (np.arange(8000) % 20000).astype("<i2")
    write = f"sys.stdout.buffer.write({pcm.tobytes()!r}); sys.stdout.flush(); time.sleep(30)"
    capture = PwRecordCapture(script(write))
    queue = capture.attach().queue
    await capture.hold()
    frames: list[Frame] = []
    try:
        while sum(f.samples.size for f in frames) < pcm.size:
            frames.append(await asyncio.wait_for(queue.get(), 5))
    finally:
        await capture.release()
    got = np.concatenate([f.samples for f in frames])
    np.testing.assert_allclose(got, pcm / 32768.0, atol=1e-6)
    for before, after in itertools.pairwise(frames):
        assert after.start >= before.end - 1e-9


async def test_an_exit_is_reported_once_then_restored_after_a_restart(tmp_path: Path) -> None:
    log = tmp_path / "starts"
    capture = PwRecordCapture(
        fake("--log", str(log), "--exit-after", "0.3", "--only-starts", "1"),
        backoff_s=(0.2,),
    )
    recorder = Recorder(capture)
    await capture.hold()
    try:
        await until(lambda: "restored" in recorder.events)
    finally:
        await capture.release()
    assert len(recorder.events) == 2
    assert recorder.events[0].startswith("lost: ")
    assert "exited with code 0" in recorder.events[0]
    assert len(log.read_text().splitlines()) == 2
    assert capture.lost_reason is None


async def test_a_stalled_process_is_lost_killed_and_restarted(tmp_path: Path) -> None:
    log = tmp_path / "starts"
    capture = PwRecordCapture(
        fake("--log", str(log), "--stall-after", "0.2", "--only-starts", "1"),
        stall_s=0.3,
        backoff_s=(0.05,),
    )
    recorder = Recorder(capture)
    await capture.hold()
    try:
        await until(lambda: "restored" in recorder.events)
    finally:
        await capture.release()
    assert recorder.events[0] == "lost: no audio from " + sys.executable + " for 0.3 s"
    assert len(log.read_text().splitlines()) == 2


async def test_the_wait_grows_while_restarts_produce_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays = record_backoff(monkeypatch)
    capture = PwRecordCapture(script("sys.exit(3)"), backoff_s=(1.0, 2.0, 4.0))
    recorder = Recorder(capture)
    await capture.hold()
    try:
        await until(lambda: len(delays) >= 5)
    finally:
        await capture.release()
    assert delays[:5] == [1.0, 2.0, 4.0, 4.0, 4.0]
    assert recorder.events == ["lost: " + sys.executable + " exited with code 3"]


async def test_the_wait_starts_over_after_a_process_that_produced_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays = record_backoff(monkeypatch)
    capture = PwRecordCapture(fake("--exit-after", "0.1"), backoff_s=(1.0, 2.0))
    recorder = Recorder(capture)
    await capture.hold()
    try:
        await until(lambda: len(delays) >= 3)
    finally:
        await capture.release()
    assert delays[:3] == [1.0, 1.0, 1.0]
    assert recorder.events[1] == "restored"


async def test_ready_returns_at_the_first_frame_or_at_a_loss() -> None:
    flowing = PwRecordCapture(fake())
    await flowing.hold()
    began = time.monotonic()
    await flowing.ready(5)
    assert time.monotonic() - began < 2
    await flowing.release()

    failing = PwRecordCapture(script("sys.exit(1)"), backoff_s=(10.0,))
    await failing.hold()
    began = time.monotonic()
    await failing.ready(5)
    assert time.monotonic() - began < 2
    assert failing.lost_reason is not None
    await failing.release()


async def test_release_ends_the_process_and_its_restarts(tmp_path: Path) -> None:
    log = tmp_path / "starts"
    capture = PwRecordCapture(fake("--log", str(log)), backoff_s=(0.05,))
    queue = capture.attach().queue
    await capture.hold()
    await asyncio.wait_for(queue.get(), 5)
    began = time.monotonic()
    await capture.release()
    assert time.monotonic() - began < 2
    await REAL_SLEEP(0.2)
    assert len(log.read_text().splitlines()) == 1


async def test_release_during_the_wait_before_a_restart_starts_nothing(tmp_path: Path) -> None:
    log = tmp_path / "starts"
    capture = PwRecordCapture(fake("--log", str(log), "--exit-after", "0"), backoff_s=(0.3,))
    recorder = Recorder(capture)
    await capture.hold()
    await until(lambda: bool(recorder.events))
    await capture.release()
    await REAL_SLEEP(0.5)
    assert len(log.read_text().splitlines()) == 1


async def test_a_missing_program_fails_the_hold() -> None:
    capture = PwRecordCapture(["/nonexistent/pw-record"])
    with pytest.raises(ToolMissing):
        await capture.hold()
