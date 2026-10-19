"""AudioteeCapture with a stand-in audiotee: its command, frames, silence with the permission hint,
exits with audiotee's error, following a process."""

import asyncio
import sys
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

from stt_daemon.audio.audiotee_capture import AudioteeCapture, audiotee_command, last_error

FAKE = Path(__file__).parents[1] / "fake_audiotee.py"
FIXTURE = Path(__file__).parents[1] / "fixtures" / "two_phrases.wav"


def fake(*extra: str) -> Callable[[str | None], list[str]]:
    """The daemon's own audiotee command line, run by the stand-in."""

    def command(pid: str | None) -> list[str]:
        return [sys.executable, str(FAKE), *audiotee_command(pid)[1:], *extra]

    return command


async def until(check: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


class Events:
    def __init__(self, capture: AudioteeCapture) -> None:
        self.seen: list[str] = []
        capture.on_lost = lambda reason: self.seen.append(f"lost: {reason}")
        capture.on_restored = lambda: self.seen.append("restored")


class StubTarget:
    """A process: `playing` is its pid, None while it plays nothing."""

    def __init__(self, pid: str) -> None:
        self.serial = pid
        self.playing: str | None = pid
        self.back = asyncio.Event()

    def current(self) -> str | None:
        return self.playing

    async def appeared(self) -> str:
        await self.back.wait()
        assert self.playing is not None
        return self.playing


def starts(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


def test_the_command_asks_for_16_khz_in_100_ms_chunks_and_includes_a_pid() -> None:
    assert audiotee_command() == ["audiotee", "--sample-rate", "16000", "--chunk-duration", "0.1"]
    assert audiotee_command("4242")[-2:] == ["--include-processes", "4242"]


async def test_frames_carry_the_wav_in_real_time() -> None:
    with wave.open(str(FIXTURE), "rb") as w:
        expected = np.frombuffer(w.readframes(16000), dtype="<i2") / 32768.0
    capture = AudioteeCapture(fake("--wav", str(FIXTURE)))
    queue = capture.attach().queue
    await capture.hold()
    try:
        got: list[np.ndarray] = []
        began = time.monotonic()
        while sum(f.size for f in got) < 16000:
            got.append((await asyncio.wait_for(queue.get(), 5)).samples)
        elapsed = time.monotonic() - began
    finally:
        await capture.release()
    samples = np.concatenate(got)[:16000]
    np.testing.assert_allclose(samples, expected, atol=1e-6)
    assert got[0].size == 1600  # one 100 ms chunk per read
    assert 0.7 <= elapsed <= 2.0


async def test_no_frames_is_a_loss_that_names_the_permission() -> None:
    capture = AudioteeCapture(fake("--silent"), stall_s=0.3, backoff_s=(10.0,))
    events = Events(capture)
    await capture.hold()
    try:
        await until(lambda: bool(events.seen))
    finally:
        await capture.release()
    [lost] = events.seen
    assert lost.startswith("lost: no audio from ")
    assert "System Audio Recording" in lost
    assert "Privacy & Security" in lost


async def test_an_exit_is_a_loss_with_audiotees_last_error_and_restarts(tmp_path: Path) -> None:
    log = tmp_path / "starts"
    capture = AudioteeCapture(fake("--exit-after", "0.2", "--log", str(log)), backoff_s=(0.05,))
    events = Events(capture)
    await capture.hold()
    try:
        await until(lambda: len(starts(log)) >= 2 and "restored" in events.seen)
    finally:
        await capture.release()
    assert events.seen[0].endswith(" exited with code 2: Audio device stopped (device: tap)")
    assert events.seen[1] == "restored"


async def test_a_process_that_goes_quiet_keeps_its_capture(tmp_path: Path) -> None:
    log = tmp_path / "starts"
    target = StubTarget("11")
    capture = AudioteeCapture(fake("--log", str(log)), target=target, backoff_s=(10.0,))
    events = Events(capture)
    queue = capture.attach().queue
    await capture.hold()
    try:
        await asyncio.wait_for(queue.get(), 5)
        target.playing = None  # listed no more: its output stopped, audiotee writes silence
        await asyncio.sleep(0.5)
        while not queue.empty():
            queue.get_nowait()
        await asyncio.wait_for(queue.get(), 5)
    finally:
        await capture.release()
    assert starts(log) == ["start 11"]
    assert events.seen == []


async def test_a_pid_that_fails_to_translate_waits_for_the_app_then_follows_it(
    tmp_path: Path,
) -> None:
    log, live = tmp_path / "starts", tmp_path / "live"
    live.write_text("")
    target = StubTarget("11")
    target.playing = None
    capture = AudioteeCapture(
        fake("--log", str(log), "--live", str(live)), target=target, backoff_s=(10.0,)
    )
    events = Events(capture)
    await capture.hold()
    try:
        await until(lambda: bool(events.seen))
        assert (
            "Failed to translate process IDs to audio objects (failed_pids: 11)" in (events.seen[0])
        )
        await asyncio.sleep(0.3)
        assert starts(log) == ["start 11"]
        live.write_text("12\n")
        target.playing = "12"
        target.back.set()
        await until(lambda: events.seen[-1] == "restored")
    finally:
        await capture.release()
    assert starts(log) == ["start 11", "start 12"]
    assert capture.command[capture.command.index("--include-processes") + 1] == "12"


async def test_an_app_on_a_new_pid_restarts_the_capture_there_without_a_loss(
    tmp_path: Path,
) -> None:
    log = tmp_path / "starts"
    target = StubTarget("11")
    capture = AudioteeCapture(fake("--log", str(log)), target=target, backoff_s=(10.0,))
    events = Events(capture)
    queue = capture.attach().queue
    await capture.hold()
    try:
        await asyncio.wait_for(queue.get(), 5)
        target.playing = "12"  # the old process went quiet, the app plays from another one
        await until(lambda: starts(log) == ["start 11", "start 12"])
        while not queue.empty():
            queue.get_nowait()
        await asyncio.wait_for(queue.get(), 5)
    finally:
        await capture.release()
    assert events.seen == []


class VanishingTarget(StubTarget):
    """Plays on `moved_to` for one look, then nowhere until `back` is set."""

    def __init__(self, pid: str, moved_to: str) -> None:
        super().__init__(pid)
        self.moved_to: str | None = moved_to

    def current(self) -> str | None:
        if self.moved_to is not None and self.playing is None:
            moved, self.moved_to = self.moved_to, None
            return moved
        return self.playing


async def test_an_app_whose_new_pid_goes_at_once_is_waited_for_not_captured_whole(
    tmp_path: Path,
) -> None:
    log = tmp_path / "starts"
    target = VanishingTarget("11", moved_to="12")
    capture = AudioteeCapture(fake("--log", str(log)), target=target, backoff_s=(0.05,))
    events = Events(capture)
    queue = capture.attach().queue
    await capture.hold()
    try:
        await asyncio.wait_for(queue.get(), 5)
        target.playing = None
        await until(lambda: bool(events.seen))
        await asyncio.sleep(0.3)
        assert starts(log) == ["start 11"]  # never an audiotee without --include-processes
        target.playing = "13"
        target.back.set()
        await until(lambda: events.seen[-1] == "restored")
    finally:
        await capture.release()
    assert starts(log) == ["start 11", "start 13"]
    assert events.seen[0].startswith("lost: ")


def test_last_error_takes_the_last_error_line_and_skips_the_rest() -> None:
    stderr = "\n".join(
        [
            'ssage": "cut by the tail"}}',
            '{"message_type": "error", "data": {"message": "first"}}',
            '{"message_type": "error", "data": {"message": "Tap failed", "context": {"a": "1"}}}',
            '{"message_type": "info", "data": {"message": "Shutting down..."}}',
            "not json",
        ]
    )
    assert last_error(stderr) == "Tap failed (a: 1)"
    assert last_error('{"message_type": "info", "data": {"message": "x"}}') is None
    assert last_error("") is None
