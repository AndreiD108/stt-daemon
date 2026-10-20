"""SpeakerMonitor: kept open or held per session, holds during the open, a missing tool."""

import asyncio
import time
from typing import ClassVar

import pytest

from stt_daemon.audio.capture import CaptureUnavailable
from stt_daemon.audio.speakers import SpeakerMonitor
from stt_daemon.audio.stream_capture import ToolMissing
from stt_daemon.audio.system_capture import SystemCapture


class FakeSpeakers(SystemCapture):
    """Starts once `opened` is set; `missing` fails the start."""

    made: ClassVar[list["FakeSpeakers"]] = []

    def __init__(self, opened: asyncio.Event | None = None, *, missing: bool = False) -> None:
        super().__init__(["true"])
        self.opened = opened
        self.missing = missing
        self.running = False
        FakeSpeakers.made.append(self)

    async def hold(self) -> None:
        if self.missing:
            raise ToolMissing("pw-record")
        if self.opened is not None:
            await self.opened.wait()
        self.running = True

    async def release(self) -> None:
        self.running = False
        await super().release()


@pytest.fixture(autouse=True)
def fresh() -> None:
    FakeSpeakers.made = []


async def test_a_hold_during_the_open_waits_for_it() -> None:
    opened = asyncio.Event()
    monitor = SpeakerMonitor(lambda: FakeSpeakers(opened), keep_open=True)
    wanting = asyncio.ensure_future(monitor.want(True))
    holding = asyncio.ensure_future(monitor.hold())
    await asyncio.sleep(0.02)
    assert not holding.done()
    opened.set()
    await asyncio.gather(wanting, holding)
    monitor.attach(time.monotonic()).detach()
    assert len(FakeSpeakers.made) == 1
    await monitor.stop()


async def test_a_held_capture_outlives_the_setup_no_longer_wanting_it() -> None:
    monitor = SpeakerMonitor(FakeSpeakers, keep_open=True)
    await monitor.want(True)
    await monitor.hold()
    await monitor.want(False)
    [capture] = FakeSpeakers.made
    assert capture.running
    monitor.attach(time.monotonic()).detach()
    await monitor.release()
    assert not capture.running
    with pytest.raises(CaptureUnavailable):
        monitor.attach(time.monotonic())


async def test_without_keep_open_sessions_hold_it_and_each_open_is_a_new_capture() -> None:
    monitor = SpeakerMonitor(FakeSpeakers, keep_open=False)
    await monitor.want(True)
    assert not FakeSpeakers.made
    await monitor.hold()
    await monitor.hold()
    await monitor.release()
    assert FakeSpeakers.made[0].running
    await monitor.release()
    assert not FakeSpeakers.made[0].running
    await monitor.hold()
    assert len(FakeSpeakers.made) == 2
    assert FakeSpeakers.made[1].running
    await monitor.stop()
    assert not FakeSpeakers.made[1].running


async def test_a_missing_tool_leaves_no_capture_and_the_next_hold_tries_again() -> None:
    tries = iter([True, False])
    monitor = SpeakerMonitor(lambda: FakeSpeakers(missing=next(tries)), keep_open=False)
    await monitor.hold()
    with pytest.raises(CaptureUnavailable):
        monitor.attach(time.monotonic())
    await monitor.release()
    await monitor.hold()
    monitor.attach(time.monotonic()).detach()
    await monitor.stop()
