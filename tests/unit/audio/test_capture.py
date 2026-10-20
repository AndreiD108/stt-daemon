import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from stt_daemon.audio import capture
from stt_daemon.audio.capture import CaptureUnavailable, MicCapture, find_input_device

RATE = 16000


class FakeStream:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = self.aborted = self.stopped = self.closed = False

    def start(self) -> None:
        self.started = True

    def abort(self) -> None:
        self.aborted = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class HangingStream(FakeStream):
    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def abort(self) -> None:
        self.aborted = True
        self.release.wait()


class RecordingEvent:
    """A shutdown event whose waits return at once and are recorded."""

    def __init__(self) -> None:
        self.flag = False
        self.waits: list[float] = []

    def set(self) -> None:
        self.flag = True

    def clear(self) -> None:
        self.flag = False

    def is_set(self) -> bool:
        return self.flag

    def wait(self, timeout: float | None = None) -> bool:
        if timeout is not None:
            self.waits.append(timeout)
        return self.flag


DEVICES = [
    {"name": "HDA Intel PCH: ALC897 Analog (hw:0,0)", "max_input_channels": 2},
    {"name": "Speakers", "max_input_channels": 0},
    {"name": "USB Mic: Audio (hw:1,0)", "max_input_channels": 1},
    {"name": "pipewire", "max_input_channels": 64},
    {"name": "default", "max_input_channels": 64},
]


class FakeSd:
    def __init__(self, devices: list[dict[str, Any]] = DEVICES, default: int = 4) -> None:
        self.devices = devices
        self.default = SimpleNamespace(device=[default, default])
        self.streams: list[FakeStream] = []
        self.failures = 0

    def query_devices(self, device: int | None = None, kind: str | None = None) -> Any:
        return self.devices if device is None else self.devices[device]

    def InputStream(self, **kwargs: Any) -> FakeStream:
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("device unavailable")
        stream = FakeStream(**kwargs)
        self.streams.append(stream)
        return stream


@pytest.fixture
def fake_sd(monkeypatch: pytest.MonkeyPatch) -> FakeSd:
    sd = FakeSd()
    monkeypatch.setattr(capture, "_sd", sd)
    return sd


def _wait_for(condition: Any, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def test_device_by_substring_skips_raw_hw_devices(fake_sd: FakeSd) -> None:
    assert find_input_device("pipe") == 3
    with pytest.raises(RuntimeError, match="hw:"):
        find_input_device("USB Mic")


def test_empty_device_name_takes_the_default_input(fake_sd: FakeSd) -> None:
    assert find_input_device("") == 4


def test_empty_device_name_falls_back_to_pipewire(fake_sd: FakeSd) -> None:
    fake_sd.default = SimpleNamespace(device=[0, 0])  # a raw hw: device
    assert find_input_device("") == 3
    fake_sd.default = SimpleNamespace(device=[-1, -1])
    assert find_input_device("") == 3


def test_no_shared_input_fails(fake_sd: FakeSd) -> None:
    fake_sd.devices = DEVICES[:3]
    fake_sd.default = SimpleNamespace(device=[0, 0])
    with pytest.raises(RuntimeError, match=r"audio\.device"):
        find_input_device("")


async def test_attach_before_open_fails(fake_sd: FakeSd) -> None:
    with pytest.raises(CaptureUnavailable):
        MicCapture().attach()


async def test_kept_open_mic_opens_at_start_and_holds_do_not_close_it(fake_sd: FakeSd) -> None:
    mic = MicCapture(device_name="pipewire")
    await mic.start()
    assert mic.is_open
    stream = fake_sd.streams[0]
    assert stream.started
    assert stream.kwargs["device"] == 3
    assert stream.kwargs["channels"] == 2
    assert stream.kwargs["samplerate"] == RATE
    await mic.hold()
    await mic.release()
    assert mic.is_open
    await mic.stop()
    assert not mic.is_open
    assert stream.stopped
    assert stream.closed


async def test_per_session_mic_opens_on_first_hold_and_closes_on_last_release(
    fake_sd: FakeSd,
) -> None:
    mic = MicCapture(keep_open=False)
    await mic.start()
    assert not mic.is_open
    await mic.hold()
    await mic.hold()
    assert mic.is_open
    mic._callback(np.full((512, 1), 0.1, dtype=np.float32), 512, None, None)  # pyright: ignore[reportPrivateUsage]
    assert mic.ring_since()
    await mic.release()
    assert mic.is_open
    await mic.release()
    assert not mic.is_open
    assert mic.ring_since() == []
    assert len(fake_sd.streams) == 1


async def test_failed_open_leaves_no_hold(fake_sd: FakeSd) -> None:
    mic = MicCapture(keep_open=False)
    fake_sd.failures = 1
    with pytest.raises(RuntimeError, match="device unavailable"):
        await mic.hold()
    await mic.hold()
    assert mic.is_open
    await mic.release()
    assert not mic.is_open


async def test_cancelled_hold_closes_the_mic_it_was_opening(
    fake_sd: FakeSd, monkeypatch: pytest.MonkeyPatch
) -> None:
    mic = MicCapture(keep_open=False)
    entered, proceed = threading.Event(), threading.Event()
    real_open = mic._open  # pyright: ignore[reportPrivateUsage]

    def slow_open() -> None:
        entered.set()
        proceed.wait()
        real_open()

    monkeypatch.setattr(mic, "_open", slow_open)
    holding = asyncio.ensure_future(mic.hold())
    assert await asyncio.to_thread(entered.wait, 1.0)
    holding.cancel()
    await asyncio.sleep(0.01)
    assert not holding.done(), "hold returned before its open finished"
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await holding
    assert not mic.is_open
    assert fake_sd.streams[0].closed
    await mic.hold()
    assert mic.is_open
    await mic.release()
    assert not mic.is_open


async def test_failed_stream_start_cleans_up_and_retries(fake_sd: FakeSd) -> None:
    class FailingStart(FakeStream):
        def start(self) -> None:
            raise RuntimeError("start failed")

    created: list[FakeStream] = []

    def failing_stream(**kwargs: Any) -> FakeStream:
        stream = FailingStart(**kwargs)
        created.append(stream)
        return stream

    mic = MicCapture(keep_open=False)
    good_stream = fake_sd.InputStream
    fake_sd.InputStream = failing_stream  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="start failed"):
        await mic.hold()
    assert created[0].aborted
    assert created[0].closed
    assert not mic.is_open
    fake_sd.InputStream = good_stream  # type: ignore[method-assign]
    await mic.hold()
    assert mic.is_open
    await mic.release()


def test_callback_averages_channels_and_drops_bad_chunks(fake_sd: FakeSd) -> None:
    mic = MicCapture()
    stereo = np.column_stack([np.full(512, 0.2), np.full(512, 0.4)]).astype(np.float32)
    mic._callback(stereo, 512, None, None)  # pyright: ignore[reportPrivateUsage]
    mic._callback(np.full((512, 1), 7, dtype=np.int16), 512, None, None)  # pyright: ignore[reportPrivateUsage]
    nan = np.full((512, 1), np.nan, dtype=np.float32)
    mic._callback(nan, 512, None, None)  # pyright: ignore[reportPrivateUsage]
    ring = mic.ring_since()
    assert len(ring) == 1
    assert ring[0].samples.dtype == np.float32
    assert ring[0].samples.shape == (512,)
    assert float(ring[0].samples[0]) == pytest.approx(0.3)


def test_callback_copies_the_portaudio_buffer(fake_sd: FakeSd) -> None:
    mic = MicCapture()
    buffer = np.full((512, 1), 0.1, dtype=np.float32)
    mic._callback(buffer, 512, None, None)  # pyright: ignore[reportPrivateUsage]
    buffer[:] = 0.9
    assert float(mic.ring_since()[0].samples[0]) == pytest.approx(0.1)


async def test_repeated_status_errors_trigger_recovery(fake_sd: FakeSd) -> None:
    mic = MicCapture(recovery_backoff_s=(0.0,), watchdog_interval_s=0.01)
    await mic.start()
    chunk = np.zeros((512, 1), dtype=np.float32)
    for _ in range(capture.STATUS_RECOVERY_THRESHOLD):
        mic._callback(chunk, 512, None, "input overflow")  # pyright: ignore[reportPrivateUsage]
    assert _wait_for(lambda: len(fake_sd.streams) == 2)
    assert fake_sd.streams[0].aborted
    await mic.stop()


async def test_stale_finished_callback_is_ignored(fake_sd: FakeSd) -> None:
    mic = MicCapture(recovery_backoff_s=(0.0,), watchdog_interval_s=0.01)
    await mic.start()
    mic._recover_stream("test")  # pyright: ignore[reportPrivateUsage]
    assert len(fake_sd.streams) == 2
    fake_sd.streams[0].kwargs["finished_callback"]()
    time.sleep(0.05)
    assert len(fake_sd.streams) == 2
    fake_sd.streams[1].kwargs["finished_callback"]()
    assert _wait_for(lambda: len(fake_sd.streams) == 3)
    await mic.stop()


async def test_request_whose_wakeup_was_cleared_still_recovers(fake_sd: FakeSd) -> None:
    mic = MicCapture(recovery_backoff_s=(0.0,), watchdog_interval_s=0.01)
    await mic.start()
    # What a request leaves when it lands between the watchdog's wait() and clear().
    mic._recovery_reason = "lost wakeup"  # pyright: ignore[reportPrivateUsage]
    assert _wait_for(lambda: len(fake_sd.streams) == 2)
    await mic.stop()


def test_recovery_aborts_requeries_and_reopens(fake_sd: FakeSd) -> None:
    mic = MicCapture(device_name="pipewire", recovery_backoff_s=(0.0,))
    old = FakeStream()
    mic._stream = old  # pyright: ignore[reportPrivateUsage]
    fake_sd.devices = [{"name": "pipewire", "max_input_channels": 1}]
    mic._recover_stream("test stall")  # pyright: ignore[reportPrivateUsage]
    assert old.aborted
    assert old.closed
    assert len(fake_sd.streams) == 1
    replacement = fake_sd.streams[0]
    assert replacement.started
    assert replacement.kwargs["device"] == 0
    assert replacement.kwargs["channels"] == 1
    assert mic.is_open
    mic._close()  # pyright: ignore[reportPrivateUsage]


def test_recovery_backoff_caps_at_its_last_delay(fake_sd: FakeSd) -> None:
    backoff = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
    mic = MicCapture(recovery_backoff_s=backoff)
    mic._stream = FakeStream()  # pyright: ignore[reportPrivateUsage]
    event = RecordingEvent()
    mic._shutdown = event  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    fake_sd.failures = len(backoff) + 1
    mic._recover_stream("test stall")  # pyright: ignore[reportPrivateUsage]
    assert event.waits[: len(backoff)] == list(backoff)
    assert event.waits[len(backoff) :] == [backoff[-1]] * 2
    assert mic.is_open


async def test_watchdog_recovers_a_stalled_stream(fake_sd: FakeSd) -> None:
    mic = MicCapture(stall_timeout_s=0.1, watchdog_interval_s=0.01, recovery_backoff_s=(0.0,))
    await mic.start()
    assert _wait_for(lambda: len(fake_sd.streams) >= 2), "watchdog did not reopen the stream"
    assert fake_sd.streams[0].aborted
    await mic.stop()
    assert mic._watchdog is None  # pyright: ignore[reportPrivateUsage]


async def test_attach_fails_while_recovering(fake_sd: FakeSd) -> None:
    mic = MicCapture(recovery_backoff_s=(30.0,), watchdog_interval_s=0.01)
    await mic.start()
    mic._request_recovery("test")  # pyright: ignore[reportPrivateUsage]
    assert _wait_for(lambda: fake_sd.streams[0].aborted)
    assert not mic.is_open
    with pytest.raises(CaptureUnavailable):
        mic.attach()
    started = time.monotonic()
    await mic.stop()
    assert time.monotonic() - started < 0.5, "stop did not interrupt the recovery backoff"
    assert len(fake_sd.streams) == 1


def test_hung_native_abort_exits_the_process(
    fake_sd: FakeSd, monkeypatch: pytest.MonkeyPatch
) -> None:
    exits: list[int] = []
    monkeypatch.setattr(capture, "_fatal_exit", exits.append)
    stream = HangingStream()
    try:
        MicCapture(stream_stop_timeout_s=0.01)._stop_stream(stream, abort=True)  # pyright: ignore[reportPrivateUsage]
    finally:
        stream.release.set()
    assert exits == [capture.HUNG_STREAM_EXIT_CODE]
