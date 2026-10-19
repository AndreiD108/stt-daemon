"""The built-in canceller as a mic track: frames from two captures, paired by capture time."""

import asyncio
import time

import numpy as np
from aec_fixture import FAR_ONLY, RATE, build

from stt_daemon.audio.aec_builtin import LOOKAHEAD_S, SETTLE_S, BuiltinEchoCancelledMic
from stt_daemon.audio.capture import CaptureSource
from stt_daemon.audio.frames import Frame

FRAME = 320
FIX = build()


class PushCapture(CaptureSource):
    """Frames pushed by the test; `lost_reason` as a system capture has it."""

    def __init__(self) -> None:
        super().__init__(sample_rate=RATE, buffer_ms=60_000, keep_open=False)
        self.lost_reason: str | None = None
        self.holds = 0

    @property
    def is_open(self) -> bool:
        return True

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    async def hold(self) -> None:
        self.holds += 1

    async def release(self) -> None:
        self.holds -= 1

    def push(self, audio: np.ndarray, start: float) -> None:
        for i in range(0, audio.size - FRAME + 1, FRAME):
            chunk = audio[i : i + FRAME].astype(np.float32)
            self._hub.push(chunk, start + (i + FRAME) / RATE)


async def collect(queue: asyncio.Queue[Frame], seconds: float) -> list[Frame]:
    frames: list[Frame] = []
    total = 0.0
    while total < seconds - 0.05:
        frame = await asyncio.wait_for(queue.get(), 5)
        frames.append(frame)
        total += frame.duration
    return frames


def level_db(audio: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(audio**2) + 1e-20))


async def test_far_frames_stamped_late_still_cancel_and_output_keeps_the_mic_times() -> None:
    mic, system = PushCapture(), PushCapture()
    far, mixed = FIX.far, FIX.mixed
    seconds = 17.5
    near_start = time.monotonic() - 30
    # The system track's frames are stamped 60 ms after the mic's for the same sound.
    system.push(far[: round(seconds * RATE)], near_start + 0.06)
    track = BuiltinEchoCancelledMic(mic, system)
    await track.hold()
    attachment = track.attach(near_start)
    mic.push(mixed[: round(seconds * RATE)], near_start)
    frames = await collect(attachment.queue, seconds - LOOKAHEAD_S)
    # The first block stands for the 10 ms before the mic's first sample: the canceller's latency.
    assert abs(frames[0].end - near_start) < 0.001
    out = np.concatenate([f.samples for f in frames]).astype(np.float64)[frames[0].samples.size :]
    for a, b in FAR_ONLY[:2]:
        part = slice(round(a * RATE), round(b * RATE))
        assert level_db(mixed[part]) - level_db(out[part]) >= 10, (a, b)
    await track.release()
    assert mic.holds == 0


async def test_a_far_end_gap_is_silence_and_a_far_restart_resets_the_canceller() -> None:
    mic, system = PushCapture(), PushCapture()
    far, mixed = FIX.far, FIX.mixed
    start = time.monotonic() - 30
    system.push(far[: 3 * RATE], start)
    system.push(far[4 * RATE : 6 * RATE], start + 4)
    track = BuiltinEchoCancelledMic(mic, system)
    await track.hold()
    attachment = track.attach(start)
    mic.push(mixed[: 6 * RATE], start)
    frames = await collect(attachment.queue, 5.8)
    assert frames[-1].end >= start + 5.7
    assert track._canceller.delay is not None  # pyright: ignore[reportPrivateUsage]
    system.lost_reason = "pw-record exited with code 1"
    mic.push(mixed[6 * RATE : 6 * RATE + FRAME], start + 6)
    await asyncio.sleep(0.05)
    system.lost_reason = None
    mic.push(mixed[6 * RATE + FRAME : 6 * RATE + 2 * FRAME], start + 6 + FRAME / RATE)
    await asyncio.sleep(0.05)
    assert track._canceller.delay is None  # pyright: ignore[reportPrivateUsage]
    await track.release()


async def test_a_mic_jump_resets_the_canceller() -> None:
    mic, system = PushCapture(), PushCapture()
    far, mixed = FIX.far, FIX.mixed
    start = time.monotonic() - 30
    system.push(far[: 8 * RATE], start)
    track = BuiltinEchoCancelledMic(mic, system)
    await track.hold()
    attachment = track.attach(start)
    mic.push(mixed[: 3 * RATE], start)
    await collect(attachment.queue, 2.8)
    assert track._canceller.delay is not None  # pyright: ignore[reportPrivateUsage]
    mic.push(mixed[5 * RATE : 6 * RATE], start + 5)
    frames = await collect(attachment.queue, 0.8)
    assert track._canceller.delay is None  # pyright: ignore[reportPrivateUsage]
    assert frames[-1].end > start + 5
    await track.release()


async def test_settle_waits_for_the_mic_audio_the_canceller_still_holds() -> None:
    mic, system = PushCapture(), PushCapture()
    seconds = 2.0
    start = time.monotonic() - seconds
    # The far end stops short, so the last mic blocks wait for it before they come out.
    system.push(FIX.far[: round((seconds - 0.05) * RATE)], start)
    track = BuiltinEchoCancelledMic(mic, system)
    await track.hold()
    attachment = track.attach(start)
    mic.push(FIX.mixed[: round(seconds * RATE)], start)
    began = time.monotonic()
    await track.settle(start + seconds)
    assert time.monotonic() - began < SETTLE_S + 0.05
    await asyncio.sleep(0)
    frames: list[Frame] = []
    while not attachment.queue.empty():
        frames.append(attachment.queue.get_nowait())
    lag = track._canceller.latency / RATE  # pyright: ignore[reportPrivateUsage]
    assert frames[-1].end >= start + seconds - lag - 0.001
    await track.release()
