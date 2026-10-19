"""ByteStreamCapture with a stand-in decoder: order, audio-time stamps, failures, backpressure."""

import asyncio
import contextlib
import itertools
import sys

import numpy as np
import pytest

from stt_daemon.audio.frames import Frame
from stt_daemon.audio.stream_capture import (
    QUEUE_FRAMES,
    STDERR_TAIL,
    ByteStreamCapture,
    DecodeFailed,
    ToolMissing,
    ffmpeg_command,
)

RATE = 16000
CAT = [
    sys.executable,
    "-c",
    "import shutil, sys; shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)",
]


def script(code: str) -> list[str]:
    return [sys.executable, "-c", f"import sys\n{code}"]


def ramp(n: int) -> bytes:
    return (np.arange(n) % 30000).astype("<i2").tobytes()


def drain(stream: ByteStreamCapture) -> list[Frame]:
    queue = stream.attach().queue
    frames: list[Frame] = []
    while not queue.empty():
        frames.append(queue.get_nowait())
    return frames


async def collect_while(stream: ByteStreamCapture, job: "asyncio.Future[None]") -> list[Frame]:
    """Takes frames until `job` is done and nothing is left."""
    queue = stream.attach().queue
    frames: list[Frame] = []
    while not (job.done() and queue.empty()):
        with contextlib.suppress(TimeoutError):
            frames.append(await asyncio.wait_for(queue.get(), 0.05))
    job.result()
    return frames


async def test_decoded_bytes_come_out_in_order_stamped_with_their_audio_position() -> None:
    stream = ByteStreamCapture(CAT)
    await stream.hold()
    data = ramp(5000)

    async def feed() -> None:
        cuts = (0, 777, 3001, 6000, len(data))  # odd cuts split a sample across writes
        for start, end in itertools.pairwise(cuts):
            await stream.write(data[start:end])
        await stream.finish()

    frames = await collect_while(stream, asyncio.ensure_future(feed()))
    samples = np.concatenate([f.samples for f in frames])
    expected = np.frombuffer(data, dtype="<i2") / 32768.0
    assert np.allclose(samples, expected)
    taken = 0
    for frame in frames:
        taken += frame.samples.size
        assert frame.end == pytest.approx(stream.origin + taken / RATE)
    assert stream.end == pytest.approx(stream.origin + 5000 / RATE)
    await stream.release()


async def test_a_decoder_that_fails_at_the_end_raises_with_its_stderr_tail() -> None:
    stream = ByteStreamCapture(
        script("sys.stdin.buffer.read(); sys.stderr.write('bad header'); sys.exit(1)")
    )
    await stream.hold()
    await stream.write(b"\x00" * 100)
    with pytest.raises(DecodeFailed, match="code 1: bad header"):
        await stream.finish()
    await stream.release()


async def test_a_decoder_that_fails_early_reports_it_while_input_still_comes() -> None:
    stream = ByteStreamCapture(script("sys.stderr.write('x' * 10000 + 'END'); sys.exit(2)"))
    failed: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    stream.on_failed = failed.set_result
    await stream.hold()
    message = await asyncio.wait_for(failed, 5)
    assert message.endswith("END")
    assert len(message) <= STDERR_TAIL + 40
    await stream.write(b"\x00" * 100_000)  # ignored, no broken-pipe error
    await stream.release()


async def test_input_with_no_audio_fails_at_the_end() -> None:
    stream = ByteStreamCapture(CAT)
    await stream.hold()
    with pytest.raises(DecodeFailed, match="no audio"):
        await stream.finish()
    await stream.release()


async def test_raw_pcm_needs_no_decoder_and_keeps_a_split_sample() -> None:
    stream = ByteStreamCapture(None)
    await stream.hold()
    data = ramp(100)
    await stream.write(data[:51])
    await stream.write(data[51:])
    await stream.finish()
    samples = np.concatenate([f.samples for f in drain(stream)])
    assert np.allclose(samples, np.frombuffer(data, dtype="<i2") / 32768.0)


async def test_writes_wait_while_the_session_is_behind() -> None:
    stream = ByteStreamCapture(None)
    await stream.hold()
    data = ramp(512 * (QUEUE_FRAMES + 10))
    writing = asyncio.ensure_future(stream.write(data))
    await asyncio.sleep(0.05)
    assert not writing.done()
    frames = await collect_while(stream, writing)
    assert len(frames) == QUEUE_FRAMES + 10


async def test_release_frees_a_write_waiting_on_a_full_queue() -> None:
    stream = ByteStreamCapture(None)
    await stream.hold()
    writing = asyncio.ensure_future(stream.write(ramp(512 * (QUEUE_FRAMES + 10))))
    await asyncio.sleep(0.05)
    assert not writing.done()
    await stream.release()
    await asyncio.wait_for(writing, 1)


async def test_release_ends_a_decoder_whose_output_nothing_reads() -> None:
    flood = "import time\nsys.stdout.buffer.write(bytes(1 << 20))\nsys.stdout.flush()\n"
    flood += "time.sleep(60)"
    stream = ByteStreamCapture(script(flood))
    await stream.hold()
    await asyncio.sleep(0.3)  # the queue and both pipe buffers fill
    await asyncio.wait_for(stream.release(), 5)


async def test_release_kills_a_decoder_still_running() -> None:
    stream = ByteStreamCapture(CAT)
    await stream.hold()
    await stream.write(b"\x00" * 64)
    await asyncio.wait_for(stream.release(), 5)
    assert not stream.is_open


async def test_a_missing_decoder_raises_tool_missing() -> None:
    stream = ByteStreamCapture(["/nonexistent/ffmpeg-for-tests"])
    with pytest.raises(ToolMissing):
        await stream.hold()


def test_ffmpeg_command_declares_raw_pcm_before_the_input() -> None:
    command = ffmpeg_command((48000, 2))
    raw = ["-f", "s16le", "-ar", "48000", "-ac", "2"]
    assert command[command.index("-i") - len(raw) : command.index("-i")] == raw
    assert command[-7:] == ["-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"]
    plain = ffmpeg_command()
    assert "-ar" not in plain[: plain.index("-i")]
