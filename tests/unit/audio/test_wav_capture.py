import asyncio
import itertools
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from stt_daemon.audio.capture import CaptureUnavailable
from stt_daemon.audio.frames import Frame
from stt_daemon.audio.wav_capture import WavCapture, read_wav

RATE = 16000


def _write_wav(path: Path, pcm: np.ndarray, rate: int = RATE, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype("<i2").tobytes())
    return path


def test_read_wav_averages_channels(tmp_path: Path) -> None:
    stereo = np.array([[8192, 16384]] * 10).reshape(-1)
    audio = read_wav(_write_wav(tmp_path / "s.wav", stereo, channels=2))
    assert audio.dtype == np.float32
    assert audio.shape == (10,)
    assert float(audio[0]) == pytest.approx(0.375)


def test_read_wav_refuses_another_rate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="16000 Hz"):
        read_wav(_write_wav(tmp_path / "s.wav", np.zeros(10), rate=8000))


async def test_plays_files_in_real_time_then_silence(tmp_path: Path) -> None:
    tone = np.full(RATE // 5, 16384)  # 0.2 s at 0.5
    source = WavCapture(_write_wav(tmp_path / "t.wav", tone), keep_open=True)
    await source.start()
    attachment = source.attach()
    began = time.monotonic()
    assert await asyncio.to_thread(source.wait_idle, 2.0)
    elapsed = time.monotonic() - began
    await asyncio.sleep(0.1)
    await source.stop()
    frames: list[Frame] = []
    while not attachment.queue.empty():
        frames.append(attachment.queue.get_nowait())
    audio = np.concatenate([f.samples for f in frames])
    assert elapsed == pytest.approx(0.2, abs=0.1)
    assert audio[: tone.size] == pytest.approx(0.5)
    assert np.all(audio[tone.size :] == 0.0)
    assert audio.size > tone.size
    gaps = [b.start - a.end for a, b in itertools.pairwise(frames)]
    assert max(map(abs, gaps)) < 1e-9


async def test_ring_holds_what_played(tmp_path: Path) -> None:
    first = _write_wav(tmp_path / "a.wav", np.full(800, 8192))
    second = _write_wav(tmp_path / "b.wav", np.full(800, 16384))
    source = WavCapture(first)
    await source.start()
    source.play(second)
    assert await asyncio.to_thread(source.wait_idle, 2.0)
    audio = np.concatenate([f.samples for f in source.ring_since()])
    await source.stop()
    assert audio[:800] == pytest.approx(0.25)
    assert audio[800:1600] == pytest.approx(0.5)


async def test_per_session_source_opens_on_hold(tmp_path: Path) -> None:
    source = WavCapture(_write_wav(tmp_path / "t.wav", np.zeros(160)), keep_open=False)
    await source.start()
    with pytest.raises(CaptureUnavailable):
        source.attach()
    await source.hold()
    assert source.is_open
    source.attach().detach()
    await source.release()
    assert not source.is_open
