"""PipeWire's echo-cancel module on this machine, through speakers and the real mic. Run with
`pytest -m live tests/live/test_aec_live.py`, speakers on, nobody talking. No ElevenLabs."""

import asyncio
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from stt_daemon.audio.aec_pipewire import PipewireAec
from stt_daemon.audio.pw_capture import pw_record_command
from stt_daemon.server.sources import SourceScan

FIXTURE = Path(__file__).parents[1] / "fixtures" / "two_phrases.wav"
RATE = 16000
WINDOW = RATE // 10
SETTLE_S = 4.0
"""The canceller adapts over the first seconds; those are not measured."""
PLAIN_MIC = ["pw-record", "--raw", "--rate", str(RATE), "--channels", "1", "--format", "s16", "-"]


async def record(command: list[str], seconds: float) -> np.ndarray:
    proc = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    assert proc.stdout is not None
    data = b""
    try:
        async with asyncio.timeout(seconds):
            while chunk := await proc.stdout.read(65536):
                data += chunk
    except TimeoutError:
        pass
    finally:
        proc.terminate()
        await proc.wait()
    return np.frombuffer(data[: len(data) - len(data) % 2], dtype="<i2") / 32768.0


def window_db(audio: np.ndarray) -> np.ndarray:
    count = audio.size // WINDOW
    frames = audio[: count * WINDOW].reshape(count, WINDOW)
    return 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)


def mean_power(db: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(10 ** (db / 10))))


async def play_twice() -> None:
    for _ in range(2):
        proc = await asyncio.create_subprocess_exec("pw-play", str(FIXTURE))
        await proc.wait()


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("pw-cli") is None, reason="PipeWire")
async def test_the_cleaned_source_hears_the_speakers_at_least_10_db_less_than_the_mic() -> None:
    scan = SourceScan(enabled=True)
    scan.start()
    aec = PipewireAec("stt-daemon-aec-live-test", nodes=scan)
    try:
        serial = await aec.load()
        seconds = 2 * 8.5
        plain, cleaned, _ = await asyncio.gather(
            record(PLAIN_MIC, seconds),
            record(pw_record_command(serial), seconds),
            play_twice(),
        )
    finally:
        await aec.unload()
        await scan.close()
    plain_db, cleaned_db = window_db(plain), window_db(cleaned)
    count = min(plain_db.size, cleaned_db.size)
    plain_db, cleaned_db = plain_db[:count], cleaned_db[:count]
    settled = np.arange(count) * WINDOW / RATE >= SETTLE_S
    # Far-end-only windows: where the plain mic hears the playback well above its floor.
    floor = np.percentile(plain_db, 20)
    far_end = settled & (plain_db > floor + 10)
    assert far_end.sum() >= 20, "the mic did not hear the speakers; turn them up"
    reduction = mean_power(plain_db[far_end]) - mean_power(cleaned_db[far_end])
    assert reduction >= 10, f"only {reduction:.1f} dB less"
