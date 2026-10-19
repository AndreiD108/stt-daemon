"""Echo cancellation on this machine, through speakers and the real mic. Run with
`pytest -m live -s tests/live/test_aec_live.py [--dump-dir DIR]`, speakers on. The first test
needs nobody talking; the second asks you to read a sentence twice. No ElevenLabs."""

import asyncio
import shutil
import sys
import time
from pathlib import Path

import aec_fixture
import numpy as np
import pytest

from stt_daemon.audio.aec_builtin import BuiltinEchoCancelledMic
from stt_daemon.audio.aec_pipewire import EchoCancelledMic, PipewireAec
from stt_daemon.audio.capture import CaptureSource, MicCapture
from stt_daemon.audio.frames import Frame
from stt_daemon.audio.pw_capture import PwRecordCapture, pw_record_command
from stt_daemon.backends.install import is_installed
from stt_daemon.backends.parakeet import RecognizerKey, load_recognizer, model_dir
from stt_daemon.server.sources import PwDumpScan

MODEL = model_dir()
"""Taken at import, before the test fixtures point the data dir elsewhere."""

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
    scan = PwDumpScan(enabled=True)
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


READ_AT = (aec_fixture.NEAR_PHRASES[0][0], aec_fixture.NEAR_PHRASES[1][0])
"""When you read: over the speakers' fourth phrase, then in their pause."""
SENTENCE = "Scribbly sends pen marks while Talkey holds the push to talk key."
FAR_ONLY_S = (4.0, 8.6)
"""Speakers only, after the built-in filter's first seconds and before the first cue to read."""
NEAR_ONLY_S = (17.4, 22.1)
READS_S = ((9.4, 14.2), (17.4, 22.4))
MODES = ("off", "pipewire", "builtin")


async def collect(source: CaptureSource, since: float, until: float) -> list[Frame]:
    attachment = source.attach(since)
    frames: list[Frame] = []
    try:
        while (remaining := until - time.monotonic()) > 0:
            try:
                async with asyncio.timeout(remaining):
                    frames.append(await attachment.queue.get())
            except TimeoutError:
                break
    finally:
        attachment.detach()
    return frames


def on_timeline(frames: list[Frame], t0: float, seconds: float) -> np.ndarray:
    out = np.zeros(round(seconds * RATE))
    for frame in frames:
        i = round((frame.start - t0) * RATE)
        part = frame.samples[: max(0, out.size - i)]
        if i >= 0:
            out[i : i + part.size] = part
    return out


def windows(db: np.ndarray, span: tuple[float, float]) -> np.ndarray:
    t = np.arange(db.size) * WINDOW / RATE
    return (t >= span[0]) & (t < span[1])


async def prompt(capsys: pytest.CaptureFixture[str], t0: float) -> None:
    for n, at in enumerate(READ_AT, 1):
        await asyncio.sleep(max(0.0, t0 + at - 1.0 - time.monotonic()))
        with capsys.disabled():
            print(f"\n>>> read {n} of 2 now: {SENTENCE}", flush=True)


def transcripts(tracks: dict[str, np.ndarray]) -> dict[str, list[str]]:
    if not is_installed(MODEL):
        return {}
    recognizer = load_recognizer(RecognizerKey(MODEL, 2, 1.5))
    heard: dict[str, list[str]] = {}
    for mode in MODES:
        heard[mode] = []
        for start, end in READS_S:
            stream = recognizer.create_stream()
            stream.accept_waveform(
                RATE, tracks[mode][round(start * RATE) : round(end * RATE)].astype(np.float32)
            )
            recognizer.decode_stream(stream)
            heard[mode].append(stream.result.text)
    return heard


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("pw-cli") is None, reason="PipeWire")
async def test_the_built_in_canceller_against_the_pipewire_module_and_none(
    capsys: pytest.CaptureFixture[str], dump_dir: Path
) -> None:
    """All three mic tracks record the same mic at once, so they compare on the same speech."""
    fixture = aec_fixture.build()
    far_wav = dump_dir / "far.wav"
    aec_fixture.write(far_wav, fixture.far)
    seconds = fixture.far.size / RATE + 0.5
    mic, system = MicCapture(keep_open=False), PwRecordCapture()
    scan = PwDumpScan(enabled=True)
    scan.start()
    module = EchoCancelledMic(
        mic,
        PipewireAec("stt-daemon-aec-live-compare", nodes=scan),
        cleaned=lambda target: PwRecordCapture(target=target),
    )
    builtin = BuiltinEchoCancelledMic(mic, system)
    sources: dict[str, CaptureSource] = {
        "system": system,
        "off": mic,
        "pipewire": module,
        "builtin": builtin,
    }
    held: list[CaptureSource] = []
    try:
        for source in (mic, system, module, builtin):
            await source.hold()
            held.append(source)
        assert module.unavailable_reason is None, module.unavailable_reason
        t0 = time.monotonic()
        player = await asyncio.create_subprocess_exec("pw-play", str(far_wav))
        prompts = asyncio.create_task(prompt(capsys, t0))
        collected = await asyncio.gather(*(collect(s, t0, t0 + seconds) for s in sources.values()))
        await prompts
        await player.wait()
    finally:
        for source in reversed(held):
            await source.release()
        await scan.close()
    tracks = {
        name: on_timeline(frames, t0, seconds)
        for name, frames in zip(sources, collected, strict=True)
    }
    for name, audio in tracks.items():
        aec_fixture.write(dump_dir / f"{name}.wav", audio)
    db = {name: window_db(audio) for name, audio in tracks.items()}
    system_floor, mic_floor = np.percentile(db["system"], 20), np.percentile(db["off"], 20)
    far_only = windows(db["off"], FAR_ONLY_S) & (db["system"] > system_floor + 10)
    near_only = (
        windows(db["off"], NEAR_ONLY_S)
        & (db["system"] < system_floor + 6)
        & (db["off"] > mic_floor + 10)
    )
    residual = {m: mean_power(db[m][far_only]) for m in MODES}
    retention = {m: mean_power(db[m][near_only]) - mean_power(db["off"][near_only]) for m in MODES}
    with capsys.disabled():
        print(f"\nrecorded to {dump_dir}")
        for mode in MODES:
            print(f"{mode:9} echo {residual[mode]:6.1f} dB   near end {retention[mode]:+5.1f} dB")
        for mode, texts in transcripts(tracks).items():
            print(f"{mode:9} " + " | ".join(texts))
    assert far_only.sum() >= 20, "the mic did not hear the speakers; turn them up"
    assert near_only.sum() >= 10, "no speech heard in the speakers' pause"
    assert residual["off"] - residual["builtin"] >= 6, residual
    assert abs(retention["builtin"]) <= 3, retention
