"""PipeWire's echo-cancel module on this machine with the daemon's arguments, on virtual devices:
the far end plays into one null sink (the module's reference), the near end into another (the
module's mic); then the echo gate on what comes out. Nothing plays through the speakers and
nobody needs to talk. Run with
`pytest -m live -s tests/live/test_aec_module_live.py [--dump-dir DIR]`. Needs Parakeet."""

import asyncio
import contextlib
import functools
import re
import shutil
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import aec_fixture
import numpy as np
import pytest

from stt_daemon.audio.aec_pipewire import Command, PipewireAec, pw_cli_command
from stt_daemon.audio.pw_capture import pw_record_command
from stt_daemon.backends import echo_gate
from stt_daemon.backends.base import PLAYBACK_LEAD_S, PLAYBACK_LOOKAHEAD_S, PlaybackSpan, Reference
from stt_daemon.backends.install import is_installed
from stt_daemon.backends.parakeet import Recognizer, RecognizerKey, load_recognizer, model_dir
from stt_daemon.server.sources import PwDumpScan

MODEL = model_dir()
"""Taken at import, before the test fixtures point the data dir elsewhere."""

RATE = aec_fixture.RATE
FAR_SINK, NEAR_SINK = "stt-daemon-test-far", "stt-daemon-test-near"
QUIET_START_DB = -12.0
QUIET_START_S = 0.6

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not (shutil.which("pw-cli") and shutil.which("pactl")),
    reason="PipeWire with pipewire-pulse",
)


@contextlib.asynccontextmanager
async def null_sinks() -> AsyncGenerator[None]:
    ids: list[str] = []
    try:
        for name in (FAR_SINK, NEAR_SINK):
            proc = await asyncio.create_subprocess_exec(
                "pactl", "load-module", "module-null-sink", f"sink_name={name}",
                stdout=asyncio.subprocess.PIPE,
            )  # fmt: skip
            out, _ = await proc.communicate()
            assert proc.returncode == 0
            ids.append(out.decode().strip())
        yield
    finally:
        for module in ids:
            proc = await asyncio.create_subprocess_exec("pactl", "unload-module", module)
            await proc.wait()


def virtual_command(*, gain_control: bool) -> Command:
    """The daemon's `pw-cli` line, with the mic and the reference on the null sinks."""

    def command(name: str) -> list[str]:
        line = pw_cli_command(name)
        args = line[-1].replace(
            "monitor.mode = true",
            f"monitor.mode = true capture.props = {{ target.object = {NEAR_SINK} "
            f"stream.capture.sink = true }} sink.props = {{ target.object = {FAR_SINK} }}",
        )
        if not gain_control:
            args = args.replace("webrtc.gain_control = true", "webrtc.gain_control = false")
        return [*line[:-1], args]

    return command


async def record(serial: str, seconds: float) -> np.ndarray:
    proc = await asyncio.create_subprocess_exec(
        *pw_record_command(serial), stdout=asyncio.subprocess.PIPE
    )
    assert proc.stdout is not None
    data = b""
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(seconds):
            while chunk := await proc.stdout.read(65536):
                data += chunk
    proc.terminate()
    await proc.wait()
    return np.frombuffer(data[: len(data) - len(data) % 2], dtype="<i2") / 32768.0


async def play(path: Path, sink: str) -> None:
    proc = await asyncio.create_subprocess_exec("pw-play", "--target", sink, str(path))
    await proc.wait()


async def through_module(
    far: np.ndarray, mic: np.ndarray, tmp: Path, *, gain_control: bool = True
) -> np.ndarray:
    """What the module's source gives while `far` plays into the reference and `mic` into the
    mic, lined up with `mic`."""
    far_wav, mic_wav = tmp / "far.wav", tmp / "mic.wav"
    aec_fixture.write(far_wav, far)
    aec_fixture.write(mic_wav, mic)
    scan = PwDumpScan(enabled=True)
    scan.start()
    name = f"stt-daemon-aec-live-{time.monotonic_ns()}"
    aec = PipewireAec(name, nodes=scan, command=virtual_command(gain_control=gain_control))
    try:
        async with null_sinks():
            serial = await aec.load()
            try:
                recording = asyncio.create_task(record(serial, mic.size / RATE + 1.5))
                await asyncio.sleep(0.3)
                await asyncio.gather(play(far_wav, FAR_SINK), play(mic_wav, NEAR_SINK))
                cleaned = await recording
            finally:
                await aec.unload()
    finally:
        await scan.close()
    return lined_up(cleaned, mic)


def lined_up(audio: np.ndarray, like: np.ndarray) -> np.ndarray:
    """`audio` shifted to where it best matches `like` (power envelopes, 10 ms steps)."""
    step = RATE // 100

    def envelope(x: np.ndarray) -> np.ndarray:
        n = x.size // step
        return np.log10(np.mean(x[: n * step].reshape(n, step) ** 2, axis=1) + 1e-9)

    a, b = envelope(audio), envelope(like)
    best = max(
        range(0, 200), key=lambda lag: np.corrcoef(a[lag : lag + b.size], b[: a.size - lag])[0, 1]
    )
    out = audio[best * step : best * step + like.size]
    return np.pad(out, (0, like.size - out.size))


def db(audio: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(audio**2) + 1e-12))


def span(audio: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    return np.concatenate([aec_fixture.clip(audio, s, e) for s, e in spans])


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


@functools.cache
def recognizer() -> Recognizer:
    return load_recognizer(RecognizerKey(MODEL, 2, 1.5))


def heard(audio: np.ndarray) -> str:
    return recognizer().decode(audio.astype(np.float32), ()).text


@pytest.fixture
def fixture() -> aec_fixture.Fixture:
    if not is_installed(MODEL):
        pytest.skip("Parakeet is not installed")
    return aec_fixture.build()


def far_only_words(fixture: aec_fixture.Fixture) -> set[str]:
    return words(heard(fixture.far)) - words(heard(fixture.near))


async def test_two_tracks_with_no_path_between_them_keep_the_near_end(
    fixture: aec_fixture.Fixture, tmp_path: Path, dump_dir: Path
) -> None:
    cleaned = await through_module(fixture.far, fixture.near, tmp_path)
    aec_fixture.write(dump_dir / "two_tracks.wav", cleaned)
    near, got = words(heard(fixture.near)), words(heard(cleaned))
    with contextlib.redirect_stdout(sys.__stdout__):
        print(f"\nnear end: {heard(fixture.near)}\ncleaned:  {heard(cleaned)}")
    assert not got & far_only_words(fixture), got & far_only_words(fixture)
    assert len(near & got) >= 0.8 * len(near), near - got


async def test_echo_mixed_into_the_mic_is_taken_out(
    fixture: aec_fixture.Fixture, tmp_path: Path, dump_dir: Path
) -> None:
    cleaned = await through_module(fixture.far, fixture.mixed, tmp_path)
    aec_fixture.write(dump_dir / "coupled.wav", cleaned)
    near, raw, got = words(heard(fixture.near)), words(heard(fixture.mixed)), words(heard(cleaned))
    with contextlib.redirect_stdout(sys.__stdout__):
        print(f"\nmic:     {heard(fixture.mixed)}\ncleaned: {heard(cleaned)}")
    far_words = far_only_words(fixture)
    assert raw & far_words, "the fixture's echo should be heard in the plain mic"
    assert not got & far_words, got & far_words
    assert len(near & got) >= 0.7 * len(near), near - got


CHUNK_S = (14.0, 22.0)
"""A commit's audio: the far end alone (14.2-17.1 s), then the near end's second phrase."""
ECHO_S = (14.2, 17.1)


def gated(audio: np.ndarray, far: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`CHUNK_S` of `audio` through the echo gate with `far` as its playback; returns the gated
    audio and which input samples it kept."""
    a, b = (round(t * RATE) for t in CHUNK_S)
    lead, ahead = round(PLAYBACK_LEAD_S * RATE), round(PLAYBACK_LOOKAHEAD_S * RATE)
    playback = np.pad(far, (lead, ahead))[a : b + lead + ahead].astype(np.float32)
    chunk = audio[a:b].astype(np.float32)
    out, gate_map = echo_gate.apply(
        chunk, Reference((PlaybackSpan(0, chunk.size, playback, lead, ahead),))
    )
    kept = np.zeros(chunk.size, dtype=bool)
    for _at, start, size in gate_map.runs:
        kept[start : start + size] = True
    return out, kept


def kept_share(kept: np.ndarray, span_s: tuple[float, float]) -> float:
    a, b = (round((t - CHUNK_S[0]) * RATE) for t in span_s)
    return float(kept[a:b].mean())


async def test_gain_control_and_the_echo_gate_on_the_cleaned_mic(
    fixture: aec_fixture.Fixture, tmp_path: Path, dump_dir: Path
) -> None:
    """One commit's audio (far end alone, then a near phrase whose first 0.6 s is 12 dB quieter)
    through the module, gain control on and off, then the echo gate with the playback as its
    reference. Prints how much of the echo and of the quiet start the gate keeps, and the text."""
    near = fixture.near.copy()
    start = round(aec_fixture.NEAR_PHRASES[1][0] * RATE)
    near[start : start + round(QUIET_START_S * RATE)] *= 10 ** (QUIET_START_DB / 20)
    mic = near + (fixture.mixed - fixture.near)
    quiet_s = (aec_fixture.NEAR_PHRASES[1][0], aec_fixture.NEAR_PHRASES[1][0] + QUIET_START_S)
    far_words = far_only_words(fixture)
    near_words = words(heard(aec_fixture.clip(near, *CHUNK_S)))
    tracks = {"plain mic": mic}
    for label, gain_control in (("gain control on", True), ("gain control off", False)):
        tracks[label] = await through_module(fixture.far, mic, tmp_path, gain_control=gain_control)
    rows: list[str] = []
    results: dict[str, set[str]] = {}
    for label, audio in tracks.items():
        out, kept = gated(audio, fixture.far)
        aec_fixture.write(dump_dir / f"gated {label}.wav", out)
        text = heard(out)
        results[label] = words(text)
        rows.append(
            f"{label:17} echo kept {kept_share(kept, ECHO_S):4.0%}   quiet start kept "
            f"{kept_share(kept, quiet_s):4.0%}   {text}"
        )
    with contextlib.redirect_stdout(sys.__stdout__):
        print("\n" + "\n".join(rows))
    for label in ("gain control on", "gain control off"):
        assert not results[label] & far_words, (label, results[label] & far_words)
        assert len(near_words & results[label]) >= 0.8 * len(near_words), (
            near_words - results[label]
        )
