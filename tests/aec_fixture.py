"""Synthetic echo for the built-in canceller's tests: TTS far end, room echo added to a TTS near
end. `python tests/aec_fixture.py DIR` writes them as WAVs, to listen to."""

import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from stt_daemon.audio.wav_capture import read_wav

RATE = 16000
FIXTURES = Path(__file__).parent / "fixtures"
ECHO_DELAY_S = 0.08
ECHO_GAIN_DB = -6.0
RT60_S = 0.3
FAR_ONLY = [(0.0, 9.7), (14.2, 17.1), (22.2, 24.6)]
NEAR_ONLY = [(11.7, 13.7), (17.7, 21.7)]
NEAR_PHRASES = [(9.7, 13.7), (17.7, 21.7)]

type Audio = npt.NDArray[np.float64]


def read(path: Path) -> Audio:
    return read_wav(path).astype(np.float64)


def write(path: Path, audio: Audio) -> None:
    pcm = np.clip(np.rint(audio * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def room_ir(delay_s: float = ECHO_DELAY_S, *, seed: int = 7) -> Audio:
    """Direct path at `delay_s`, then exponentially decaying noise (RT60 0.3 s); unit energy."""
    rng = np.random.default_rng(seed)
    start = round(delay_s * RATE)
    tail = round(RT60_S * RATE)
    t = np.arange(tail) / RATE
    ir = np.zeros(start + tail)
    ir[start:] = rng.standard_normal(tail) * 10 ** (-3 * t / RT60_S) * 0.3
    ir[start] = 1.0
    return ir / np.sqrt(np.sum(ir**2))


def echo_of(far: Audio, ir: Audio, gain_db: float = ECHO_GAIN_DB) -> Audio:
    return np.convolve(far, ir)[: far.size] * 10 ** (gain_db / 20)


def clip(audio: Audio, start_s: float, end_s: float) -> Audio:
    return audio[round(start_s * RATE) : round(end_s * RATE)]


@dataclass(frozen=True)
class Fixture:
    far: Audio
    near: Audio
    """The clean near end."""
    mixed: Audio
    """`near` plus the far end's echo: what the mic hears."""
    far_only: list[tuple[float, float]]
    near_only: list[tuple[float, float]]


def build() -> Fixture:
    """Far end: two_phrases.wav's phrases at 0, 3.1, 5.7, 8.8, 14.2 and 22.2 s. Near end:
    jargon.wav's first phrase at 9.7 s, over the far end's fourth, and its second at 17.7 s, in a
    far-end pause."""
    phrases = read(FIXTURES / "two_phrases.wav")
    jargon = read(FIXTURES / "jargon.wav")
    first, second = clip(phrases, 0.9, 3.8), clip(phrases, 5.6, 8.0)
    total = round(25.2 * RATE)
    far, near = np.zeros(total), np.zeros(total)
    for at, part in (
        (0.0, first),
        (3.1, second),
        (5.7, first),
        (8.8, second),
        (14.2, first),
        (22.2, second),
    ):
        i = round(at * RATE)
        far[i : i + part.size] += part
    phrases = (clip(jargon, 0.9, 4.9), clip(jargon, 7.3, 11.3))
    for (at, _), part in zip(NEAR_PHRASES, phrases, strict=True):
        i = round(at * RATE)
        near[i : i + part.size] += part
    return Fixture(
        far=far,
        near=near,
        mixed=near + echo_of(far, room_ir()),
        far_only=FAR_ONLY,
        near_only=NEAR_ONLY,
    )


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    fixture = build()
    write(out / "far.wav", fixture.far)
    write(out / "near.wav", fixture.near)
    write(out / "mixed.wav", fixture.mixed)
