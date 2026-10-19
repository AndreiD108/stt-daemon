"""The echo gate on synthetic level patterns: which frames the playback explains, the gate's two
thresholds, and decoder times mapped back through the cuts."""

import numpy as np
import pytest
from aec_fixture import echo_of, room_ir

from stt_daemon.backends import echo_gate
from stt_daemon.backends.base import PlaybackSpan, Reference
from stt_daemon.backends.echo_gate import FRAME, GateMap, apply, echo_frames, gate

RATE = 16000
LEAD = 800 * RATE // 1000
LOOKAHEAD = 100 * RATE // 1000


def bursts(frames: int, seed: int, db: float) -> np.ndarray:
    """Noise in syllable-like bursts (60-200 ms) at `db` dBFS, with dips 25 dB under
    (40-160 ms) between them."""
    rng = np.random.default_rng(seed)
    envelope = np.full(frames, 10 ** (-25 / 20))
    at = 0
    while at < frames:
        on = int(rng.integers(3, 10))
        envelope[at : at + on] = rng.uniform(0.5, 1.0)
        at += on + int(rng.integers(2, 8))
    noise = rng.normal(0, 1, frames * FRAME)
    return (noise * np.repeat(envelope, FRAME) * 10 ** (db / 20)).astype(np.float32)


def delayed(x: np.ndarray, frames: int, db: float) -> np.ndarray:
    """`x` through a room (RT60 0.3 s: 4 dB per frame), shifted by `frames`, at `db`."""
    wet = echo_of(x.astype(np.float64), room_ir(0.0), 0.0)
    out = np.zeros_like(x)
    shift = frames * FRAME
    if shift >= 0:
        out[shift:] = wet[: x.size - shift]
    else:
        out[:shift] = wet[-shift:]
    return (out * 10 ** (db / 20)).astype(np.float32)


def room(x: np.ndarray, seed: int = 9) -> np.ndarray:
    """The mic's noise floor, -70 dBFS, under `x`."""
    floor = np.random.default_rng(seed).normal(0, 10 ** (-70 / 20), x.size)
    return (x + floor).astype(np.float32)


def span_over(playback: np.ndarray, mic_frames: int) -> PlaybackSpan:
    """One stretch of `mic_frames`; `playback` covers lead + stretch + lookahead."""
    return PlaybackSpan(0, mic_frames * FRAME, playback, LEAD, LOOKAHEAD)


def with_margins(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(LEAD, np.float32), x, np.zeros(LOOKAHEAD, np.float32)])


N = 250  # 5 s


def active(x: np.ndarray) -> np.ndarray:
    return echo_gate.frame_db(x) > -60


def test_delayed_echo_is_found() -> None:
    played = bursts(N, 1, -20)
    mic = room(delayed(played, 6, -15))
    found = echo_frames(N, mic, [span_over(with_margins(played), N)])
    on = active(mic)
    assert found[on].mean() > 0.9


def test_near_speech_over_the_echo_is_not_echo() -> None:
    played = bursts(N, 1, -20)
    near = bursts(N, 2, -15)
    alone = 3 * N // 4  # one delay and coupling per chunk: echo alone must lead
    near[: alone * FRAME] = 0
    echo = delayed(played, 6, -15)
    mic = room(echo + near)
    found = echo_frames(N, mic, [span_over(with_margins(played), N)])
    over = echo_gate.frame_db(near) > echo_gate.frame_db(echo) + 10
    assert found[over].mean() < 0.1
    assert found[:alone][active(mic)[:alone]].mean() > 0.9


def test_playback_stamped_after_its_echo_is_found() -> None:
    played = bursts(N, 1, -20)
    mic = room(delayed(played, -4, -15))  # the echo comes 80 ms before the playback's stamp
    found = echo_frames(N, mic, [span_over(with_margins(played), N)])
    assert found[active(mic)].mean() > 0.9


def test_a_mic_that_does_not_follow_the_playback_has_no_echo() -> None:
    played = bursts(N, 1, -20)
    mic = room(bursts(N, 3, -30))
    assert not echo_frames(N, mic, [span_over(with_margins(played), N)]).any()


def test_silent_playback_has_no_echo() -> None:
    mic = room(bursts(N, 3, -30))
    silent = np.zeros(LEAD + N * FRAME + LOOKAHEAD, np.float32)
    assert not echo_frames(N, mic, [span_over(silent, N)]).any()


def test_each_span_matches_the_playback_at_its_own_time() -> None:
    """Two stretches with a pause cut between: each span's playback lines up with its own audio
    only, so the echo is found in both."""
    played = bursts(2 * N, 1, -20)
    mic_all = room(delayed(played, 6, -15))
    half = N * FRAME
    mic = np.concatenate([mic_all[: half // 2], mic_all[half + half // 2 :]])
    frames = N // 2
    first = with_margins(played)[: LEAD + frames * FRAME + LOOKAHEAD]
    second_start = half + half // 2
    second = with_margins(played)[second_start : second_start + LEAD + frames * FRAME + LOOKAHEAD]
    spans = [
        PlaybackSpan(0, frames * FRAME, first, LEAD, LOOKAHEAD),
        PlaybackSpan(frames * FRAME, frames * FRAME, second, LEAD, LOOKAHEAD),
    ]
    found = echo_frames(N, mic, spans)
    on = active(mic)
    assert found[:frames][on[:frames]].mean() > 0.9
    assert found[frames:][on[frames:]].mean() > 0.9


def steady(frames: int, db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 1, frames * FRAME) * 10 ** (db / 20)).astype(np.float32)


def kept_frames(audio: np.ndarray, echo: np.ndarray) -> np.ndarray:
    """Which input frames survive the gate."""
    _, kept = gate(audio, echo)
    out = np.zeros(len(audio) // FRAME, bool)
    for _, chunk, samples in kept.runs:
        out[chunk // FRAME : (chunk + samples) // FRAME] = True
    return out


@pytest.mark.parametrize(("echo", "kept"), [(True, False), (False, True)])
def test_the_body_cuts_echo_harder(echo: bool, kept: bool) -> None:
    """A stretch 30 dB under the voice: past the echo threshold (25), within the other (35)."""
    audio = np.concatenate([steady(50, -20), steady(60, -50, 1), steady(50, -20, 2)])
    mask = np.zeros(160, bool)
    mask[50:110] = echo
    assert kept_frames(audio, mask)[60:100].all() == kept
    assert kept_frames(audio, mask)[60:100].any() == kept


@pytest.mark.parametrize(("echo", "kept"), [(True, False), (False, True)])
def test_the_start_cuts_echo_harder(echo: bool, kept: bool) -> None:
    """Before the first loud frame, 8 dB under the voice: past the echo start threshold (6),
    within the other (10)."""
    audio = np.concatenate([steady(60, -28), steady(100, -20, 1)])
    mask = np.zeros(160, bool)
    mask[:60] = echo
    assert kept_frames(audio, mask)[:50].all() == kept
    assert kept_frames(audio, mask)[:50].any() == kept


def test_cut_stretches_are_joined_by_near_silence() -> None:
    audio = np.concatenate([steady(50, -20), np.zeros(60 * FRAME, np.float32), steady(50, -20)])
    gated, kept = gate(audio, np.zeros(160, bool))
    assert len(kept.runs) == 2
    joint = gated[kept.runs[0][2] : kept.runs[1][0]]
    assert joint.size == round(echo_gate.JOINT_S * RATE)
    assert 20 * np.log10(np.sqrt((joint.astype(np.float64) ** 2).mean())) < -60


def test_short_cuts_are_filled() -> None:
    audio = np.concatenate([steady(50, -20), np.zeros(24 * FRAME, np.float32), steady(50, -20)])
    gated, kept = gate(audio, np.zeros(124, bool))
    assert len(kept.runs) == 1
    assert gated.size == audio.size


GAP = 1600  # a joint, 100 ms


@pytest.mark.parametrize(
    ("gated_s", "chunk_s"),
    [
        ((0.05, 0.08), (0.05, 0.08)),  # inside the first stretch
        ((0.21, 0.25), (0.51, 0.55)),  # inside the second
        ((0.09, 0.25), (0.5, 0.55)),  # across the joint, mostly in the second
        ((0.07, 0.11), (0.07, 0.1)),  # across the joint, mostly in the first
    ],
)
def test_word_times_map_back_through_the_cuts(
    gated_s: tuple[float, float], chunk_s: tuple[float, float]
) -> None:
    kept = GateMap(((0, 0, 1600), (1600 + GAP, 8000, 1600)))
    assert kept.word_span(*gated_s) == pytest.approx(chunk_s)


def test_without_a_reference_only_the_level_gate_applies() -> None:
    audio = np.concatenate([steady(50, -20), steady(60, -50, 1), steady(50, -20, 2)])
    gated, _ = apply(audio, None)
    assert gated.size == audio.size  # 30 dB under is within the non-echo body threshold
    gated, _ = apply(audio, Reference(()))
    assert gated.size == audio.size
