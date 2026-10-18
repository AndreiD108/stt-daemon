from pathlib import Path

import numpy as np
import pytest

from stt_daemon.audio.frames import Frame, Samples
from stt_daemon.audio.vad import (
    EDGE_LOOKBACK_WINDOWS,
    WINDOW,
    SpeechEnded,
    SpeechStarted,
    StreamingVAD,
    VadEvent,
)
from stt_daemon.audio.wav_capture import read_wav

RATE = 16000
FIXTURE = Path(__file__).parents[1] / "fixtures" / "two_phrases.wav"
# Where the fixture was built to put speech (TTS phrases, -40 dBFS edges), seconds from its start.
SPEECH = ((1.0, 3.74), (5.74, 7.81))
START_TOLERANCE_S = 0.08
END_TOLERANCE_S = 0.12
T0 = 1000.0


def _feed(vad: StreamingVAD, audio: Samples, chunk: int, t0: float = T0) -> list[VadEvent]:
    events: list[VadEvent] = []
    for i in range(0, audio.size, chunk):
        part = audio[i : i + chunk]
        events += vad.push(Frame(part, t0 + (i + part.size) / RATE, RATE))
    return events


@pytest.fixture(scope="module")
def speech() -> Samples:
    return read_wav(FIXTURE)


@pytest.mark.parametrize("chunk", [160, 512, 777, 3200])
def test_events_carry_the_true_speech_edges(speech: Samples, chunk: int) -> None:
    events = _feed(StreamingVAD(), speech, chunk)
    assert [type(e) for e in events] == [SpeechStarted, SpeechEnded] * 2
    assert [e.speech_id for e in events] == ["s1", "s1", "s2", "s2"]
    pairs = zip(events[::2], events[1::2], strict=True)
    for (start, end), (began, ended) in zip(SPEECH, pairs, strict=True):
        assert began.at - T0 == pytest.approx(start, abs=START_TOLERANCE_S)
        assert ended.at - T0 == pytest.approx(end, abs=END_TOLERANCE_S)


def test_events_come_after_the_minimum_durations(speech: Samples) -> None:
    vad = StreamingVAD(min_speech_ms=250, min_silence_ms=600)
    confirmed: list[tuple[VadEvent, float]] = []
    for i in range(0, speech.size, 512):
        frame = Frame(speech[i : i + 512], T0 + (i + 512) / RATE, RATE)
        confirmed += [(e, frame.end) for e in vad.push(frame)]
    for event, when in confirmed:
        if isinstance(event, SpeechStarted):
            assert 0.256 - 0.001 <= when - event.at <= vad.start_delay_s + 0.001
        else:
            assert when - event.at == pytest.approx(vad.end_delay_s, abs=0.001)
    assert vad.start_delay_s == pytest.approx(0.256 + EDGE_LOOKBACK_WINDOWS * WINDOW / RATE)
    assert vad.end_delay_s == pytest.approx(0.608)


@pytest.mark.parametrize("phase", range(0, WINDOW, 32))
def test_starts_stay_at_the_edge_whatever_the_window_phase(speech: Samples, phase: int) -> None:
    audio = np.concatenate([np.zeros(phase, np.float32), speech])
    starts = [e for e in _feed(StreamingVAD(), audio, 320) if isinstance(e, SpeechStarted)]
    assert len(starts) == 2
    for began, (start, _) in zip(starts, SPEECH, strict=True):
        assert began.at - T0 - phase / RATE == pytest.approx(start, abs=0.03)


def test_speech_shorter_than_the_minimum_is_ignored(speech: Samples) -> None:
    start = int(SPEECH[0][0] * RATE)
    burst = np.concatenate(
        [np.zeros(RATE, np.float32), speech[start : start + RATE // 10], np.zeros(RATE, np.float32)]
    )
    assert _feed(StreamingVAD(min_speech_ms=250), burst, 512) == []


def test_ids_keep_counting_after_reset(speech: Samples) -> None:
    vad = StreamingVAD()
    first = _feed(vad, speech, 512)
    vad.reset()
    second = _feed(vad, speech, 512, t0=T0 + 100)
    assert [e.speech_id for e in first + second] == ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"]
    assert second[0].at - T0 - 100 == pytest.approx(SPEECH[0][0], abs=START_TOLERANCE_S)


def test_a_gap_in_frame_times_ends_speech_at_the_gap(speech: Samples) -> None:
    vad = StreamingVAD()
    cut = int(2.0 * RATE)  # mid first phrase
    before = _feed(vad, speech[:cut], 512)
    assert [type(e) for e in before] == [SpeechStarted]
    assert vad.speech_id == "s1"
    after = _feed(vad, speech[cut:], 512, t0=T0 + 30)
    assert isinstance(after[0], SpeechEnded)
    assert after[0].speech_id == "s1"
    assert after[0].at == pytest.approx(T0 + cut / RATE)
    assert vad.speech_id is None
    assert [type(e) for e in after[1:]] == [SpeechStarted, SpeechEnded] * 2
    assert after[1].speech_id == "s2"


def test_rejects_other_sample_rates() -> None:
    with pytest.raises(ValueError, match="16 kHz"):
        StreamingVAD(sample_rate=8000)


def test_long_speech_keeps_little_timing_history(speech: Samples) -> None:
    vad = StreamingVAD()
    start = int(SPEECH[0][0] * RATE)
    phrase = speech[start : int(SPEECH[0][1] * RATE)]
    events = _feed(vad, np.tile(phrase, 4), 80)
    assert [type(e) for e in events] == [SpeechStarted]
    assert len(vad._anchors) < 50  # pyright: ignore[reportPrivateUsage]
