import numpy as np
import pytest

from stt_daemon.audio.frames import Frame
from stt_daemon.audio.vad import SpeechEnded, SpeechStarted, VadEvent
from stt_daemon.session.pause_gate import Outgoing, PauseGate

RATE = 16000
STEP = 0.02


def frame(end: float) -> Frame:
    return Frame(np.zeros(round(STEP * RATE), np.float32), end, RATE)


def feed(
    gate: PauseGate, start: float, stop: float, events: dict[int, VadEvent] | None = None
) -> list[Outgoing]:
    """20 ms frames ending at start+STEP … stop; `events` keyed by frame number from `start`."""
    out: list[Outgoing] = []
    for i in range(round((stop - start) / STEP)):
        event = (events or {}).get(i)
        out += gate.push(frame(start + (i + 1) * STEP), [event] if event else [])
    return out


def span(out: list[Outgoing]) -> tuple[float, float]:
    return out[0].frame.start, out[-1].frame.end


def test_silence_before_the_first_speech_is_held_back() -> None:
    gate = PauseGate(hold_s=0.4, pad_s=0.1)
    assert feed(gate, 0.0, 3.0) == []


def test_speech_start_sends_held_audio_from_the_pad_before_its_edge() -> None:
    gate = PauseGate(hold_s=0.4, pad_s=0.1)
    # Speech begins at 2.0 s and is confirmed at the frame ending 2.3 s.
    out = feed(gate, 0.0, 2.3, {114: SpeechStarted("s1", 2.0)})
    assert span(out) == pytest.approx((1.9, 2.3))
    assert out[0].cut
    assert not any(o.cut for o in out[1:])
    assert gate.speaking
    more = feed(gate, 2.3, 2.5)
    assert span(more) == pytest.approx((2.3, 2.5))
    assert not any(o.cut for o in more)


def test_held_audio_keeps_at_least_hold_s() -> None:
    gate = PauseGate(hold_s=0.4, pad_s=0.1)
    feed(gate, 0.0, 5.0)
    out = gate.flush()
    start, end = span(out)
    assert end == pytest.approx(5.0)
    assert 0.4 <= end - start <= 0.4 + 2 * STEP


def test_short_pause_stays_whole_and_long_pause_is_cut() -> None:
    gate = PauseGate(hold_s=0.4, pad_s=0.1)
    feed(gate, 0.0, 1.0, {44: SpeechStarted("s1", 0.8)})
    # Pause confirmed at 1.6 s; speech back at 1.65 s, confirmed at 1.9 s: nothing dropped.
    sent = feed(gate, 1.0, 2.0, {29: SpeechEnded("s1", 1.0), 44: SpeechStarted("s2", 1.65)})
    assert span(sent) == pytest.approx((1.0, 2.0))
    assert not any(o.cut for o in sent)
    # Pause confirmed at 2.6 s; speech back at 6.0 s: only the pad before it is sent.
    sent = feed(gate, 2.0, 6.3, {29: SpeechEnded("s2", 2.0), 214: SpeechStarted("s3", 6.0)})
    sent_before_cut = [o for o in sent if o.frame.end <= 2.6 + 1e-9]
    after = sent[len(sent_before_cut) :]
    # The frame that confirmed the pause is already held.
    assert span(sent_before_cut) == pytest.approx((2.0, 2.58))
    assert after[0].cut
    assert span(after) == pytest.approx((5.9, 6.3))


def test_flush_sends_everything_held() -> None:
    gate = PauseGate(hold_s=0.4, pad_s=0.1)
    feed(gate, 0.0, 0.2)
    out = gate.flush()
    assert span(out) == pytest.approx((0.0, 0.2))
    assert gate.flush() == []
