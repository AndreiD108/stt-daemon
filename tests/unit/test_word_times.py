from dataclasses import replace

import numpy as np
import pytest

from stt_daemon.audio.frames import Frame
from stt_daemon.backends.base import Committed, Word
from stt_daemon.session.pause_gate import SpeechLog
from stt_daemon.session.word_times import (
    TAIL_PAD_S,
    estimate,
    speech_spans,
    syllables,
    with_times,
)

RATE = 16000


def tone(seconds: float, level: float = 0.1) -> np.ndarray:
    return np.full(round(seconds * RATE), level, np.float32)


def flat(spans: list[tuple[float, float]]) -> list[float]:
    return [t for span in spans for t in span]


PHRASE_1: str = "The quick brown fox jumps over the lazy dog."
PHRASE_2: str = "Please send the report before noon tomorrow."


@pytest.mark.parametrize(
    ("word", "count"),
    [
        ("the", 1),
        ("table", 2),
        ("idea", 3),
        ("PipeWire", 2),
        ("VAD", 3),
        ("make", 1),
        ("little", 2),
        ("area", 3),
        ("tea", 1),
        ("daemon", 2),
        ("Parakeet", 3),
        ("tomorrow.", 3),
        ("I", 1),
        ("don't", 1),
        ("WWW", 9),
        ("42", 2),
        ("café", 2),
    ],
)
def test_syllables(word: str, count: int) -> None:
    assert syllables(word) == count


def test_a_span_is_split_by_syllables() -> None:
    # 1 + 2 + 3 syllables over 1.2 s
    times = estimate(["the", "table", "tomorrow"], tone(2.0), [(0.5, 1.7)], RATE)
    assert flat(times) == pytest.approx([0.5, 0.7, 0.7, 1.1, 1.1, 1.7])


def test_no_word_straddles_a_pause() -> None:
    words = f"{PHRASE_1} {PHRASE_2}".split()
    # The second phrase is spoken faster: by duration share alone it would take fewer words.
    times = estimate(words, tone(6.0), [(0.1, 2.84), (3.54, 5.61)], RATE)
    first = times[: len(PHRASE_1.split())]
    second = times[len(PHRASE_1.split()) :]
    assert first[0][0] == pytest.approx(0.1)
    assert first[-1][1] == pytest.approx(2.84)
    assert second[0][0] == pytest.approx(3.54)
    assert second[-1][1] == pytest.approx(5.61)


def test_without_punctuation_words_follow_the_spans_durations() -> None:
    # Ten one-syllable words over spans of 1 s and 4 s: two and eight.
    times = estimate(["word"] * 10, tone(6.0), [(0.0, 1.0), (2.0, 6.0)], RATE)
    assert [s for s, _ in times if s < 1.0] == pytest.approx([0.0, 0.5])
    assert times[2][0] == pytest.approx(2.0)


def test_times_are_in_order_and_never_overlap() -> None:
    words = ["a", "b", "c", "d", "e", "f", "g", "h"]
    times = estimate(words, tone(10.0), [(0.3, 1.0), (2.0, 2.2), (5.0, 9.0)], RATE)
    assert flat(times) == sorted(flat(times))


def test_a_span_ends_where_its_audio_goes_quiet() -> None:
    # Loud until 1.2 s, then 40 dB down: the VAD's late end at 1.7 s is moved back.
    audio = np.concatenate([tone(1.2), tone(0.8, 0.001)])
    times = estimate(["one", "two"], audio, [(0.5, 1.7)], RATE)
    assert times[-1][1] == pytest.approx(1.2 + TAIL_PAD_S, abs=0.011)


def test_text_without_latin_vowels_is_split_by_characters() -> None:
    times = estimate(["привет", "мир"], tone(1.0), [(0.0, 0.9)], RATE)
    assert flat(times) == pytest.approx([0.0, 0.6, 0.6, 0.9])


def test_without_speech_spans_the_whole_audio_counts() -> None:
    assert flat(estimate(["one", "two"], tone(2.0), [], RATE)) == pytest.approx(
        [0.0, 1.0, 1.0, 2.0]
    )


def test_speech_log_keeps_open_speech_and_forgets_old_spans() -> None:
    log = SpeechLog()
    log.started(1.0)
    log.ended(2.0)
    log.started(3.0)
    assert log.between(1.5, 4.0) == [(1.5, 2.0), (3.0, 4.0)]
    log.forget_before(2.5)
    assert log.between(0.0, 4.0) == [(3.0, 4.0)]


def test_speech_spans_map_capture_time_to_chunk_seconds_across_a_cut() -> None:
    log = SpeechLog()
    log.started(10.1)
    log.ended(10.9)
    log.started(20.05)
    # Two 0.5 s frames back to back, a cut, then one 0.5 s frame.
    pieces = [(10.0, 8000), (10.5, 8000), (20.0, 8000)]
    spans = speech_spans(pieces, log.between, 16000)
    assert flat(spans) == pytest.approx([0.1, 0.9, 1.05, 1.5])


def committed(text: str, words: tuple[Word, ...] | None) -> Committed:
    return Committed(1, 1, text, None, words, 3.0)


def test_estimated_times_count_from_the_connection_start() -> None:
    log = SpeechLog()
    log.started(10.0)
    log.ended(11.0)
    frames = [Frame(tone(1.0), 11.0, RATE)]
    event = with_times(committed("one two", None), frames, log.between, 32000)
    assert event.words == (Word("one", 2.0, 2.5), Word("two", 2.5, 3.0))


def test_estimated_words_keep_the_speakers_given_with_the_text() -> None:
    log = SpeechLog()
    log.started(0.0)
    log.ended(1.0)
    frames = [Frame(tone(1.0), 1.0, RATE)]
    event = replace(committed("one two", None), speakers=("speaker_0", "speaker_1"))
    words = with_times(event, frames, log.between, 0).words
    assert words is not None
    assert [w.speaker for w in words] == ["speaker_0", "speaker_1"]


def test_text_with_an_empty_word_list_gets_times() -> None:
    log = SpeechLog()
    log.started(0.0)
    log.ended(1.0)
    frames = [Frame(tone(1.0), 1.0, RATE)]
    event = with_times(committed("hello", ()), frames, log.between, 0)
    assert event.words == (Word("hello", 0.0, 1.0),)


def test_silent_audio_still_gets_ordered_times() -> None:
    times = estimate(["one", "two"], tone(1.0, 0.0), [(0.2, 0.8)], RATE)
    assert flat(times) == pytest.approx([0.2, 0.5, 0.5, 0.8])


def test_answers_with_word_times_or_no_text_are_left_alone() -> None:
    frames = [Frame(tone(1.0), 1.0, RATE)]
    timed = committed("one", (Word("one", 0.1, 0.2),))
    assert with_times(timed, frames, SpeechLog().between, 0) is timed
    blank = committed(" ", None)
    assert with_times(blank, frames, SpeechLog().between, 0) is blank
