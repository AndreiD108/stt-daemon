import pytest

from stt_daemon.session.timeline import Timeline

RATE = 16000


def test_contiguous_sends_map_linearly() -> None:
    tl = Timeline(origin=100.0)
    tl.sent(1, 100.5, RATE, cut=False)
    tl.sent(1, 101.5, RATE // 2, cut=False)
    assert tl.to_capture(1, 0.0, "start") == pytest.approx(100.5)
    assert tl.to_capture(1, 1.25, "start") == pytest.approx(101.75)
    assert tl.to_ms(tl.to_capture(1, 1.25, "start")) == 1750


def test_cut_pause_maps_both_sides_back_to_capture_time() -> None:
    tl = Timeline(origin=10.0)
    tl.sent(1, 10.0, RATE, cut=False)  # sent 0-1 s ← captured 10-11 s
    tl.sent(1, 13.0, RATE // 2, cut=True)  # sent 1-1.5 s ← captured 13-13.5 s
    tl.sent(1, 20.0, RATE, cut=True)  # sent 1.5-2.5 s ← captured 20-21 s
    assert tl.to_capture(1, 0.4, "start") == pytest.approx(10.4)
    assert tl.to_capture(1, 1.2, "start") == pytest.approx(13.2)
    assert tl.to_capture(1, 2.0, "end") == pytest.approx(20.5)


def test_times_at_a_cut_snap_to_the_side_of_the_word() -> None:
    tl = Timeline(origin=10.0)
    tl.sent(1, 10.0, RATE, cut=False)
    tl.sent(1, 13.0, RATE, cut=True)
    # A word starting at the cut begins after the held-back pause; one ending there ends before.
    assert tl.to_capture(1, 1.0, "start") == pytest.approx(13.0)
    assert tl.to_capture(1, 1.0, "end") == pytest.approx(11.0)
    assert tl.to_capture(1, 0.0, "end") == pytest.approx(10.0)
    assert tl.to_capture(1, 2.0, "start") == pytest.approx(14.0)


def test_a_word_across_a_cut_goes_whole_into_the_side_holding_more_of_it() -> None:
    tl = Timeline(origin=10.0)
    tl.sent(1, 10.0, RATE, cut=False)  # sent 0-1 s ← captured 10-11 s
    tl.sent(1, 13.0, RATE, cut=True)  # sent 1-2 s ← captured 13-14 s
    assert tl.word_span(1, 0.7, 1.5) == (pytest.approx(13.0), pytest.approx(13.5))
    assert tl.word_span(1, 0.5, 1.1) == (pytest.approx(10.5), pytest.approx(11.0))
    assert tl.word_span(1, 0.2, 0.4) == (pytest.approx(10.2), pytest.approx(10.4))


def test_small_jitter_continues_a_run_but_a_capture_gap_starts_one() -> None:
    tl = Timeline(origin=0.0)
    tl.sent(1, 0.0, RATE, cut=False)
    tl.sent(1, 1.003, RATE, cut=False)  # callback jitter: the sample clock is kept
    tl.sent(1, 2.5, RATE, cut=False)  # stream reopened half a second later
    assert tl.to_capture(1, 1.5, "start") == pytest.approx(1.5)
    assert tl.to_capture(1, 2.5, "start") == pytest.approx(3.0)


def test_each_connection_counts_from_zero() -> None:
    tl = Timeline(origin=0.0)
    tl.sent(1, 0.0, 2 * RATE, cut=False)
    tl.sent(2, 2.0, RATE, cut=False)
    assert tl.sent_samples(1) == 2 * RATE
    assert tl.sent_samples(2) == RATE
    assert tl.to_capture(2, 0.5, "start") == pytest.approx(2.5)


def test_times_past_the_audio_sent_clamp_and_ms_never_go_negative() -> None:
    tl = Timeline(origin=5.0)
    tl.sent(1, 4.0, RATE, cut=False)  # pre-start audio from before origin can't happen, but clamp
    assert tl.to_capture(1, 9.0, "end") == pytest.approx(5.0)
    assert tl.to_capture(1, -1.0, "start") == pytest.approx(4.0)
    assert tl.to_ms(4.0) == 0
    assert tl.to_capture(3, 1.0, "start") == 5.0
