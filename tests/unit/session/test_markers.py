from stt_daemon.session.markers import MarkerQueue, PendingMarker, TimedWord, place

WORDS = (
    TimedWord("move", 1000, 1200),
    TimedWord("this,", 1300, 1500),
    TimedWord("up", 1800, 1900),
)
TEXT = "move this, up"


def m(at_ms: int, marker: str = "⟦A⟧") -> PendingMarker:
    return PendingMarker(marker, at_ms)


def test_marker_goes_into_the_nearest_word_gap() -> None:
    text, placed = place(TEXT, WORDS, (900, 2000), [m(1650)])
    assert text == "move this, ⟦A⟧ up"
    assert placed[0].after_word == 1


def test_marker_inside_a_word_goes_to_its_nearer_side() -> None:
    assert place(TEXT, WORDS, (900, 2000), [m(1050)])[0] == "⟦A⟧ move this, up"
    assert place(TEXT, WORDS, (900, 2000), [m(1150)])[0] == "move ⟦A⟧ this, up"


def test_marker_before_the_first_word_goes_first_and_keeps_its_time() -> None:
    text, placed = place(TEXT, WORDS, (900, 2000), [m(500)])
    assert text == "⟦A⟧ move this, up"
    assert (placed[0].after_word, placed[0].at_ms) == (-1, 500)


def test_marker_after_the_last_word_goes_last() -> None:
    assert place(TEXT, WORDS, (900, 2000), [m(5000)])[0] == "move this, up ⟦A⟧"


def test_markers_in_one_gap_keep_their_order() -> None:
    text, _ = place(TEXT, WORDS, (900, 2000), [m(1600, "⟦A⟧"), m(1700, "⟦B⟧")])
    assert text == "move this, ⟦A⟧ ⟦B⟧ up"


def test_words_missing_from_the_text_fall_back_to_token_order() -> None:
    words = (TimedWord("mov", 1000, 1200), TimedWord("xx", 1300, 1500), TimedWord("up", 1800, 1900))
    assert place(TEXT, words, (900, 2000), [m(1650)])[0] == "move this, ⟦A⟧ up"


def test_without_word_times_the_span_spreads_over_the_characters() -> None:
    # "aaaa bbbb cccc": token midpoints at 2/14, 7/14, 12/14 of the 1400 ms span.
    text, placed = place("aaaa bbbb cccc", None, (0, 1400), [m(600)])
    assert text == "aaaa ⟦A⟧ bbbb cccc"
    assert placed[0].after_word == 0
    assert place("aaaa bbbb cccc", None, (0, 1400), [m(100)])[0] == "⟦A⟧ aaaa bbbb cccc"


def test_markers_in_empty_text() -> None:
    text, placed = place("", (), (3000, 3000), [m(100, "⟦A⟧"), m(200, "⟦B⟧")])
    assert text == "⟦A⟧ ⟦B⟧"
    assert [p.after_word for p in placed] == [-1, -1]


def test_queue_takes_markers_up_to_a_time_then_all() -> None:
    queue = MarkerQueue()
    queue.add("⟦A⟧", 100)
    queue.add("⟦B⟧", 900)
    queue.add("⟦C⟧", 300)
    assert [p.marker for p in queue.take(500)] == ["⟦A⟧", "⟦C⟧"]
    assert len(queue) == 1
    assert [p.marker for p in queue.take()] == ["⟦B⟧"]
    assert len(queue) == 0
