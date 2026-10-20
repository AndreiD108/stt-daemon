from stt_daemon.session.markers import PlacedMarker
from stt_daemon.session.transcript import Transcript


def add(
    t: Transcript, text: str, markers: tuple[PlacedMarker, ...] = (), plain: str | None = None
) -> None:
    plain = text if plain is None else plain
    t.add(text, plain=plain, language=None, span=(0, 0), words=None, markers=markers)


def test_final_text_joins_with_one_space_unless_whitespace_is_there() -> None:
    t = Transcript()
    for text in ("Hello there.", "How are", " you?\n", "Fine"):
        add(t, text)
    assert t.final_text == "Hello there. How are you?\nFine"


def test_empty_segments_are_dropped_unless_they_hold_markers() -> None:
    t = Transcript()
    add(t, "one")
    add(t, "  ")
    add(t, "⟦A⟧", (PlacedMarker("⟦A⟧", 10, -1),), plain="")
    assert [s.index for s in t.segments] == [0, 1]
    assert t.final_text == "one ⟦A⟧"
    assert t.plain_text == "one"


def add_tracked(
    t: Transcript, track: str, text: str, start_ms: int, plain: str | None = None
) -> None:
    t.add(
        text,
        plain=text if plain is None else plain,
        language=None,
        span=(start_ms, start_ms + 500),
        words=None,
        markers=(),
        track=track,
    )


def test_tracked_segments_come_one_line_each_in_start_order_with_their_track() -> None:
    t = Transcript()
    add_tracked(t, "system", "welcome everyone", 1000)
    add_tracked(t, "mic", " thanks ", 400)
    add_tracked(t, "system", "  ", 1200)
    add_tracked(t, "mic", "first point", 1000)
    add_tracked(t, "system", "next ⟦A⟧ slide", 2500, plain="next slide")
    assert t.final_text == (
        "[mic] thanks\n[system] welcome everyone\n[mic] first point\n[system] next ⟦A⟧ slide"
    )
    assert t.plain_text == (
        "[mic] thanks\n[system] welcome everyone\n[mic] first point\n[system] next slide"
    )
    assert [s.index for s in t.segments] == [0, 1, 2, 3]
