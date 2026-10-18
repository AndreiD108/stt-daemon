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
