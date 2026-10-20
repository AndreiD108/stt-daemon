import pytest

from stt_daemon.backends.base import merge_keyterms, trim_previous_text


def test_keyterms_keep_config_first_and_drop_repeats() -> None:
    merged = merge_keyterms(["Scribbly", "Talkey"], ["talkey", "  canvas   doc ", "", "Huion"])
    assert merged == ["Scribbly", "Talkey", "canvas doc", "Huion"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, None),
        ("   ", None),
        ("  short text  ", "short text"),
        ("a" * 50, "a" * 50),
        # The cut lands inside the b's, so the partial word is skipped.
        ("aaa bbbbbb " + "c" * 44, "c" * 44),
        # The cut lands on a word start; nothing is skipped.
        ("xxxx " + "b" * 48 + " c", "b" * 48 + " c"),
        # One long word with no space to advance to: kept as cut.
        ("z" * 80, "z" * 50),
    ],
)
def test_previous_text_keeps_the_tail_from_a_word_start(
    text: str | None, expected: str | None
) -> None:
    assert trim_previous_text(text, 50) == expected
