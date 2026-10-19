"""A session's committed segments and the final text built from them."""

from collections.abc import Iterable
from dataclasses import dataclass

from .markers import PlacedMarker, TimedWord


@dataclass(frozen=True)
class Segment:
    index: int
    text: str
    """With markers inline."""
    language: str | None
    start_ms: int
    end_ms: int
    words: tuple[TimedWord, ...] | None
    """None when the client didn't ask for word times; then `after_word` isn't sent either."""
    markers: tuple[PlacedMarker, ...]
    speaker: str | None = None
    """Set when the backend named one speaker for all its words."""
    track: str | None = None


class Transcript:
    """One session's segments, from all its tracks."""

    def __init__(self) -> None:
        self.segments: list[Segment] = []
        self._plain: list[str] = []

    def add(
        self,
        text: str,
        *,
        plain: str,
        language: str | None,
        span: tuple[int, int],
        words: tuple[TimedWord, ...] | None,
        markers: tuple[PlacedMarker, ...],
        speaker: str | None = None,
        track: str | None = None,
    ) -> Segment | None:
        """The new segment; None when it has neither text nor markers. `plain` is `text`
        without markers."""
        if not text.strip() and not markers:
            return None
        segment = Segment(len(self.segments), text, language, *span, words, markers, speaker, track)
        self.segments.append(segment)
        self._plain.append(plain)
        return segment

    @property
    def final_text(self) -> str:
        """Segment texts joined by one space, unless one side already has whitespace; with
        tracks, one `[track] text` line per segment in start-time order."""
        return self._text([segment.text for segment in self.segments])

    @property
    def plain_text(self) -> str:
        """`final_text` without markers."""
        return self._text(self._plain)

    def _text(self, texts: list[str]) -> str:
        if all(segment.track is None for segment in self.segments):
            return _join(texts)
        order = sorted(range(len(texts)), key=lambda i: self.segments[i].start_ms)
        return "\n".join(
            f"[{self.segments[i].track}] {texts[i].strip()}" for i in order if texts[i].strip()
        )


def _join(texts: Iterable[str]) -> str:
    out = ""
    for text in texts:
        if not text.strip():
            continue
        if out and not out[-1].isspace() and not text[0].isspace():
            out += " "
        out += text
    return out
