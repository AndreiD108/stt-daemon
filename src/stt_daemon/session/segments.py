"""Backend answers → segments: session times through the timeline, markers placed in the text."""

from ..backends.base import Committed, Word
from .markers import MarkerQueue, TimedWord, place
from .timeline import Timeline
from .transcript import Segment, Transcript


class Segments:
    def __init__(self, *, auto_language: bool, word_timestamps: bool) -> None:
        self.timeline = Timeline(0.0, 16000)
        self.markers = MarkerQueue()
        self.transcript = Transcript()
        self._auto_language = auto_language
        self._word_timestamps = word_timestamps
        self._boundary_s: dict[int, float] = {}

    def committed(self, event: Committed, final: bool) -> Segment | None:
        """`final` places every marker left; else only those up to the segment's end."""
        tl, conn = self.timeline, event.connection_id
        boundary = self._boundary_s.get(conn, 0.0)
        end_s = max(event.audio_end_s, boundary)
        self._boundary_s[conn] = end_s
        start_ms = tl.to_ms(tl.to_capture(conn, boundary, "start"))
        end_ms = max(tl.to_ms(tl.to_capture(conn, end_s, "end")), start_ms)
        words: tuple[TimedWord, ...] | None = None
        speaker = None
        if event.words is not None:
            words = tuple(self._timed(conn, w) for w in event.words)
            speakers = {w.speaker for w in event.words}
            if len(speakers) == 1:
                speaker = speakers.pop()
        markers = self.markers.take(None if final else end_ms)
        text, placed = place(event.text, words, (start_ms, end_ms), markers)
        return self.transcript.add(
            text,
            plain=event.text,
            language=event.language if self._auto_language else None,
            span=(start_ms, end_ms),
            words=words if self._word_timestamps else None,
            markers=tuple(placed),
            speaker=speaker,
        )

    def leftover(self, at: float) -> Segment | None:
        """Markers no answer covered, in an empty segment at `at` (monotonic)."""
        if not self.markers:
            return None
        at_ms = self.timeline.to_ms(at)
        text, placed = place("", (), (at_ms, at_ms), self.markers.take())
        return self.transcript.add(
            text,
            plain="",
            language=None,
            span=(at_ms, at_ms),
            words=() if self._word_timestamps else None,
            markers=tuple(placed),
        )

    def _timed(self, conn: int, word: Word) -> TimedWord:
        tl = self.timeline
        start = tl.to_ms(tl.to_capture(conn, word.start_s, "start"))
        end = tl.to_ms(tl.to_capture(conn, word.end_s, "end"))
        return TimedWord(word.text, start, max(end, start), word.speaker)
