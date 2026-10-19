"""Backend answers → segments: session times through the timeline, markers placed in the text.

A session with several tracks has one `Segments` per track, all sharing the marker queue and
the transcript. A marker goes to the first segment, of any track, that has text and reaches its
time; a segment without text takes it only once every other track has passed that time too.
"""

from ..backends.base import Committed, Word
from .markers import MarkerQueue, PendingMarker, TimedWord, place
from .timeline import Timeline
from .transcript import Segment, Transcript


class Segments:
    def __init__(
        self,
        *,
        auto_language: bool,
        word_timestamps: bool,
        track: str | None = None,
        markers: MarkerQueue | None = None,
        transcript: Transcript | None = None,
    ) -> None:
        self.timeline = Timeline(0.0, 16000)
        self.markers = MarkerQueue() if markers is None else markers
        self.transcript = Transcript() if transcript is None else transcript
        self.track = track
        self.peers: list[Segments] = []
        """The session's other tracks."""
        self.reached_ms = 0
        """End of the audio this track has committed, in session ms."""
        self.finished = False
        """Its final commit was answered."""
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
        self.reached_ms = max(self.reached_ms, end_ms)
        self.finished = self.finished or final
        words: tuple[TimedWord, ...] | None = None
        speaker = None
        if event.words is not None:
            words = tuple(self._timed(conn, w) for w in event.words)
            speakers = {w.speaker for w in event.words}
            if len(speakers) == 1:
                speaker = speakers.pop()
        markers = self._take_markers(end_ms, final, has_text=bool(event.text.strip()))
        text, placed = place(event.text, words, (start_ms, end_ms), markers)
        return self.transcript.add(
            text,
            plain=event.text,
            language=event.language if self._auto_language else None,
            span=(start_ms, end_ms),
            words=words if self._word_timestamps else None,
            markers=tuple(placed),
            speaker=speaker,
            track=self.track,
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
            track=self.track,
        )

    def _take_markers(self, end_ms: int, final: bool, *, has_text: bool) -> list[PendingMarker]:
        if not self.peers:
            return self.markers.take(None if final else end_ms)
        behind = [p.reached_ms for p in self.peers if not p.finished]
        if final and not behind:
            return self.markers.take()
        if has_text:
            return self.markers.take(end_ms)
        return self.markers.take(end_ms, also=lambda m: all(r >= m.at_ms for r in behind))

    def _timed(self, conn: int, word: Word) -> TimedWord:
        tl = self.timeline
        start = tl.to_ms(tl.to_capture(conn, word.start_s, "start"))
        end = tl.to_ms(tl.to_capture(conn, word.end_s, "end"))
        return TimedWord(word.text, start, max(end, start), word.speaker)
