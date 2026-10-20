"""Markers shared by a session's tracks: which track's segment takes each marker."""

from stt_daemon.backends.base import Committed
from stt_daemon.session.markers import MarkerQueue
from stt_daemon.session.segments import Segments
from stt_daemon.session.timeline import Timeline
from stt_daemon.session.transcript import Segment, Transcript

RATE = 16000


def two_tracks() -> tuple[Segments, Segments, MarkerQueue]:
    """Both tracks sent 10 s of audio on connection 1, captured from session ms 0."""
    markers, transcript = MarkerQueue(), Transcript()
    tracks: list[Segments] = []
    for name in ("system", "mic"):
        segments = Segments(
            auto_language=False,
            word_timestamps=True,
            track=name,
            markers=markers,
            transcript=transcript,
        )
        segments.timeline = Timeline(100.0, RATE)
        segments.timeline.sent(1, 100.0, 10 * RATE, cut=False)
        tracks.append(segments)
    system, mic = tracks
    system.peers, mic.peers = [mic], [system]
    return system, mic, markers


def commit(segments: Segments, text: str, end_s: float, *, final: bool = False) -> Segment | None:
    return segments.committed(Committed(1, None, text, None, None, end_s), final)


def test_the_first_segment_with_text_past_the_marker_takes_it() -> None:
    system, mic, markers = two_tracks()
    markers.add("⟦A⟧", 1500)
    assert commit(system, "", 2.0) is None
    seg = commit(mic, "so as I said", 2.0)
    assert seg is not None
    assert seg.track == "mic"
    assert [m.marker for m in seg.markers] == ["⟦A⟧"]


def test_a_segment_without_text_takes_a_marker_once_the_other_track_passed_it() -> None:
    system, mic, markers = two_tracks()
    markers.add("⟦A⟧", 1500)
    markers.add("⟦B⟧", 3500)
    commit(mic, "", 2.0)  # the mic passed 1.5 s with no text there
    seg = commit(system, "", 4.0)
    assert seg is not None
    assert seg.track == "system"
    assert [m.marker for m in seg.markers] == ["⟦A⟧"]
    assert len(markers) == 1


def test_a_final_commit_takes_every_marker_left_only_when_it_is_the_last_track() -> None:
    system, mic, markers = two_tracks()
    markers.add("⟦A⟧", 9000)
    assert commit(system, "", 5.0, final=True) is None
    seg = commit(mic, "", 6.0, final=True)
    assert seg is not None
    assert seg.track == "mic"
    assert [m.marker for m in seg.markers] == ["⟦A⟧"]


def test_one_track_alone_keeps_the_single_track_rule() -> None:
    segments = Segments(auto_language=False, word_timestamps=False)
    segments.timeline = Timeline(100.0, RATE)
    segments.timeline.sent(1, 100.0, 10 * RATE, cut=False)
    segments.markers.add("⟦A⟧", 1500)
    seg = commit(segments, "", 2.0)
    assert seg is not None
    assert seg.track is None
    assert seg.text == "⟦A⟧"
