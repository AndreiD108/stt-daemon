# Session unit tests

Unit tests for `stt_daemon/session/`: timeline, pause cutting, commits, markers and transcripts, recovery, tracks, playback reference, upload sessions. See `../CLAUDE.md`.

## Scope

Everything under `tests/unit/session/`. No nested CLAUDE.md.

## File tree

Use this tree as a map — scan it to find the parts your task touches, then go straight there.

<!-- BEGIN FILE TREE -->
```
__init__.py                 package marker
test_markers.py             word-gap placement, char fallback, empty text, queue
test_pause_gate.py          held silence, pre-speech pad, short vs long pauses, flush
test_recovery.py            link alone: replay tail and pacing, retries, give-up, idle drop, hung send, rollover
test_recovery_edges.py      close-time ack, unknown provider cuts, final-ack failure, in-flight commit/stop, replay keeps the reference
test_reference.py           PlaybackHistory: jitter vs jumps, forget, old playback dropped, cover waits and gives up; reference spans
test_session_audio.py       fixture sessions: word times through a cut, several commits, chunk cap, stop/cancel
test_session_lifecycle.py   silent sessions: leftover markers, marker refusals, failure, failed open
test_session_playback.py    mic sessions with playback: reference per commit, none for other backends, quiet stretch, speakers released
test_session_recovery.py    fixture sessions: word times, markers, give-up, bounded stop during recovery
test_session_tracks.py      system sessions: per-track reconnect, capture loss, fatal error on one track, mic held before the system capture
test_timeline.py            runs, cuts, edge snapping, words across a cut, jitter vs gaps, per-connection offsets
test_track_segments.py      markers shared by two tracks: which segment takes each
test_transcript.py          final-text joining, empty segments, tracks as lines in start order
test_upload_session.py      upload session: segments per speaker, markers need at, failures, cancel, input cap
test_word_times.py          syllable counts, words over spans, sentence ends, non-Latin split, capture → chunk spans
```
<!-- END FILE TREE -->
