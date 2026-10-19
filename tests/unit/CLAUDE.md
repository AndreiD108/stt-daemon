# Unit tests

Pure-logic tests for the `stt_daemon` package: local sockets only, never the mic or the internet.

## Scope

Everything under `tests/unit/`. No nested CLAUDE.md.

## File tree

Use this tree as a map — scan it to find the parts your task touches, then go straight there.

<!-- BEGIN FILE TREE -->
```
conftest.py           fixtures: local fake ElevenLabs server, backend pointed at it
fake_elevenlabs.py    that server: records requests and messages, scripted commit replies
test_backend_base.py  keyterm merge, previous_text trim
test_fake_backend.py  reply order, empty commits, delays, partials, failures, time limit
test_elevenlabs_open.py    open retries and budget, HTTP and message refusals, key kept out of logs
test_elevenlabs_stream.py  URL params, chunking, commits, padding, acks, partials, errors, self-commit
test_elevenlabs_batch.py   upload fields, PCM/WAV body, words and speaker split, HTTP codes, retry, cancel
test_frames.py        ring order, since-trimming, eviction, multi-attach, attach atomicity
test_capture.py       device choice, holds, callback, stall/status recovery, backoff, hung stop
test_wav_capture.py   WAV reading, real-time pacing, per-session open
test_pw_capture.py    system capture with a stand-in: frames, lost/restored, backoff, stall, release, following an app
test_sources.py       pw-dump parsing on the recording, ids, scan restarts, following an app's next stream, nodes by name
test_sources_macos.py  macOS scan on a stand-in Core Audio: names, ids, listener, polling, read failures, follow by bundle id, scan per OS
test_audiotee_capture.py  audiotee with a stand-in: command, frames, permission hint, exit error, quiet process kept, new pid followed
test_aec.py           built-in canceller on synthetic echo: echo drop, near end kept, delay search and changes, bypass, CPU; Parakeet (slow)
test_aec_builtin.py   built-in canceller as mic track: late-stamped far frames, far gap and restart, mic jump, settle at stop
test_aec_pipewire.py  echo-cancel module via a pw-cli stand-in: node wait, load errors, kill, plain-mic fallback; mode; install check
test_stream_capture.py  byte stream with a stand-in decoder: order, audio-time stamps, failures, backpressure
test_vad.py           edges on the fixture vs chunk size and window phase, delays, ids, gap reset
test_speech_watch.py  daemon-wide VAD: events, speech starts by id and expiry, mic going quiet
test_timeline.py      runs, cuts, edge snapping, words across a cut, jitter vs gaps, per-connection offsets
test_pause_gate.py    held silence, pre-speech pad, short vs long pauses, flush
test_word_times.py    syllable counts, words over spans, sentence ends, non-Latin split, capture → chunk spans
test_markers.py       word-gap placement, char fallback, empty text, queue
test_transcript.py    final-text joining, empty segments, tracks as lines in start order
test_track_segments.py  markers shared by two tracks: which segment takes each
test_session_audio.py   fixture sessions: word times through a cut, several commits, chunk cap, stop/cancel
test_session_lifecycle.py  silent sessions: leftover markers, marker refusals, failure, failed open
test_upload_session.py  upload session: segments per speaker, markers need at, failures, cancel, input cap
test_recovery.py      link alone: replay tail and pacing, retries, give-up, idle drop, hung send, rollover
test_recovery_edges.py  close-time ack, unknown provider cuts, final-ack failure, in-flight commit/stop
test_session_tracks.py  system sessions: per-track reconnect, capture loss, fatal error on one track, mic held before the system capture
test_session_recovery.py  fixture sessions: word times, markers, give-up, bounded stop during recovery
test_config.py        paths per OS, defaults, type errors, Parakeet limits, config writer and key removal
test_credentials.py   mode/owner checks, precedence, writer, stdin parsing, no key in errors
test_slot.py          slot states, sticky evictable, takeover order and failures, client_gone
slot_harness.py       Slot test rig: recorded events and slot changes, session makers, test clients
test_slot_priority.py  dictation priority: victim choice, resume order, backend gone, concurrency refusal retry
test_peer.py          peer pid/cmdline over real Unix sockets; macOS parsing on fixture bytes
test_messages.py      envelope parsing, encoding, per-command field checks
test_backend_set.py   setup and default, language default and checks, features, hints, context lookup
test_cli.py           run (broken config, --keys-stdin), key set and backend install default rule, set-default, backend remove, service files
test_transcribe_client.py  `transcribe` CLI vs a scripted daemon: file bytes, text and JSON output, exit codes
test_plugins.py       plugin loader and failures, statuses, config and limits, adapter: chunks, s16le, turns, timeout, options; docs example
test_cli_plugins.py   backend list, key set / install / set-default by plugin name
test_parakeet.py      token → word grouping, per-commit decoding, failures, model check
test_echo_gate.py     echo frames: delayed / early-stamped echo, near speech over it, per-span match; gate thresholds, joints, word times back
test_playback.py      playback clock, lookahead wait, reference spans; SpeakerMonitor holds vs want, open race, missing tool; replay keeps reference
test_session_playback.py  mic sessions with playback: reference per commit, none for other backends, quiet stretch trimmed, speakers released
test_tdt.py           keyterm BPE split; trie scoring: whole, broken, restarted, open at the end, nested
test_parakeet_engine.py  load at open, one recognizer for all, decodes in turn, idle unload, load failure, reload
test_install.py       unpack once, partial model replaced, bad download / archive leaves nothing
test_service.py       unit/plist rendering, ExecStart escaping, systemctl calls, remove guard, PortAudio, ffmpeg and audiotee hints
```
<!-- END FILE TREE -->
