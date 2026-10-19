# stt-daemon

A local speech-to-text daemon in Python: one process owns the mic, many clients connect over a Unix
socket, and one session runs at a time. The wire protocol is in `docs/protocol.md`; code against it.

## Scope

Overview of the whole repository. No nested CLAUDE.md.

## File tree

Use this tree as a map — scan it to find the parts your task touches, then go straight there.

<!-- BEGIN FILE TREE -->
```
.github/                  CI
  workflows/
    ci.yml                ruff, pyright, pytest (unit + integration) on push and PR
docs/                     agent- and client-facing docs
  protocol.md             wire protocol: every command and event
  media/                  README GIFs: hero (listen), client (socket lines + app view), backends (cloud vs local)
src/                      package source
  stt_daemon/             the `stt-daemon` package
    __init__.py           __version__ from package metadata
    audio/                mic capture, recent-audio ring, VAD
      __init__.py         empty package marker
      frames.py           Frame, ring + attachments (FrameHub); one lock so attach sees each frame once
      capture.py          CaptureSource base (keep-open vs per-session holds), MicCapture: PortAudio (loaded on first use), watchdog
      wav_capture.py      WavCapture: WAV files in real time, silence between; mic stand-in for tests
      stream_capture.py   ByteStreamCapture: client bytes via ffmpeg (or raw PCM) → frames on audio time, bounded queue
      pw_capture.py       PwRecordCapture: system audio via pw-record (default output's monitor or one app's stream); restarts, re-attach
      aec.py              EchoCanceller: NumPy AEC per 10 ms block; GCC-PHAT delay, PBFDAF filter pair, double-talk guard, residual gain
      aec_builtin.py      BuiltinEchoCancelledMic: mic track through EchoCanceller, system capture as reference, paired by capture time
      aec_pipewire.py     PipewireAec: echo-cancel module per session via pw-cli; EchoCancelledMic: mic track on its source, plain-mic fallback
      vad.py              StreamingVAD: sherpa-onnx silero per window, own debounce, low-threshold edge lookback
      silero_vad.onnx     bundled silero VAD model (from sherpa-onnx releases)
      silero_vad.LICENSE  its MIT license
    backends/             transcription backends behind one interface; BACKENDS registry in __init__
      __init__.py         BACKENDS: name → backend class
      base.py             the contract: Backend, Connection, FileBackend (whole files), events, Capabilities; keyterm merge, previous_text trim
      elevenlabs.py       Scribe v2 Realtime: retrying open, padded short commits, in-order acks, pad-free word times
      elevenlabs_wire.py  EL language codes (639-3 → 639-1), error message → code, HTTP status → code with EL text
      elevenlabs_batch.py  file upload API: transcribe only, bytes as sent, one retry, 20–30 s speaker runs → commits
      fake.py             FakeBackend for tests: scripted replies, partials, delays, failures, warnings, time limit
      parakeet.py         local Parakeet TDT on sherpa-onnx: shared recognizer (load at open, decodes in turn, idle unload), token → word times
      install.py          `backend install parakeet`: download, SHA-256 check, unpack, bpe.vocab from tokens.txt
    server/               the socket side: connections, commands, mic and backend slots, event routing
      __init__.py         empty package marker
      socket.py           Server: hello gate, per-connection command order, transcribe/audio, routing, reload, shutdown
      connection.py       Connection (send, slow-reader close), line reading, socket file prep, SocketInUse
      messages.py         JSONL envelope; per-command field checks → bad_request / invalid naming the field
      wire.py             session and VAD events → protocol event data (words, after_word only when asked)
      backend_set.py      Setup: backends set up here, default; resolve a start (language, features, hints)
      starts.py           Starts mixin: start_session, transcribe, start_system_session → sessions; start errors as replies
      slot.py             Slot: mic slot idle/busy/evictable, takeover order, backend-only (transcribe) starts, slot changes
      slot_state.py       Client, Holder, SlotState (wire form), StartRefused (busy / session_active details)
      backend_slots.py    BackendSlots: per-backend max_sessions, the sessions holding them, suspended ones
      speech.py           SpeechWatch: daemon-wide VAD on all mic audio; VAD events, speech starts by id
      sources.py          SourceScan: pw-dump -m output → playing apps (sources, ids, events), nodes by name; Follow: one app's next stream
      peer.py             socket peer's pid and cmdline: SO_PEERCRED + /proc; macOS LOCAL_PEERPID + KERN_PROCARGS2
    session/              one session: capture → VAD → pause cutting → backend → segments
      __init__.py         empty package marker
      session.py          Session: one or two tracks; start, stop (bounded) / cancel / error end, suspend / resume, capture loss
      tracks.py           Track: one source's VAD → pause gate → link, commit cadence (wall or audio time), audio loop
      upload.py           UploadSession: transcribe on a file backend; bytes kept, one upload at stop, no progress
      options.py          SessionOptions, MarkerRefused, start_point (where a started session takes its audio from)
      recovery.py         Link: unanswered-audio log, in-order sends, rollover, reconnect, paced replay, suspend (int16 log) / resume
      segments.py         backend answers → segments: session times via timeline, markers placed, leftover markers
      moving.py           while_moving: wait bounded by no progress, paused time not counted
      events.py           what a session emits: started, partial, error, ended (+ evicted_by); segments from transcript
      pause_gate.py       PauseGate: sends speech and short pauses, holds long ones back to a pre-speech pad
      timeline.py         sent audio per connection ↔ capture time; backend seconds → session ms
      markers.py          pending markers; placement in word gaps or by character length
      transcript.py       Segment, committed segments, final text (with and without markers)
      context.py          previous_text store by (client label, context_key), in memory
    service/              user service install: systemd unit + limits drop-in, launchd plist; PortAudio check
      __init__.py         empty package marker
      common.py           Runner (injectable for tests), ServiceError, the installed stt-daemon script's path
      systemd.py          unit and limits.conf rendering, ExecStart escaping; install/remove/status/logs
      launchd.py          macOS agent plist (TMPDIR kept for the socket); launchctl calls
      preflight.py        PortAudio load check, tools on PATH (ffmpeg, pw-record, pw-dump, pw-cli), echo-cancel module files; install command per distro
    cli.py                `stt-daemon` entry point: run, service, key set, backend set-default/install, listen, transcribe
    control.py            one blocking request to the running daemon (hello, command, reply); `reload` after key set
    listen.py             `listen`: session client printing events; terminal keys (Enter, marker key, Ctrl-C)
    transcribe_client.py  `transcribe`: streams files as audio chunks, one session each; text or JSON out
    daemon.py             `run`: config, credentials, mic, server; SIGTERM/SIGINT shutdown
    config.py             config.toml loading, defaults, validation; config/socket paths per OS; in-place key writer
    credentials.py        API keys: mode-600 credentials.toml, <BACKEND>_API_KEY env, stdin keys; key writer
tests/
  aec_fixture.py          synthetic echo fixture: TTS far end through a synthetic room into TTS near end; writes WAVs
  conftest.py             marks tests/live as `live` (skipped by default); a temp XDG_DATA_HOME per test; `--dump-dir`
  fake_pw_record.py       pw-record stand-in: WAV then silence in real time; exits, stalls, start log, exits when its target goes
  fake_pw_dump.py         pw-dump -m stand-in, and FakePipeWire: the test side that plays and stops streams, adds sources
  fake_pw_cli.py          pw-cli load-module stand-in: announces the echo-cancel source in a FakePipeWire; load error, exit, ignored TERM
  fake_elevenlabs_batch.py  local stand-in for the EL upload API: records multipart fields, scripted answers, hold
  fixtures/
    two_phrases.wav       TTS speech, 16 kHz: phrases at 1.00–3.74 s and 5.74–7.81 s
    jargon.wav            TTS speech, 16 kHz: three phrases with jargon words, 2 s pauses
    pw_dump_monitor.json  recorded pw-dump -m -N (props trimmed): registry, a stream playing and ending, id reuse
  unit/                   pure logic; local sockets only, never the mic or the internet
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
    test_aec.py           built-in canceller on synthetic echo: echo drop, near-end kept, delay estimate and change over the search range, bypass, CPU; Parakeet (slow)
    test_aec_builtin.py   built-in canceller as mic track: late-stamped far frames, far gap and restart, mic jump, settle at stop
    test_aec_pipewire.py  echo-cancel module via a pw-cli stand-in: node wait, load errors, kill, plain-mic fallback; mode; install check
    test_stream_capture.py  byte stream with a stand-in decoder: order, audio-time stamps, failures, backpressure
    test_vad.py           edges on the fixture vs chunk size and window phase, delays, ids, gap reset
    test_speech_watch.py  daemon-wide VAD: events, speech starts by id and expiry, mic going quiet
    test_timeline.py      runs, cuts, edge snapping, jitter vs gaps, per-connection offsets
    test_pause_gate.py    held silence, pre-speech pad, short vs long pauses, flush
    test_markers.py       word-gap placement, char fallback, empty text, queue
    test_transcript.py    final-text joining, empty segments, tracks as lines in start order
    test_track_segments.py  markers shared by two tracks: which segment takes each
    test_session_audio.py   fixture sessions: word times through a cut, several commits, chunk cap, stop/cancel
    test_session_lifecycle.py  silent sessions: leftover markers, marker refusals, failure, failed open
    test_upload_session.py  upload session: segments per speaker, markers need at, failures, cancel, input cap
    test_recovery.py      link alone: replay tail and pacing, retries, give-up, idle drop, hung send, rollover
    test_recovery_edges.py  close-time ack, unknown provider cuts, final-ack failure, in-flight commit/stop
    test_session_tracks.py  system sessions: per-track reconnect, capture loss, fatal error on one track
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
    test_parakeet.py      token → word grouping, hotwords, per-commit decoding, failures, model check
    test_parakeet_engine.py  load at open, one recognizer for all, decodes in turn, idle unload, load failure, reload
    test_install.py       bpe.vocab, unpack once, partial model replaced, bad download / archive leaves nothing
    test_service.py       unit/plist rendering, ExecStart escaping, systemctl calls, remove guard, PortAudio and ffmpeg hints
  integration/            real daemon over a socket, WAV capture, fake backend
    conftest.py           `daemon` fixture
    harness.py            in-process Server on a temp socket; test clients; fixture helpers
    test_daemon_audio.py      word times through a cut, several commits, rollover and replay
    test_daemon_clients.py    hello gate, welcome, errors, busy and eviction, observers, disconnects, reload
    test_daemon_lifecycle.py  socket file, shutdown, context store, marker times
    test_daemon_speech.py     VAD event routing, speech_id starts (edge, expiry, mic closed, eviction)
    test_daemon_transcribe.py transcribe: ffmpeg and raw PCM, file-time word times, slots beside mic, decoder errors
    test_daemon_batch.py      elevenlabs-batch over the socket: default rule, speakers, refusals, markers, disconnect
    test_daemon_priority.py   transcribe suspended by a dictation: same output as unsuspended; blocked writer stalls no one
    test_daemon_system.py     system sessions: two tracks interleaved, dictation alongside, suspend/resume, fatal track, tool missing
    test_daemon_sources.py    sources in welcome and events (opt-in), app sessions, unknown app, re-attach (also while suspended)
    test_daemon_aec.py        mic track on the echo-cancelled source; no module → plain mic, one aec_unavailable; pw-cli ending mid-session; built-in canceller keeps echo out
    test_cli_daemon.py        key set reloads the daemon; listen prints segments, exit codes
    test_parakeet_daemon.py   `slow`, real model: commits at pauses, word times at speech edges, keyterms per stream, two tracks
  live/                   opt-in (`pytest -m live`): real mic, real backend key
    test_elevenlabs_live.py  fixture to real EL: word times at the speech edges (realtime and upload), bad key
    test_aec_live.py      fixture through the speakers: cleaned source ≥ 10 dB below the mic; off / module / built-in side by side while you read
pyproject.toml            package, deps, ruff/pyright/pytest settings
README.md                 public README: pitch, client features, install, backends
```
<!-- END FILE TREE -->

## Discipline

- Source code files cap at ~500 lines (tests included). Split by concern if crossing.
- Above rule applies even to files you have not authored yourself. If you touch it, you fix it.
- A directory's listing — its files and its subdirectories with their one-liners — reads as the complete picture of one module at that level.
- CLAUDE.md file trees mirror reality. Update the relevant CLAUDE.md in the same change as a file add/remove/rename/repurpose.
- Tree descriptions: few words, elliptical, what the file is or a rough overview of its contents — not a sentence; ≤120 chars.
- Directories with their own CLAUDE.md get a folder summary, not a recursive tree.

## Checks

`uv run ruff check`, `uv run ruff format --check`, `uv run pyright` (strict), `uv run pytest`. CI runs the
same. Dependency resolution is limited to releases at least a week old (`[tool.uv] exclude-newer`).

## API keys never reach output

No error message, log line or repr may hold a key. Errors about keys name the file, line or variable,
never the value; tests assert this.

## sherpa-onnx needs sherpa-onnx-core pinned beside it

The Linux wheels of `sherpa-onnx` load `libonnxruntime.so` from `sherpa-onnx-core`, but uv resolves
`sherpa-onnx`'s deps from wheels that don't declare it. Keep both as direct deps at the same `==`
version, and bump them together.

## The pre-speech hold follows the VAD's delay

The VAD confirms a speech start `StreamingVAD.start_delay_s` late (about 0.45 s at defaults, edge
lookback included), and its `at` is the true edge. The session's `PauseGate` keeps that delay plus
the pad (`vad.pause_keep_ms`) plus two windows of held audio, computed in `session/session.py`.
Keep it derived: a fixed hold shorter than the delay cuts the first syllable after every pause.

## Recovery drops audio only when the backend answers

The recovery log (`session/recovery.py`) keeps every audio frame and commit that no connection has
answered; a commit's answer trims it. After a reconnect the new connection gets the whole log
first. Trimming on send instead of on answer loses audio when a connection fails.

## Parakeet's `bpe.vocab` is generated and unverified

sherpa-onnx's Parakeet export has no `bpe.vocab`, which hotword decoding needs.
`backends/install.py` writes one from `tokens.txt`, score = minus the token id. This is a
workaround checked only on TTS clips, not against the real SentencePiece scores. Tokens that
start a word begin with a space, not `▁`.

## Local commit hooks

`.git/hooks/pre-commit` and `commit-msg` exist only locally, not in the repo: they reject staged
text and commit messages with words from `.git/hooks/deny-words.sh`. Change the text; never
bypass them.
