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
src/                      package source
  stt_daemon/             the `stt-daemon` package
    __init__.py           __version__ from package metadata
    audio/                mic capture, recent-audio ring, VAD
      __init__.py         empty package marker
      frames.py           Frame, ring + attachments (FrameHub); one lock so attach sees each frame once
      capture.py          CaptureSource base (keep-open vs per-session holds), MicCapture: PortAudio (loaded on first use), watchdog
      wav_capture.py      WavCapture: WAV files in real time, silence between; mic stand-in for tests
      vad.py              StreamingVAD: sherpa-onnx silero per window, own debounce, low-threshold edge lookback
      silero_vad.onnx     bundled silero VAD model (from sherpa-onnx releases)
      silero_vad.LICENSE  its MIT license
    backends/             transcription backends behind one interface; BACKENDS registry in __init__
      __init__.py         BACKENDS: name → backend class
      base.py             the contract: Backend, Connection, events, Capabilities; keyterm merge, previous_text trim
      elevenlabs.py       Scribe v2 Realtime: retrying open, padded short commits, in-order acks, pad-free word times
      elevenlabs_wire.py  EL language codes, error message → (code, retryable)
      fake.py             FakeBackend for tests: scripted replies, partials, delays, failures, warnings, time limit
    server/               the socket side: connections, commands, the session slot, event routing
      __init__.py         empty package marker
      socket.py           Server: socket file, hello gate, per-connection command order, routing, reload, shutdown
      messages.py         JSONL envelope; per-command field checks → bad_request / invalid naming the field
      wire.py             session and VAD events → protocol event data (words, after_word only when asked)
      backend_set.py      Setup: backends set up here, default; resolve a start (language, features, hints)
      slot.py             Slot: idle/busy/evictable, sticky evictable timer, takeover order, slot changes
      speech.py           SpeechWatch: daemon-wide VAD on all mic audio; VAD events, speech starts by id
      peer.py             socket peer's pid and cmdline: SO_PEERCRED + /proc; macOS LOCAL_PEERPID + KERN_PROCARGS2
    session/              one session: capture → VAD → pause cutting → backend → segments
      __init__.py         empty package marker
      session.py          Session: start, audio loop, commit cadence, stop (bounded) / cancel / error end
      recovery.py         Link: unanswered-audio log, in-order sends, rollover, reconnect, paced replay
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
      launchd.py          macOS agent plist (TMPDIR kept for the socket); launchctl calls; untested on a Mac
      preflight.py        PortAudio load check; install command per distro from /etc/os-release
    cli.py                `stt-daemon` entry point: run, service, key set, backend set-default, listen
    control.py            one blocking request to the running daemon (hello, command, reply); `reload` after key set
    listen.py             `listen`: session client printing events; terminal keys (Enter, marker key, Ctrl-C)
    daemon.py             `run`: config, credentials, mic, server; SIGTERM/SIGINT shutdown
    config.py             config.toml loading, defaults, validation; config/socket paths per OS; in-place key writer
    credentials.py        API keys: mode-600 credentials.toml, <BACKEND>_API_KEY env, stdin keys; key writer
tests/
  conftest.py             marks everything under tests/live as `live` (skipped by default)
  fixtures/
    two_phrases.wav       TTS speech, 16 kHz: phrases at 1.00–3.74 s and 5.74–7.81 s
  unit/                   pure logic; local sockets only, never the mic or the internet
    conftest.py           fixtures: local fake ElevenLabs server, backend pointed at it
    fake_elevenlabs.py    that server: records requests and messages, scripted commit replies
    test_backend_base.py  keyterm merge, previous_text trim
    test_fake_backend.py  reply order, empty commits, delays, partials, failures, time limit
    test_elevenlabs_open.py    open retries and budget, HTTP and message refusals, key kept out of logs
    test_elevenlabs_stream.py  URL params, chunking, commits, padding, acks, partials, errors, self-commit
    test_frames.py        ring order, since-trimming, eviction, multi-attach, attach atomicity
    test_capture.py       device choice, holds, callback, stall/status recovery, backoff, hung stop
    test_wav_capture.py   WAV reading, real-time pacing, per-session open
    test_vad.py           edges on the fixture vs chunk size and window phase, delays, ids, gap reset
    test_speech_watch.py  daemon-wide VAD: events, speech starts by id and expiry, mic going quiet
    test_timeline.py      runs, cuts, edge snapping, jitter vs gaps, per-connection offsets
    test_pause_gate.py    held silence, pre-speech pad, short vs long pauses, flush
    test_markers.py       word-gap placement, char fallback, empty text, queue
    test_transcript.py    final-text joining, empty segments
    test_session_audio.py   fixture sessions: word times through a cut, several commits, stop/cancel
    test_session_lifecycle.py  silent sessions: leftover markers, marker refusals, failure, failed open
    test_recovery.py      link alone: replay tail and pacing, retries, give-up, idle drop, hung send, rollover
    test_recovery_edges.py  close-time ack, unknown provider cuts, final-ack failure, in-flight commit/stop
    test_session_recovery.py  fixture sessions: word times, markers, give-up, bounded stop during recovery
    test_config.py        paths per OS, defaults, type errors, config writer
    test_credentials.py   mode/owner checks, precedence, writer, stdin parsing, no key in errors
    test_slot.py          slot states, sticky evictable, takeover order and failures, client_gone
    test_peer.py          peer pid/cmdline over real Unix sockets; macOS parsing on fixture bytes
    test_messages.py      envelope parsing, encoding, per-command field checks
    test_backend_set.py   setup and default, language default and checks, features, hints, context lookup
    test_cli.py           run (broken config, --keys-stdin), key set default rule, set-default, service install files
    test_service.py       unit/plist rendering, ExecStart escaping, systemctl calls, remove guard, PortAudio hints
  integration/            real daemon over a socket, WAV capture, fake backend
    conftest.py           `daemon` fixture
    harness.py            in-process Server on a temp socket; test clients; fixture helpers
    test_daemon_audio.py      word times through a cut, several commits, rollover and replay
    test_daemon_clients.py    hello gate, welcome, errors, busy and eviction, observers, disconnects, reload
    test_daemon_lifecycle.py  socket file, shutdown, context store, marker times
    test_daemon_speech.py     VAD event routing, speech_id starts (edge, expiry, mic closed, eviction)
    test_cli_daemon.py        key set reloads the daemon; listen prints segments, exit codes
  live/                   opt-in (`pytest -m live`): real mic, real backend key
    test_elevenlabs_live.py  fixture to real EL: word times at the speech edges, padded tail, bad key
pyproject.toml            package, deps, ruff/pyright/pytest settings
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
