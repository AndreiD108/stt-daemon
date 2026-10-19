# stt-daemon

A local speech-to-text daemon in Python: one process owns the mic, many clients connect over a Unix
socket. One mic session runs at a time; file (`transcribe`) and system-audio sessions run next to
it on backend slots. The wire protocol is in `docs/protocol.md`; code against it.

## Scope

Overview of the whole repository. `tests/unit/` has its own CLAUDE.md for the unit tests.

## File tree

Use this tree as a map — scan it to find the parts your task touches, then go straight there.

<!-- BEGIN FILE TREE -->
```
.github/                  CI
  workflows/
    ci.yml                ruff, pyright, pytest (unit + integration) on push and PR
docs/                     agent- and client-facing docs
  protocol.md             wire protocol: every command and event
  plugins.md              backend plugins: the contract, an example plugin, dependencies, status
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
      system_capture.py   SystemCapture: a capture subprocess's s16 stdout → frames; stall/exit loss, restarts, re-attach, follows an app
      pw_capture.py       PwRecordCapture: pw-record on the default output's monitor or one app's stream (Linux)
      audiotee_capture.py  AudioteeCapture: audiotee on all output or one pid (macOS); permission hint on silence, its JSON error on exit
      aec.py              EchoCanceller: NumPy AEC per 10 ms block; GCC-PHAT delay, PBFDAF filter pair, double-talk guard, residual gain
      aec_builtin.py      BuiltinEchoCancelledMic: mic track through EchoCanceller, system capture as reference, paired by capture time
      aec_pipewire.py     PipewireAec: echo-cancel module per session via pw-cli; EchoCancelledMic: mic track on its source, plain-mic fallback
      vad.py              StreamingVAD: sherpa-onnx silero per window, own debounce, low-threshold edge lookback
      silero_vad.onnx     bundled silero VAD model (from sherpa-onnx releases)
      silero_vad.LICENSE  its MIT license
    backends/             transcription backends behind one interface; BACKENDS registry in __init__
      __init__.py         BACKENDS: name → built-in backend class
      base.py             the contract: Backend, Connection, FileBackend (whole files), events, Capabilities; keyterm merge, previous_text trim
      elevenlabs.py       Scribe v2 Realtime: retrying open, padded short commits, in-order acks, pad-free word times
      elevenlabs_wire.py  EL language codes (639-3 → 639-1), error message → code, HTTP status → code with EL text
      elevenlabs_batch.py  file upload API: transcribe only, bytes as sent, one retry, 20–30 s speaker runs → commits
      fake.py             FakeBackend for tests: scripted replies, partials, delays, failures, warnings, time limit, text only
      chunked.py          ChunkedConnection: audio buffered, each commit's chunk transcribed in order; Turns (one at a time, dictation first)
      parakeet.py         local Parakeet TDT on sherpa-onnx: shared recognizer (load at open, decodes in turn, idle unload), token → word times
      plugins.py          plugin files: import by path, checks, setup per load, limits; PluginBackend on the chunked connection, 120 s timeout
      install.py          `backend install parakeet`: download, SHA-256 check, unpack, bpe.vocab from tokens.txt
    server/               the socket side: connections, commands, mic and backend slots, event routing
      __init__.py         empty package marker
      socket.py           Server: hello gate, per-connection command order, transcribe/audio, routing, reload, shutdown
      connection.py       Connection (send, slow-reader close), line reading, socket file prep, SocketInUse
      messages.py         JSONL envelope; per-command field checks → bad_request / invalid naming the field
      wire.py             session and VAD events → protocol event data (words, after_word only when asked)
      backend_set.py      Setup: backends set up here (plugins too), default, statuses, failure log; resolve a start (language, features, hints)
      starts.py           Starts mixin: start_session, transcribe, start_system_session → sessions; start errors as replies
      slot.py             Slot: mic slot idle/busy/evictable, takeover order, backend-only starts, dictation suspends / resumes, slot changes
      slot_state.py       Client, Holder, SlotState (wire form), StartRefused (busy / session_active details)
      backend_slots.py    BackendSlots: per-backend max_sessions, the sessions holding them, suspended ones
      speech.py           SpeechWatch: daemon-wide VAD on all mic audio; VAD events, speech starts by id
      sources.py          SourceScan base (events, waits, Follow: an app's next stream); PwDumpScan: pw-dump -m → sources, nodes; default_scan
      sources_macos.py    CoreAudioScan: Core Audio process list (polled + listener, PyObjC) → sources by pid, bundle id as binary
      peer.py             socket peer's pid and cmdline: SO_PEERCRED + /proc; macOS LOCAL_PEERPID + KERN_PROCARGS2
    session/              one session: capture → VAD → pause cutting → backend → segments
      __init__.py         empty package marker
      session.py          Session: one or two tracks; start, stop (bounded) / cancel / error end, suspend / resume, capture loss
      tracks.py           Track: one source's VAD → pause gate → link, commits by pause length and audio sent, audio loop
      upload.py           UploadSession: transcribe on a file backend; bytes kept, one upload at stop, no progress
      options.py          SessionOptions, MarkerRefused, start_point (where a started session takes its audio from)
      recovery.py         Link: unanswered-audio log, in-order sends, rollover, reconnect, paced replay, suspend / resume, text-only word times
      link_log.py         the link's log items: audio (int16 while suspended), commit points
      segments.py         backend answers → segments: session times via timeline, markers placed, leftover markers
      moving.py           while_moving: wait bounded by no progress, paused time not counted
      events.py           what a session emits: started, partial, error, ended (+ evicted_by); segments from transcript
      pause_gate.py       PauseGate: sends speech and short pauses, holds long ones back to a pre-speech pad; SpeechLog of spans
      word_times.py       word times estimated for text-only answers: words over speech spans, split by syllables
      timeline.py         sent audio per connection ↔ capture time; backend seconds → session ms
      markers.py          pending markers; placement in word gaps or by character length
      transcript.py       Segment, committed segments, final text (with and without markers)
      context.py          previous_text store by (client label, context_key), in memory
    service/              user service install: systemd unit + limits drop-in, launchd plist; PortAudio check
      __init__.py         empty package marker
      common.py           Runner (injectable for tests), ServiceError, the installed stt-daemon script's path
      systemd.py          unit and limits.conf rendering, ExecStart escaping; install/remove/status/logs
      launchd.py          macOS agent plist (TMPDIR and PATH kept); launchctl calls
      preflight.py        PortAudio check, tools on PATH (ffmpeg, pw-*, audiotee), echo-cancel module files; install command per distro
    plugin.py             public plugin contract: Backend, Capabilities, Options, Result, Word
    cli.py                `stt-daemon` entry point: run, service, key set, backend set-default/install/remove/list, listen, transcribe
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
  fake_audiotee.py        audiotee stand-in: its flags, WAV then silence in chunks, JSON stderr; pid not live, exit, stall, silent
  fake_pw_dump.py         pw-dump -m stand-in, and FakePipeWire: the test side that plays and stops streams, adds sources
  fake_pw_cli.py          pw-cli load-module stand-in: announces the echo-cancel source in a FakePipeWire; load error, exit, ignored TERM
  fake_elevenlabs_batch.py  local stand-in for the EL upload API (any multipart POST): records fields, scripted answers, hold
  fixtures/
    two_phrases.wav       TTS speech, 16 kHz: phrases at 1.00–3.74 s and 5.74–7.81 s
    jargon.wav            TTS speech, 16 kHz: three phrases with jargon words, 2 s pauses
    pw_dump_monitor.json  recorded pw-dump -m -N (props trimmed): registry, a stream playing and ending, id reuse
    plugins/              plugin files for loader, CLI and daemon tests
      openai_transcriptions.py  the docs' example: OpenAI-compatible /audio/transcriptions over httpx, text only, needs a key
      local_model.py      install() writes a model file, available() checks it; timed words
      raising_setup.py    setup raises (two-line message)
      builtin_name.py     named parakeet: refused as a built-in name
      two_classes.py      two Backend subclasses in one file: refused
  unit/                   pure logic, local sockets only (own CLAUDE.md): per-module tests, fake EL server, slot rig
  integration/            real daemon over a socket, WAV capture, fake backend
    conftest.py           `daemon` fixture
    harness.py            in-process Server on a temp socket; test clients; fixture helpers
    test_daemon_audio.py      word times through a cut, several commits, rollover and replay
    test_daemon_word_times.py  text-only backend: estimated times inside each phrase, markers, later commits
    test_daemon_clients.py    hello gate, welcome, errors, busy and eviction, observers, disconnects, reload
    test_daemon_lifecycle.py  socket file, shutdown, context store, marker times
    test_daemon_speech.py     VAD event routing, speech_id starts (edge, expiry, mic closed, eviction)
    test_daemon_transcribe.py transcribe: ffmpeg and raw PCM, file-time word times, slots beside mic, decoder errors
    test_daemon_plugins.py    example plugin on a local HTTP stand-in: dictation (estimated times, marker), transcribe, API error, key set + reload
    test_daemon_batch.py      elevenlabs-batch over the socket: default rule, speakers, refusals, markers, disconnect
    test_daemon_priority.py   transcribe suspended by a dictation: same output as unsuspended; blocked writer stalls no one
    test_daemon_system.py     system sessions on pw-record/audiotee stand-ins: two tracks, dictation beside, suspend/resume, fatal track, no tool, pid
    test_daemon_sources.py    sources in welcome and events (opt-in), app sessions, unknown app, re-attach (also while suspended)
    test_daemon_aec.py        mic track on the echo-cancelled source; no module → plain mic, one aec_unavailable; pw-cli exit; built-in AEC
    test_cli_daemon.py        key set reloads the daemon; listen prints segments, exit codes
    test_parakeet_daemon.py   `slow`, real model: commits at pauses, word times at speech edges, estimate vs real times, keyterms, two tracks
  live/                   opt-in (`pytest -m live`): real mic, real backend key
    test_elevenlabs_live.py  fixture to real EL: word times at the speech edges (realtime and upload), bad key
    test_aec_live.py      fixture through the speakers: cleaned source ≥ 10 dB below the mic; off / module / built-in side by side while you read
pyproject.toml            package, deps, ruff/pyright/pytest settings
README.md                 public README: pitch, client features, install, backends, files, meetings, plugins
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

## Parakeet's `bpe.vocab` is generated

sherpa-onnx's Parakeet export has no `bpe.vocab`, which hotword decoding needs.
`backends/install.py` writes one from `tokens.txt`, score = minus the token id, not the real
SentencePiece scores. Tokens that start a word begin with a space, not `▁`.

## PipeWire and Core Audio

- `pw-dump -m` prints JSON arrays back to back, not one per line; `ArraySplitter` cuts them by
  bracket depth. Registry ids are reused, `object.serial` is not: sources and `pw-record
  --target` go by serial.
- The echo-cancel module lives as long as its `pw-cli -m load-module` process; ending `pw-cli`
  unloads it. A session's end must end that process (TERM, then KILL).
- PyObjC is a darwin-only dependency. Import `CoreAudio`, `objc` and `AppKit` inside functions,
  after the `sys.platform` check, never at module level.

## Local commit hooks

`.git/hooks/pre-commit` and `commit-msg` exist only locally, not in the repo: they reject staged
text and commit messages with words from `.git/hooks/deny-words.sh`. Change the text; never
bypass them.
