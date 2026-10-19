# stt-daemon protocol

One daemon owns the mic; many clients connect. A client runs sessions: a session transcribes the
mic, what the machine plays, or audio the client sends. One mic session runs at a time; the other
kinds run next to it. [Session kinds](#session-kinds) has the slots each kind takes and which
session gives way.

A minimal client: connect, send `hello`, read `welcome`, send `start_session`, read `segment`
events, send `stop_session`, read `session_ended`. Error codes are in [Error codes](#error-codes).

## Conventions

- Transport: JSONL (one JSON object per line, UTF-8, `\n`) over a Unix socket. Socket file
  permissions are the access control. Path: `$XDG_RUNTIME_DIR/stt-daemon/stt.sock` on Linux,
  `$TMPDIR/stt-daemon/stt.sock` on macOS; `STT_DAEMON_SOCKET` overrides it for the daemon and
  for clients. A line longer than 2 MiB gets `bad_request` and the daemon closes the connection.
- Envelope, both directions:

  ```json
  {"type": "<name>", "id": "<client-chosen>", "reply_to": "<id>", "ts": 1760000000000, "data": {}}
  ```

  `id` (optional, string or integer) on commands; the reply carries it as `reply_to`. A command
  without `id` gets a reply without `reply_to`. Unsolicited events have no `reply_to`. `ts` is the
  send time; the daemon ignores it on commands. `data` defaults to `{}`.
- Wall-clock times (`ts`, `*_at`): Unix epoch ms, UTC.
- Session times (`*_ms`): ms since the session's `started_at`, on the real capture clock. The daemon
  cuts pauses before it sends audio to the backend and reconnects the backend on rollover; it maps
  the backend's times back, so session times include the pauses and run without seams. The daemon
  counts on a monotonic clock and converts to UTC with one offset taken when it starts, so a
  wall-clock change can't move words. In a `transcribe` session, session times are positions in
  the audio sent: `words[0].start_ms` of 61500 is 61.5 s into the file, however fast it arrived.
- The first client message must be `hello`. Any other message first gets `error {code:
  "protocol"}`, and a line that isn't a valid message gets `bad_request`; either way the daemon
  closes the connection.
- A client holds at most one session of any kind, so session commands act on the client's own
  session and carry no id. A client that runs a mic session and a `transcribe` at once uses two
  connections. The daemon generates a `session_id` and puts it on every session event, for logs.
- Commands from one connection run one at a time, in order; `ping` is answered at once. When 16
  commands wait, the daemon stops reading the connection until one is done, `ping` included: a
  client streaming `audio` faster than the session takes it just waits on its socket writes.
- Fields shown in an event's example are always present unless the text says when they are
  missing; a field that can be `null` says so.

## Session kinds

| Kind | Command | Transcribes | Slots it takes | Evictable | When a dictation needs its slot |
| --- | --- | --- | --- | --- | --- |
| `mic` | `start_session` | the daemon's mic | the mic slot, one backend slot | after 5 min with no speech | never gives way |
| `system` | `start_system_session` | what the machine plays, and the mic with `with_mic` | one backend slot | never | suspended, after `transcribe` sessions |
| `transcribe` | `transcribe` | the bytes the client sends: a file ffmpeg reads, or raw PCM | one backend slot | never | suspended first |

The mic slot (the slot) is the right to run the one mic session: a client takes it with
`start_session` and gives it back when its session ends. Each backend has slots too: every session
takes one on its backend, and a backend runs at most `max_sessions` sessions at once (config
`backends.<name>.max_sessions`, default 2). Capacity counts sessions, not clients. A `system`
session with `with_mic` records the mic without the mic slot, so a dictation runs next to it.
`system` sessions run on Linux and macOS (14.2 or later). A `transcribe` session runs as fast as
its audio arrives.

A backend takes the kinds in its `kinds` in `welcome`. `elevenlabs-batch` takes `transcribe` only.

Dictation comes first. When a `start_session` finds its backend full and a session of another kind
holds a slot there, the daemon suspends that session instead of refusing the start: a `transcribe`
session before a `system` one, then the most recently started. A suspended session loses nothing.
It drops its backend connection and keeps its audio; when a slot on its backend frees, it resumes
and the backend gets the kept audio, longest suspended first. Its client gets `session_suspended` and
`session_resumed`. Only a `mic` session suspends, and never another `mic` session; those starts
get `busy`. On Parakeet, a `mic` session's next decode also goes before other kinds' waiting
decodes; a running decode is never interrupted.

What ends a session:

- Any kind: `stop_session` (the rest is committed), `cancel_session` (the rest is dropped), the
  client's disconnect (cancelled), or a fatal backend error (`session_error` with `fatal: true`,
  then `session_ended` with reason `error`). The daemon never ends a session because nobody
  speaks.
- `mic`: also another client's start once the slot is `evictable` (reason `evicted`). At daemon
  shutdown it stops as for `stop_session`.
- `system` and `transcribe`: daemon shutdown cancels them. So does a `reload` that removes the
  backend of a suspended one (`session_error {code: "config", fatal: true}`).
- `system`: lost system audio never ends it; the track goes silent and the daemon retries (see
  [`start_system_session`](#start_system_session)).
- `transcribe`: input ffmpeg can't decode (`invalid_request`), and a backend that answers nothing
  for 60 s (`backend_error`), end it. On `elevenlabs-batch` a `transcribe` session is never
  suspended, and its stop waits up to 10 minutes for the answer (see [`transcribe`](#transcribe)).

## Client → daemon

### `hello`

```json
{"protocol": 1, "label": "Scribbly", "observe_all": false, "vad_events": true, "sources": true}
```

- `label` (optional): shown to other clients in `slot`; also scopes `context_key`. `""` is no
  label.
- `observe_all` (optional, default false): receive every session event of every client, like a log.
  See [Observers](#observers). Refused with `observers_disabled` when the daemon's config has
  `server.observers = false`.
- `vad_events` (optional, default false): receive `speech_started` and `speech_ended`; see
  [VAD events](#vad-events).
- `sources` (optional, default false): receive `source_added` and `source_removed`; see
  [Sources](#sources). Observers get them only with this flag too.

Reply: `welcome`. Or, and the daemon closes the connection: `error {code: "protocol_unsupported",
supported: [1]}`, `observers_disabled`, or `bad_request` for a field of the wrong type. A second `hello` gets
`bad_request`.

### `start_session`

```json
{
  "label": "desks.html",
  "backend": "elevenlabs",
  "language": "auto",
  "keyterms": ["Scribbly"],
  "previous_text": "…",
  "context_key": "docs/desks.html",
  "partials": true,
  "word_timestamps": true,
  "speech_id": "s41"
}
```

All fields optional.

- `label`: the session's own label, next to the client's.
- `speech_id`: start the session's audio where that speech began (from `speech_started`), less
  a short pad (config `vad.pause_keep_ms`, 100 ms), not at the command.
- Without `speech_id`, the session's audio starts `audio.pre_start_ms` (1 s) before the command
  arrived, but not before the previous session's audio ended or the oldest audio in the buffer.
  When the mic isn't kept open (`mic_open: false` in `welcome`), it starts when the mic is ready.
  Either way that point is `started_at`.
- `backend`: default `default_backend` from `welcome`.
- `language`: a code from the backend's `languages` in `welcome`, or `"auto"` (detect). Default:
  the backend's `language` in config (`backends.elevenlabs.language`, itself `"auto"` unless
  changed); for a backend with no such setting, `"auto"` when it lists it. A language the backend
  doesn't have fails the start with `unsupported_language`; a backend that takes no language at
  all fails it with `unsupported`.
- `keyterms`: bias words, merged after the config defaults.
- `previous_text`: context hint for the backend (trimmed to its limit, 50 chars for ElevenLabs).
  `""` sends none and skips the `context_key` lookup.
- `context_key`: when `previous_text` is missing, the daemon uses the last text stored under
  (client `label`, `context_key`). It stores a session's `final_text` there, with markers removed,
  when the session ends `stopped` or `evicted` with non-empty text; other endings leave the stored
  text as it was. In memory only, at most 1000 keys (the oldest is dropped); lost on restart.
  Without a client `label`, the store lives only as long as the connection. Without
  `context_key`, nothing is looked up or stored.
- `partials`: send `partial` events. Default false.
- `word_timestamps`: put `words` on `segment` events, and `after_word` on their `markers`.
  Default false.

- `diarize`, `num_speakers`: as for [`transcribe`](#transcribe). No built-in backend that takes
  `mic` sessions has `diarize`, so there `diarize: true` fails with `unsupported_option`; a
  plugin backend with `diarize: true` in `welcome` takes it in every kind.

Hints (`keyterms`, `previous_text`, `context_key`) are dropped when the backend can't use them; the
backend's entry in `welcome` says which it takes. Features (`partials`, `word_timestamps`) fail the
start with `unsupported` when the backend's entry has them `false`; so does a backend whose `kinds`
lacks the session's kind.

Checks run in this order, before either slot changes: field types (`bad_request`), `speech_id`
(`speech_expired`), the mic slot (`session_active`, `busy`), the backend, language and features,
then room on the backend (`busy`, unless a session of another kind there can be suspended; see
[Session kinds](#session-kinds)). So a request for a missing backend sent while the mic slot is
`busy` gets `busy`. An evictable holder's session doesn't count against its backend's capacity: a
start that would evict it is refused only when the backend is full without it, and then nothing is
evicted.

When the slot is `evictable`, the daemon takes it over in this order:

1. Check the whole request. A bad request fails and the holder's session goes on.
2. Hold the slot for this client; any other start now gets `busy`. Copy this session's audio
   out of the buffer now (from `speech_id`'s start, or the pre-start point), so it can't expire
   during the wait below; audio captured during the wait also goes to this session. If the mic
   can't be held here (`mic_unavailable`), the holder's session goes on.
3. Stop the holder's session as `stop_session` would (commit what is left, bounded wait). The
   holder gets `session_ended` (reason `evicted`, or `error` if the backend fails during the stop)
   with everything committed by then; text the backend can't flush in time is lost.
4. Start this session: `slot`, then `session_started`. If this client disconnects or its start
   fails from step 3 on, the copy is dropped and the slot goes idle.

The holder's session keeps its backend slot until its `session_ended` goes out.

When the mic slot is `busy`, or the backend has no room, the start fails with `busy`; clients that
follow `slot` events don't send it then.

Reply: `session_started`, or `error` with one of the start codes in [Error codes](#error-codes).

### `transcribe`

```json
{"backend": "parakeet", "language": "en", "word_timestamps": true, "pcm": {"rate": 48000, "channels": 2}}
```

Starts a `transcribe` session: the client then sends the audio with `audio` commands and ends it
with `stop_session`. All fields optional. `label`, `backend`, `language`, `keyterms`,
`previous_text`, `context_key`, `partials` and `word_timestamps` work as for `start_session`;
`speech_id` is refused (`bad_request`).

- `backend`: default `default_backend`, except that a default of `elevenlabs` gives
  `elevenlabs-batch` when it is set up (it is whenever `elevenlabs` is).
- `diarize` (default false): tell speakers apart; words and segments get `speaker`. Only on a
  backend with `diarize: true` in `welcome`, else `unsupported_option` (`field: "diarize"`).
- `num_speakers` (1–32): the most speakers the audio has. Only with `diarize: true`, else
  `bad_request` (`field: "num_speakers"`).
- `pcm`: the bytes are raw PCM, s16le, at `rate` (8000–192000) with `channels` (1 or 2). Raw PCM
  has no header, so without `pcm` the bytes must be a file in a format ffmpeg reads (wav, mp3,
  m4a, ogg, webm, mp4, …); ffmpeg decides, the daemon does not look at the bytes.
- 16 kHz mono `pcm` goes to the session as it is. Anything else goes through an `ffmpeg` process
  on the daemon's machine, which must be installed: without it the start fails with
  `tool_unavailable` and the message names the install command.

Checks run in this order: field types (`bad_request`), this client's session (`session_active`),
the backend, language and features, ffmpeg (`tool_unavailable`), then room on the backend
(`busy`, `full_slot: "backend"`). A full backend refuses the start; the daemon does not queue it.

The session commits as a live one does, by audio sent (see [`segment`](#segment)). It sends
`progress` events as it goes. It takes no more audio while 60 s of it (or twice the chunk cap,
if more) waits for the backend's text; a backend that then answers nothing for 60 s ends the
session with a fatal `backend_error`.

While suspended for a dictation, the session goes on taking in audio up to that 60 s limit, then
stops like any session that is behind: the daemon stops reading the connection, so commands sent
after that `audio`, `stop_session` and `cancel_session` included, wait for the resume. To drop a
suspended session at once, disconnect. Suspended time doesn't count as the backend answering
nothing.

On `elevenlabs-batch` the session works another way. The daemon keeps the bytes as sent, with no
ffmpeg and no decoding, and uploads them in one request at `stop_session`; ElevenLabs reads the
container itself. Raw `pcm` goes as it is at 16 kHz mono, else with a WAV header for its rate and
channels. So:

- The session commits only at the stop and is never suspended. It sends neither `partial` nor
  `progress` events, and its start never fails with `tool_unavailable`.
- At the stop, all the text comes at once, as segments of 20 to 30 s (cut at a sentence end where
  there is one) and, with `diarize`, at each speaker change. The daemon sends them as fast as the
  client reads, then `session_ended`. The stop waits for the answer however long it takes (the
  daemon gives up after 10 minutes with no answer, with `network`).
- `add_marker` needs `at`, since the audio's length is known only from the answer.
- `duration_ms` in `session_ended` is the end of the last word, not the file's length.
- The input may be at most 256 MiB; more ends the session with a fatal `invalid_request`.
- A failed upload is tried once more after 2 s on a `network` or `backend_error`; any other
  failure, or a second one, ends the session with a fatal `session_error`.

Reply: `session_started` (`kind: "transcribe"`), or `error` with one of the start codes.

### `start_system_session`

```json
{"backend": "parakeet", "with_mic": true, "word_timestamps": true, "app": "src_7"}
```

Starts a `system` session: it transcribes what the machine plays, from the default output (it
follows the user's output switches) or from one application, until `stop_session` or
`cancel_session`. All fields
optional. `label`, `backend`, `language`, `keyterms`, `previous_text`, `context_key`, `partials`
and `word_timestamps` work as for `start_session`; `speech_id` is refused (`bad_request`); the
session starts at the command.

- `with_mic` (default false): also transcribe the mic, as a second track. The mic track takes no
  mic slot, so dictation runs next to it. With speakers on, the mic also hears what they play.
- `app` (optional): a source id from `welcome.sources` or `source_added` (see [Sources](#sources)).
  The system track then captures only that application's stream. An id the daemon doesn't list
  now gives `invalid` with `field: "app"`. Works the same with `with_mic`.

Each track (`system`, and `mic` with `with_mic`) runs on its own: its own pause cuts, commits,
backend connection (or, on Parakeet, its own turns on the shared model) and reconnects, on the
session's clock. The session takes one slot on its backend for both tracks. `partial`, `segment`
and `session_error` events carry the `track` they come from. `session_ended.final_text` has one
line per segment of either track, in start-time order, each line prefixed `[system] ` or `[mic] `.

Markers belong to the session, not to a track. A marker goes into the first segment, of either
track, with text whose audio reaches `at`. A segment with no text takes it only once every other
track has passed `at`.

The system track is a capture process: `pw-record` (PipeWire's tools) on Linux, `audiotee` (Core
Audio process taps) on macOS 14.2 or later. Without it the start fails with `tool_unavailable` and
the message says how to install it. Other platforms answer `unsupported`.

On macOS, the first system session makes macOS ask for the System Audio Recording permission, for
the process that runs the daemon: `stt-daemon` when the launchd agent runs it, the terminal app
when it runs from a terminal. Until it is granted, `audiotee` runs but captures nothing, so the
session gets `capture_lost` after 3 s, and its message names the setting: System Settings >
Privacy & Security > Screen & System Audio Recording.

Only backend failures end the session. When the system audio stops (the capture process exits, or
sends nothing for 3 s), its track goes silent: the client gets `session_error {code:
"capture_lost", track, fatal: false}`, and the daemon starts it again after 1, 2, 4, 8, then every 10 s,
until audio comes again: `session_error {code: "capture_restored", track, fatal: false}`. A fatal
backend error on either track ends the session, with the text of both tracks. `session_started`
waits up to 2 s for the first system audio; if none comes, `capture_lost` follows it.

On Linux, the `mic` track has echo cancellation: PipeWire's echo-cancel module takes what the
speakers play out of the mic, so the call's other side is not transcribed twice. When the module
can't be loaded, or stops during the session, the track takes the plain mic and the client gets
one `session_error {code: "aec_unavailable", track: "mic", fatal: false}`. On macOS, the daemon's
own echo canceller does the same; it needs nothing installed.

With `app`, the track ends with the application's stream (the app closed, or stopped playing).
The client gets `capture_lost`, and the daemon does not retry against the gone stream: it waits
until a source of the same application appears (same `binary`; same `name` when the source had no
`binary`), captures that stream, and sends `capture_restored`. A new source of that app while the
session is suspended is picked up the same way; the audio is there on the resume.

On macOS, a source is a process, and with `app` the track captures that process's output. When the
process stops playing, the track stays on it and gets silence, with no `capture_lost`. When the
same application plays from another process (same `binary`, the bundle id), the track moves to
that process. When the capture process exits (the app quit), the track waits as above.

`stop_session` commits both tracks and waits as for a mic session. While suspended for a dictation,
both tracks keep their audio, and a stop finishes after the resume, as for
[`stop_session`](#stop_session) after a suspension.

Checks run in this order: field types (`bad_request`), the platform (`unsupported`), `app`
(`invalid`), this client's session (`session_active`), the backend, language and features, the
capture tool (`tool_unavailable`), then room on the backend (`busy`, `full_slot: "backend"`). With `with_mic`,
a mic that doesn't open gives `mic_unavailable`.

Reply: `session_started` (`kind: "system"`, with `tracks`; with `app`, also `app` and `app_name`),
or `error` with one of the start codes.

### `audio`

```json
{"data": "UklGRiQAAABXQVZF…"}
```

- `data`: the next bytes of the input, base64, at most 1 MiB decoded (`invalid`, `field: "data"`
  when larger or not base64). Send the input in order, in chunks of any size up to that.

No reply when the bytes are taken. While the session is behind, the daemon takes them later and
stops reading the connection (see [Conventions](#conventions)); the client needs no pacing of
its own. Errors: `no_session` (the client has no `transcribe` session, or it is ending), `invalid`.

### `add_marker`

```json
{"marker": "⟦A⟧", "at": 1760000012345}
```

- `marker`: the text to place, non-blank.
- `at` (optional, epoch ms UTC): when the event happened on the client. Missing: the time the
  message arrived. Earlier than `started_at`: the marker is refused (`out_of_range`). Later than
  the arrival time: the arrival time is used.
- In a `transcribe` session, `at` is `started_at` plus the marker's position in the audio, in ms,
  and may be past the audio sent so far. Missing: the end of the audio the session has taken in.

Placement. No early commit; every accepted marker shows up in exactly one segment, unless the
session is cancelled (`cancel_session`, or the client disconnects), which drops markers not yet
placed:

- It goes into the first segment not yet committed when the marker arrives whose audio reaches
  `at`; committed text never changes. (A commit can be on its way when the marker arrives; a
  marker from after that commit's cut waits for the next segment.)
- Inside that segment it goes into the gap between words nearest `at`: from word times when the
  segment has them (all but a whole-file backend without them), else from character lengths across
  the segment's span. When `at` is before the
  segment's first word (a marker that arrived late, or in a pause), it goes first (`after_word: -1`);
  `at_ms` keeps the true time.
- Markers still waiting at stop, including in a session with no speech at all, come in a last
  segment. If nothing more was said, that segment has no words, its `text` holds only the markers,
  and `start_ms` equals `end_ms`.

Reply: `marker_accepted {marker, at_ms}`, or `error`: `no_session`, `session_stopping` (the session
is ending), `invalid` (`marker` missing or blank, or a field of the wrong type), `out_of_range`.

### `stop_session`

```json
{}
```

Commits what is left, waiting at most `session.stop_wait_ms` (5 s) for the backend's last text.
Reply: `session_ended` (reason `stopped`; `error`, after a `session_error`, when the backend fails
during the stop), or `error {code: "no_session"}`: the client has no session (for example, it was
evicted), or the session is already ending and its `session_ended` follows.

In a `transcribe` session, `stop_session` means the input is complete. The daemon decodes and
transcribes the rest, which can be minutes of audio, then answers `session_ended` with the whole
text. It waits as long as the session moves on and gives up after `session.stop_wait_ms` with no
audio taken and no text from the backend. Input ffmpeg can't decode, or no audio at all, ends the
session with `session_error {code: "invalid_request", fatal: true}`, whose message holds the end
of ffmpeg's error output, and `session_ended` (reason `error`).

A session that was suspended, or is suspended when the stop comes, finishes after it resumes: it
transcribes all the audio it kept, then answers `session_ended`, which can come minutes later.
Waiting for a slot doesn't count against the stop wait; as for `transcribe`, the stop gives up only
after `session.stop_wait_ms` with no progress once resumed. `cancel_session` drops the kept audio.

### `cancel_session`

```json
{}
```

Drops what is left. Reply: `session_ended` (reason `cancelled`), or `error {code: "no_session"}`,
as for `stop_session`. A `cancel_session` sent after `stop_session` waits behind it (commands run
in order); to drop a `transcribe` session that is still finishing, disconnect. A disconnect or a
daemon shutdown cancels a `transcribe` session at any point, finishing or not.

### `reload`

```json
{}
```

Re-reads config and credentials, rescans installed backends and imports the plugin files again. Sent by the `stt-daemon` CLI after
`key set`, `backend install`, `backend remove` and `backend set-default`. A running session keeps
its backend, even one no longer set up; a removed Parakeet model leaves memory when its last
session ends. New `backends.parakeet.num_threads` and `hotwords_score` values apply once no
Parakeet session is running. A new `max_sessions` applies to the next start; a backend over its new
capacity keeps its sessions until they end. A suspended session whose backend is no longer set up
ends with `session_error {code: "config", fatal: true}` and `session_ended` with what it
committed. When backends or capacities change, a `slot` event
follows the `backends` event. Changes to `[audio]` and `[slot]` take effect when the daemon
restarts. `[vad]` changes apply to
sessions started after the reload, but VAD events keep the old settings until a restart. Reply: `backends`; every
other client gets the same `backends` event. Or `error {code: "config"}` when the config or
credentials file is invalid (bad TOML, a bad value, credentials not mode 600); the daemon keeps its
last setup.

### `backends_status`

```json
{}
```

Reply: `backends_status`, every backend the daemon knows with its status, and the plugin files that
did not load. `stt-daemon backend list` prints it.

```json
{
  "backends": [
    {"name": "elevenlabs", "status": "ready", "default": true},
    {"name": "parakeet", "status": "not installed", "default": false},
    {"name": "openai", "status": "no key", "default": false},
    {"name": "local-whisper", "status": "failed: RuntimeError: no GPU found", "default": false}
  ],
  "failed_files": [{"file": "broken.py", "status": "failed: SyntaxError: invalid syntax"}]
}
```

`status` is `ready`, `no key`, `not installed`, or `failed: ` and the first line of the error.
Only `ready` backends are in `welcome` and `backends`.

### `ping`

Reply: `pong {}`, at once, even while another command from this connection runs (a `stop_session`
sent during a slow start waits for the start; a `ping` doesn't).

## Daemon → client

### `welcome`

```json
{
  "protocol": 1,
  "daemon_version": "0.1.0",
  "client_id": "c7",
  "backends": [
    {
      "name": "elevenlabs",
      "languages": ["auto", "en", "ro"],
      "word_timestamps": true,
      "partials": true,
      "hints": ["keyterms", "previous_text"],
      "diarize": false,
      "kinds": ["mic", "system", "transcribe"]
    }
  ],
  "default_backend": "elevenlabs",
  "mic_open": true,
  "slot": {"state": "idle", "backends": {"elevenlabs": {"max_sessions": 2, "sessions": []}}},
  "sources": [{"id": "src_7", "name": "Zoom", "binary": "zoom", "media_name": "Playback"}]
}
```

- `backends` lists only the backends set up on this machine; empty when none is. The built-in
  ones: `elevenlabs` and `elevenlabs-batch` (an API key is set; the batch one has `partials: false`,
  `hints: ["keyterms"]`, `diarize: true`, `kinds: ["transcribe"]`) and `parakeet` (local model
  installed; `languages: ["auto"]`, `partials: false`, `hints: ["keyterms"]`), then each plugin
  backend set up, by its name (see [plugins.md](plugins.md)). Plugins have `partials: false`,
  `word_timestamps: true` and every kind.
- `diarize`: the backend tells speakers apart (see [`transcribe`](#transcribe)).
- `kinds`: the session kinds the backend takes.
- `default_backend`: the user's global default, `null` when there is no backend; the first backend
  set up, changed with `stt-daemon backend set-default <backend>`. Never a backend that takes only
  `transcribe` sessions.
- `slot` has the `slot` event's shape. Unlike `slot` events, it can show a holder whose session is
  still starting (`since: null`), and that session in `backends`.
- `mic_open`: the daemon keeps the mic open between sessions (config `audio.keep_mic_open`,
  default true). When false, the mic opens at each start: VAD events flow only during sessions,
  sessions start at the command, and each start waits for the device.
- `sources`: the applications playing audio now; see [Sources](#sources). Always present, `[]`
  off Linux and macOS.

### `backends`

```json
{"backends": [], "default_backend": null}
```

Sent to every client after a `reload`. Same shapes as in `welcome`.

### `slot`

Sent to every client when the mic slot changes state or holder, when a backend slot is taken or
released, and when a `reload` changes the backends or their `max_sessions`.

```json
{
  "state": "busy",
  "holder": {
    "client_id": "c7",
    "pid": 41230,
    "cmdline": ["scribbly", "serve"],
    "client_label": "Scribbly",
    "session_label": "desks.html",
    "session_id": "…",
    "since": 1760000000000,
    "last_speech_at": 1760000300000
  },
  "backends": {
    "elevenlabs": {
      "max_sessions": 2,
      "sessions": [
        {"session_id": "…", "kind": "mic", "pid": 41230, "client_label": "Scribbly", "suspended": false}
      ]
    },
    "parakeet": {"max_sessions": 2, "sessions": []}
  }
}
```

- `state` and `holder` describe the mic slot. `state`: `idle` (no `holder`) | `busy` |
  `evictable`.
- `evictable`: no speech detected for 5 minutes (config `slot.evictable_after_s`). Once evictable,
  the session stays evictable until it ends; later speech doesn't turn it back to `busy`.
- Every `holder` field is present; all but `client_id` can be `null`.
- `pid` and `cmdline` come from the socket peer, not from the client: on Linux `SO_PEERCRED` and
  `/proc/<pid>/cmdline`; on macOS `LOCAL_PEERPID` and the process table. `null` when the platform
  can't tell.
- `since`: the session's `started_at`; `null` while the session is still starting (in `welcome`
  and in the `slot` of a `busy` error). `slot` events come only once it has started.
- `last_speech_at`: when speech was last heard in this session (the time of this event while
  speech goes on); `null` when there was none yet. The slot doesn't send an event for each speech.
- `backends`: every backend set up, by name, with its `max_sessions` and the sessions holding its
  slots, oldest first; `{}` when none is set up. A backend that a `reload` dropped stays listed
  while it still has sessions. `pid` and `client_label` can be `null`. During a takeover, the
  evicted session stays listed until its `session_ended`, so a backend can briefly list more
  sessions than `max_sessions`.
- `suspended`: the session is suspended for a dictation and waits for a slot; it doesn't count
  against `max_sessions`. Suspended sessions come after the others, longest suspended first.

### `session_started`

```json
{
  "session_id": "…",
  "client_id": "c7",
  "client_label": "Scribbly",
  "session_label": "desks.html",
  "kind": "mic",
  "backend": "elevenlabs",
  "language": "auto",
  "started_at": 1760000000000,
  "previous_text": "…"
}
```

- `kind`: `mic`, `system` or `transcribe`; see [Session kinds](#session-kinds).
- `tracks`: only for a `system` session: `["system"]`, or `["system", "mic"]` with `with_mic`.
- `app`, `app_name`: only for a `system` session started with `app`: the source id, and that
  source's `name`.
- `started_at`: capture time of the session's first sample. For `transcribe`, the time the
  session started; its session times count from there in audio time.
- `previous_text`: the context hint the session uses, from the command or the `context_key` store,
  before the backend trims it; `null` when none.
- `client_label`, `session_label`, `language`: `null` when not set.
- The client fields let observers tell sessions apart without waiting for `slot`.
- Parakeet sends `session_started` without waiting for its model to load (about 1 s when it is
  not in memory; it leaves memory 10 minutes after the last Parakeet session, config
  `backends.parakeet.unload_after_minutes`, where 0 keeps it). The audio is kept
  and transcribed once the model is in; a model that fails to load ends the session with
  `model_error`.

### `partial`

```json
{"session_id": "…", "text": "move this up next to"}
```

The backend's current guess for the text after the last segment. Each `partial` replaces the
previous one; a `segment` replaces the partial. Only with `partials: true`. In a `system`
session, `track` names the track, and a partial replaces only that track's previous one.

### `segment`

```json
{
  "session_id": "…",
  "index": 3,
  "text": "move this ⟦A⟧ up next to the header",
  "language": "en",
  "start_ms": 41200,
  "end_ms": 44950,
  "words": [{"text": "move", "start_ms": 41200, "end_ms": 41480}],
  "markers": [{"marker": "⟦A⟧", "at_ms": 41650, "after_word": 1}]
}
```

- Committed text; it never changes after this.
- `start_ms`, `end_ms`: the span of captured audio this segment covers, from the first sample sent
  after the previous commit to the last sample before this one (pauses inside included). A commit
  with no text and no markers sends no segment, so a segment can start later than the previous
  segment's `end_ms`.
- `text` holds any markers placed in it, inline; `words` never does.
- `track`: only in a `system` session: `system` or `mic`. `index` counts the session's segments
  across both tracks.
- `language`: the detected language, when the session used `"auto"` and the backend reports one
  (ElevenLabs does, Parakeet doesn't); missing otherwise.
- `markers`: those placed in this segment; missing when there are none. `at_ms` is the
  marker's time (from `at` or arrival).
- `words`, and `after_word` on each marker: only with `word_timestamps: true`. `after_word` is
  the index in `words` of the word the marker follows, `-1` before the first. A backend that
  answers with text only still yields `words`: the daemon estimates their times from the audio,
  splitting each stretch of speech over its words by syllable count.
- `speaker` (on the segment and on each word, as `"speaker_0"`, `"speaker_1"`, …): only in a
  session with `diarize: true`. Segments split where the speaker changes, so a segment has one;
  missing when the backend named none.
- When segments commit depends on how much audio was sent since the last commit (silence the
  session cut is not sent, so it doesn't count). Early in a segment only a long pause commits, so
  a segment keeps its sentence when the speaker stops to think; as it grows, shorter pauses do.
  Pauses come from the VAD and need speech sent since the last commit.

  | Audio sent | Pause that commits |
  |---|---|
  | 0–8 s | 3.5 s |
  | 8–14 s | 1 s (ElevenLabs: 2 s) |
  | 14–19 s | 0.8 s |
  | 19–23 s | 0.6 s |
  | 23–27 s | 0.3 s |
  | 27–30 s | 0.15 s (not ElevenLabs) |

  With no such pause, a segment commits at 25 s of audio for ElevenLabs (it commits on its own
  at about 36 s) and at `backends.<name>.max_chunk_seconds` (default 30) for Parakeet and plugins.
  Every session also commits at stop, and ElevenLabs before each connection rollover (about every
  13 minutes). Parakeet sessions share one model and take turns to transcribe, so a commit can
  wait for another session's.

  On Parakeet, mic audio goes through a level gate before it is transcribed. In each commit's
  audio, sound 35 dB or more under the voice is cut (10 dB before the first loud part); sound
  that is echo of what the speakers played is cut from 25 dB under (6 dB before the first loud
  part). This keeps most background talk and leftover echo out of the text; word times still
  point into the session's audio. The gate does not replace echo cancellation: a
  commit with only echo in it is transcribed. To tell echo, the daemon records the speakers
  (`pw-record` on Linux, `audiotee` on macOS, with the same permission as
  [`start_system_session`](#start_system_session)) while the mic is open: from daemon start with
  `audio.keep_mic_open`, else during each session. A commit waits up to 0.4 s for the speakers'
  audio of its last moments. Config `backends.parakeet.echo_gate = false` turns the gate off; the
  speakers are then not recorded once running sessions end.

### `progress`

```json
{"session_id": "…", "audio_seconds": 61.536}
```

How much of a `transcribe` session's audio the session has taken in, in seconds from its start.
At most one a second while the audio comes, and one more once the input is complete
(`stop_session`), with the total.

### `session_suspended`

```json
{"session_id": "…", "reason": "preempted", "by": {"kind": "mic", "pid": 5120, "client_label": "Talkey"}}
```

A dictation took this session's slot on its backend; see [Session kinds](#session-kinds). Only to
the session's client and observers. `by` is the dictation's session: `pid` and `client_label` can be `null`. The session
keeps its audio and its markers; `segment` events stop until it resumes.

### `session_resumed`

```json
{"session_id": "…", "suspended_seconds": 42.5}
```

The session has a slot again and its backend connection is open (for Parakeet, it is in the decode
queue again); the audio kept meanwhile goes first. `suspended_seconds`: wall time suspended.

### `session_ended`

```json
{
  "session_id": "…",
  "reason": "stopped",
  "final_text": "…",
  "duration_ms": 312000,
  "segment_count": 14,
  "evicted_by": {"client_id": "c9", "pid": 5120, "client_label": "Talkey"}
}
```

- `reason`: `stopped` | `cancelled` | `evicted` | `client_gone` | `error`.
- `final_text`: all segments joined, markers inline. Kept on `evicted`. In a `system` session,
  one line per segment in start-time order, each prefixed `[system] ` or `[mic] `.
- `evicted_by`: only with reason `evicted`; `pid` and `client_label` can be `null`.
- `client_gone`: the holder disconnected; the session is cancelled. Only observers see it.
- `cancelled` also ends a `transcribe` or `system` session the daemon drops when it shuts down.

### `session_error`

```json
{"session_id": "…", "code": "network", "message": "…", "fatal": true}
```

A backend failure during a session, or `internal` when the daemon fails. The daemon reopens a
stalled mic by itself and doesn't report it here. In a `system` session, `track` names the track;
the system track reports `capture_lost` and `capture_restored` here, and the mic track
`aec_unavailable` (all non-fatal; see [`start_system_session`](#start_system_session)). A
suspended session whose backend a `reload` removed ends with `config`.

- `fatal: true` is followed by `session_ended` (reason `error`) with every segment committed so far.
- `fatal: false`: the session goes on. For a lost backend connection, the daemon reconnects and
  replays the audio the backend hasn't answered yet. It gives up, fatally, after 3 failed
  reconnects with no answered commit between them, or at once for a code that can't be retried
  (see [Error codes](#error-codes)). Stop and eviction still use their bounded wait.

When a backend returns text without saying where it cut the audio, the daemon holds that text
until the next known commit boundary or the end of the session. If the connection fails first, it
replays the audio instead of publishing that text twice. This can delay a segment, but does not
discard the audio.

### `error`

```json
{"code": "busy", "message": "…"}
```

The reply to a command that failed. The message says what happened and what to do next; codes are
below. A command field with the wrong type gives `bad_request` (`invalid` for `add_marker`), with
`field` naming it. A line that isn't a JSON object with a `type`, or whose `id` or `data` has the
wrong type, gets `bad_request` without `field` or `reply_to`.

A connection whose unread events pile up past 8 MiB is closed.

## Error codes

Codes of `error` replies, and of `session_error` for the backend codes.

| Code | Sent for | Meaning; what to do |
| --- | --- | --- |
| `protocol` | first message | Not `hello`. Reconnect and send `hello` first. |
| `protocol_unsupported` | `hello` | `supported` lists the daemon's protocol versions. |
| `observers_disabled` | `hello` | The config has `server.observers = false`; connect without `observe_all`. |
| `bad_request` | any | Not a valid message or command; `field` names a field of the wrong type. |
| `busy` | `start_session`, `transcribe`, `start_system_session` | A slot is full and no session there can be suspended; `slot` has every slot. `full_slot`: `mic` (another client holds the mic slot; wait for a `slot` event with `idle` or `evictable`) or `backend` (the backend named in `backend` runs `max_sessions` sessions; wait for room or use another backend). `holder` (`pid`, `client_label`, `kind`; the first two can be `null`) is the session in the way: the mic holder, or the oldest on the backend. |
| `session_active` | `start_session`, `transcribe`, `start_system_session` | This client already holds a session; `slot` has it. Stop it first. |
| `speech_expired` | `start_session` | The `speech_id` is unknown or its start left the buffer. Start without it. |
| `no_backend` | `start_session`, `transcribe`, `start_system_session` | No backend set up; the message names `stt-daemon key set elevenlabs` and `stt-daemon backend install parakeet`. |
| `unknown_backend` | `start_session`, `transcribe`, `start_system_session` | `backend` isn't set up here; pick one from `welcome.backends`. |
| `unsupported_language` | `start_session`, `transcribe`, `start_system_session` | The backend lacks that language (or the config's default language). |
| `unsupported` | `start_session`, `transcribe`, `start_system_session` | The backend lacks a requested feature, or doesn't take this session kind (`elevenlabs-batch` with `start_session`), or the platform can't capture system audio (`start_system_session` off Linux and macOS). Start without the feature, or on another backend. |
| `unsupported_option` | `start_session`, `transcribe`, `start_system_session` | The backend lacks an option; `field` names it (`diarize`). Send without it. |
| `mic_unavailable` | `start_session`, `start_system_session` with `with_mic` | The mic didn't open. |
| `tool_unavailable` | `transcribe`, `start_system_session` | A program the daemon needs isn't installed; `tool` names it (`ffmpeg`, `pw-record`, `audiotee`), the message says how to install it. A `transcribe` with raw 16 kHz mono `pcm` needs none. |
| `no_session` | `add_marker`, `audio`, `stop_session`, `cancel_session` | No session, or it is already ending. |
| `session_stopping` | `add_marker` | The session is ending. |
| `invalid` | `add_marker`, `audio`, `start_system_session` | `marker` missing or blank, `data` not base64 or over 1 MiB, or `app` not a listed source; `field` names the field. |
| `out_of_range` | `add_marker` | `at` is before `started_at`. |
| `config` | `reload`; `session_error`, `fatal: true` | Config or credentials invalid; the last setup stays. In `session_error`: a `reload` removed the backend of a suspended session. |
| `capture_lost`, `capture_restored` | `session_error`, `fatal: false` | A `system` session's track stopped getting audio, or gets it again; `track` names it. The session goes on. |
| `aec_unavailable` | `session_error`, `fatal: false`, `track: "mic"` | The `mic` track of a `system` session takes the plain mic: echo cancellation could not start, or stopped. Sent once; the message says why and names the install command when something is missing. |
| `internal` | any | The daemon failed; its log says why. |

Backend codes come as the reply to a start command when the backend can't
connect, and in `session_error` during a session. For a code the daemon doesn't retry, the
session ends at once (`fatal: true`), and starting again gives the same error until the user fixes
the cause.

| Code | Retried | Meaning |
| --- | --- | --- |
| `auth` | no | The API key was refused; the user sets a new one with `stt-daemon key set`. |
| `quota` | no | The account is out of quota (ElevenLabs HTTP 402). |
| `invalid_request` | no | The backend rejected the request or the audio; for `transcribe`, also input ffmpeg can't decode. |
| `model_error` | no | The local model didn't load; the message names the directory to delete before reinstalling. |
| `concurrency_limit` | no | The provider refused one more connection: the account's concurrent sessions are in use. A `start_session` that gets it suspends a session of another kind on that backend, if there is one, and tries once more. |
| `rate_limited` | yes, during a session | Throttled. At start (HTTP 429) it isn't retried; start again later. |
| `network` | yes | The connection failed or timed out. |
| `backend_error` | yes (ElevenLabs), no (Parakeet decode, plugins) | The backend failed. |
| `session_limit`, `idle_drop` | yes | The backend ended a connection (time limit, no audio). |

"Retried" means the daemon reconnects; it still gives up after 3 failed reconnects, with that
code and `fatal: true`. Act on `fatal`, not on the code. For an HTTP failure, `message` starts
with the status and holds the provider's own text, as in `ElevenLabs HTTP 402: …`.

## VAD events

For clients that sent `vad_events: true`, and for observers during sessions. The daemon's VAD runs
on CPU the whole time the mic is open, and it reports speech:

`speech_started`:

```json
{"speech_id": "s41", "at": 1760000012000, "expires_at": 1760000017000}
```

`speech_ended`:

```json
{"speech_id": "s41", "at": 1760000015200}
```

- While the slot is idle, they go to every client that sent `vad_events: true`. During a session,
  only to the holder, if it sent `vad_events: true`, and to every observer.
- A session that starts during speech already in progress: its holder gets that speech's
  `speech_ended` (with the `speech_id` sent earlier), then the events that follow.
- Both events arrive late, and `at` is the edge, found in the buffered audio (to a 32 ms window).
  At defaults, `speech_started` comes up to about 0.45 s after its `at` (speech must last
  `vad.min_speech_ms`, 250 ms, plus a look back for a soft onset), and `speech_ended` about
  `vad.min_silence_ms` (600 ms) after its `at`; add up to one capture chunk (`audio.chunk_ms`,
  200 ms). A session started from `speech_id` takes its audio from that edge, so the first
  syllable isn't cut.
- `expires_at`: `at` plus the audio buffer (`audio.buffer_ms`, 5 s). Until about then,
  `start_session {speech_id}` can start a session from that speech; the check is whether the
  speech's start is still in the buffer, so a start right at `expires_at` can go either way.
- A client can drive sessions from these alone: start on `speech_started`, stop on `speech_ended`
  or later. The daemon doesn't stop sessions on its own.

## Sources

The daemon watches which applications play audio. On Linux it reads `pw-dump` (PipeWire's tools),
and each playback stream is a source. On macOS it reads Core Audio's process list every 2 s (and
at once when Core Audio reports a change), and each process whose audio output runs is a source. `welcome.sources` lists them, and clients that sent
`sources: true` get these events as streams appear and go:

`source_added`:

```json
{"source": {"id": "src_7", "name": "Zoom", "binary": "zoom", "media_name": "Playback"}}
```

`source_removed`:

```json
{"id": "src_7"}
```

- `id`: the daemon's id for the stream, kept while the stream lives and never reused. Ids hold for
  one daemon run; after a reconnect, read `welcome.sources` again. When `pw-dump` exits, every
  source gets `source_removed`; once it runs again, the streams come back as new sources.
- `name`: the application's name, else the stream's node name. `binary`: the program, `null` when
  PipeWire doesn't say. `media_name`: what the stream says it plays (a title, `Playback`); `null`
  when unset.
- On macOS, `name` is the app's name, else its bundle id, else the process name. `binary` is the
  bundle id, `null` for a process without one. `media_name` is always `null`. A source goes when
  its process stops its output, so a paused player goes, and comes back as a new source when it
  plays again. Output running doesn't mean audible: an app that plays silence is listed.
- One application can have several sources (a browser can have one per tab that plays). A source
  goes when the app closes its stream, which some apps do when they pause; playing again is then
  a new source.
- Without `pw-dump`, or while it isn't running, `sources` is empty and no events come; the daemon
  tries again every 30 s. On macOS, while Core Audio's list can't be read, `sources` is empty.
  Elsewhere, `sources` is always empty.

## Observers

A client with `observe_all: true` gets every `session_started`, `partial`, `segment`, `progress`,
`session_suspended`, `session_resumed`, `session_ended` and `session_error`, for every client's
session (sent with only the options that
session asked for), and the VAD events during sessions. It maps `session_id` to a client through
the client fields on `session_started`. Copies sent to observers have no `reply_to`, since ids
belong to the connection that sent the command. An observer gets its own session's events once,
as the holder, with `reply_to` on the replies to its commands.

Observers see other clients' transcripts. That is the normal desktop model: the socket is reachable
by the user's own processes, which can already read the user's files, clipboard and windows, so the
trust boundary is the user account, and an observer opens no hole that wasn't there. A user who
wants no observers sets `server.observers = false` in `config.toml`; `hello` with `observe_all`
then gets `observers_disabled`.
