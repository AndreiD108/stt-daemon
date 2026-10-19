# stt-daemon protocol

One daemon owns the mic; many clients connect; one mic session runs at a time. The mic slot (the
slot) is the right to run that session: a client takes it with `start_session` and gives it back
when its session ends. Each backend also has slots: a session takes one on its backend, and a
backend runs at most `max_sessions` sessions at once (config `backends.<name>.max_sessions`,
default 2). Capacity counts sessions, not clients. Every session today is a mic session (`kind:
"mic"`), so it takes the mic slot and one backend slot.

A minimal client: connect, send `hello`, read `welcome`, send `start_session`, read `segment`
events, send `stop_session`, read `session_ended`. Error codes are in [Error codes](#error-codes).

## Conventions

- Transport: JSONL (one JSON object per line, UTF-8, `\n`) over a Unix socket. Socket file
  permissions are the access control. Path: `$XDG_RUNTIME_DIR/stt-daemon/stt.sock` on Linux,
  `$TMPDIR/stt-daemon/stt.sock` on macOS; `STT_DAEMON_SOCKET` overrides it for the daemon and
  for clients. A line longer than 1 MiB gets `bad_request` and the daemon closes the connection.
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
  wall-clock change can't move words.
- The first client message must be `hello`. Any other message first gets `error {code:
  "protocol"}`, and a line that isn't a valid message gets `bad_request`; either way the daemon
  closes the connection.
- A client holds at most one session, so session commands act on the client's own session and
  carry no id. The daemon generates a `session_id` and puts it on every session event, for logs.
- Commands from one connection run one at a time, in order; `ping` is answered at once.
- Fields shown in an event's example are always present unless the text says when they are
  missing; a field that can be `null` says so.

## Client → daemon

### `hello`

```json
{"protocol": 1, "label": "Scribbly", "observe_all": false, "vad_events": true}
```

- `label` (optional): shown to other clients in `slot`; also scopes `context_key`. `""` is no
  label.
- `observe_all` (optional, default false): receive every session event of every client, like a log.
  See [Observers](#observers). Refused with `observers_disabled` when the daemon's config has
  `server.observers = false`.
- `vad_events` (optional, default false): receive `speech_started` and `speech_ended`; see
  [VAD events](#vad-events).

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

Hints (`keyterms`, `previous_text`, `context_key`) are dropped when the backend can't use them; the
backend's entry in `welcome` says which it takes. Features (`partials`, `word_timestamps`) fail the
start with `unsupported` when the backend lacks them.

Checks run in this order, before either slot changes: field types (`bad_request`), `speech_id`
(`speech_expired`), the mic slot (`session_active`, `busy`), the backend, language and features,
then room on the backend (`busy`). So a request for a missing backend sent while the mic slot is
`busy` gets `busy`. An evictable holder's session doesn't count against its backend's capacity:
a start that would evict it is refused only when the backend is full without it, and then nothing
is evicted.

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

### `add_marker`

```json
{"marker": "⟦A⟧", "at": 1760000012345}
```

- `marker`: the text to place, non-blank.
- `at` (optional, epoch ms UTC): when the event happened on the client. Missing: the time the
  message arrived. Earlier than `started_at`: the marker is refused (`out_of_range`). Later than
  the arrival time: the arrival time is used.

Placement. No early commit; every accepted marker shows up in exactly one segment, unless the
session is cancelled (`cancel_session`, or the client disconnects), which drops markers not yet
placed:

- It goes into the first segment not yet committed when the marker arrives whose audio reaches
  `at`; committed text never changes. (A commit can be on its way when the marker arrives; a
  marker from after that commit's cut waits for the next segment.)
- Inside that segment it goes into the gap between words nearest `at`: from word times when the
  backend has them, else from character lengths across the segment's span. When `at` is before the
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

### `cancel_session`

```json
{}
```

Drops what is left. Reply: `session_ended` (reason `cancelled`), or `error {code: "no_session"}`,
as for `stop_session`.

### `reload`

```json
{}
```

Re-reads config and credentials and rescans installed backends. Sent by the `stt-daemon` CLI after
`key set`, `backend install`, `backend remove` and `backend set-default`. A running session keeps
its backend, even one no longer set up; a removed Parakeet model leaves memory when its last
session ends. New `backends.parakeet.num_threads` and `hotwords_score` values apply once no
Parakeet session is running. A new `max_sessions` applies to the next start; a backend over its new
capacity keeps its sessions until they end. When backends or capacities change, a `slot` event
follows the `backends` event. Changes to `[audio]` and `[slot]` take effect when the daemon
restarts. `[vad]` changes apply to
sessions started after the reload, but VAD events keep the old settings until a restart. Reply: `backends`; every
other client gets the same `backends` event. Or `error {code: "config"}` when the config or
credentials file is invalid (bad TOML, a bad value, credentials not mode 600); the daemon keeps its
last setup.

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
      "hints": ["keyterms", "previous_text"]
    }
  ],
  "default_backend": "elevenlabs",
  "mic_open": true,
  "slot": {"state": "idle", "backends": {"elevenlabs": {"max_sessions": 2, "sessions": []}}}
}
```

- `backends` lists only the backends set up on this machine; empty when none is. Today:
  `elevenlabs` (an API key is set) and `parakeet` (local model installed; `languages: ["auto"]`,
  `partials: false`, `hints: ["keyterms"]`).
- `default_backend`: the user's global default, `null` when there is no backend; the first backend
  set up, changed with `stt-daemon backend set-default <backend>`.
- `slot` has the `slot` event's shape. Unlike `slot` events, it can show a holder whose session is
  still starting (`since: null`), and that session in `backends`.
- `mic_open`: the daemon keeps the mic open between sessions (config `audio.keep_mic_open`,
  default true). When false, the mic opens at each start: VAD events flow only during sessions,
  sessions start at the command, and each start waits for the device.

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
      "sessions": [{"session_id": "…", "kind": "mic", "pid": 41230, "client_label": "Scribbly"}]
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

- `kind`: `mic`, a session on the daemon's mic, the only kind today.
- `started_at`: capture time of the session's first sample.
- `previous_text`: the context hint the session uses, from the command or the `context_key` store,
  before the backend trims it; `null` when none.
- `client_label`, `session_label`, `language`: `null` when not set.
- The client fields let observers tell sessions apart without waiting for `slot`.
- Parakeet sends `session_started` without waiting for its model to load (about 1 s when it is
  not in memory; it leaves memory 10 minutes after the last Parakeet session). The audio is kept
  and transcribed once the model is in; a model that fails to load ends the session with
  `model_error`.

### `partial`

```json
{"session_id": "…", "text": "move this up next to"}
```

The backend's current guess for the text after the last segment. Each `partial` replaces the
previous one; a `segment` replaces the partial. Only with `partials: true`.

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
- `language`: the detected language, when the session used `"auto"` and the backend reports one
  (ElevenLabs does, Parakeet doesn't); missing otherwise.
- `markers`: those placed in this segment; missing when there are none. `at_ms` is the
  marker's time (from `at` or arrival).
- `words`, and `after_word` on each marker: only with `word_timestamps: true`. `after_word` is
  the index in `words` of the word the marker follows, `-1` before the first.
- When segments commit is up to the backend. Timed commits are checked every 5 s and need speech
  sent since the last commit, so they can cut mid-speech. ElevenLabs: 25 s after the last commit
  (ElevenLabs advises 20–30 s), before each connection rollover (about every 13 minutes), and at
  stop. Parakeet, which transcribes after the fact: at each pause the VAD reports, once 30 s of
  audio was sent with no pause (config `backends.parakeet.max_chunk_seconds`), and at stop.
  Parakeet sessions share one model and take turns to transcribe, so a commit can wait for
  another session's.

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
- `final_text`: all segments joined, markers inline. Kept on `evicted`.
- `evicted_by`: only with reason `evicted`; `pid` and `client_label` can be `null`.
- `client_gone`: the holder disconnected; the session is cancelled. Only observers see it.

### `session_error`

```json
{"session_id": "…", "code": "network", "message": "…", "fatal": true}
```

A backend failure during a session, or `internal` when the daemon fails. The daemon reopens a
stalled mic by itself and doesn't report it here.

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
| `busy` | `start_session` | A slot is full; `slot` has every slot. `full_slot`: `mic` (another client holds the mic slot; wait for a `slot` event with `idle` or `evictable`) or `backend` (`backend` runs `max_sessions` sessions; wait for room or use another backend). `holder` (`pid`, `client_label`, `kind`; the first two can be `null`) is the session in the way: the mic holder, or the oldest on the backend. |
| `session_active` | `start_session` | This client already holds a session; `slot` has it. Stop it first. |
| `speech_expired` | `start_session` | The `speech_id` is unknown or its start left the buffer. Start without it. |
| `no_backend` | `start_session` | No backend set up; the message names `stt-daemon key set elevenlabs` and `stt-daemon backend install parakeet`. |
| `unknown_backend` | `start_session` | `backend` isn't set up here; pick one from `welcome.backends`. |
| `unsupported_language` | `start_session` | The backend lacks that language (or the config's default language). |
| `unsupported` | `start_session` | The backend lacks a requested feature. Start without it. |
| `mic_unavailable` | `start_session` | The mic didn't open. |
| `no_session` | `add_marker`, `stop_session`, `cancel_session` | No session, or it is already ending. |
| `session_stopping` | `add_marker` | The session is ending. |
| `invalid` | `add_marker` | `marker` missing or blank; `field` names a field of the wrong type. |
| `out_of_range` | `add_marker` | `at` is before `started_at`. |
| `config` | `reload` | Config or credentials invalid; the last setup stays. |
| `internal` | any | The daemon failed; its log says why. |

Backend codes come as the reply to `start_session` when the backend can't connect, and in
`session_error` during a session. For a code the daemon doesn't retry, the session ends at once
(`fatal: true`), and starting again gives the same error until the user fixes the cause.

| Code | Retried | Meaning |
| --- | --- | --- |
| `auth` | no | The API key was refused; the user sets a new one with `stt-daemon key set`. |
| `quota` | no | The account is out of quota. |
| `invalid_request` | no | The backend rejected the request or the audio. |
| `model_error` | no | The local model didn't load; the message names the directory to delete before reinstalling. |
| `rate_limited` | yes, during a session | Throttled. At start (HTTP 429) it isn't retried; start again later. |
| `network` | yes | The connection failed or timed out. |
| `backend_error` | yes (ElevenLabs), no (Parakeet decode) | The backend failed. |
| `session_limit`, `idle_drop` | yes | The backend ended a connection (time limit, no audio). |

"Retried" means the daemon reconnects; it still gives up after 3 failed reconnects, with that
code and `fatal: true`. Act on `fatal`, not on the code.

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

## Observers

A client with `observe_all: true` gets every `session_started`, `partial`, `segment`,
`session_ended` and `session_error`, for every client's session (sent with only the options that
session asked for), and the VAD events during sessions. It maps `session_id` to a client through
the client fields on `session_started`. Copies sent to observers have no `reply_to`, since ids
belong to the connection that sent the command. An observer gets its own session's events once,
as the holder, with `reply_to` on the replies to its commands.

Observers see other clients' transcripts. That is the normal desktop model: the socket is reachable
by the user's own processes, which can already read the user's files, clipboard and windows, so the
trust boundary is the user account, and an observer opens no hole that wasn't there. A user who
wants no observers sets `server.observers = false` in `config.toml`; `hello` with `observe_all`
then gets `observers_disabled`.
