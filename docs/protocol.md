# stt-daemon protocol

One daemon owns the mic; many clients connect; one session runs at a time (the slot).

## Conventions

- Transport: JSONL (one JSON object per line, UTF-8, `\n`; at most 1 MiB, a longer line closes
  the connection) over a Unix socket. Socket file
  permissions are the access control. Path: `$XDG_RUNTIME_DIR/stt-daemon/stt.sock` on Linux,
  `$TMPDIR/stt-daemon/stt.sock` on macOS; `STT_DAEMON_SOCKET` overrides it for the daemon and
  for clients.
- Envelope, both directions:

  ```json
  {"type": "<name>", "id": "<client-chosen>", "reply_to": "<id>", "ts": 1760000000000, "data": {}}
  ```

  `id` on commands; `reply_to` on the reply to a command; `ts` is the send time. Unsolicited
  events have no `reply_to`.
- Wall-clock times (`ts`, `*_at`): Unix epoch ms, UTC.
- Session times (`*_ms`): ms since the session's `started_at`, on the real capture clock. The daemon
  cuts pauses before it sends audio to the backend and reconnects the backend on rollover; it maps
  the backend's times back, so session times include the pauses and run without seams. The daemon
  counts on a monotonic clock and converts to UTC once, at `started_at`, so a wall-clock change
  can't move words.
- The first client message must be `hello`. Anything else first gets `error {code: "protocol"}`
  and the daemon closes the connection.
- A client holds at most one session, so session commands act on the client's own session and
  carry no id. The daemon generates a `session_id` and puts it on every session event, for logs.

## Client → daemon

### `hello`

```json
{"protocol": 1, "label": "Scribbly", "observe_all": false, "vad_events": true}
```

- `label` (optional): shown to other clients in `slot`; also scopes `context_key`.
- `observe_all` (optional, default false): receive every session event of every client, like a log.
- `vad_events` (optional, default false): receive `speech_started` and `speech_ended`; see
  [VAD events](#vad-events).

Reply: `welcome`, or `error {code: "protocol_unsupported", supported: [1]}` and close.

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
  a short pad (config `vad.pause_keep_ms`, 100 ms), not at the command. Without it, the session's audio starts 1 s before the command arrived when the
  mic is kept open, and when the mic is ready otherwise. Either way that point is `started_at`.
- `backend`: default `default_backend` from `welcome`.
- `language`: a code from the backend's list, or `"auto"` (detect). Default: the backend's
  `language` in config (`backends.elevenlabs.language`, itself `"auto"` unless changed); for a
  backend with no such setting, `"auto"` when it lists it.
- `keyterms`: bias words, merged after the config defaults.
- `previous_text`: context hint for the backend (trimmed to its limit, 50 chars for EL).
- `context_key`: when `previous_text` is missing, the daemon uses the last text stored under
  (client `label`, `context_key`). It stores a session's `final_text` there, with markers removed,
  when the session ends `stopped` or `evicted` with non-empty text; other endings leave the stored
  text as it was. In memory only; lost on restart. Without a client `label`, the store lives only
  as long as the connection. Without `context_key`, nothing is looked up or stored.
- `partials`: send `partial` events. Default false.
- `word_timestamps`: put `words` on `segment` events, and `after_word` on their `markers`.
  Default false.

Hints (`keyterms`, `previous_text`, `context_key`) are dropped when the backend can't use them; the
backend's entry in `welcome` says which it takes. Features (`language`, `partials`,
`word_timestamps`) fail the start with `unsupported` when the backend lacks them.

When the slot is `evictable`, the daemon takes it over in this order:

1. Check the whole request first. A bad request fails and the holder's session goes on.
2. Hold the slot for this client; any other start now gets `busy`. Copy this session's audio
   out of the ring now (from `speech_id`'s start, or 1 s before the command), so it can't expire
   during the wait below; audio captured during the wait also goes to this session. If this client
   disconnects or its start fails after this point, the copy is dropped and the slot goes idle.
3. Stop the holder's session as `stop_session` would (commit what is left, bounded wait). The
   holder gets `session_ended` (reason `evicted`) with everything committed by then; text the
   backend can't flush in time is lost.
4. Start this session: `slot`, then `session_started`.

When the slot is `busy`, the start fails; clients that follow `slot` events don't send it then.

Reply: `session_started`, or `error` with code `busy` (data has the `slot`), `session_active`
(this client already holds one), `speech_expired` (the speech start left the buffer),
`no_backend` (none set up; the message names `stt-daemon key set elevenlabs` and
`stt-daemon backend install parakeet`), `unknown_backend`, `unsupported_language`, `unsupported`,
`bad_request`, `mic_unavailable` (the mic didn't open), or the backend's code when it can't
connect (`auth`, `quota`, `network`, …, as in `session_error`).

### `add_marker`

```json
{"marker": "⟦A⟧", "at": 1760000012345}
```

- `at` (optional, epoch ms UTC): when the event happened on the client. Missing: the time the
  message arrived.

- `at` earlier than `started_at`: the marker is refused (`out_of_range`). `at` later than the
  arrival time: the arrival time is used.

Placement. No early commit; every accepted marker shows up in exactly one segment:

- It goes into the first segment not yet committed when the marker arrives whose audio reaches
  `at`; committed text never changes. (A commit can be on its way when the marker arrives; a
  marker from after that commit's cut waits for the next segment.)
- Inside that segment it goes into the gap between words nearest `at`: from word times when the
  backend has them, else from character lengths across the segment's span. When `at` is before the
  segment's first word (a marker that arrived late, or in a pause), it goes first (`after_word: -1`);
  `at_ms` keeps the true time.
- Markers still waiting at stop, including in a session with no speech at all, come in a last
  segment, with empty `text` if nothing was said.

Reply: `marker_accepted {marker, at_ms}`, or `error` (`no_session`, `session_stopping`, `invalid`,
`out_of_range`).

### `stop_session`

```json
{}
```

Commits what is left. Reply: `session_ended` (reason `stopped`), or `error {code: "no_session"}`
(for example, the session was evicted).

### `cancel_session`

```json
{}
```

Drops what is left. Reply: `session_ended` (reason `cancelled`), or `error {code: "no_session"}`.

### `reload`

```json
{}
```

Re-reads config and credentials and rescans installed backends. Sent by the `stt-daemon` CLI after
`key set`, `backend install` and `backend set-default`. A running session keeps its backend;
changes to `[audio]` and `[slot]` take effect when the daemon restarts. Reply: `backends`; every
other client gets the same `backends` event. Or `error {code: "config"}` when the config or
credentials file can't be read; the daemon keeps its last setup.

### `ping`

Reply: `pong {}`. Answered at once; other commands from one connection run one at a time, in order
(a `stop_session` sent during a slow start waits for the start).

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
  "slot": {}
}
```

`backends` lists only the backends set up on this machine; empty when none is. `default_backend`:
the user's global default (`null` when there is no backend); the first backend set up, changed with
`stt-daemon backend set-default <backend>`. `slot` has the
`slot` event's shape. `mic_open`: the daemon keeps the mic open between sessions
(config `audio.keep_mic_open`, default true). When false, the mic opens at each start: VAD events
flow only during sessions, sessions start at the command, and each start waits for the device.

### `backends`

```json
{"backends": [], "default_backend": "elevenlabs"}
```

Sent to every client after a `reload`. Same shapes as in `welcome`.

### `slot`

Sent to every client when the slot changes state or holder.

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
  }
}
```

- `state`: `idle` (no `holder`) | `busy` | `evictable`.
- `evictable`: no speech detected for 5 minutes (config). Once evictable, the session stays
  evictable until it ends; later speech doesn't turn it back to `busy`.
- `pid` and `cmdline` come from the socket peer, not from the client: on Linux `SO_PEERCRED` and
  `/proc/<pid>/cmdline`; on macOS `LOCAL_PEERPID` and the process table. `null` when the platform
  can't tell.
- `since`: the session's `started_at`. A `busy` error that comes while the holder's session is
  still starting has `since: null`; `slot` events come only once it has started.
- `last_speech_at`: as of this event; the slot doesn't send an event for each speech.

### `session_started`

```json
{
  "session_id": "…",
  "client_id": "c7",
  "client_label": "Scribbly",
  "session_label": "desks.html",
  "backend": "elevenlabs",
  "language": "auto",
  "started_at": 1760000000000,
  "previous_text": "…"
}
```

`started_at`: capture time of the session's first sample. `previous_text`: what the daemon sent,
if any. The client fields let observers tell sessions apart without waiting for `slot`.

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
  after the previous commit to the last sample before this one (pauses inside included).
- `text` holds any markers placed in it, inline; `words` never does.
- `language`: the detected language when the session used `"auto"`.
- `markers`: those placed in this segment; missing when there are none. `at_ms` is the
  marker's time (from `at` or arrival).
- `words`, and `after_word` on each marker: only with `word_timestamps: true`. `after_word` is
  the index in `words` of the word the marker follows, `-1` before the first.
- When segments commit is up to the backend. EL: 25 s of wall time after the last commit, once
  speech was sent (EL advises 20–30 s), and at stop. A backend that transcribes after the fact
  (local Whisper, OpenRouter) may commit at pauses or only at stop.

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
- `evicted_by`: only with reason `evicted`.
- `client_gone`: the holder disconnected; the session is cancelled. Only observers see it.

### `session_error`

```json
{"session_id": "…", "code": "network", "message": "…", "fatal": true}
```

A backend or capture failure during a session. `fatal: true` is followed by `session_ended`
(reason `error`) with every segment committed so far. `fatal: false`: the session goes on; for a
lost backend connection, the daemon reconnects and replays audio not covered by an acknowledgement.
It gives up, fatally, after 3 reconnects in a row without an acknowledged commit, or at once when
the failure cannot be retried (`auth`, `quota`). Stop and eviction still use their bounded wait.

When a provider returns text without saying where it cut the audio, the daemon holds that text
until the next known commit boundary. If the connection fails first, it replays the audio instead
of publishing that text twice. This can delay a segment, but does not discard the audio.

### `error`

```json
{"code": "busy", "message": "…"}
```

The reply to a command that failed. The message says what happened and what to do next. A field
with the wrong type gives `bad_request` (`invalid` for `add_marker`), with `field` naming it. A line
that isn't a JSON object with a `type` gets `bad_request` without `reply_to`; an unknown `type`
gets `bad_request`. `internal`: the daemon failed; its log says why. A connection whose unread
events pile up past 8 MiB is closed.

## VAD events

For clients that sent `vad_events: true`. The daemon's VAD runs on CPU the whole time the mic is
open, and it reports speech:

`speech_started`:

```json
{"speech_id": "s41", "at": 1760000012000, "expires_at": 1760000017000}
```

`speech_ended`:

```json
{"speech_id": "s41", "at": 1760000015200}
```

- While the slot is idle, they go to every client that asked. During a session, only to the
  holder, if it asked, and to observers. Observers get them with or without `vad_events`.
- A session that starts during speech already in progress: its holder gets that speech's
  `speech_ended` (with the `speech_id` sent earlier), then the events that follow.
- Both events arrive late: the VAD only switches once speech or silence has lasted a minimum
  (config). `at` is the true edge, found in the buffered audio; a session started from `speech_id`
  takes its audio from that true start, so the first syllable isn't cut.
- `expires_at`: until then, `start_session {speech_id}` can start a session from that speech
  start. The audio buffer (5 s, config `audio.buffer_ms`) sets it.
- A client can drive sessions from these alone: start on `speech_started`, stop on `speech_ended`
  or later. The daemon doesn't stop sessions on its own.

## Observers

A client with `observe_all: true` gets every `session_started`, `partial`, `segment`,
`session_ended`, `session_error`, `speech_started` and `speech_ended`, for every client's session (sent with
only the options that session asked for). It maps `session_id` to a client through the client
fields on `session_started`. Copies sent to observers have no `reply_to`, since ids belong to the
connection that sent the command.
