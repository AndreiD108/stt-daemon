# stt-daemon

Your mic, as a service.

stt-daemon is a small local service that owns the microphone and turns speech
into text for any app on your machine. Apps connect over a Unix socket, get the
text as you speak, and never touch the mic themselves. It also transcribes what
your machine plays, a call or a video, and audio files.

It handles what typically goes wrong with voice input: cut-off first words,
your speakers in the transcript, sentences splitting at every pause,
connections that drop mid-sentence. No recording code, no model or API to
manage.

Transcribe in the cloud with ElevenLabs Scribe, locally with Parakeet, or with
a backend you add yourself.

![stt-daemon listen prints what it hears, word by word](docs/media/hero.gif)

## Build your own client

The protocol is a few JSON lines over a socket, so a client gets written in
minutes, in any language. Say hello, start a session, read what comes back:

- **Streaming.** Final segments as they commit, and on ElevenLabs partial text
  while you speak.
- **Word times.** Every word with its start and end, on the capture clock,
  pauses included.
- **Voice activity.** `speech_started` and `speech_ended` events. A client can
  start a session from a detected speech, and the first word is in it.
- **Key terms.** Names and jargon the backend should expect.
- **Markers.** Drop a marker at a moment and get it back between the words said then.

![A client starts a session with key terms and gets VAD events, partials, and a segment with word times](docs/media/client.gif)

Everything is in [docs/protocol.md](docs/protocol.md). Point your agent at it
and you're good to go.

## How it works

The daemon runs as a user service and holds the mic, with recent audio in a
buffer, so nothing gets chopped off the start of what you say. You can turn
that off and have the mic open only when transcribing. On Linux the mic goes
through PipeWire's echo cancellation, so what your speakers play stays out of
what you dictate.

Apps connect over a Unix socket and ask for a session. One dictation runs at a
time; file and meeting transcriptions run next to it. Backends:

- **ElevenLabs Scribe v2 Realtime**, in the cloud. Needs an API key.
- **ElevenLabs Scribe v2**, for files: one upload, same key. Tells speakers apart.
- **Parakeet TDT 0.6B v3**, local. Runs on the CPU, and keeps most background
  voices out of what you dictate.
- **Your own**, as a plugin you [write yourself](#write-your-own-backend).

![The same sentence through ElevenLabs in the cloud and through Parakeet on the CPU](docs/media/backends.gif)

## Install

Copy and run it in your terminal:

```sh
uv tool install git+https://github.com/AndreiD108/stt-daemon && stt-daemon service install
```

The service starts at login.

## Set up a backend

```sh
stt-daemon key set elevenlabs          # cloud: asks for your API key
stt-daemon backend install parakeet    # local: downloads the model, 487 MB
```

One is enough. The first one you set up becomes the default;
`stt-daemon backend set-default parakeet` changes it, and
`stt-daemon backend remove parakeet` deletes the model again. `stt-daemon backend list` shows
each backend, whether it is ready, and the default. Keys live in
`~/.config/stt-daemon/credentials.toml`, readable by you only. Then:

```sh
stt-daemon listen --words
```

Speak, press Enter, and read what the daemon heard.

## Transcribe a file

```sh
stt-daemon transcribe talk.m4a notes.mp3           # the text of each, a blank line between
stt-daemon transcribe --json --backend parakeet talk.m4a   # text, words and segments as JSON
stt-daemon transcribe --backend elevenlabs-batch --diarize interview.mp3   # each line names its speaker
```

Any audio or video file ffmpeg reads works.

## Transcribe a meeting

A client can transcribe what your machine plays, a call or a video, with
`start_system_session`. With `with_mic`, your mic is a second track, so both sides of a call come
out as one transcript, each line marked `[system]` or `[mic]`. It runs next to your dictation,
and the mic track has echo cancellation, so speakers are fine.

![A note-taking client picks the call app, and both sides of the call come out as one transcript](docs/media/meeting.gif)

To take one application's audio only, say the call app and not the music next to it, a client
picks it from the sources the daemon lists (every app that plays audio, with its name) and starts
the session with that app's id. When the app stops playing and starts again, the session picks
it up again on its own.

## Write your own backend

A backend plugin is one Python file in `~/.config/stt-daemon/backends/`. It implements one call,
`transcribe`: 16 kHz audio in, text out, with or without word times. The daemon does the rest:
dictation, files and meetings, markers, estimated word times, keys and the default. Clients pick
it by its name, and `stt-daemon backend list` shows whether it loaded. The contract and a complete
example for any OpenAI-compatible endpoint are in [docs/plugins.md](docs/plugins.md).

## Update

stt-daemon never updates itself or checks for updates. To update:

```sh
uv tool upgrade stt-daemon
stt-daemon service install
```

## Remove

```sh
stt-daemon service remove
uv tool uninstall stt-daemon
```

This keeps your settings, keys, and downloaded models in
`~/.config/stt-daemon` and `~/.local/share/stt-daemon`; delete those too if
you want them gone.

## License

MIT.
