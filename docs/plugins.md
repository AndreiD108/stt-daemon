# Backend plugins

A plugin adds a transcription backend to the daemon: one Python file in
`~/.config/stt-daemon/backends/` (`$XDG_CONFIG_HOME/stt-daemon/backends/`, on macOS too). The
daemon imports every `*.py` there at startup and at each `reload`. A plugin implements one call,
`transcribe`, and gets everything else from the daemon: dictation, `transcribe` and system
sessions, recovery, markers, word times, keys, capacity and the default backend.

## The contract

Each file defines exactly one subclass of `stt_daemon.plugin.Backend`. Everything a plugin uses
comes from `stt_daemon.plugin`; nothing else in the package is part of the contract.

```python
class Backend:
    name: ClassVar[str]  # the backend's name: lowercase letters, digits, "-"
    capabilities: ClassVar[Capabilities] = Capabilities()
    needs_key: ClassVar[bool] = False  # True: not set up until `key set <name>`

    def setup(self, key: str | None, config: Mapping[str, Any]) -> None: ...
    async def transcribe(self, pcm: bytes, options: Options) -> Result: ...
    async def install(self, data_dir: Path) -> None: ...  # optional
    def available(self) -> bool: ...  # optional; default True
```

- `name` must start with a letter and must not be a built-in backend's name (`elevenlabs`,
  `elevenlabs-batch`, `parakeet`). Clients pick the backend by this name, as `backend` in
  `start_session`, `transcribe` and `start_system_session`.
- `setup` runs when the daemon loads the plugin and at each `reload`. `key` is the plugin's key
  from `credentials.toml` (`[<name>] api_key`, or the env var `<NAME>_API_KEY` with `-` as `_`);
  `config` is the `[backends.<name>]` table from `config.toml`, `{}` without one. Keep it quick:
  load models on the first `transcribe`, not here.
- `transcribe` gets the audio of one chunk, 16 kHz mono signed 16-bit little-endian PCM, and
  returns its text. The daemon cuts chunks at pauses in speech, and at
  `backends.<name>.max_chunk_seconds` (default 30, at least 5) of speech with no pause. It calls
  `transcribe` once per chunk, in order, one call at a time for the plugin; a dictation's chunks
  go before other sessions' waiting ones. A chunk under 0.1 s gets empty text without a call. It runs on the daemon's event loop, so blocking work
  (a local model, a blocking HTTP client) belongs in `asyncio.to_thread`. A call that raises, or
  takes longer than 120 s, ends the session with `session_error {code: "backend_error"}` and the
  exception's text as the message, with the plugin's key replaced by `[key]`. A call past its
  timeout is not cancelled; the next call waits until it ends.
- `install` runs for `stt-daemon backend install <name>`, with `data_dir` set to
  `~/.local/share/stt-daemon/models/<name>/` (created first). `setup` runs before it.
- `available` returning False marks the backend `not installed`: it is left out of `backends`
  and selecting it gets `unknown_backend`.

`Capabilities`:

| Field | Default | Meaning |
| --- | --- | --- |
| `languages` | `("auto",)` | codes a client may send as `language`; `"auto"`: the backend detects it |
| `word_times` | `False` | `Result.words` carry times |
| `diarize` | `False` | takes `diarize` and names each word's speaker |
| `keyterms` | `False` | takes the client's key terms |
| `previous_text` | `False` | takes the text before this session as context |

The daemon passes on only what the capabilities allow. A client asking for `diarize` on a plugin
without it gets `unsupported_option`, and a language outside `languages` gets
`unsupported_language`; key terms and `previous_text` are hints, dropped when the plugin does not
take them. Plugins have no partials.

`Options`: `language` (None when the backend should detect it), `keyterms`, `previous_text`,
`diarize`, `num_speakers`.

`Result`: `text`, and `words`, each a `Word` with `text`, `start` and `end` (seconds from the
start of the chunk), and `speaker`. Give `text`, `words` or both; with words only, the text is
the words joined by spaces. When any word has no times, or there are no words, the daemon
estimates each word's times from the chunk's audio, so clients asking for `word_timestamps` get
words either way; speakers given on words without times stay on those words.

## An example

A backend for any OpenAI-compatible `/audio/transcriptions` endpoint, in
`~/.config/stt-daemon/backends/openai_transcriptions.py`:

```python
"""An stt-daemon backend for any OpenAI-compatible `/audio/transcriptions` endpoint."""

import io
import wave
from collections.abc import Mapping
from typing import Any

import httpx

from stt_daemon.plugin import Backend, Capabilities, Options, Result

DEFAULT_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"


class OpenAITranscriptions(Backend):
    name = "openai"
    needs_key = True
    # Text only: the daemon estimates word times from the audio.
    capabilities = Capabilities(languages=("auto", "en", "de", "fr", "es"), previous_text=True)

    def setup(self, key: str | None, config: Mapping[str, Any]) -> None:
        self.key = key
        self.endpoint = str(config.get("endpoint", DEFAULT_ENDPOINT))
        self.model = str(config.get("model", "whisper-1"))

    async def transcribe(self, pcm: bytes, options: Options) -> Result:
        fields = {"model": self.model, "response_format": "json"}
        if options.language:
            fields["language"] = options.language
        if options.previous_text:
            fields["prompt"] = options.previous_text
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.key}"},
                data=fields,
                files={"file": ("audio.wav", to_wav(pcm), "audio/wav")},
            )
        if response.is_error:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        return Result(text=response.json()["text"])


def to_wav(pcm: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    return out.getvalue()
```

Its settings and key:

```toml
# ~/.config/stt-daemon/config.toml
[backends.openai]
endpoint = "https://api.openai.com/v1/audio/transcriptions"
model = "whisper-1"
max_sessions = 2
```

```sh
stt-daemon key set openai              # stores [openai] in credentials.toml, reloads the daemon
stt-daemon backend set-default openai  # optional
stt-daemon listen --backend openai
```

## Dependencies

A plugin imports from the daemon's own environment. Add what it needs to that environment:

```sh
uv tool install stt-daemon --with <package>
```

`httpx`, which the example uses, is already there.

## Status

```sh
stt-daemon backend list
```

prints each backend, built-in and plugin, with its status: `ready`, `no key`, `not installed`, or
`failed:` and the first line of the error, and marks the default. A file that does not import,
defines no `Backend` subclass or more than one, uses a built-in name, or raises in `setup` or
`available` is `failed`; the daemon logs it once with its traceback and goes on without it. The
list comes from the running daemon (`backends_status` on the socket), else from the files.

The daemon never writes into the plugin directory. A plugin runs in the daemon's process with
your permissions, like any code you run.
