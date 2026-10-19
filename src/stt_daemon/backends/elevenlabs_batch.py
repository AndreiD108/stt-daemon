"""ElevenLabs speech-to-text over its file upload API: one request per `transcribe` session.

The bytes go as the client sent them (any container ElevenLabs reads); headerless PCM goes as
`pcm_s16le_16` when it is 16 kHz mono, else inside a WAV header. The answer is split into
`Committed` runs of 20 to 30 s and at speaker changes, so no segment grows with the file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from typing import Any, Self, cast

import httpx

from ..config import Config, ElevenLabsBatchConfig, ElevenLabsConfig
from ..credentials import Credentials
from .base import (
    BackendUnavailable,
    Capabilities,
    CommitPolicy,
    Committed,
    FileAudio,
    FileBackend,
    OpenFailed,
    OpenOptions,
    Word,
    merge_keyterms,
)
from .elevenlabs_wire import ISO3_TO_LANGUAGE, LANGUAGES, http_failure

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
RETRY_AFTER_S = 2.0
SEGMENT_MIN_S = 20.0
SEGMENT_MAX_S = 30.0
_SENTENCE_ENDS = (".", "?", "!", "…")
_MAX_KEYTERMS = 1000
_MAX_KEYTERM_CHARS = 49
_MAX_KEYTERM_WORDS = 5
_KEYTERM_BANNED = frozenset("<>{}[]\\")

CAPABILITIES = Capabilities(
    languages=("auto", *LANGUAGES),
    word_timestamps=True,
    partials=False,
    hints=frozenset({"keyterms"}),
    max_connection_s=None,
    commit_policy=CommitPolicy(),
    paced_replay=False,
    diarize=True,
)


class ElevenLabsBatchBackend(FileBackend):
    name = "elevenlabs-batch"
    capabilities = CAPABILITIES
    default_for = "elevenlabs"

    def __init__(
        self, config: ElevenLabsBatchConfig, shared: ElevenLabsConfig, api_key: str
    ) -> None:
        self._config = config
        self._shared = shared
        self._api_key = api_key

    def __repr__(self) -> str:
        return f"ElevenLabsBatchBackend(model_id={self._config.model_id!r})"

    @property
    def max_sessions(self) -> int:
        return self._config.max_sessions

    @property
    def default_language(self) -> str:
        return self._shared.language

    @classmethod
    def create(cls, config: Config, credentials: Credentials) -> Self:
        key = credentials.get("elevenlabs")
        if not key:
            raise BackendUnavailable(
                "no_key", "no ElevenLabs API key; set one with `stt-daemon key set elevenlabs`"
            )
        return cls(config.backends.elevenlabs_batch, config.backends.elevenlabs, key)

    async def transcribe_file(self, audio: FileAudio, options: OpenOptions) -> list[Committed]:
        fields = self._fields(options)
        content, file_format = _file_body(audio)
        if file_format is not None:
            fields.append(("file_format", file_format))
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                return await self._post(client, fields, content)
            except OpenFailed as exc:
                if not exc.retryable:
                    raise
                logger.warning("ElevenLabs upload failed (%s); retrying once", exc.message)
            await asyncio.sleep(RETRY_AFTER_S)
            return await self._post(client, fields, content)

    async def _post(
        self, client: httpx.AsyncClient, fields: list[tuple[str, str]], content: bytes
    ) -> list[Committed]:
        try:
            response = await client.post(
                self._config.endpoint,
                headers={"xi-api-key": self._api_key},
                data=_multi(fields),
                files={"file": ("audio", content, "application/octet-stream")},
            )
        except httpx.TimeoutException as exc:
            raise OpenFailed(
                "network", f"ElevenLabs upload timed out ({type(exc).__name__})", retryable=True
            ) from None
        except httpx.TransportError as exc:
            raise OpenFailed(
                "network", f"ElevenLabs upload failed: {type(exc).__name__}", retryable=True
            ) from None
        if response.status_code != 200:
            raise http_failure(response.status_code, response.content, self._api_key)
        try:
            data: Any = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise OpenFailed(
                "backend_error",
                "ElevenLabs answered with something that isn't JSON",
                retryable=False,
            ) from None
        if not isinstance(data, dict):
            raise OpenFailed(
                "backend_error", "ElevenLabs answered with no transcript", retryable=False
            )
        return committed_runs(cast(dict[str, Any], data))

    def _fields(self, options: OpenOptions) -> list[tuple[str, str]]:
        fields = [
            ("model_id", self._config.model_id),
            ("timestamps_granularity", "word"),
            ("tag_audio_events", "false"),
            ("diarize", "true" if options.diarize else "false"),
        ]
        if options.diarize and options.num_speakers is not None:
            fields.append(("num_speakers", str(options.num_speakers)))
        language = options.language or self._shared.language
        if language != "auto":
            fields.append(("language_code", language))
        fields.extend(("keyterms", term) for term in self._keyterms(options.keyterms))
        return fields

    def _keyterms(self, extra: tuple[str, ...]) -> list[str]:
        terms = merge_keyterms(self._shared.keyterms, extra)
        bad = [
            t
            for t in terms
            if len(t) > _MAX_KEYTERM_CHARS
            or len(t.split()) > _MAX_KEYTERM_WORDS
            or _KEYTERM_BANNED.intersection(t)
        ]
        if bad:
            logger.warning("dropping keyterms ElevenLabs refuses: %s", bad)
            terms = [t for t in terms if t not in bad]
        if len(terms) > _MAX_KEYTERMS:
            logger.warning(
                "dropping %d keyterms past the first %d", len(terms) - _MAX_KEYTERMS, _MAX_KEYTERMS
            )
            terms = terms[:_MAX_KEYTERMS]
        return terms


def _multi(fields: list[tuple[str, str]]) -> dict[str, str | list[str]]:
    """Repeated names (keyterms) become lists, which httpx sends as repeated fields."""
    out: dict[str, str | list[str]] = {}
    for name, value in fields:
        if name == "keyterms":
            cast(list[str], out.setdefault(name, [])).append(value)
        else:
            out[name] = value
    return out


def _file_body(audio: FileAudio) -> tuple[bytes, str | None]:
    if audio.pcm is None:
        return audio.data, None
    rate, channels = audio.pcm
    if (rate, channels) == (16000, 1):
        return audio.data, "pcm_s16le_16"
    return wav_header(len(audio.data), rate, channels) + audio.data, None


def wav_header(data_bytes: int, rate: int, channels: int) -> bytes:
    """A 44-byte RIFF header for s16le PCM."""
    block = 2 * channels
    return (
        struct.pack("<4sI4s", b"RIFF", 36 + data_bytes, b"WAVE")
        + struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, channels, rate, rate * block, block, 16)
        + struct.pack("<4sI", b"data", data_bytes)
    )


def committed_runs(data: dict[str, Any]) -> list[Committed]:
    """The answer as `Committed` runs of words by one speaker, each at most `SEGMENT_MAX_S`
    long and cut at a sentence end past `SEGMENT_MIN_S`. `spacing` entries join the text but
    carry no word; audio events are dropped."""
    raw: list[Any] = data.get("words") or []
    entries = [cast(dict[str, Any], e) for e in raw if isinstance(e, dict)]
    code = data.get("language_code")
    language = ISO3_TO_LANGUAGE.get(code, code) if isinstance(code, str) and code else None
    runs: list[tuple[list[str], list[Word]]] = []
    end_s = 0.0
    for entry in entries:
        kind, text = entry.get("type"), str(entry.get("text") or "")
        if isinstance(end := entry.get("end"), int | float):
            end_s = max(end_s, float(end))
        if kind == "spacing" and runs:
            runs[-1][0].append(text)
        if kind != "word" or not text.strip():
            continue
        word = _word(entry, text, runs)
        if not runs or _cut_before(runs[-1][1], word):
            runs.append(([], []))
        runs[-1][0].append(text)
        runs[-1][1].append(word)
    if len(runs) <= 1:
        words = tuple(runs[0][1]) if runs else ()
        return [Committed(0, 1, str(data.get("text") or "").strip(), language, words, end_s)]
    out: list[Committed] = []
    for i, (texts, words) in enumerate(runs, start=1):
        run_end = end_s if i == len(runs) else words[-1].end_s
        out.append(Committed(0, i, "".join(texts).strip(), language, tuple(words), run_end))
    return out


def _cut_before(run: list[Word], word: Word) -> bool:
    if run[-1].speaker != word.speaker or word.end_s - run[0].start_s > SEGMENT_MAX_S:
        return True
    return run[-1].end_s - run[0].start_s >= SEGMENT_MIN_S and run[-1].text.endswith(_SENTENCE_ENDS)


def _word(entry: dict[str, Any], text: str, runs: list[tuple[list[str], list[Word]]]) -> Word:
    last_end = runs[-1][1][-1].end_s if runs else 0.0
    start = entry.get("start")
    start_s = float(start) if isinstance(start, int | float) else last_end
    end = entry.get("end")
    end_s = float(end) if isinstance(end, int | float) else start_s
    speaker = entry.get("speaker_id")
    return Word(
        text.strip(), start_s, max(end_s, start_s), speaker if isinstance(speaker, str) else None
    )
