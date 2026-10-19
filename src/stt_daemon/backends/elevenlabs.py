"""ElevenLabs Scribe v2 Realtime over its WebSocket API, with manual commits.

Wire rules this code depends on (checked against the live API):
- Audio waits for `session_started`; every `input_audio_chunk` carries `sample_rate`.
- A commit with under 0.3 s of uncommitted audio gets `commit_throttled`, and the server closes
  the connection. Short tails are padded with silence; a commit with no audio is answered here.
- `committed_transcript_with_timestamps` follows every commit when timestamps are on, also for
  empty text, sometimes before the plain `committed_transcript`; it alone acknowledges a commit.
  EL echoes no commit id, so acknowledgements are matched in order.
- Word times count from the connection's first audio, padding included.
- `previous_text` goes only on the first audio chunk.
- Every error message is followed by the server closing the connection.
- The server commits by itself after ~36 s of uncommitted audio.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Self
from urllib.parse import urlencode

import numpy as np
import websockets

from ..audio.frames import Samples
from ..config import Config, ElevenLabsConfig
from ..credentials import Credentials
from .base import (
    SAMPLE_RATE,
    Backend,
    BackendUnavailable,
    Capabilities,
    Closed,
    CommitPolicy,
    Committed,
    Connection,
    EventSink,
    Failed,
    OpenFailed,
    OpenOptions,
    Partial,
    Reference,
    Word,
    merge_keyterms,
    trim_previous_text,
)
from .elevenlabs_wire import ERRORS, LANGUAGES, http_failure

logger = logging.getLogger(__name__)
# websockets logs the handshake headers, the API key among them, at DEBUG.
_ws_logger = logging.getLogger(f"{__name__}.websocket")
_ws_logger.setLevel(logging.INFO)

# Per attempt: handshake + session_started. Healthy connects measured P99 ~460 ms, max ~930 ms;
# the whole budget has to fit in the capture ring.
_ATTEMPT_TIMEOUTS_S = (0.6,) * 5
_MIN_COMMIT_AUDIO_S = 0.3
_PAD_TO_S = 0.35
_SELF_COMMIT_AFTER_S = 32.0
_MAX_CHUNK_SAMPLES = SAMPLE_RATE
_PREVIOUS_TEXT_LIMIT = 50
_MAX_KEYTERMS = 50
_MAX_KEYTERM_CHARS = 20
_CLOSE_TIMEOUT_S = 2.0

CAPABILITIES = Capabilities(
    languages=("auto", *LANGUAGES),
    word_timestamps=True,
    partials=True,
    hints=frozenset({"keyterms", "previous_text"}),
    max_connection_s=15 * 60.0,
    commit_policy=CommitPolicy(
        pause_steps=((0, 3.5), (8, 2.0), (14, 0.8), (19, 0.6), (23, 0.3)), max_chunk_s=25.0
    ),
)


class ElevenLabsBackend(Backend):
    name = "elevenlabs"
    takes_key = True
    capabilities = CAPABILITIES

    def __init__(self, config: ElevenLabsConfig, api_key: str) -> None:
        self._config = config
        self._api_key = api_key

    def __repr__(self) -> str:
        return f"ElevenLabsBackend(model_id={self._config.model_id!r})"

    @property
    def max_sessions(self) -> int:
        return self._config.max_sessions

    @property
    def default_language(self) -> str:
        return self._config.language

    @classmethod
    def create(cls, config: Config, credentials: Credentials) -> Self:
        key = credentials.get(cls.name)
        if not key:
            raise BackendUnavailable(
                "no_key", "no ElevenLabs API key; set one with `stt-daemon key set elevenlabs`"
            )
        return cls(config.backends.elevenlabs, key)

    async def open(
        self, options: OpenOptions, connection_id: int, sink: EventSink
    ) -> ElevenLabsConnection:
        language = options.language or self._config.language
        if language not in CAPABILITIES.languages:
            raise OpenFailed(
                "unsupported_language", f"ElevenLabs has no language {language!r}", retryable=False
            )
        url = self._url(language, self._keyterms(options.keyterms))
        previous_text = trim_previous_text(options.previous_text, _PREVIOUS_TEXT_LIMIT)
        for attempt, budget_s in enumerate(_ATTEMPT_TIMEOUTS_S, start=1):
            conn = ElevenLabsConnection(connection_id, sink, previous_text, options.partials)
            try:
                try:
                    await asyncio.wait_for(conn.connect(url, self._api_key), budget_s)
                except TimeoutError:
                    raise OpenFailed(
                        "network", f"connect timed out after {budget_s:.1f} s", retryable=True
                    ) from None
            except OpenFailed as exc:
                await conn.close()
                if not exc.retryable or attempt == len(_ATTEMPT_TIMEOUTS_S):
                    raise
                logger.warning(
                    "ElevenLabs connect attempt %d/%d failed (%s); retrying",
                    attempt,
                    len(_ATTEMPT_TIMEOUTS_S),
                    exc.message,
                )
                continue
            except BaseException:
                await asyncio.shield(conn.close())
                raise
            if attempt > 1:
                logger.info("ElevenLabs connect succeeded on attempt %d", attempt)
            return conn
        raise AssertionError("unreachable")

    def _keyterms(self, extra: tuple[str, ...]) -> list[str]:
        terms = merge_keyterms(self._config.keyterms, extra)
        too_long = [t for t in terms if len(t) > _MAX_KEYTERM_CHARS]
        if too_long:
            logger.warning("dropping keyterms over %d characters: %s", _MAX_KEYTERM_CHARS, too_long)
            terms = [t for t in terms if len(t) <= _MAX_KEYTERM_CHARS]
        if len(terms) > _MAX_KEYTERMS:
            logger.warning("dropping keyterms past the first %d: %s", _MAX_KEYTERMS, terms[50:])
            terms = terms[:_MAX_KEYTERMS]
        return terms

    def _url(self, language: str, keyterms: list[str]) -> str:
        params = [
            ("model_id", self._config.model_id),
            ("audio_format", f"pcm_{SAMPLE_RATE}"),
            ("commit_strategy", "manual"),
            ("include_timestamps", "true"),
            ("include_language_detection", "true" if language == "auto" else "false"),
        ]
        if language != "auto":
            params.append(("language_code", language))
        params.extend(("keyterms", t) for t in keyterms)
        return f"{self._config.endpoint}?{urlencode(params)}"


@dataclass(frozen=True)
class _PendingCommit:
    commit_id: int | None
    audio_end_s: float
    answered_here: bool
    """No audio since the last commit, so nothing was sent; answered in turn without EL."""


class ElevenLabsConnection(Connection):
    def __init__(
        self, connection_id: int, sink: EventSink, previous_text: str | None, partials: bool
    ) -> None:
        super().__init__(connection_id)
        self._sink = sink
        self._previous_text = previous_text
        self._partials = partials
        self._ws: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._started = asyncio.Event()
        self._open = False
        self._ended = False
        self._open_failure: Failed | None = None
        self._send_lock = asyncio.Lock()
        self._sent_samples: int = 0
        self._committed_samples: int = 0
        self._wire_samples: int = 0
        self._committed_wire_samples: int = 0
        self._pads: list[tuple[int, int]] = []
        """(wire start, length) of each silence pad, in samples."""
        self._next_commit_id = 1
        self._pending: collections.deque[_PendingCommit] = collections.deque()
        self._last_partial = ""
        self._cut_unknown = False

    async def connect(self, url: str, api_key: str) -> None:
        try:
            # open() bounds the whole attempt.
            self._ws = await websockets.connect(
                url,
                additional_headers={"xi-api-key": api_key},
                open_timeout=None,
                close_timeout=_CLOSE_TIMEOUT_S,
                logger=_ws_logger,
            )
        except websockets.InvalidStatus as exc:
            raise http_failure(exc.response.status_code, exc.response.body, api_key) from None
        except websockets.InvalidURI as exc:
            raise OpenFailed("invalid_request", str(exc), retryable=False) from None
        except (OSError, websockets.WebSocketException) as exc:
            raise OpenFailed("network", f"connect failed: {exc}", retryable=True) from None
        self._reader = asyncio.create_task(self._read(), name="elevenlabs-reader")
        await self._started.wait()
        if self._open_failure is not None:
            f = self._open_failure
            raise OpenFailed(f.code, f.message, retryable=f.retryable)
        self._open = True

    async def send_audio(self, samples: Samples) -> None:
        async with self._send_lock:
            for i in range(0, len(samples), _MAX_CHUNK_SAMPLES):
                if self._ended:
                    return
                await self._send_chunk(samples[i : i + _MAX_CHUNK_SAMPLES])
                if (
                    self._wire_samples - self._committed_wire_samples
                    >= _SELF_COMMIT_AFTER_S * SAMPLE_RATE
                ):
                    logger.info(
                        "committing after %.0f s of uncommitted audio", _SELF_COMMIT_AFTER_S
                    )
                    await self._commit(None)

    async def commit(self, reference: Reference | None = None) -> int:
        async with self._send_lock:
            commit_id = self._next_commit_id
            self._next_commit_id += 1
            if not self._ended:
                await self._commit(commit_id)
            return commit_id

    async def close(self) -> None:
        self._end(None)
        reader = self._reader
        if reader is not None and reader is not asyncio.current_task():
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as exc:
                logger.debug("closing the ElevenLabs socket: %s", exc)

    async def _send_chunk(self, samples: Samples) -> None:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        msg: dict[str, Any] = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm).decode(),
            "commit": False,
            "sample_rate": SAMPLE_RATE,
        }
        if self._sent_samples == 0 and self._previous_text:
            msg["previous_text"] = self._previous_text
        self._sent_samples += len(samples)
        self._wire_samples += len(samples)
        await self._send(msg)

    async def _commit(self, commit_id: int | None) -> None:
        audio_end_s = self._sent_samples / SAMPLE_RATE
        if self._sent_samples == self._committed_samples:
            self._pending.append(_PendingCommit(commit_id, audio_end_s, answered_here=True))
            self._answer_waiting_here()
            return
        uncommitted = 0 if self._cut_unknown else self._wire_samples - self._committed_wire_samples
        self._cut_unknown = False
        if uncommitted < _MIN_COMMIT_AUDIO_S * SAMPLE_RATE:
            pad = round(_PAD_TO_S * SAMPLE_RATE) - uncommitted
            self._pads.append((self._wire_samples, pad))
            self._wire_samples += pad
            await self._send(
                {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": base64.b64encode(bytes(2 * pad)).decode(),
                    "commit": False,
                    "sample_rate": SAMPLE_RATE,
                }
            )
        self._committed_samples = self._sent_samples
        self._committed_wire_samples = self._wire_samples
        self._pending.append(_PendingCommit(commit_id, audio_end_s, answered_here=False))
        await self._send(
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "commit": True,
                "sample_rate": SAMPLE_RATE,
            }
        )
        logger.debug("commit %s sent at %.2f s", commit_id, audio_end_s)

    async def _send(self, msg: dict[str, Any]) -> None:
        if self._ended:
            return
        try:
            await self._ws.send(json.dumps(msg))
        except websockets.ConnectionClosed as exc:
            self._end(Failed(self.connection_id, "network", f"connection lost: {exc}", True))

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("non-JSON frame from ElevenLabs: %r", raw[:200])
                    continue
                if isinstance(data, dict):
                    self._dispatch(data)  # pyright: ignore[reportUnknownArgumentType]
                if self._ended:
                    return
            self._end(
                Failed(self.connection_id, "network", "ElevenLabs closed the connection", True)
            )
        except websockets.ConnectionClosed as exc:
            self._end(Failed(self.connection_id, "network", f"connection lost: {exc}", True))
        except Exception as exc:
            logger.exception("ElevenLabs reader failed")
            self._end(Failed(self.connection_id, "backend_error", f"reader failed: {exc}", True))

    def _dispatch(self, data: dict[str, Any]) -> None:
        kind = data.get("message_type")
        if kind == "session_started":
            logger.info("ElevenLabs session %s started", data.get("session_id"))
            self._started.set()
        elif kind == "partial_transcript":
            text = str(data.get("text") or "")
            if self._partials and text and text != self._last_partial:
                self._last_partial = text
                self._sink(Partial(self.connection_id, text))
        elif kind == "committed_transcript_with_timestamps":
            self._acknowledge(data)
        elif kind == "warning":
            logger.warning("ElevenLabs warning: %s", data.get("warning"))
        elif isinstance(kind, str) and kind in ERRORS:
            code, retryable = ERRORS[kind]
            message = f"ElevenLabs {kind}: {data.get('error', '')}"
            self._end(Failed(self.connection_id, code, message, retryable))
        elif kind != "committed_transcript":
            logger.debug("ignoring ElevenLabs message %r", kind)

    def _acknowledge(self, data: dict[str, Any]) -> None:
        pending = self._pending.popleft() if self._pending else None
        if pending is None:
            # The server committed by itself at a point it doesn't say: claim no audio, and pad
            # the next commit in full since its uncommitted length is unknown.
            self._cut_unknown = True
        text = str(data.get("text") or "")
        raw_words: list[dict[str, Any]] = data.get("words") or []
        words = tuple(
            Word(str(w["text"]).strip(), self._unpad(w["start"]), self._unpad(w["end"]))
            for w in raw_words
            if w.get("type") == "word" and str(w.get("text", "")).strip()
        )
        self._last_partial = text
        self._sink(
            Committed(
                self.connection_id,
                pending.commit_id if pending else None,
                text,
                data.get("language_code") or None,
                words,
                pending.audio_end_s if pending else self._committed_samples / SAMPLE_RATE,
            )
        )
        self._answer_waiting_here()

    def _answer_waiting_here(self) -> None:
        while self._pending and self._pending[0].answered_here:
            p = self._pending.popleft()
            self._sink(Committed(self.connection_id, p.commit_id, "", None, (), p.audio_end_s))

    def _unpad(self, wire_s: float) -> float:
        """Wire time to the time in the audio the session sent, padding taken out."""
        wire = wire_s * SAMPLE_RATE
        removed = 0
        for start, length in self._pads:
            if wire <= start:
                break
            removed += min(length, wire - start)
        return (wire - removed) / SAMPLE_RATE

    def _end(self, failure: Failed | None) -> None:
        if self._ended:
            return
        self._ended = True
        self._started.set()
        if not self._open:
            self._open_failure = failure or Failed(
                self.connection_id, "network", "closed before session_started", True
            )
            return
        if failure is not None:
            logger.warning(
                "ElevenLabs connection %d failed: %s", self.connection_id, failure.message
            )
            self._sink(failure)
        self._sink(Closed(self.connection_id))
        if self._reader is not None and self._reader is not asyncio.current_task():
            self._reader.cancel()
