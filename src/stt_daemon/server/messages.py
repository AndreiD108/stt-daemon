"""The JSONL envelope and the checks on each command's `data`."""

import base64
import binascii
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

PROTOCOL = 1
AUDIO_CHUNK_LIMIT = 1 << 20
"""Most bytes one `audio` command may carry, decoded."""

type MessageId = str | int


class RequestError(Exception):
    """A command that fails with an `error` reply; `extra` goes into the reply's data."""

    def __init__(self, code: str, message: str, **extra: object) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra

    def to_wire(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, **self.extra}


@dataclass(frozen=True)
class Message:
    type: str
    id: MessageId | None
    data: Mapping[str, Any]


@dataclass(frozen=True)
class Hello:
    label: str | None
    observe_all: bool
    vad_events: bool


@dataclass(frozen=True)
class StartRequest:
    label: str | None = None
    backend: str | None = None
    language: str | None = None
    keyterms: tuple[str, ...] = ()
    previous_text: str | None = None
    context_key: str | None = None
    partials: bool = False
    word_timestamps: bool = False
    speech_id: str | None = None


@dataclass(frozen=True)
class TranscribeRequest:
    start: StartRequest
    pcm: tuple[int, int] | None = None
    """(rate, channels) of headerless s16le input."""


@dataclass(frozen=True)
class MarkerRequest:
    marker: str
    at: int | None


def parse_line(line: bytes) -> Message:
    """One received line. Raises `RequestError("bad_request")`; its `id` is lost then."""
    try:
        raw = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RequestError(
            "bad_request", "the line is not JSON; send one JSON object per line"
        ) from None
    if not isinstance(raw, dict):
        raise RequestError("bad_request", "the line is not a JSON object")
    obj = cast(dict[str, Any], raw)
    kind = obj.get("type")
    if not isinstance(kind, str) or not kind:
        raise RequestError("bad_request", "type is missing; every message needs a type")
    message_id = obj.get("id")
    if message_id is not None and (
        isinstance(message_id, bool) or not isinstance(message_id, str | int)
    ):
        raise RequestError("bad_request", "id must be a string or an integer")
    data = obj.get("data", {})
    if not isinstance(data, dict):
        raise RequestError("bad_request", "data must be a JSON object")
    return Message(kind, message_id, cast(dict[str, Any], data))


def encode(kind: str, data: Mapping[str, object], reply_to: MessageId | None = None) -> bytes:
    envelope: dict[str, object] = {"type": kind}
    if reply_to is not None:
        envelope["reply_to"] = reply_to
    envelope["ts"] = round(time.time() * 1000)
    envelope["data"] = data
    return json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"


def parse_hello(data: Mapping[str, Any]) -> Hello:
    if data.get("protocol") != PROTOCOL or isinstance(data.get("protocol"), bool):
        raise RequestError(
            "protocol_unsupported",
            f"this daemon speaks protocol {PROTOCOL}; send hello with protocol: {PROTOCOL}",
            supported=[PROTOCOL],
        )
    fields = _Fields(data, "bad_request")
    return Hello(
        label=fields.text("label") or None,
        observe_all=fields.flag("observe_all"),
        vad_events=fields.flag("vad_events"),
    )


def parse_start(data: Mapping[str, Any]) -> StartRequest:
    fields = _Fields(data, "bad_request")
    return StartRequest(
        label=fields.text("label") or None,
        backend=fields.text("backend"),
        language=fields.text("language"),
        keyterms=fields.texts("keyterms"),
        previous_text=fields.text("previous_text"),
        context_key=fields.text("context_key"),
        partials=fields.flag("partials"),
        word_timestamps=fields.flag("word_timestamps"),
        speech_id=fields.text("speech_id"),
    )


def parse_transcribe(data: Mapping[str, Any]) -> TranscribeRequest:
    if "speech_id" in data:
        raise RequestError(
            "bad_request",
            "transcribe takes no speech_id; it reads the audio you send",
            field="speech_id",
        )
    start = parse_start(data)
    pcm = data.get("pcm")
    if pcm is None:
        return TranscribeRequest(start)
    if not isinstance(pcm, dict):
        raise RequestError("bad_request", "pcm must be an object: {rate, channels}", field="pcm")
    fields = cast(dict[str, Any], pcm)
    rate, channels = fields.get("rate"), fields.get("channels")
    if not _int_in(rate, 8000, 192000):
        raise RequestError(
            "bad_request", "pcm.rate must be an integer, 8000 to 192000", field="pcm"
        )
    if not _int_in(channels, 1, 2):
        raise RequestError("bad_request", "pcm.channels must be 1 or 2", field="pcm")
    return TranscribeRequest(start, (cast(int, rate), cast(int, channels)))


def parse_audio(data: Mapping[str, Any]) -> bytes:
    value = data.get("data")
    if not isinstance(value, str) or not value:
        raise RequestError("invalid", "data must be the audio bytes in base64", field="data")
    if len(value) > (AUDIO_CHUNK_LIMIT + 2) // 3 * 4:
        raise _too_big()
    try:
        audio = base64.b64decode(value, validate=True)
    except binascii.Error:
        raise RequestError("invalid", "data is not valid base64", field="data") from None
    if len(audio) > AUDIO_CHUNK_LIMIT:
        raise _too_big()
    return audio


def _too_big() -> RequestError:
    message = f"data holds more than {AUDIO_CHUNK_LIMIT} bytes; send smaller chunks"
    return RequestError("invalid", message, field="data")


def _int_in(value: object, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def parse_marker(data: Mapping[str, Any]) -> MarkerRequest:
    fields = _Fields(data, "invalid")
    marker = fields.text("marker")
    if not marker or not marker.strip():
        raise RequestError("invalid", "marker is missing; send the marker's text in marker")
    return MarkerRequest(marker, fields.integer("at"))


class _Fields:
    """Optional fields of one command; a wrong type fails with `code`, naming the field."""

    def __init__(self, data: Mapping[str, Any], code: str) -> None:
        self._data = data
        self._code = code

    def _fail(self, name: str, expected: str) -> RequestError:
        return RequestError(self._code, f"{name} must be {expected}", field=name)

    def text(self, name: str) -> str | None:
        value = self._data.get(name)
        if value is not None and not isinstance(value, str):
            raise self._fail(name, "a string")
        return value

    def flag(self, name: str) -> bool:
        value = self._data.get(name, False)
        if not isinstance(value, bool):
            raise self._fail(name, "true or false")
        return value

    def integer(self, name: str) -> int | None:
        value = self._data.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise self._fail(name, "an integer (epoch ms)")
        return value

    def texts(self, name: str) -> tuple[str, ...]:
        value = self._data.get(name)
        if value is None:
            return ()
        items = cast(list[Any], value) if isinstance(value, list) else None
        if items is None or not all(isinstance(item, str) for item in items):
            raise self._fail(name, "a list of strings")
        return tuple(cast(list[str], items))
