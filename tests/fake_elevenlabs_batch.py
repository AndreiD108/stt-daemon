"""A local stand-in for the ElevenLabs file upload endpoint: records each request, answers from a
script."""

import email.parser
import email.policy
import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

from stt_daemon.config import Config


@dataclass
class Upload:
    headers: dict[str, str]
    fields: list[tuple[str, str]]
    file: bytes
    file_type: str
    file_name: str | None

    def values(self, name: str) -> list[str]:
        return [v for n, v in self.fields if n == name]

    def value(self, name: str) -> str | None:
        found = self.values(name)
        return found[0] if found else None


def answer(text: str, words: list[dict[str, Any]], language: str = "en") -> dict[str, Any]:
    return {"language_code": language, "language_probability": 0.99, "text": text, "words": words}


def word(text: str, start: float, end: float, speaker: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"text": text, "type": "word", "start": start, "end": end}
    if speaker is not None:
        entry["speaker_id"] = speaker
    return entry


def spacing(start: float, end: float, speaker: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"text": " ", "type": "spacing", "start": start, "end": end}
    if speaker is not None:
        entry["speaker_id"] = speaker
    return entry


@dataclass
class FakeBatch:
    replies: list[tuple[int, bytes]] = field(default_factory=list[tuple[int, bytes]])
    """Answered in order; once used up, the default answer."""
    default: tuple[int, bytes] = (200, json.dumps(answer("hello there", [])).encode())
    uploads: list[Upload] = field(default_factory=list[Upload])
    hold: threading.Event | None = None
    """Set: each request waits on it before answering."""
    url: str = ""

    def reply(self, status: int, body: dict[str, Any] | str) -> None:
        raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.replies.append((status, raw))

    def configure(self, config: Config) -> None:
        config.backends.elevenlabs_batch.endpoint = self.url

    def _next(self) -> tuple[int, bytes]:
        return self.replies.pop(0) if self.replies else self.default


def _parse(content_type: str, body: bytes) -> tuple[list[tuple[str, str]], Message | None]:
    head = f"Content-Type: {content_type}\r\n\r\n".encode()
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(head + body)
    fields: list[tuple[str, str]] = []
    file_part = None
    for part in msg.iter_parts():  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        part = cast(Message, part)
        name = part.get_param("name", header="content-disposition")
        if name == "file":
            file_part = part
        else:
            payload = cast(bytes, part.get_payload(decode=True))
            fields.append((str(name), payload.decode()))
    return fields, file_part


@contextmanager
def serve_batch() -> Generator[FakeBatch]:
    fake = FakeBatch()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            fields, file_part = _parse(self.headers.get("Content-Type", ""), body)
            file = cast(bytes, file_part.get_payload(decode=True)) if file_part else b""
            fake.uploads.append(
                Upload(
                    {k.lower(): v for k, v in self.headers.items()},
                    fields,
                    file,
                    file_part.get_content_type() if file_part else "",
                    file_part.get_filename() if file_part else None,
                )
            )
            if fake.hold is not None:
                fake.hold.wait(10)
            status, raw = fake._next()  # pyright: ignore[reportPrivateUsage]
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except OSError:
                pass

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    fake.url = f"http://127.0.0.1:{server.server_address[1]}/v1/speech-to-text"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield fake
    finally:
        if fake.hold is not None:
            fake.hold.set()
        server.shutdown()
        server.server_close()
