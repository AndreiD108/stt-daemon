"""A local stand-in for the ElevenLabs realtime endpoint, and helpers for its tests."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

from websockets.asyncio.server import Server, ServerConnection
from websockets.http11 import Request, Response

from stt_daemon.backends.base import BackendEvent, Committed

TEST_KEY = "sk_test_key_4d1f9e"

type Reply = Callable[[int], list[dict[str, Any]]]


def committed(
    text: str, words: list[dict[str, Any]] | None = None, language: str | None = "en"
) -> list[dict[str, Any]]:
    return [
        {"message_type": "committed_transcript", "text": text},
        {
            "message_type": "committed_transcript_with_timestamps",
            "text": text,
            "language_code": language,
            "words": words,
        },
    ]


def el_word(text: str, start: float, end: float) -> dict[str, Any]:
    return {"text": text, "start": start, "end": end, "type": "word", "logprob": -0.1}


class FakeElevenLabs:
    """Records requests and client messages; answers each commit with `reply(n)`."""

    def __init__(self) -> None:
        self.server: Server | None = None
        self.requests: list[Request] = []
        self.messages: list[dict[str, Any]] = []
        self.status: int | None = None
        """Answer the handshake with this HTTP status instead of upgrading."""
        self.greeting: list[dict[str, Any]] = [
            {"message_type": "session_started", "session_id": "sess-1", "config": {}}
        ]
        self.reply: Reply = lambda n: committed(f"text {n}")
        self.commits = 0
        self.connections: list[ServerConnection] = []

    @property
    def url(self) -> str:
        assert self.server is not None
        port = next(iter(self.server.sockets)).getsockname()[1]
        return f"ws://127.0.0.1:{port}/v1/speech-to-text/realtime"

    def audio_messages(self) -> list[dict[str, Any]]:
        return [m for m in self.messages if m["audio_base_64"]]

    async def push(self, *messages: dict[str, Any]) -> None:
        for m in messages:
            await self.connections[-1].send(json.dumps(m))

    def process_request(self, conn: ServerConnection, request: Request) -> Response | None:
        self.requests.append(request)
        if self.status is not None:
            return conn.respond(self.status, "refused\n")
        return None

    async def handle(self, conn: ServerConnection) -> None:
        self.connections.append(conn)
        await self.push(*self.greeting)
        async for raw in conn:
            msg = json.loads(raw)
            self.messages.append(msg)
            if msg["commit"]:
                self.commits += 1
                await self.push(*self.reply(self.commits))


async def wait_until(check: Callable[[], bool], timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.005)


def commits(events: list[BackendEvent]) -> list[Committed]:
    return [e for e in events if isinstance(e, Committed)]
