import asyncio
import socket
from typing import Self

import pytest

from stt_daemon.backends.base import (
    Capabilities,
    CommitPolicy,
    Committed,
    FileAudio,
    FileBackend,
    OpenFailed,
    OpenOptions,
    Word,
)
from stt_daemon.config import Config
from stt_daemon.credentials import Credentials
from stt_daemon.server import wire
from stt_daemon.server.connection import Connection
from stt_daemon.server.slot_state import Client
from stt_daemon.session import upload
from stt_daemon.session.events import SessionEnded, SessionError, SessionEvent, SessionStarted
from stt_daemon.session.options import MarkerRefused, SessionOptions
from stt_daemon.session.transcript import Segment
from stt_daemon.session.upload import UploadSession

TWO_SPEAKERS = [
    Committed(
        0, 1, "Hi there.", "en", (Word("Hi", 0.5, 0.7, "a"), Word("there.", 0.8, 1.2, "a")), 1.2
    ),
    Committed(0, 2, "Hello.", "en", (Word("Hello.", 2.0, 2.5, "b"),), 3.0),
]


class FileFake(FileBackend):
    name = "files"
    capabilities = Capabilities(
        languages=("auto",),
        word_timestamps=True,
        partials=False,
        hints=frozenset(),
        max_connection_s=None,
        commit_policy=CommitPolicy(),
        diarize=True,
    )

    def __init__(self) -> None:
        self.answers: list[Committed] = list(TWO_SPEAKERS)
        self.failure: OpenFailed | None = None
        self.gate: asyncio.Event | None = None
        self.calls: list[tuple[FileAudio, OpenOptions]] = []

    @classmethod
    def create(cls, config: Config, credentials: Credentials) -> Self:
        return cls()

    async def transcribe_file(self, audio: FileAudio, options: OpenOptions) -> list[Committed]:
        self.calls.append((audio, options))
        if self.gate is not None:
            await self.gate.wait()
        if self.failure is not None:
            raise self.failure
        return self.answers


async def started(
    backend: FileFake, *, words: bool = True, pcm: tuple[int, int] | None = None
) -> tuple[UploadSession, list[SessionEvent]]:
    events: list[SessionEvent] = []
    options = SessionOptions(OpenOptions(language="auto", diarize=True), word_timestamps=words)
    session = UploadSession(backend=backend, options=options, pcm=pcm, emit=events.append)
    await session.start()
    return session, events


def segments(events: list[SessionEvent]) -> list[Segment]:
    return [e for e in events if isinstance(e, Segment)]


async def test_stop_uploads_the_bytes_as_sent_and_emits_a_segment_per_speaker() -> None:
    backend = FileFake()
    session, events = await started(backend, pcm=(44100, 2))
    await session.write(b"abc")
    await session.write(b"def")
    ended = await session.stop()
    [(audio, options)] = backend.calls
    assert audio == FileAudio(b"abcdef", (44100, 2))
    assert options.diarize
    first, second = segments(events)
    assert (first.text, first.speaker, first.start_ms, first.end_ms) == ("Hi there.", "a", 0, 1200)
    assert (second.text, second.speaker, second.start_ms, second.end_ms) == (
        "Hello.",
        "b",
        1200,
        3000,
    )
    assert first.words is not None
    assert first.words[0].start_ms == 500
    assert first.words[0].speaker == "a"
    assert first.language == "en"
    assert isinstance(events[0], SessionStarted)
    assert events[0].kind == "transcribe"
    assert ended == SessionEnded("stopped", "Hi there. Hello.", 3000, 2)
    assert events[-1] == ended


async def test_speaker_stays_on_segments_without_word_times() -> None:
    session, events = await started(FileFake(), words=False)
    await session.write(b"x")
    await session.stop()
    assert [(s.speaker, s.words) for s in segments(events)] == [("a", None), ("b", None)]


async def test_markers_need_at_and_land_by_time() -> None:
    session, events = await started(FileFake())
    assert session.started is not None
    with pytest.raises(MarkerRefused) as refused:
        session.add_marker("m0")
    assert refused.value.code == "invalid"
    assert session.add_marker("m1", session.started.started_at + 1500) == 1500
    await session.write(b"x")
    await session.stop()
    first, second = segments(events)
    assert first.markers == ()
    assert [(m.marker, m.after_word) for m in second.markers] == [("m1", -1)]
    assert second.text.startswith("m1")


async def test_backend_failure_ends_with_its_code() -> None:
    backend = FileFake()
    backend.failure = OpenFailed("quota", "ElevenLabs HTTP 402: Out of credits", retryable=False)
    session, events = await started(backend)
    await session.write(b"x")
    ended = await session.stop()
    assert ended.reason == "error"
    assert SessionError("quota", "ElevenLabs HTTP 402: Out of credits", True) in events


async def test_stop_with_no_audio_fails_without_calling_the_backend() -> None:
    backend = FileFake()
    session, events = await started(backend)
    ended = await session.stop()
    assert ended.reason == "error"
    assert backend.calls == []
    assert any(isinstance(e, SessionError) and e.code == "invalid_request" for e in events)


async def test_cancel_during_the_upload_drops_the_answer() -> None:
    backend = FileFake()
    backend.gate = asyncio.Event()
    session, events = await started(backend)
    assert session.started is not None
    session.add_marker("m", session.started.started_at)
    await session.write(b"x")
    stopping = asyncio.create_task(session.stop())
    async with asyncio.timeout(1):
        while not backend.calls:
            await asyncio.sleep(0)
    ended = await session.cancel()
    assert ended.reason == "cancelled"
    assert await stopping == ended
    assert segments(events) == []
    backend.gate.set()
    await asyncio.sleep(0)
    assert segments(events) == []


async def test_cancel_before_stop_never_uploads() -> None:
    backend = FileFake()
    session, _ = await started(backend)
    await session.write(b"x")
    assert (await session.cancel("client_gone")).reason == "client_gone"
    assert backend.calls == []


async def test_input_over_the_limit_ends_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upload, "INPUT_LIMIT", 4)
    backend = FileFake()
    session, events = await started(backend)
    await session.write(b"abc")
    await session.write(b"de")
    ended = await session.wait_ended()
    assert ended.reason == "error"
    assert any(isinstance(e, SessionError) and e.code == "invalid_request" for e in events)
    assert backend.calls == []


async def test_never_suspends() -> None:
    session, _ = await started(FileFake())
    assert not session.suspendable
    from stt_daemon.session.events import PreemptedBy

    assert not await session.suspend(PreemptedBy("mic", None, None))


SEGMENTS = 1000
"""About 10 KiB of JSON each: past `WRITE_BUFFER_LIMIT` in all."""


def long_answer(segments: int, words_each: int) -> list[Committed]:
    out: list[Committed] = []
    for n in range(segments):
        base = n * 20.0
        words = tuple(
            Word(f"word{i}", base + i * 0.05, base + i * 0.05 + 0.04) for i in range(words_each)
        )
        out.append(Committed(0, n + 1, " ".join(w.text for w in words), "en", words, base + 20.0))
    return out


async def read_session(drain: bool) -> tuple[int, bool]:
    """A long answer over a real socket to a client that reads with short pauses: the lines it
    got before `session_ended` or the end, and whether the daemon closed the connection."""
    backend = FileFake()
    backend.answers = long_answer(SEGMENTS, 200)
    a, b = socket.socketpair()
    _, writer = await asyncio.open_connection(sock=a)
    reader, peer_writer = await asyncio.open_connection(sock=b)
    conn = Connection(Client("c1"), writer)

    def emit(event: SessionEvent) -> None:
        if isinstance(event, Segment):
            conn.send("segment", wire.segment(event, "s"))
        elif isinstance(event, SessionEnded):
            conn.send("session_ended", {"reason": event.reason})

    options = SessionOptions(OpenOptions(language="auto"), word_timestamps=True)
    session = UploadSession(
        backend=backend, options=options, pcm=None, emit=emit, drain=conn.drain if drain else None
    )
    await session.start()
    await session.write(b"abc")
    stop = asyncio.create_task(session.stop())
    lines = 0
    while line := await reader.readline():
        lines += 1
        if b'"session_ended"' in line:
            break
        if lines % 20 == 0:
            await asyncio.sleep(0.001)
    await stop
    writer.close()
    peer_writer.close()
    return lines, conn.closed


async def test_a_long_answer_goes_out_as_fast_as_the_client_reads() -> None:
    lines, closed = await read_session(drain=True)
    assert not closed
    assert lines == SEGMENTS + 1


async def test_without_waiting_for_the_client_a_long_answer_closes_it() -> None:
    """Pins that the test above is a real check: the burst alone overflows the write buffer."""
    lines, closed = await read_session(drain=False)
    assert closed
    assert lines < SEGMENTS + 1
