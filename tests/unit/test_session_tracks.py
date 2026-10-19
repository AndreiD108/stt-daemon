"""A `system` session's two tracks: separate links, a lost system capture, the session's end."""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest
from test_session_audio import fake, segments

from stt_daemon.audio.pw_capture import PwRecordCapture
from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.base import EventSink, OpenFailed, OpenOptions
from stt_daemon.backends.fake import FakeBackend, FakeConnection, FakeReply
from stt_daemon.config import Config
from stt_daemon.session import recovery
from stt_daemon.session.events import SessionError, SessionEvent, SessionStarted
from stt_daemon.session.options import SessionOptions
from stt_daemon.session.session import Session

FIXTURES = Path(__file__).parents[1] / "fixtures"
FAKE_PW_RECORD = Path(__file__).parents[1] / "fake_pw_record.py"


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recovery, "RECONNECT_COOLDOWN_S", 0.0)


def system_capture(*args: str, backoff_s: tuple[float, ...] = (0.1,)) -> PwRecordCapture:
    return PwRecordCapture(
        [sys.executable, str(FAKE_PW_RECORD), *args], stall_s=0.5, backoff_s=backoff_s
    )


async def system_session(
    backend: FakeBackend, capture: PwRecordCapture, mic: WavCapture | None = None
) -> tuple[Session, list[SessionEvent]]:
    events: list[SessionEvent] = []
    options = SessionOptions(OpenOptions(language="auto"), word_timestamps=True)
    session = Session(
        config=Config(), capture=capture, backend=backend, options=options,
        emit=events.append, kind="system", mic=mic,
    )  # fmt: skip
    await session.start()
    return session, events


def errors(events: list[SessionEvent]) -> list[tuple[str, str | None, bool]]:
    return [(e.code, e.track, e.fatal) for e in events if isinstance(e, SessionError)]


async def test_one_tracks_reconnect_leaves_the_other_tracks_connection_alone() -> None:
    # Replies go in commit order across connections: each track's first commit gets a slow one,
    # so the system track's log is still whole when its connection fails.
    slow = [FakeReply("lost", delay_s=5.0), FakeReply("held", delay_s=5.0)]
    backend = fake([*slow, "a", "b", "c", "d"], interval_s=1.5)
    mic = WavCapture(FIXTURES / "jargon.wav")
    await mic.start()
    capture = system_capture("--wav", str(FIXTURES / "two_phrases.wav"))
    session, events = await system_session(backend, capture, mic)
    try:
        while len(backend.connections) < 2 or not backend.connections[0].commit_ids:
            await asyncio.sleep(0.01)
        system_conn, mic_conn = backend.connections
        await asyncio.sleep(0.5)
        system_conn.fail("network", True)
        while len(backend.connections) < 3:
            await asyncio.sleep(0.01)
        await asyncio.sleep(2.0)
        mic_alive = not mic_conn.ended
        ended = await session.stop()
    finally:
        await mic.stop()
    replacement = backend.connections[2]
    assert mic_alive
    assert len(backend.connections) == 3  # the mic track kept its one connection
    assert np.array_equal(replacement.audio[: system_conn.audio.size], system_conn.audio)
    assert errors(events) == [("network", "system", False)]
    assert ended.reason == "stopped"
    assert {s.track for s in segments(events)} == {"system", "mic"}


async def test_a_lost_system_capture_is_reported_then_restored_and_the_session_goes_on(
    tmp_path: Path,
) -> None:
    capture = system_capture(
        "--log", str(tmp_path / "starts"), "--exit-after", "0.5", "--only-starts", "1"
    )
    session, events = await system_session(fake([]), capture)
    while len(errors(events)) < 2:
        await asyncio.sleep(0.01)
    assert session.state == "running"
    ended = await session.stop()
    assert errors(events) == [
        ("capture_lost", "system", False),
        ("capture_restored", "system", False),
    ]
    assert ended.reason == "stopped"


async def test_a_capture_with_no_audio_at_the_start_still_starts_then_reports_it() -> None:
    capture = PwRecordCapture([sys.executable, "-c", "pass"], backoff_s=(10.0,))
    session, events = await system_session(fake([]), capture)
    try:
        assert isinstance(events[0], SessionStarted)
        assert events[0].tracks == ("system",)
        assert errors(events) == [("capture_lost", "system", False)]
        error = events[1]
        assert isinstance(error, SessionError)
        assert "exited with code 0" in error.message
    finally:
        await session.cancel()


async def test_a_fatal_backend_error_on_one_track_ends_the_session_with_both_tracks_text() -> None:
    backend = fake(["one", "two", "three", "four", "five", "six"], interval_s=1.0)
    mic = WavCapture(FIXTURES / "jargon.wav")
    await mic.start()
    capture = system_capture("--wav", str(FIXTURES / "two_phrases.wav"))
    session, events = await system_session(backend, capture, mic)
    try:
        while {s.track for s in segments(events)} != {"system", "mic"}:
            await asyncio.sleep(0.01)
        backend.connections[1].fail("auth", False)
        ended = await session.wait_ended()
    finally:
        await mic.stop()
    assert errors(events) == [("auth", "mic", True)]
    assert ended.reason == "error"
    lines = ended.final_text.splitlines()
    assert {line.split(" ")[0] for line in lines} == {"[system]", "[mic]"}
    assert len(lines) == len(segments(events))


async def test_a_track_that_fails_for_good_before_the_start_ends_fails_the_start() -> None:
    backend = fake([])
    second_open = asyncio.Event()
    go_on = asyncio.Event()
    plain_open = backend.open

    async def gated_open(
        options: OpenOptions, connection_id: int, sink: EventSink
    ) -> FakeConnection:
        if backend.open_calls == 1:
            second_open.set()
            await go_on.wait()
        return await plain_open(options, connection_id, sink)

    backend.open = gated_open  # type: ignore[method-assign]
    mic = WavCapture(keep_open=False)
    await mic.start()
    capture = system_capture("--wav", str(FIXTURES / "two_phrases.wav"))
    events: list[SessionEvent] = []
    session = Session(
        config=Config(), capture=capture, backend=backend,
        options=SessionOptions(OpenOptions(language="auto")), emit=events.append,
        kind="system", mic=mic,
    )  # fmt: skip
    try:
        starting = asyncio.create_task(session.start())
        await second_open.wait()
        backend.connections[0].fail("auth", False)
        await asyncio.sleep(0.05)
        go_on.set()
        with pytest.raises(OpenFailed) as failed:
            await starting
    finally:
        await mic.stop()
    assert failed.value.code == "auth"
    assert events == []
    assert session.state == "new"
    assert all(c.ended for c in backend.connections)
    assert not capture.is_open
    assert not mic.is_open  # its hold was released
