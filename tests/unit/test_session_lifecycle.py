"""Session start, markers, failures and endings, on a silent capture."""

import asyncio
import time

import pytest

from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.base import Committed, OpenFailed, OpenOptions
from stt_daemon.backends.fake import FakeBackend, FakeReply
from stt_daemon.config import Config
from stt_daemon.session.events import SessionEnded, SessionError, SessionEvent
from stt_daemon.session.session import MarkerRefused, Session, SessionOptions, start_point
from stt_daemon.session.transcript import Segment


def new_session(
    capture: WavCapture,
    backend: FakeBackend,
    events: list[SessionEvent],
    config: Config | None = None,
) -> Session:
    return Session(
        config=config or Config(),
        capture=capture,
        backend=backend,
        options=SessionOptions(OpenOptions(language="en"), word_timestamps=True),
        emit=events.append,
    )


def test_start_point_never_reaches_into_the_previous_session() -> None:
    assert start_point(100.0, 1000, None) == 99.0
    assert start_point(100.0, 1000, 99.5) == 99.5
    assert start_point(100.0, 1000, 90.0) == 99.0


async def test_markers_in_a_silent_session_come_in_a_last_segment() -> None:
    capture = WavCapture()
    await capture.start()
    events: list[SessionEvent] = []
    session = new_session(capture, FakeBackend(), events)
    try:
        await session.start()
        await asyncio.sleep(0.3)
        at_ms = session.add_marker("⟦A⟧")
        ended = await session.stop()
    finally:
        await capture.stop()
    [segment] = [e for e in events if isinstance(e, Segment)]
    assert segment.text == "⟦A⟧"
    assert segment.words == ()
    assert [(m.after_word, m.at_ms) for m in segment.markers] == [(-1, at_ms)]
    assert segment.language is None
    assert ended == SessionEnded("stopped", "⟦A⟧", ended.duration_ms, 1)


async def test_marker_times_are_checked_and_clamped() -> None:
    capture = WavCapture()
    await capture.start()
    session = new_session(capture, FakeBackend(), [])
    try:
        started = await session.start()
        await asyncio.sleep(0.2)
        with pytest.raises(MarkerRefused) as refused:
            session.add_marker("⟦A⟧", at=started.started_at - 1)
        assert refused.value.code == "out_of_range"
        future = session.add_marker("⟦B⟧", at=round(time.time() * 1000) + 60_000)
        assert 150 <= future <= 1000
        await session.stop()
        with pytest.raises(MarkerRefused) as refused:
            session.add_marker("⟦C⟧")
        assert refused.value.code == "session_stopping"
    finally:
        await capture.stop()


async def test_backend_failure_ends_the_session_with_an_error() -> None:
    capture = WavCapture()
    await capture.start()
    events: list[SessionEvent] = []
    backend = FakeBackend()
    session = new_session(capture, backend, events)
    try:
        await session.start()
        session.add_marker("⟦A⟧")
        backend.fail("quota", retryable=False)
        ended = await asyncio.wait_for(session.wait_ended(), 2)
        assert await session.stop() == ended
    finally:
        await capture.stop()
    assert isinstance(events[-3], Segment)
    assert events[-3].text == "⟦A⟧"
    assert events[-2] == SessionError("quota", "fake failure", fatal=True)
    assert events[-1] == ended
    assert ended.reason == "error"


async def test_failed_open_lets_go_of_the_mic() -> None:
    capture = WavCapture(keep_open=False)
    events: list[SessionEvent] = []
    backend = FakeBackend(open_failures=[OpenFailed("auth", "bad key", retryable=False)])
    session = new_session(capture, backend, events)
    with pytest.raises(OpenFailed):
        await session.start()
    assert not capture.is_open
    assert events == []


async def test_text_answered_after_cancel_is_dropped() -> None:
    capture = WavCapture()
    await capture.start()
    events: list[SessionEvent] = []
    session = new_session(capture, FakeBackend(), events)
    try:
        await session.start()
        cancelling = asyncio.create_task(session.cancel())
        await asyncio.sleep(0)
        late = Committed(1, 1, "late", None, (), 0.5)
        session._on_backend_event(late)  # pyright: ignore[reportPrivateUsage]
        ended = await cancelling
    finally:
        await capture.stop()
    assert not [e for e in events if isinstance(e, Segment)]
    assert ended == SessionEnded("cancelled", "", ended.duration_ms, 0)


async def test_a_cancelled_stop_caller_still_ends_the_session() -> None:
    capture = WavCapture(keep_open=False)
    config = Config()
    config.session.stop_wait_ms = 300
    events: list[SessionEvent] = []
    session = new_session(capture, FakeBackend([FakeReply("slow", delay_s=5.0)]), events, config)
    await session.start()
    await asyncio.sleep(0.1)
    stopping = asyncio.create_task(session.stop())
    await asyncio.sleep(0.05)
    stopping.cancel()
    ended = await asyncio.wait_for(session.wait_ended(), 2)
    assert ended.reason == "stopped"
    assert await session.stop() == ended
    assert not capture.is_open
    assert [e for e in events if isinstance(e, SessionEnded)] == [ended]


async def test_backend_failure_during_stop_ends_with_an_error() -> None:
    capture = WavCapture()
    await capture.start()
    events: list[SessionEvent] = []
    backend = FakeBackend([FakeReply("lost", delay_s=1.0)])
    session = new_session(capture, backend, events)
    try:
        await session.start()
        await asyncio.sleep(0.1)
        session.add_marker("⟦A⟧")
        stopping = asyncio.create_task(session.stop())
        await asyncio.sleep(0.1)
        backend.fail("quota", retryable=False)
        ended = await asyncio.wait_for(stopping, 2)
    finally:
        await capture.stop()
    assert ended.reason == "error"
    assert ended.final_text == "⟦A⟧"
    assert events[-2] == SessionError("quota", "fake failure", fatal=True)
    assert [e for e in events if isinstance(e, SessionEnded)] == [ended]
