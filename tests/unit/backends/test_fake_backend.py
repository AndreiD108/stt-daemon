import asyncio
import dataclasses

import numpy as np
import pytest

from stt_daemon.audio.frames import Samples
from stt_daemon.backends.base import (
    BackendEvent,
    Closed,
    Committed,
    Failed,
    OpenFailed,
    OpenOptions,
    Partial,
    Word,
)
from stt_daemon.backends.fake import FAKE_CAPABILITIES, FakeBackend, FakeReply


def _audio(seconds: float) -> Samples:
    return np.zeros(round(seconds * 16000), np.float32)


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_replies_answer_commits_in_order_with_spread_words() -> None:
    backend = FakeBackend(["hello there", FakeReply("again", words=[("again", 0.1, 0.4)])])
    events: list[BackendEvent] = []
    conn = await backend.open(OpenOptions(), 7, events.append)
    await conn.send_audio(_audio(1.0))
    assert await conn.commit() == 1
    await conn.send_audio(_audio(0.5))
    assert await conn.commit() == 2
    await _settle()
    assert events == [
        Committed(7, 1, "hello there", None, (Word("hello", 0, 0.5), Word("there", 0.5, 1.0)), 1.0),
        Committed(7, 2, "again", None, (Word("again", 0.1, 0.4),), 1.5),
    ]


async def test_commit_without_audio_is_answered_empty_without_using_a_reply() -> None:
    backend = FakeBackend(["kept for later"])
    events: list[BackendEvent] = []
    conn = await backend.open(OpenOptions(), 1, events.append)
    await conn.commit()
    await conn.send_audio(_audio(0.2))
    await conn.commit()
    await _settle()
    assert [(e.commit_id, e.text) for e in events if isinstance(e, Committed)] == [
        (1, ""),
        (2, "kept for later"),
    ]


async def test_a_delayed_reply_holds_back_later_ones() -> None:
    backend = FakeBackend([FakeReply("slow", delay_s=0.05), "fast"])
    events: list[BackendEvent] = []
    conn = await backend.open(OpenOptions(), 1, events.append)
    for _ in range(2):
        await conn.send_audio(_audio(0.1))
        await conn.commit()
    await _settle()
    assert events == []
    await asyncio.sleep(0.08)
    assert [e.text for e in events if isinstance(e, Committed)] == ["slow", "fast"]


async def test_partials_follow_the_next_reply_only_when_asked() -> None:
    backend = FakeBackend(
        [FakeReply("a b", partials=("a", "a b")), FakeReply("c", partials=("c",))]
    )
    events: list[BackendEvent] = []
    conn = await backend.open(OpenOptions(partials=True), 1, events.append)
    for _ in range(3):
        await conn.send_audio(_audio(0.1))
    await conn.commit()
    await conn.send_audio(_audio(0.1))
    await _settle()
    assert [e.text for e in events if isinstance(e, Partial)] == ["a", "a b", "c"]

    quiet = FakeBackend([FakeReply("x", partials=("x",))])
    quiet_events: list[BackendEvent] = []
    quiet_conn = await quiet.open(OpenOptions(), 1, quiet_events.append)
    await quiet_conn.send_audio(_audio(0.1))
    assert quiet_events == []


async def test_failing_reply_ends_the_connection_and_later_calls_do_nothing() -> None:
    backend = FakeBackend([FakeReply(fail=("idle_drop", True)), "never"])
    events: list[BackendEvent] = []
    conn = await backend.open(OpenOptions(), 3, events.append)
    await conn.send_audio(_audio(0.1))
    await conn.commit()
    await _settle()
    assert events == [Failed(3, "idle_drop", "fake failure", True), Closed(3)]
    await conn.send_audio(_audio(0.1))
    await conn.commit()
    await conn.close()
    await _settle()
    assert len(events) == 2
    assert [r.text for r in backend.replies] == ["never"]


async def test_fail_drops_the_newest_live_connection() -> None:
    backend = FakeBackend()
    old: list[BackendEvent] = []
    new: list[BackendEvent] = []
    await backend.open(OpenOptions(), 1, old.append)
    await backend.open(OpenOptions(), 2, new.append)
    backend.fail("network")
    assert old == []
    assert new == [Failed(2, "network", "fake failure", True), Closed(2)]


async def test_close_ends_with_closed_only_and_drops_pending_replies() -> None:
    backend = FakeBackend([FakeReply("late", delay_s=0.02)])
    events: list[BackendEvent] = []
    conn = await backend.open(OpenOptions(), 1, events.append)
    await conn.send_audio(_audio(0.1))
    await conn.commit()
    await conn.close()
    await asyncio.sleep(0.04)
    assert events == [Closed(1)]


async def test_connection_time_limit() -> None:
    caps = dataclasses.replace(FAKE_CAPABILITIES, max_connection_s=0.02)
    backend = FakeBackend(capabilities=caps)
    events: list[BackendEvent] = []
    await backend.open(OpenOptions(), 1, events.append)
    await asyncio.sleep(0.05)
    assert [type(e) for e in events] == [Failed, Closed]
    first = events[0]
    assert isinstance(first, Failed)
    assert (first.code, first.retryable) == ("session_limit", True)


async def test_open_failures_are_raised_in_turn() -> None:
    backend = FakeBackend(open_failures=[OpenFailed("network", "down", retryable=True)])
    with pytest.raises(OpenFailed):
        await backend.open(OpenOptions(), 1, lambda _: None)
    await backend.open(OpenOptions(), 2, lambda _: None)
    assert backend.open_calls == 2
    assert [c.connection_id for c in backend.connections] == [2]


async def test_records_what_the_session_sent() -> None:
    backend = FakeBackend()
    options = OpenOptions(language="en", keyterms=("Scribbly",), previous_text="before")
    conn = await backend.open(options, 1, lambda _: None)
    await conn.send_audio(np.full(800, 0.5, np.float32))
    await conn.send_audio(_audio(0.05))
    recorded = backend.connections[0]
    assert recorded.options == options
    assert recorded.sent_s == pytest.approx(0.1)
    assert recorded.audio[:800].tolist() == [0.5] * 800
