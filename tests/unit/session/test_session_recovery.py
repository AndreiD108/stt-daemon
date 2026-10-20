"""Fixture sessions across connection ends: times, markers and text survive rollover and replay."""

import asyncio
import re
import time
from dataclasses import replace
from typing import Literal

import numpy as np
import pytest

from stt_daemon.audio.wav_capture import read_wav
from stt_daemon.backends.fake import FakeReply
from stt_daemon.session import recovery
from stt_daemon.session.events import SessionEnded, SessionError
from unit.session.test_session_audio import FIXTURE, capture_ms, fake, play_until, run, segments


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recovery, "RECONNECT_COOLDOWN_S", 0.0)


async def test_word_times_stay_true_across_a_rollover_and_a_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 30 s limit: soft rollover 3 s into a connection, at the pause after the first phrase.
    monkeypatch.setattr(recovery, "SOFT_ROLLOVER_MARGIN_S", 27.0)
    monkeypatch.setattr(recovery, "HARD_ROLLOVER_MARGIN_S", 10.0)
    first = [("one", 0.2, 0.5), ("two", 1.5, 1.8)]
    second = [("three", 0.3, 0.6), ("four", 1.2, 1.5)]
    backend = fake([FakeReply("one two", words=first), FakeReply("three four", words=second)])
    backend.capabilities = replace(backend.capabilities, max_connection_s=30.0)
    capture, session, events = await run(backend)
    try:
        await play_until(session, 6.6)
        assert len(backend.connections) == 2
        backend.fail()
        await asyncio.to_thread(capture.wait_idle, 15)
        ended = await session.stop()
    finally:
        await capture.stop()
    old, lost, replayed = backend.connections
    assert lost.audio.size > 0
    assert np.array_equal(replayed.audio[: lost.audio.size], lost.audio)
    fixture = read_wav(FIXTURE)
    a, b = segments(events)
    assert a.words is not None
    assert b.words is not None
    for word, (_, start_s, end_s) in zip(a.words, first, strict=True):
        assert abs(word.start_ms - capture_ms(fixture, old.audio, start_s)) <= 1
        assert abs(word.end_ms - capture_ms(fixture, old.audio, end_s)) <= 1
    for word, (_, start_s, end_s) in zip(b.words, second, strict=True):
        assert abs(word.start_ms - capture_ms(fixture, replayed.audio, start_s)) <= 1
        assert abs(word.end_ms - capture_ms(fixture, replayed.audio, end_s)) <= 1
    assert a.end_ms <= b.start_ms
    assert 5600 <= b.start_ms <= 5700  # the replayed audio starts at the second phrase's pad
    assert [e.code for e in events if isinstance(e, SessionError)] == ["network"]
    assert ended.reason == "stopped"
    assert ended.final_text == "one two three four"


async def test_markers_survive_a_replay_once() -> None:
    replies: list[FakeReply | str] = [
        FakeReply("lost", delay_s=1.0),
        "first",
        "second",
        "third",
    ]
    backend = fake(replies, chunk_s=2.0)
    capture, session, events = await run(backend)
    try:
        while not backend.connections[0].commit_ids:
            await asyncio.sleep(0.01)
        session.add_marker("⟦SENT⟧")  # after the commit went out, before its answer
        await asyncio.sleep(0.2)
        backend.fail()
        session.add_marker("⟦REPLAY⟧")
        await asyncio.to_thread(capture.wait_idle, 15)
        ended = await session.stop()
    finally:
        await capture.stop()
    text = ended.final_text
    assert text.count("⟦SENT⟧") == 1
    assert text.count("⟦REPLAY⟧") == 1
    assert text.index("⟦SENT⟧") < text.index("⟦REPLAY⟧")
    assert " ".join(re.sub(r"⟦\w+⟧", "", text).split()) == "first second third"
    assert [s.index for s in segments(events)] == list(range(len(segments(events))))


async def test_exhausted_recovery_ends_with_the_committed_text() -> None:
    fail = FakeReply(fail=("network", True))
    backend = fake(["kept", fail, fail, fail, fail], chunk_s=1.5)
    capture, session, events = await run(backend)
    try:
        ended = await asyncio.wait_for(session.wait_ended(), 15)
    finally:
        await capture.stop()
    assert ended.reason == "error"
    assert ended.final_text == "kept"
    assert isinstance(events[-2], SessionError)
    assert (events[-2].code, events[-2].fatal) == ("network", True)
    errors = [e for e in events if isinstance(e, SessionError)]
    assert [e.fatal for e in errors] == [False, False, False, True]
    assert len(backend.connections) == 4


@pytest.mark.parametrize("reason", ["stopped", "evicted"])
async def test_stop_during_replacement_open_is_bounded(
    reason: Literal["stopped", "evicted"],
) -> None:
    backend = fake(["kept"], chunk_s=1.5)
    capture, session, events = await run(backend)
    try:
        await play_until(session, 2.8)
        assert segments(events)[0].text == "kept"
        backend.open_delay_s = 10
        backend.fail()
        async with asyncio.timeout(2):
            while backend.open_calls < 2:
                await asyncio.sleep(0.01)
        session._config.session.stop_wait_ms = 200  # pyright: ignore[reportPrivateUsage]
        began = time.monotonic()
        ended = await asyncio.wait_for(session.stop(reason), 1)
        assert time.monotonic() - began < 0.8
        assert ended.reason == reason
        assert ended.final_text == "kept"
        assert [e for e in events if isinstance(e, SessionEnded)] == [ended]
        assert capture._holds == 0  # pyright: ignore[reportPrivateUsage]
    finally:
        await capture.stop()


async def test_final_answer_then_quota_ends_session_with_error() -> None:
    backend = fake([FakeReply("kept", then_fail=("quota", False))])
    capture, session, events = await run(backend)
    try:
        await play_until(session, 2.0)
        ended = await asyncio.wait_for(session.stop(), 2)
    finally:
        await capture.stop()
    assert ended.reason == "error"
    assert ended.final_text == "kept"
    assert isinstance(events[-2], SessionError)
    assert events[-2].fatal
