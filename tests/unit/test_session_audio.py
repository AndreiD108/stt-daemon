"""Sessions over the TTS fixture (speech at 1.00-3.74 s and 5.74-7.81 s), played in real time."""

import asyncio
import itertools
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from stt_daemon.audio.wav_capture import WavCapture, read_wav
from stt_daemon.backends.base import CommitPolicy, OpenOptions
from stt_daemon.backends.fake import FAKE_CAPABILITIES, FakeBackend, FakeReply
from stt_daemon.config import Config
from stt_daemon.session.events import PartialText, SessionEnded, SessionEvent
from stt_daemon.session.options import SessionOptions
from stt_daemon.session.session import Session
from stt_daemon.session.transcript import Segment

FIXTURE = Path(__file__).parents[1] / "fixtures" / "two_phrases.wav"
RATE = 16000


def fake(replies: list[FakeReply | str], chunk_s: float | None = None) -> FakeBackend:
    policy = CommitPolicy(max_chunk_s=chunk_s)
    return FakeBackend(replies, capabilities=replace(FAKE_CAPABILITIES, commit_policy=policy))


async def run(
    backend: FakeBackend, config: Config | None = None, *, partials: bool = False
) -> tuple[WavCapture, Session, list[SessionEvent]]:
    capture = WavCapture(FIXTURE)
    await capture.start()
    await asyncio.sleep(0.1)  # the ring then starts at the fixture's first sample
    events: list[SessionEvent] = []
    options = SessionOptions(OpenOptions(language="auto", partials=partials), word_timestamps=True)
    session = Session(
        config=config or Config(), capture=capture, backend=backend, options=options,
        emit=events.append,
    )  # fmt: skip
    await session.start()
    return capture, session, events


async def play_until(session: Session, seconds: float) -> None:
    assert session.started is not None
    started = session.started.started_at / 1000
    await asyncio.sleep(max(0.0, started + seconds - time.time()))


def segments(events: list[SessionEvent]) -> list[Segment]:
    return [e for e in events if isinstance(e, Segment)]


def capture_ms(fixture: np.ndarray, audio: np.ndarray, seconds: float) -> float:
    """Where in the fixture the audio sent at `seconds` was captured, in ms."""
    block = audio[round(seconds * RATE) :][:160]
    for i in np.flatnonzero(fixture == block[0]):
        if np.array_equal(fixture[i : i + block.size], block):
            return i / RATE * 1000
    raise AssertionError(f"audio at {seconds} s not found in the fixture")


async def test_word_times_map_back_through_a_cut_pause() -> None:
    words = [("one", 0.3, 0.5), ("two", 1.5, 1.7), ("three", 3.6, 3.8), ("four", 4.5, 4.7)]
    backend = fake([FakeReply("one two three four", words=words)])
    capture, session, events = await run(backend)
    try:
        await asyncio.to_thread(capture.wait_idle, 15)
        ended = await session.stop()
    finally:
        await capture.stop()
    sent = backend.connections[0]
    assert sent.sent_s < 9.3 - 1.5  # the 2 s pause and the leading silence were cut
    [segment] = segments(events)
    assert segment.words is not None
    fixture = read_wav(FIXTURE)
    for word, (_, start_s, end_s) in zip(segment.words, words, strict=True):
        assert abs(word.start_ms - capture_ms(fixture, sent.audio, start_s)) <= 1
        assert abs(word.end_ms - capture_ms(fixture, sent.audio, end_s)) <= 1
    assert segment.words[2].start_ms > 5740  # after the cut, in the second phrase
    assert 850 <= segment.start_ms <= 960  # 100 ms before the first speech
    assert segment.end_ms >= 9000
    assert ended == SessionEnded("stopped", "one two three four", ended.duration_ms, 1)
    assert ended.duration_ms >= 9000


async def test_several_commits_on_one_connection() -> None:
    replies: list[FakeReply | str] = [
        FakeReply("alpha", partials=("al",)),
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
    ]
    backend = fake(replies, chunk_s=1.5)
    capture, session, events = await run(backend, partials=True)
    try:
        await asyncio.to_thread(capture.wait_idle, 15)
        ended = await session.stop()
    finally:
        await capture.stop()
    assert len(backend.connections) == 1
    assert len(backend.connections[0].commit_ids) >= 4
    segs = segments(events)
    assert len(segs) >= 4
    assert [s.index for s in segs] == list(range(len(segs)))
    for before, after in itertools.pairwise(segs):
        assert before.end_ms <= after.start_ms
    for s in segs:
        assert s.words is not None
        assert all(s.start_ms <= w.start_ms <= w.end_ms <= s.end_ms for w in s.words)
    assert ended.final_text == " ".join(s.text for s in segs)
    assert ended.segment_count == len(segs)
    assert events.index(PartialText("al")) < events.index(segs[0])


async def test_marker_during_a_commit_goes_by_its_time() -> None:
    backend = fake([FakeReply("first", delay_s=1.0), "second"], chunk_s=2.0)
    capture, session, events = await run(backend)
    try:
        while not backend.connections[0].commit_ids:
            await asyncio.sleep(0.01)
        assert session.started is not None
        session.add_marker("⟦IN⟧", at=session.started.started_at + 1000)
        await asyncio.sleep(0.1)
        session.add_marker("⟦NEXT⟧")
        await play_until(session, 3.5)
        await session.stop()
    finally:
        await capture.stop()
    first, second = segments(events)
    assert first.text == "⟦IN⟧ first"
    assert [m.marker for m in second.markers] == ["⟦NEXT⟧"]


async def test_stop_waits_a_bounded_time_and_keeps_leftover_markers() -> None:
    config = Config()
    config.session.stop_wait_ms = 300
    backend = fake([FakeReply("too late", delay_s=5.0)])
    capture, session, events = await run(backend, config)
    try:
        await play_until(session, 2.5)
        session.add_marker("⟦A⟧")
        began = time.monotonic()
        ended = await session.stop()
        took = time.monotonic() - began
    finally:
        await capture.stop()
    assert took < 1.0
    [segment] = segments(events)
    assert segment.text == "⟦A⟧"
    assert ended.reason == "stopped"
    assert ended.final_text == "⟦A⟧"


async def test_cancel_keeps_committed_text_and_drops_the_rest() -> None:
    backend = fake(["kept", FakeReply("dropped", delay_s=2.0)], chunk_s=1.5)
    capture, session, events = await run(backend)
    try:
        while not segments(events):
            await asyncio.sleep(0.01)
        await play_until(session, 3.2)
        session.add_marker("⟦A⟧")
        ended = await session.cancel()
        await asyncio.sleep(0.2)
    finally:
        await capture.stop()
    assert [s.text for s in segments(events)] == ["kept"]
    assert ended == SessionEnded("cancelled", "kept", ended.duration_ms, 1)
    assert events[-1] == ended


async def test_a_chunk_without_a_pause_commits_at_the_cap() -> None:
    policy = CommitPolicy(max_chunk_s=1.0)
    backend = FakeBackend(
        ["a", "b", "c", "d", "e", "f"],
        capabilities=replace(FAKE_CAPABILITIES, commit_policy=policy),
    )
    capture, session, _ = await run(backend)
    try:
        await asyncio.to_thread(capture.wait_idle, 15)
        await session.stop()
    finally:
        await capture.stop()
    spans = backend.connections[0].commit_spans
    assert len(spans) >= 4
    for start, end in spans[:-1]:
        assert 1.0 <= end - start < 1.1, spans
    assert spans[-1][1] - spans[-1][0] < 1.1


@pytest.mark.parametrize(("pause_from_s", "commits_at_pause"), [(2.0, True), (4.0, False)])
async def test_the_pause_that_commits_depends_on_the_audio_sent_not_the_time_passed(
    pause_from_s: float, commits_at_pause: bool
) -> None:
    # The fixture's 2 s pause comes after ~3.4 s of sent audio and ~4.5 s of session time.
    policy = CommitPolicy(pause_steps=((0, 3.5), (pause_from_s, 1.0)))
    backend = FakeBackend(["a", "b"], capabilities=replace(FAKE_CAPABILITIES, commit_policy=policy))
    capture, session, _ = await run(backend)
    try:
        await asyncio.to_thread(capture.wait_idle, 15)
        await session.stop()
    finally:
        await capture.stop()
    first_end = backend.connections[0].commit_spans[0][1]
    if commits_at_pause:
        assert first_end == pytest.approx(3.44, abs=0.25)
    else:
        assert first_end > 6  # both phrases


async def test_the_pause_tail_after_a_short_pause_commit_does_not_commit_on_its_own() -> None:
    # A 0.3 s pause commits before the gate closes at 0.6 s; the silence sent meanwhile is no
    # speech, so it waits for the next phrase instead of becoming a chunk of its own.
    policy = CommitPolicy(pause_steps=((0, 0.3),))
    backend = FakeBackend(
        ["a", "b", "c"], capabilities=replace(FAKE_CAPABILITIES, commit_policy=policy)
    )
    capture, session, _ = await run(backend)
    try:
        await asyncio.to_thread(capture.wait_idle, 15)
        await session.stop()
    finally:
        await capture.stop()
    spans = backend.connections[0].commit_spans
    assert len(spans) == 3, spans  # each phrase, then the held audio at stop
