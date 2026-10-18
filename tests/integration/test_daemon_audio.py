"""Sessions over the socket on the fixture: word times through cuts, commits, rollover, replay."""

import asyncio
import itertools
import time
from dataclasses import replace

import pytest
from harness import FIXTURE, Harness, Msg, TestClient, capture_ms, fake

from stt_daemon.audio.wav_capture import WavCapture, read_wav
from stt_daemon.backends.fake import FakeBackend, FakeReply
from stt_daemon.session import recovery

JITTER_MS = 20
"""WavCapture stamps each 20 ms chunk when it arrives; under load one can come a chunk late."""


def segments(client_msgs: list[Msg]) -> list[Msg]:
    return [m["data"] for m in client_msgs if m["type"] == "segment"]


async def run_fixture(
    daemon: Harness,
    replies: list[FakeReply | str],
    *,
    interval_s: float | None = None,
    max_connection_s: float | None = None,
) -> tuple[FakeBackend, WavCapture, TestClient]:
    """A word-timed session from the fixture's first sample, while the fixture plays."""
    backend = fake(replies, interval_s=interval_s)
    backend.capabilities = replace(backend.capabilities, max_connection_s=max_connection_s)
    capture = WavCapture(FIXTURE)
    await daemon.start(backend, capture=capture)
    client, _ = await daemon.connect("Scribbly")
    started = await client.request("start_session", {"word_timestamps": True})
    assert started["type"] == "session_started", started
    return backend, capture, client


async def test_word_times_map_back_through_a_pause_cut(daemon: Harness) -> None:
    words = [("one", 0.3, 0.5), ("two", 1.5, 1.7), ("three", 3.6, 3.8), ("four", 4.5, 4.7)]
    backend, capture, client = await run_fixture(
        daemon, [FakeReply("one two three four", words=words)]
    )
    await asyncio.to_thread(capture.wait_idle, 15)
    ended = await client.request("stop_session")
    assert ended["type"] == "session_ended"
    assert ended["data"]["reason"] == "stopped"
    assert ended["data"]["final_text"] == "one two three four"
    [conn] = backend.connections
    assert conn.sent_s < 9.3 - 1.5  # the 2 s pause and the leading silence were cut
    [segment] = segments(client.received)
    fixture = read_wav(FIXTURE)
    for word, (text, start_s, end_s) in zip(segment["words"], words, strict=True):
        assert word["text"] == text
        assert abs(word["start_ms"] - capture_ms(fixture, conn.audio, start_s)) <= JITTER_MS
        assert abs(word["end_ms"] - capture_ms(fixture, conn.audio, end_s)) <= JITTER_MS
    assert segment["words"][2]["start_ms"] > 5740  # after the cut, in the second phrase


async def test_several_commits_on_one_connection(daemon: Harness) -> None:
    replies: list[FakeReply | str] = [f"w{i}a w{i}b" for i in range(8)]
    backend, capture, client = await run_fixture(daemon, replies, interval_s=1.5)
    await asyncio.to_thread(capture.wait_idle, 15)
    ended = await client.request("stop_session")
    [conn] = backend.connections
    segs = segments(client.received)
    assert len(segs) >= 4
    # The fake spreads each reply's two words over its commit's audio: the first starts at the
    # commit's first sent sample, the second at its middle.
    spans = [(start, end) for start, end in conn.commit_spans if end > start]
    fixture = read_wav(FIXTURE)
    for seg, (start_s, end_s) in zip(segs, spans, strict=False):
        first, second = seg["words"]
        assert abs(first["start_ms"] - capture_ms(fixture, conn.audio, start_s)) <= JITTER_MS
        middle = capture_ms(fixture, conn.audio, (start_s + end_s) / 2)
        assert abs(second["start_ms"] - middle) <= JITTER_MS
    assert any(seg["start_ms"] > 5740 for seg in segs)  # commits after the cut map back too
    assert [s["index"] for s in segs] == list(range(len(segs)))
    for before, after in itertools.pairwise(segs):
        assert before["end_ms"] <= after["start_ms"]
    for s in segs:
        assert all(s["start_ms"] <= w["start_ms"] <= w["end_ms"] <= s["end_ms"] for w in s["words"])
    assert ended["data"]["segment_count"] == len(segs)


async def test_word_times_stay_true_across_a_rollover_and_a_replay(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 30 s limit: soft rollover 3 s into a connection, at the pause after the first phrase.
    monkeypatch.setattr(recovery, "SOFT_ROLLOVER_MARGIN_S", 27.0)
    monkeypatch.setattr(recovery, "HARD_ROLLOVER_MARGIN_S", 10.0)
    monkeypatch.setattr(recovery, "RECONNECT_COOLDOWN_S", 0.0)
    first = [("one", 0.2, 0.5), ("two", 1.5, 1.8)]
    second = [("three", 0.3, 0.6), ("four", 1.2, 1.5)]
    replies: list[FakeReply | str] = [
        FakeReply("one two", words=first),
        FakeReply("three four", words=second),
    ]
    backend, capture, client = await run_fixture(daemon, replies, max_connection_s=30.0)
    started = client.of_type("session_started")[0]["data"]["started_at"] / 1000
    await asyncio.sleep(max(0.0, started + 6.6 - time.time()))
    assert len(backend.connections) == 2
    backend.fail()
    await asyncio.to_thread(capture.wait_idle, 15)
    ended = await client.request("stop_session")
    old, lost, replayed = backend.connections
    assert lost.audio.size > 0
    fixture = read_wav(FIXTURE)
    a, b = segments(client.received)
    for word, (_, start_s, end_s) in zip(a["words"], first, strict=True):
        assert abs(word["start_ms"] - capture_ms(fixture, old.audio, start_s)) <= JITTER_MS
        assert abs(word["end_ms"] - capture_ms(fixture, old.audio, end_s)) <= JITTER_MS
    for word, (_, start_s, end_s) in zip(b["words"], second, strict=True):
        assert abs(word["start_ms"] - capture_ms(fixture, replayed.audio, start_s)) <= JITTER_MS
        assert abs(word["end_ms"] - capture_ms(fixture, replayed.audio, end_s)) <= JITTER_MS
    assert a["end_ms"] <= b["start_ms"]
    errors = client.of_type("session_error")
    assert [e["data"]["code"] for e in errors] == ["network"]
    assert errors[0]["data"]["fatal"] is False
    assert ended["data"]["final_text"] == "one two three four"
