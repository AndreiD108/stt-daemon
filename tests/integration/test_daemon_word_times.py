"""A backend that answers with text only: word times estimated from the audio, over the socket."""

import asyncio
import time
from dataclasses import replace
from pathlib import Path

from harness import FIXTURE, Harness, Msg, TestClient

from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.base import CommitPolicy
from stt_daemon.backends.fake import FAKE_CAPABILITIES, FakeBackend, FakeReply

JARGON = Path(__file__).parents[1] / "fixtures" / "jargon.wav"
TWO_PHRASES = [
    ((1000, 3740), "The quick brown fox jumps over the lazy dog."),
    ((5740, 7810), "Please send the report before noon tomorrow."),
]
JARGON_PHRASES = [
    ((1080, 5190), "Restart the stt-daemon, then open kitty and check the Huion tablet."),
    ((7240, 10980), "Scribbly sends pen marks while Talkey holds the push to talk key."),
    ((13390, 16730), "PipeWire routes the mic through sherpa onnx before the VAD runs."),
]
EDGE_MS = 150


async def session(
    daemon: Harness,
    fixture: Path,
    replies: list[FakeReply | str],
    *,
    interval_s: float | None = None,
    marker_at_ms: int | None = None,
) -> tuple[list[Msg], Msg]:
    """Segments and the end of a word-timed session from the fixture's first sample."""
    policy = CommitPolicy(interval_s=interval_s, check_every_s=0.1)
    capabilities = replace(FAKE_CAPABILITIES, commit_policy=policy)
    backend = FakeBackend(replies, capabilities=capabilities, text_only=True)
    capture = WavCapture(fixture)
    await daemon.start(backend, capture=capture)
    client, welcome = await daemon.connect("Scribbly")
    assert welcome["data"]["backends"][0]["word_timestamps"] is True
    started = await client.request("start_session", {"word_timestamps": True})
    assert started["type"] == "session_started", started
    if marker_at_ms is not None:
        at = started["data"]["started_at"] + marker_at_ms
        await asyncio.sleep(max(0.0, at / 1000 - time.time()))
        reply = await client.request("add_marker", {"marker": "⟦A⟧", "at": at})
        assert reply["type"] == "marker_accepted", reply
    await asyncio.to_thread(capture.wait_idle, 25)
    ended = await client.request("stop_session")
    assert ended["type"] == "session_ended", ended
    return segments(client), ended["data"]


def segments(client: TestClient) -> list[Msg]:
    return [m["data"] for m in client.of_type("segment") if m["data"]["text"]]


def check_phrases(words: list[Msg], phrases: list[tuple[tuple[int, int], str]]) -> None:
    edges = [t for w in words for t in (w["start_ms"], w["end_ms"])]
    assert edges == sorted(edges), "times out of order or overlapping"
    at = 0
    for (start_ms, end_ms), text in phrases:
        phrase = words[at : at + len(text.split())]
        at += len(phrase)
        assert [w["text"] for w in phrase] == text.split()
        for word in phrase:
            assert start_ms - EDGE_MS <= word["start_ms"] <= word["end_ms"] <= end_ms + EDGE_MS
        assert abs(phrase[0]["start_ms"] - start_ms) <= EDGE_MS, phrase[0]
        assert abs(phrase[-1]["end_ms"] - end_ms) <= EDGE_MS, phrase[-1]


async def test_text_only_answers_get_word_times_inside_their_phrases(daemon: Harness) -> None:
    text = " ".join(t for _, t in TWO_PHRASES)
    # Marker during "report before", in the second phrase.
    [segment], ended = await session(daemon, FIXTURE, [text], marker_at_ms=6500)
    check_phrases(segment["words"], TWO_PHRASES)
    [marker] = segment["markers"]
    assert len(TWO_PHRASES[0][1].split()) <= marker["after_word"] < len(segment["words"]) - 1
    assert "⟦A⟧" in ended["final_text"].split("Please", 1)[1]


async def test_three_phrases_with_jargon(daemon: Harness) -> None:
    text = " ".join(t for _, t in JARGON_PHRASES)
    [segment], _ = await session(daemon, JARGON, [text])
    check_phrases(segment["words"], JARGON_PHRASES)


async def test_later_commits_on_a_connection_map_to_their_own_audio(daemon: Harness) -> None:
    replies: list[FakeReply | str] = [f"w{i}a w{i}b w{i}c" for i in range(12)]
    segs, _ = await session(daemon, FIXTURE, replies, interval_s=1.0)
    assert len(segs) >= 4
    words = [w for s in segs for w in s["words"]]
    edges = [t for w in words for t in (w["start_ms"], w["end_ms"])]
    assert edges == sorted(edges)
    for seg in segs:
        # A commit of trailing silence has no speech to anchor to: its words spread over it all.
        speech = any(s < seg["end_ms"] and seg["start_ms"] < e for (s, e), _ in TWO_PHRASES)
        for word in seg["words"]:
            assert seg["start_ms"] <= word["start_ms"] <= word["end_ms"] <= seg["end_ms"]
            inside = [s - EDGE_MS <= word["start_ms"] <= e + EDGE_MS for (s, e), _ in TWO_PHRASES]
            assert any(inside) or not speech, word
