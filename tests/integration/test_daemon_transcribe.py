"""`transcribe` sessions over the socket: files through ffmpeg, raw PCM, audio-time word times,
backend slots next to mic sessions, decoder failures."""

import asyncio
import base64
import contextlib
import io
import os
import shutil
import time
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from harness import FIXTURE, RATE, Harness, Msg, TestClient, capture_ms, fake

from stt_daemon.audio.wav_capture import read_wav
from stt_daemon.backends.base import CommitPolicy
from stt_daemon.backends.fake import FAKE_CAPABILITIES, FakeBackend, FakeReply
from stt_daemon.config import Config, SessionConfig
from stt_daemon.server import starts as server_starts
from stt_daemon.session import session as session_module

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
PCM_16K = {"rate": RATE, "channels": 1}


def missing(tool: str) -> str:
    return f"{tool} is missing"


def fixture_pcm() -> bytes:
    return (read_wav(FIXTURE) * 32768).astype("<i2").tobytes()


def send_bytes(client: TestClient, data: bytes, chunk: int = 48 * 1024) -> None:
    for at in range(0, len(data), chunk):
        client.send("audio", {"data": base64.b64encode(data[at : at + chunk]).decode()})


def segments(client: TestClient) -> list[Msg]:
    return [m["data"] for m in client.of_type("segment")]


@needs_ffmpeg
async def test_a_wav_file_in_chunks_commits_at_pauses_with_word_times_in_file_time(
    daemon: Harness,
) -> None:
    backend = fake(["one two", "three four"])
    await daemon.start(backend)
    client, _ = await daemon.connect("Scribbly")
    started = await client.request("transcribe", {"word_timestamps": True})
    assert started["type"] == "session_started", started
    assert started["data"]["kind"] == "transcribe"
    began = time.monotonic()
    send_bytes(client, FIXTURE.read_bytes())
    ended = await client.request("stop_session")
    elapsed = time.monotonic() - began
    assert ended["type"] == "session_ended", ended
    assert ended["data"]["reason"] == "stopped"
    assert ended["data"]["final_text"] == "one two three four"
    fixture = read_wav(FIXTURE)
    duration_s = fixture.size / RATE
    assert elapsed < duration_s / 2  # unpaced: faster than the audio plays
    first, second = segments(client)
    [conn] = backend.connections
    for seg, (low, high) in zip((first, second), ((1000, 3740), (5740, 7810)), strict=True):
        words = seg["words"]
        # The fake starts a commit's first word at its first sample: the pre-speech pad.
        assert low - 150 <= words[0]["start_ms"] <= low
        assert all(low - 150 <= w["start_ms"] <= w["end_ms"] <= high + 1000 for w in words)
    spans = [(start, end) for start, end in conn.commit_spans if end > start]
    # A third commit sends the silence held at the end; its empty text makes no segment.
    for seg, (start_s, _end_s) in zip((first, second), spans[:2], strict=True):
        # Exact: file time, no capture jitter.
        assert abs(seg["words"][0]["start_ms"] - capture_ms(fixture, conn.audio, start_s)) <= 1
    progress = [m["data"]["audio_seconds"] for m in client.of_type("progress")]
    assert progress[-1] == pytest.approx(duration_s, abs=0.001)
    assert len(progress) <= elapsed + 2


async def test_raw_pcm_at_16k_mono_needs_no_ffmpeg(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_starts, "check_tool", missing)
    await daemon.start(fake(["one two", "three four"]))
    client, _ = await daemon.connect("Scribbly")
    started = await client.request("transcribe", {"pcm": PCM_16K})
    assert started["type"] == "session_started", started
    send_bytes(client, fixture_pcm(), chunk=1001)  # odd sizes split samples across chunks
    ended = await client.request("stop_session")
    assert ended["data"]["final_text"] == "one two three four"
    assert [s["start_ms"] for s in segments(client)][1] > 5600


async def test_without_ffmpeg_a_file_gets_tool_unavailable_and_the_slot_stays_free(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_starts, "check_tool", missing)
    await daemon.start(fake())
    client, _ = await daemon.connect("Scribbly")
    refused = await client.request("transcribe", {})
    assert refused["type"] == "error"
    assert refused["data"]["code"] == "tool_unavailable"
    assert refused["data"]["tool"] == "ffmpeg"
    _, welcome = await daemon.connect("Talkey")
    assert welcome["data"]["slot"]["backends"]["fake"]["sessions"] == []


async def test_a_second_transcribe_on_a_full_backend_is_busy(daemon: Harness) -> None:
    backend = fake()
    backend.capacity = 1
    await daemon.start(backend)
    first, _ = await daemon.connect("Scribbly")
    assert (await first.request("transcribe", {"pcm": PCM_16K}))["type"] == "session_started"
    second, _ = await daemon.connect("Talkey")
    refused = await second.request("transcribe", {"pcm": PCM_16K})
    assert refused["data"]["code"] == "busy"
    assert refused["data"]["full_slot"] == "backend"
    assert refused["data"]["holder"]["kind"] == "transcribe"


async def test_a_mic_session_and_a_transcribe_run_at_once(daemon: Harness) -> None:
    await daemon.start(fake(["mic words", "file words"]))
    talkey, _ = await daemon.connect("Talkey")
    scribbly, _ = await daemon.connect("Scribbly")
    assert (await talkey.request("start_session"))["type"] == "session_started"
    started = await scribbly.request("transcribe", {"pcm": PCM_16K})
    assert started["type"] == "session_started", started
    slot = await talkey.wait(
        lambda m: m["type"] == "slot" and len(m["data"]["backends"]["fake"]["sessions"]) == 2
    )
    kinds = {s["kind"] for s in slot["data"]["backends"]["fake"]["sessions"]}
    assert kinds == {"mic", "transcribe"}
    assert slot["data"]["state"] == "busy"  # the mic slot is the mic session's alone
    send_bytes(scribbly, fixture_pcm())
    ended = await scribbly.request("stop_session")
    assert ended["data"]["reason"] == "stopped"
    assert (await talkey.request("stop_session"))["data"]["reason"] == "stopped"


async def test_audio_commands_are_checked(daemon: Harness) -> None:
    await daemon.start(fake())
    client, _ = await daemon.connect("Scribbly")
    none = await client.request("audio", {"data": "AAAA"})
    assert none["data"]["code"] == "no_session"
    assert (await client.request("transcribe", {"pcm": PCM_16K}))["type"] == "session_started"
    bad = await client.request("audio", {"data": "not base64!"})
    assert (bad["data"]["code"], bad["data"]["field"]) == ("invalid", "data")
    big = base64.b64encode(bytes((1 << 20) + 1)).decode()
    too_big = await client.request("audio", {"data": big})
    assert (too_big["data"]["code"], too_big["data"]["field"]) == ("invalid", "data")
    rate = await client.request("transcribe", {"pcm": {"rate": 4000, "channels": 1}})
    assert rate["data"]["code"] == "bad_request"


@needs_ffmpeg
async def test_input_ffmpeg_cannot_read_ends_the_session_with_its_message(
    daemon: Harness,
) -> None:
    await daemon.start(fake())
    client, _ = await daemon.connect("Scribbly")
    assert (await client.request("transcribe", {}))["type"] == "session_started"
    send_bytes(client, b"this is not audio at all " * 200)
    client.send("stop_session")
    ended = await client.event("session_ended")
    assert ended["data"]["reason"] == "error"
    [error] = client.of_type("session_error")
    assert error["data"]["code"] == "invalid_request"
    assert error["data"]["fatal"] is True
    assert "ffmpeg" in error["data"]["message"]


async def test_markers_land_by_file_time(daemon: Harness) -> None:
    await daemon.start(fake(["one two", "three four"]))
    client, _ = await daemon.connect("Scribbly")
    started = await client.request("transcribe", {"pcm": PCM_16K})
    marker_at = started["data"]["started_at"] + 5000  # in the pause between the phrases
    accepted = await client.request("add_marker", {"marker": "¶", "at": marker_at})
    assert accepted["data"]["at_ms"] == 5000
    send_bytes(client, fixture_pcm())
    await client.request("stop_session")
    first, second = segments(client)
    assert "markers" not in first
    assert [m["at_ms"] for m in second["markers"]] == [5000]


async def test_cancel_and_disconnect_free_the_backend_slot(daemon: Harness) -> None:
    await daemon.start(fake())
    client, _ = await daemon.connect("Scribbly")
    assert (await client.request("transcribe", {"pcm": PCM_16K}))["type"] == "session_started"
    send_bytes(client, np.zeros(RATE, dtype="<i2").tobytes())
    cancelled = await client.request("cancel_session")
    assert cancelled["data"]["reason"] == "cancelled"
    assert (await client.request("transcribe", {"pcm": PCM_16K}))["type"] == "session_started"
    watcher, _ = await daemon.connect("Talkey")
    await client.close()
    await watcher.wait(
        lambda m: m["type"] == "slot" and m["data"]["backends"]["fake"]["sessions"] == []
    )


async def test_a_full_line_of_audio_fits(daemon: Harness) -> None:
    await daemon.start(fake())
    client, _ = await daemon.connect("Scribbly")
    assert (await client.request("transcribe", {"pcm": PCM_16K}))["type"] == "session_started"
    send_bytes(client, bytes(1 << 20), chunk=1 << 20)
    ended = await client.request("stop_session")
    assert ended["data"]["reason"] == "stopped"
    progress = client.of_type("progress")
    assert progress[-1]["data"]["audio_seconds"] == pytest.approx((1 << 19) / RATE, abs=0.001)


def stalled_backend() -> FakeBackend:
    """Commits every 0.25 s of audio; the first answer never comes."""
    policy = CommitPolicy(max_chunk_s=0.25)
    return FakeBackend(
        [FakeReply("never", delay_s=3600)],
        capabilities=replace(FAKE_CAPABILITIES, commit_policy=policy),
    )


def wav_bytes(seconds: float) -> bytes:
    fixture = read_wav(FIXTURE)
    tiled = np.tile(fixture, int(seconds * RATE) // fixture.size + 1)[: round(seconds * RATE)]
    pcm = (tiled * 32768).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())
    return out.getvalue()


def ffmpeg_children() -> list[int]:
    found: list[int] = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        with contextlib.suppress(OSError):
            text = stat.read_text()
            name = text[text.index("(") + 1 : text.rindex(")")]
            ppid = int(text[text.rindex(")") + 2 :].split()[1])
            if name == "ffmpeg" and ppid == os.getpid():
                found.append(int(stat.parent.name))
    return found


async def stalled_transcribe(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch, stop_wait_ms: int, seconds: float = 3.0
) -> tuple[TestClient, TestClient]:
    """A file session whose audio loop waits on the backend while ffmpeg's output is queued."""
    monkeypatch.setattr(session_module, "MIN_BACKLOG_S", 0.5)
    await daemon.start(stalled_backend(), config=Config(session=SessionConfig(stop_wait_ms)))
    watcher, _ = await daemon.connect("Talkey")
    client, _ = await daemon.connect("Scribbly")
    assert (await client.request("transcribe", {}))["type"] == "session_started"
    send_bytes(client, wav_bytes(seconds))
    return client, watcher


async def backend_freed(watcher: TestClient) -> None:
    await watcher.wait(
        lambda m: m["type"] == "slot" and m["data"]["backends"]["fake"]["sessions"] == []
    )


@needs_ffmpeg
async def test_a_stalled_backend_cannot_hold_a_stop_past_its_wait(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, watcher = await stalled_transcribe(daemon, monkeypatch, stop_wait_ms=300)
    ended = await client.request("stop_session")
    assert ended["data"]["reason"] == "stopped"
    await backend_freed(watcher)
    assert ffmpeg_children() == []


@needs_ffmpeg
async def test_a_disconnect_ends_a_stopping_transcribe(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, watcher = await stalled_transcribe(daemon, monkeypatch, stop_wait_ms=60_000)
    client.send("stop_session")
    await asyncio.sleep(0.3)
    await client.close()
    await backend_freed(watcher)
    assert ffmpeg_children() == []


@needs_ffmpeg
async def test_shutdown_ends_a_stopping_transcribe(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = await stalled_transcribe(daemon, monkeypatch, stop_wait_ms=60_000)
    client.send("stop_session")
    await asyncio.sleep(0.3)
    server, daemon.server = daemon.server, None
    assert server is not None
    async with asyncio.timeout(10):
        await server.close()
    assert ffmpeg_children() == []
    ended = await client.event("session_ended")
    assert ended["data"]["reason"] == "cancelled"


async def test_a_backend_failure_frees_a_connection_waiting_to_send_audio(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_module, "MIN_BACKLOG_S", 0.5)
    backend = stalled_backend()
    await daemon.start(backend)
    client, _ = await daemon.connect("Scribbly")
    assert (await client.request("transcribe", {"pcm": PCM_16K}))["type"] == "session_started"
    send_bytes(client, fixture_pcm(), chunk=16 * 1024)  # more than the session's queue holds
    await asyncio.sleep(0.3)
    backend.fail("auth", retryable=False)
    ended = await client.event("session_ended")
    assert ended["data"]["reason"] == "error"
    assert (await client.request("transcribe", {"pcm": PCM_16K}))["type"] == "session_started"


@needs_ffmpeg
async def test_a_backend_that_answers_nothing_fails_a_session_whose_input_waits(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_module, "STALL_S", 1.0)
    # 30 s fills ffmpeg's pipes and the command queue: the input waits on the backend.
    client, watcher = await stalled_transcribe(daemon, monkeypatch, 60_000, seconds=30.0)
    client.send("stop_session")
    ended = await client.event("session_ended")
    assert ended["data"]["reason"] == "error"
    [error] = client.of_type("session_error")
    assert (error["data"]["code"], error["data"]["fatal"]) == ("backend_error", True)
    await backend_freed(watcher)
    assert ffmpeg_children() == []
    assert (await client.request("transcribe", {"pcm": PCM_16K}))["type"] == "session_started"
