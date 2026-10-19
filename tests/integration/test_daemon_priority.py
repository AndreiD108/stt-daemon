"""Dictation over the socket on a full backend: a `transcribe` suspended and resumed whole."""

import asyncio
import base64
import os

import pytest
from harness import FIXTURE, RATE, Harness, Msg, TestClient, fake

from stt_daemon.audio.wav_capture import read_wav
from stt_daemon.session import session as session_module

PCM_16K = {"rate": RATE, "channels": 1}
SPLIT_S = 5.0
"""Past the first phrase's pause (1.00 to 3.74 s), before the second (5.74 s)."""


def fixture_pcm() -> bytes:
    return (read_wav(FIXTURE) * 32768).astype("<i2").tobytes()


def send_bytes(client: TestClient, data: bytes, chunk: int = 48 * 1024) -> None:
    for at in range(0, len(data), chunk):
        client.send("audio", {"data": base64.b64encode(data[at : at + chunk]).decode()})


def segments(client: TestClient) -> list[Msg]:
    return [
        {k: v for k, v in m["data"].items() if k != "session_id"} for m in client.of_type("segment")
    ]


async def transcribe(daemon: Harness, *, dictation: bool) -> TestClient:
    """The fixture through `transcribe`, split at `SPLIT_S`; with `dictation`, a mic session
    takes the backend's one slot between the halves and ends only after the stop was sent."""
    backend = fake(["one two", "three four"])
    backend.capacity = 1
    await daemon.start(backend)
    scribbly, _ = await daemon.connect("Scribbly")
    started = await scribbly.request("transcribe", {"pcm": PCM_16K, "word_timestamps": True})
    assert started["type"] == "session_started", started
    pcm = fixture_pcm()
    split = round(SPLIT_S * RATE) * 2
    send_bytes(scribbly, pcm[:split])
    await scribbly.event("segment", index=0)
    if dictation:
        talkey, _ = await daemon.connect("Talkey")
        assert (await talkey.request("start_session"))["type"] == "session_started"
        suspended = await scribbly.event("session_suspended")
        assert suspended["data"]["reason"] == "preempted"
        assert suspended["data"]["by"] == {
            "kind": "mic",
            "pid": os.getpid(),
            "client_label": "Talkey",
        }
        slot = await talkey.wait(
            lambda m: (
                m["type"] == "slot"
                and any(s["suspended"] for s in m["data"]["backends"]["fake"]["sessions"])
            )
        )
        kinds = [(s["kind"], s["suspended"]) for s in slot["data"]["backends"]["fake"]["sessions"]]
        assert kinds == [("mic", False), ("transcribe", True)]
        other, _ = await daemon.connect("Batch")
        refused = await other.request("transcribe", {"pcm": PCM_16K})
        assert refused["data"]["code"] == "busy"
        send_bytes(scribbly, pcm[split:])
        scribbly.send("stop_session")
        await asyncio.sleep(0.3)
        assert scribbly.of_type("session_ended") == []
        assert (await talkey.request("cancel_session"))["data"]["reason"] == "cancelled"
        resumed = await scribbly.event("session_resumed")
        assert resumed["data"]["suspended_seconds"] >= 0.3
    else:
        send_bytes(scribbly, pcm[split:])
        scribbly.send("stop_session")
    ended = await scribbly.event("session_ended")
    assert ended["data"]["reason"] == "stopped"
    return scribbly


async def test_a_suspended_transcribe_comes_out_as_if_never_suspended() -> None:
    plain_daemon, suspended_daemon = Harness(), Harness()
    try:
        plain = await transcribe(plain_daemon, dictation=False)
        suspended = await transcribe(suspended_daemon, dictation=True)
    finally:
        await plain_daemon.stop()
        await suspended_daemon.stop()
    assert [s["text"] for s in segments(plain)] == ["one two", "three four"]
    assert segments(suspended) == segments(plain)
    [plain_end] = plain.of_type("session_ended")
    [suspended_end] = suspended.of_type("session_ended")
    assert suspended_end["data"]["final_text"] == plain_end["data"]["final_text"]


async def test_a_transcribe_blocked_on_writes_while_suspended_stalls_no_one(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_module, "MIN_BACKLOG_S", 0.5)
    monkeypatch.setattr(session_module, "STALL_S", 0.5)
    backend = fake(interval_s=0.25)  # a commit cap under half the backlog limit, as real ones
    backend.capacity = 1
    await daemon.start(backend)
    scribbly, _ = await daemon.connect("Scribbly")
    assert (await scribbly.request("transcribe", {"pcm": PCM_16K}))["type"] == "session_started"
    talkey, _ = await daemon.connect("Talkey")
    assert (await talkey.request("start_session"))["type"] == "session_started"
    await scribbly.event("session_suspended")
    many = fixture_pcm() * 20  # far past the backlog limit and the command queue
    send_bytes(scribbly, many, chunk=16 * 1024)
    await asyncio.sleep(1.0)  # past STALL_S: suspended time is not a stall
    assert (await talkey.request("ping"))["type"] == "pong"
    observer, _ = await daemon.connect("Watch")
    assert (await observer.request("ping"))["type"] == "pong"
    assert scribbly.of_type("session_ended") == [], scribbly.of_type("session_error")
    assert (await talkey.request("stop_session"))["data"]["reason"] == "stopped"
    await scribbly.event("session_resumed")
    ended = await scribbly.request("stop_session", timeout=30)
    assert ended["data"]["reason"] == "stopped", ended


async def test_a_client_gone_while_its_suspended_transcribe_waits_on_input_is_dropped(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_module, "MIN_BACKLOG_S", 0.5)
    backend = fake(interval_s=0.25)
    backend.capacity = 1
    await daemon.start(backend)
    watch, _ = await daemon.connect("Watch", observe_all=True)
    scribbly, _ = await daemon.connect("Scribbly")
    assert (await scribbly.request("transcribe", {"pcm": PCM_16K}))["type"] == "session_started"
    talkey, _ = await daemon.connect("Talkey")
    assert (await talkey.request("start_session"))["type"] == "session_started"
    await scribbly.event("session_suspended")
    send_bytes(scribbly, fixture_pcm() * 20, chunk=16 * 1024)
    await asyncio.sleep(0.5)  # the daemon has stopped reading it
    scribbly.writer.transport.abort()  # close() would wait to send the buffered audio
    await watch.event("session_ended", timeout=3, reason="client_gone")
    await watch.wait(
        lambda m: (
            m["type"] == "slot"
            and [x["kind"] for x in m["data"]["backends"]["fake"]["sessions"]] == ["mic"]
        )
    )
    assert (await talkey.request("stop_session"))["data"]["reason"] == "stopped"
