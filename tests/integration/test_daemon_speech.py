"""VAD events over the socket and sessions started from a speech, on the fixture."""

import asyncio
import time

import numpy as np
from harness import FIXTURE, Harness, Msg, TestClient, fake

from stt_daemon.audio.wav_capture import WavCapture, read_wav
from stt_daemon.config import Config

RATE = 16000
JITTER_MS = 20
"""WavCapture stamps each 20 ms chunk when it arrives; under load one can come a chunk late."""
PAD_MS = Config().vad.pause_keep_ms
ONSET = 0.01
"""Fixture silence stays under this level; its speech goes over it at 1.00 s."""


def speech_events(client: TestClient) -> list[tuple[str, str]]:
    return [
        (m["type"], m["data"]["speech_id"])
        for m in client.received
        if m["type"] in ("speech_started", "speech_ended")
    ]


def onset(audio: np.ndarray) -> int:
    return int(np.flatnonzero(np.abs(audio) > ONSET)[0])


def assert_speech_whole(audio: np.ndarray, max_lead_ms: float) -> None:
    """The audio holds the fixture's first speech from its onset on, with some lead before it."""
    fixture = read_wav(FIXTURE)
    lead, start = onset(audio), onset(fixture)
    assert 0 < lead <= max_lead_ms * RATE / 1000
    assert np.array_equal(audio[lead : lead + RATE], fixture[start : start + RATE])


def index_of(client: TestClient, match: Msg) -> int:
    return next(i for i, m in enumerate(client.received) if m is match)


async def test_events_follow_the_slot_and_a_mid_speech_holder_gets_the_end(
    daemon: Harness,
) -> None:
    await daemon.start(capture=WavCapture(FIXTURE))
    talkey, _ = await daemon.connect("Talkey", vad_events=True)
    other, _ = await daemon.connect("Scribbly", vad_events=True)
    observer, _ = await daemon.connect("monitor", observe_all=True)
    plain, _ = await daemon.connect("plain")

    began = await talkey.event("speech_started", speech_id="s1")
    data = began["data"]
    assert data["expires_at"] - data["at"] == Config().audio.buffer_ms
    assert abs(data["at"] / 1000 - time.time()) < 1
    started = await talkey.request("start_session")  # mid-speech
    assert started["type"] == "session_started"
    ended = await talkey.event("speech_ended", speech_id="s1")
    assert index_of(talkey, ended) > index_of(talkey, started)
    await talkey.event("speech_ended", speech_id="s2")
    await talkey.request("stop_session")
    await observer.event("session_ended")

    s1 = [("speech_started", "s1"), ("speech_ended", "s1")]
    s2 = [("speech_started", "s2"), ("speech_ended", "s2")]
    assert speech_events(talkey) == s1 + s2
    assert speech_events(other) == s1[:1]
    assert speech_events(observer) == s1[1:] + s2
    assert speech_events(plain) == []


async def test_a_session_from_a_speech_id_takes_the_audio_from_the_speech_edge(
    daemon: Harness,
) -> None:
    backend = fake()
    capture = WavCapture(FIXTURE)
    await daemon.start(backend, capture=capture)
    client, _ = await daemon.connect("Talkey", vad_events=True)
    began = await client.event("speech_started", speech_id="s1")
    await asyncio.sleep(1.0)  # the start comes from the ring, well after the speech began
    started = await client.request("start_session", {"speech_id": "s1"})
    assert started["type"] == "session_started", started
    lead_ms = began["data"]["at"] - started["data"]["started_at"]
    assert abs(lead_ms - PAD_MS) <= JITTER_MS
    await asyncio.to_thread(capture.wait_idle, 15)
    await client.request("stop_session")
    [conn] = backend.connections
    assert_speech_whole(conn.audio, PAD_MS + 80)


async def test_unknown_and_expired_speech_ids_fail(daemon: Harness) -> None:
    config = Config()
    config.audio.buffer_ms = config.audio.pre_start_ms = 2000
    await daemon.start(capture=WavCapture(FIXTURE, buffer_ms=2000), config=config)
    client, _ = await daemon.connect("Talkey", vad_events=True)
    unknown = await client.request("start_session", {"speech_id": "s9"})
    assert unknown["data"]["code"] == "speech_expired"
    began = await client.event("speech_started", speech_id="s1")
    assert began["data"]["expires_at"] - began["data"]["at"] == 2000
    await asyncio.sleep(began["data"]["expires_at"] / 1000 - time.time() + 0.1)
    expired = await client.request("start_session", {"speech_id": "s1"})
    assert expired["data"]["code"] == "speech_expired"
    assert client.of_type("slot") == []


async def test_with_the_mic_closed_between_sessions_events_flow_only_in_sessions(
    daemon: Harness,
) -> None:
    capture = WavCapture(FIXTURE, keep_open=False)
    await daemon.start(capture=capture)
    client, welcome = await daemon.connect("Talkey", vad_events=True)
    assert welcome["data"]["mic_open"] is False
    await asyncio.sleep(1.5)
    assert speech_events(client) == []
    await client.request("start_session")  # the fixture plays from here
    await client.event("speech_started", speech_id="s1")
    await client.request("stop_session")
    reply = await client.request("start_session", {"speech_id": "s1"})
    assert reply["data"]["code"] == "speech_expired"


async def test_an_evicting_start_from_a_speech_id_gets_that_speech(daemon: Harness) -> None:
    backend = fake()
    capture = WavCapture()
    config = Config()
    config.slot.evictable_after_s = 1
    await daemon.start(backend, capture=capture, config=config)
    holder, _ = await daemon.connect("Scribbly", vad_events=True)
    talkey, _ = await daemon.connect("Talkey", vad_events=True, observe_all=True)
    await holder.request("start_session")
    await talkey.event("slot", state="evictable", timeout=3)
    capture.play(FIXTURE)
    await talkey.event("speech_started", speech_id="s1")
    await asyncio.sleep(0.5)
    started = await talkey.request("start_session", {"speech_id": "s1"})
    assert started["type"] == "session_started", started
    ended = await holder.event("session_ended")
    assert ended["data"]["reason"] == "evicted"
    await asyncio.to_thread(capture.wait_idle, 15)
    await talkey.request("stop_session")
    _, conn = backend.connections
    assert_speech_whole(conn.audio, PAD_MS + 80)
