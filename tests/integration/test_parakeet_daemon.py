"""Parakeet sessions over the socket on a TTS fixture; needs the model installed."""

import base64
import itertools
import logging
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from harness import FIXTURE, Harness, Msg

from stt_daemon.audio.frames import Frame
from stt_daemon.audio.pw_capture import PwRecordCapture
from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.base import Committed
from stt_daemon.backends.install import is_installed
from stt_daemon.backends.parakeet import (
    Engine,
    ParakeetBackend,
    RecognizerKey,
    load_recognizer,
    model_dir,
)
from stt_daemon.config import ParakeetConfig
from stt_daemon.session import recovery
from stt_daemon.session.word_times import Speech

MODEL = model_dir()
JARGON = Path(__file__).parents[1] / "fixtures" / "jargon.wav"
FAKE_PW_RECORD = Path(__file__).parents[1] / "fake_pw_record.py"
"""18.7 s, three phrases; speech above -40 dBFS at 1.08-5.19, 7.24-10.98, 13.39-16.73 s:
"Restart the stt-daemon, then open kitty and check the Huion tablet."
"Scribbly sends pen marks while Talkey holds the push to talk key."
"PipeWire routes the mic through sherpa onnx before the VAD runs." """
PHRASE_STARTS_MS = (1080, 7240, 13390)
KEYTERMS = ["stt-daemon", "kitty", "Huion", "Scribbly", "Talkey", "PipeWire", "sherpa onnx"]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_installed(MODEL), reason="the Parakeet model is not installed"),
]


async def _session(
    daemon: Harness, keyterms: list[str], loaded: list[RecognizerKey]
) -> tuple[list[Msg], Msg]:
    def loader(key: RecognizerKey) -> object:
        loaded.append(key)
        return load_recognizer(key)

    backend = ParakeetBackend(ParakeetConfig(), MODEL, Engine(loader))
    await daemon.start(capture=WavCapture(JARGON), backends={"parakeet": backend})
    client, welcome = await daemon.connect("Scribbly")
    assert welcome["data"]["backends"][0]["name"] == "parakeet"
    started = await client.request("start_session", {"word_timestamps": True, "keyterms": keyterms})
    assert started["type"] == "session_started", started
    # The 2.0 and 2.4 s pauses are too short to commit early in a chunk; the silence after
    # the file commits the third phrase.
    await client.event("segment", timeout=30.0, index=1)
    stopped = await client.request("stop_session")
    assert stopped["type"] == "session_ended", stopped
    return [m["data"] for m in client.of_type("segment")], stopped["data"]


def _phrase_starts(spoken: list[Msg]) -> list[Msg]:
    """Each segment's first word and every word after one that ends a sentence."""
    return [
        word
        for segment in spoken
        for before, word in zip([None, *segment["words"]], segment["words"], strict=False)
        if before is None or before["text"].endswith(".")
    ]


async def test_segments_commit_at_pauses_with_word_times(daemon: Harness) -> None:
    loaded: list[RecognizerKey] = []
    segments, ended = await _session(daemon, [], loaded)
    assert len(loaded) == 1
    texts = [s["text"] for s in segments if s["text"]]
    assert len(texts) == 2, texts
    assert "tablet" in texts[0]
    assert "pen marks" in texts[0]
    assert "routes the mic" in texts[1]
    assert ended["final_text"].lower().startswith("restart ")  # the next word varies with the gate
    spoken = [s for s in segments if s["text"]]
    for word, edge_ms in zip(_phrase_starts(spoken), PHRASE_STARTS_MS, strict=True):
        assert abs(word["start_ms"] - edge_ms) <= 200, (word, edge_ms)
    for segment in spoken:
        words = segment["words"]
        assert " ".join(w["text"] for w in words) == segment["text"]
        assert all(a["end_ms"] <= b["start_ms"] + 1 for a, b in itertools.pairwise(words))
        assert segment["start_ms"] <= words[0]["start_ms"]
        assert words[-1]["end_ms"] <= segment["end_ms"]


async def test_estimated_word_times_come_near_parakeets_own(
    daemon: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    errors_ms: list[float] = []
    real_with_times = recovery.with_times

    def compare(event: Committed, frames: list[Frame], speech: Speech, offset: int) -> Committed:
        if event.words:
            estimated = real_with_times(replace(event, words=None), frames, speech, offset)
            assert estimated.words is not None
            for real, guess in zip(event.words, estimated.words, strict=True):
                errors_ms.append((guess.start_s - real.start_s) * 1000)
        return event

    monkeypatch.setattr(recovery, "with_times", compare)
    await _session(daemon, [], [])
    caplog.set_level(logging.INFO)
    median = statistics.median(abs(e) for e in errors_ms)
    logging.getLogger(__name__).info(
        "estimate - Parakeet start, ms: median |error| %.0f, %s",
        median,
        [round(e) for e in errors_ms],
    )
    assert len(errors_ms) >= 30
    assert median < 120


async def test_keyterms_go_in_per_stream(daemon: Harness) -> None:
    loaded: list[RecognizerKey] = []
    segments, ended = await _session(daemon, KEYTERMS, loaded)
    assert len(loaded) == 1
    texts = [s["text"] for s in segments if s["text"]]
    assert len(texts) == 2, texts
    # Without keyterms: "Talkie", "Pipewire".
    assert "Talkey" in texts[0]
    assert "PipeWire" in texts[1]
    assert ended["final_text"]


async def test_a_file_through_transcribe_gets_word_times_in_file_time(daemon: Harness) -> None:
    backend = ParakeetBackend(ParakeetConfig(), MODEL, Engine(load_recognizer))
    await daemon.start(backends={"parakeet": backend})
    client, _ = await daemon.connect("Scribbly")
    started = await client.request("transcribe", {"word_timestamps": True})
    assert started["type"] == "session_started", started
    assert started["data"]["kind"] == "transcribe"
    data = JARGON.read_bytes()
    for at in range(0, len(data), 64 * 1024):
        client.send("audio", {"data": base64.b64encode(data[at : at + 64 * 1024]).decode()})
    stopped = await client.request("stop_session", timeout=30.0)
    assert stopped["data"]["reason"] == "stopped", stopped
    spoken = [m["data"] for m in client.of_type("segment") if m["data"]["text"]]
    assert len(spoken) == 2, [s["text"] for s in spoken]
    assert "pen marks" in spoken[0]["text"]
    assert "routes the mic" in spoken[1]["text"]
    for word, edge_ms in zip(_phrase_starts(spoken), PHRASE_STARTS_MS, strict=True):
        assert abs(word["start_ms"] - edge_ms) <= 200, (word, edge_ms)


async def test_a_system_session_decodes_both_tracks_on_one_model(daemon: Harness) -> None:
    loaded: list[RecognizerKey] = []

    def loader(key: RecognizerKey) -> object:
        loaded.append(key)
        return load_recognizer(key)

    backend = ParakeetBackend(ParakeetConfig(), MODEL, Engine(loader))
    mic = WavCapture()
    await daemon.start(
        capture=mic,
        backends={"parakeet": backend},
        system_capture=lambda _: PwRecordCapture(
            [sys.executable, str(FAKE_PW_RECORD), "--wav", str(FIXTURE)]
        ),
    )
    client, _ = await daemon.connect("Scribbly")
    started = await client.request(
        "start_system_session", {"with_mic": True, "word_timestamps": True}
    )
    assert started["type"] == "session_started", started
    mic.play(JARGON)

    def spoken(track: str) -> list[Msg]:
        return [
            m["data"]
            for m in client.of_type("segment")
            if m["data"]["track"] == track and m["data"]["text"]
        ]

    await client.wait(lambda _m: len(spoken("system")) == 1 and len(spoken("mic")) == 2, 30.0)
    stopped = await client.request("stop_session")
    assert stopped["data"]["reason"] == "stopped", stopped
    assert len(loaded) == 1
    assert "tablet" in spoken("mic")[0]["text"]
    assert "pen marks" in spoken("mic")[0]["text"]
    assert "routes the mic" in spoken("mic")[1]["text"]
    lines = stopped["data"]["final_text"].splitlines()
    assert {line.split(" ", 1)[0] for line in lines} == {"[system]", "[mic]"}
