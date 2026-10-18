"""Parakeet sessions over the socket on a TTS fixture; needs the model installed."""

import itertools
from pathlib import Path

import pytest
from harness import Harness, Msg

from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.install import is_installed
from stt_daemon.backends.parakeet import (
    Engine,
    ParakeetBackend,
    RecognizerKey,
    load_recognizer,
    model_dir,
)
from stt_daemon.config import ParakeetConfig

MODEL = model_dir()
JARGON = Path(__file__).parents[1] / "fixtures" / "jargon.wav"
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
    await client.event("segment", timeout=25.0, index=2)
    stopped = await client.request("stop_session")
    assert stopped["type"] == "session_ended", stopped
    return [m["data"] for m in client.of_type("segment")], stopped["data"]


async def test_segments_commit_at_pauses_with_word_times(daemon: Harness) -> None:
    loaded: list[RecognizerKey] = []
    segments, ended = await _session(daemon, [], loaded)
    assert [k.beam for k in loaded] == [False]
    texts = [s["text"] for s in segments if s["text"]]
    assert len(texts) == 3, texts
    assert "tablet" in texts[0]
    assert "pen marks" in texts[1]
    assert "routes the mic" in texts[2]
    assert ended["final_text"].startswith("Restart the")
    spoken = [s for s in segments if s["text"]]
    for segment, edge_ms in zip(spoken, PHRASE_STARTS_MS, strict=True):
        words = segment["words"]
        assert " ".join(w["text"] for w in words) == segment["text"]
        assert abs(words[0]["start_ms"] - edge_ms) <= 200, (words[0], edge_ms)
        assert all(a["end_ms"] <= b["start_ms"] + 1 for a, b in itertools.pairwise(words))
        assert segment["start_ms"] <= words[0]["start_ms"]
        assert words[-1]["end_ms"] <= segment["end_ms"]


async def test_keyterms_take_the_beam_search_path(daemon: Harness) -> None:
    loaded: list[RecognizerKey] = []
    segments, ended = await _session(daemon, KEYTERMS, loaded)
    assert [k.beam for k in loaded] == [True]
    texts = [s["text"] for s in segments if s["text"]]
    assert len(texts) == 3, texts
    assert "pen marks" in texts[1]
    assert "routes the mic" in texts[2]
    assert ended["final_text"]
