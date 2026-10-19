"""The echo gate over the socket: the mic hears the speakers' echo 16 dB under the voice, then the
user's phrase, in one commit; a stand-in speakers capture plays the far end. Needs the model."""

import asyncio
import contextlib
import itertools
import time
from pathlib import Path

import aec_fixture
import numpy as np
import pytest
from harness import Harness, Msg

from stt_daemon.audio.frames import Attachment
from stt_daemon.audio.speakers import SpeakerMonitor
from stt_daemon.audio.system_capture import SystemCapture
from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.install import is_installed
from stt_daemon.backends.parakeet import Engine, ParakeetBackend, model_dir
from stt_daemon.config import ParakeetConfig

MODEL = model_dir()
RATE = 16000
FRAME = 320
MIC_NOISE_DB = -80.0
NEAR_TEXT = "Scribbly sends pen marks while Talkey holds the push to talk key."
FAR_AT_S = 1.0
"""two_phrases.wav's first phrase, 2.9 s: "The quick brown fox jumps over the lazy dog." """
NEAR_AT_S = 4.2
ECHO_DB = -16.0
"""The echo's level under the voice."""

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_installed(MODEL), reason="the Parakeet model is not installed"),
]


class Speakers(SystemCapture):
    """What the speakers play: silence, then `play`'s audio; 20 ms frames in real time."""

    def __init__(self) -> None:
        super().__init__(["true"], buffer_ms=3000)
        self._queued = np.zeros(0, np.float32)
        self._pushing: asyncio.Task[None] | None = None
        self.attached = 0

    def play(self, audio: np.ndarray) -> None:
        self._queued = audio.astype(np.float32)

    async def hold(self) -> None:
        self._pushing = asyncio.create_task(self._push_frames())

    async def release(self) -> None:
        if self._pushing is not None:
            self._pushing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pushing
        await super().release()

    def attach(self, since: float | None = None) -> Attachment:
        self.attached += 1
        return super().attach(since)

    async def _push_frames(self) -> None:
        began = time.monotonic()
        for k in itertools.count():
            await asyncio.sleep(max(0.0, began + (k + 1) * FRAME / RATE - time.monotonic()))
            frame, self._queued = self._queued[:FRAME], self._queued[FRAME:]
            frame = np.pad(frame, (0, FRAME - frame.size))
            self._hub.push(frame, time.monotonic())


def words(text: str) -> list[str]:
    return [w.strip(".,!?").lower() for w in text.split() if w.strip(".,!?")]


def voice_db(audio: np.ndarray) -> float:
    frames = audio[: audio.size // FRAME * FRAME].reshape(-1, FRAME)
    return float(np.percentile(10 * np.log10((frames**2).mean(axis=1) + 1e-12), 95))


def fixture() -> tuple[np.ndarray, np.ndarray]:
    """(mic, far): the far end's room echo at `ECHO_DB`, then jargon.wav's second phrase."""
    jargon = aec_fixture.read(aec_fixture.FIXTURES / "jargon.wav")
    phrases = aec_fixture.read(aec_fixture.FIXTURES / "two_phrases.wav")
    total = round(10.0 * RATE)
    near, far = np.zeros(total), np.zeros(total)
    spoken = aec_fixture.clip(jargon, 7.3, 11.3)
    near[round(NEAR_AT_S * RATE) :][: spoken.size] += spoken
    played = aec_fixture.clip(phrases, 0.9, 3.8)
    far[round(FAR_AT_S * RATE) :][: played.size] += played
    echo = aec_fixture.echo_of(far, aec_fixture.room_ir(), 0.0)
    echo *= 10 ** ((voice_db(near) + ECHO_DB - voice_db(echo)) / 20)
    noise = np.random.default_rng(1).standard_normal(total) * 10 ** (MIC_NOISE_DB / 20)
    return near + echo + noise, far


async def transcribe(
    daemon: Harness, tmp_path: Path, *, echo_gate: bool
) -> tuple[list[Msg], int, Speakers]:
    """The session's segments, where in it the fixture began (ms), and the speakers."""
    mic_audio, far = fixture()
    path = tmp_path / "mic.wav"
    aec_fixture.write(path, mic_audio)
    mic, speakers = WavCapture(), Speakers()
    backend = ParakeetBackend(ParakeetConfig(echo_gate=echo_gate), MODEL, Engine())
    await daemon.start(
        capture=mic,
        backends={"parakeet": backend},
        speakers=SpeakerMonitor(lambda: speakers, keep_open=True),
    )
    client, _ = await daemon.connect("Talkey")
    started = await client.request("start_session", {"word_timestamps": True})
    assert started["type"] == "session_started", started
    await asyncio.sleep(0.5)
    mic.play(path)
    speakers.play(far.astype(np.float32))
    played_ms = round(time.time() * 1000) - started["data"]["started_at"]
    await asyncio.to_thread(mic.wait_idle, 60)
    stopped = await client.request("stop_session", timeout=30)
    assert stopped["type"] == "session_ended", stopped
    segments = [m["data"] for m in client.of_type("segment") if m["data"]["text"]]
    return segments, played_ms, speakers


def echo_words(segments: list[Msg], played_ms: int) -> list[str]:
    """Words that start while only the echo plays."""
    start, end = FAR_AT_S * 1000, NEAR_AT_S * 1000 - 200
    return [
        w["text"] for s in segments for w in s["words"] if start <= w["start_ms"] - played_ms < end
    ]


async def test_echo_before_the_users_phrase_is_not_transcribed(
    daemon: Harness, tmp_path: Path
) -> None:
    segments, played_ms, speakers = await transcribe(daemon, tmp_path, echo_gate=True)
    assert speakers.attached == 1
    assert echo_words(segments, played_ms) == []
    heard = {w for s in segments for w in words(s["text"])}
    near = set(words(NEAR_TEXT))
    assert len(near & heard) >= len(near) - 2, heard
    # The gate cut the echo out of the decoded audio; times still point into the session's.
    [first] = [w for s in segments for w in s["words"] if words(w["text"]) == ["scribbly"]]
    assert abs(first["start_ms"] - played_ms - NEAR_AT_S * 1000) <= 300, (first, played_ms)


async def test_without_the_gate_the_echo_is_transcribed(daemon: Harness, tmp_path: Path) -> None:
    segments, played_ms, speakers = await transcribe(daemon, tmp_path, echo_gate=False)
    assert speakers.attached == 0
    assert len(echo_words(segments, played_ms)) >= 4
