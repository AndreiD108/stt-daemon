"""The daemon-wide VAD on pushed frames: events, speech starts by id, the mic going quiet."""

import asyncio
import time

from stt_daemon.audio.capture import CaptureSource
from stt_daemon.audio.vad import GAP_RESET_S, SpeechEnded, SpeechStarted, VadEvent
from stt_daemon.audio.wav_capture import read_wav
from stt_daemon.config import VadConfig
from stt_daemon.server.speech import SpeechWatch
from unit.paths import TWO_PHRASES as FIXTURE

"""Phrases at 1.00-3.74 s and 5.74-7.81 s."""
RATE = 16000
CHUNK = 320
START_TOLERANCE_S = 0.08
END_TOLERANCE_S = 0.12


class PushedCapture(CaptureSource):
    """Never opens anything; tests push frames into its hub."""

    def __init__(self, buffer_ms: int = 5000) -> None:
        super().__init__(sample_rate=RATE, buffer_ms=buffer_ms, keep_open=True)

    @property
    def is_open(self) -> bool:
        return True

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    def push_audio(self, seconds: float, origin: float) -> None:
        """The fixture's first `seconds`, stamped as if captured from `origin` on."""
        audio = read_wav(FIXTURE)[: round(seconds * RATE)]
        for i in range(0, audio.size, CHUNK):
            chunk = audio[i : i + CHUNK].copy()
            self._hub.push(chunk, origin + (i + chunk.size) / RATE)


async def watching(capture: CaptureSource) -> tuple[SpeechWatch, list[VadEvent]]:
    events: list[VadEvent] = []
    watch = SpeechWatch(capture=capture, vad=VadConfig(), buffer_ms=5000, on_event=events.append)
    watch.start()
    await asyncio.sleep(0)  # the watch attaches before frames come
    return watch, events


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_events_carry_the_true_edges_and_starts_last_while_the_ring_holds_them() -> None:
    capture = PushedCapture()
    watch, events = await watching(capture)
    origin = time.monotonic() - 9.3
    capture.push_audio(9.3, origin)
    await settle()
    await watch.close()
    assert [type(e) for e in events] == [SpeechStarted, SpeechEnded, SpeechStarted, SpeechEnded]
    assert [e.speech_id for e in events] == ["s1", "s1", "s2", "s2"]
    for event, edge in zip(events, (1.00, 3.74, 5.74, 7.81), strict=True):
        tolerance = START_TOLERANCE_S if isinstance(event, SpeechStarted) else END_TOLERANCE_S
        assert abs(event.at - origin - edge) <= tolerance
    assert watch.start_of("s1") is None  # the 5 s ring starts at 4.3 s
    assert watch.start_of("s2") == events[2].at
    assert watch.start_of("s9") is None


async def test_speech_ends_when_frames_stop() -> None:
    capture = PushedCapture()
    watch, events = await watching(capture)
    origin = time.monotonic() - 2.0
    capture.push_audio(2.0, origin)  # cut inside the first phrase
    await settle()
    assert [type(e) for e in events] == [SpeechStarted]
    await asyncio.sleep(GAP_RESET_S + 0.2)
    await watch.close()
    assert [type(e) for e in events] == [SpeechStarted, SpeechEnded]
    assert abs(events[1].at - (origin + 2.0)) < 0.001
