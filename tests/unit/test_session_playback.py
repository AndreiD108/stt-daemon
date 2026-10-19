"""Mic sessions with the speakers' playback: which backends get a reference, the playback kept
in a quiet stretch, and the speakers released at the end."""

import asyncio
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from stt_daemon.audio.speakers import SpeakerMonitor
from stt_daemon.audio.system_capture import SystemCapture
from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.base import PLAYBACK_LEAD_S, OpenOptions
from stt_daemon.backends.fake import FAKE_CAPABILITIES, FakeBackend
from stt_daemon.config import Config
from stt_daemon.session.events import SessionEvent
from stt_daemon.session.options import SessionOptions
from stt_daemon.session.session import Session

FIXTURE = Path(__file__).parents[1] / "fixtures" / "two_phrases.wav"
RATE = 16000


class PlayingSpeakers(SystemCapture):
    """20 ms frames in real time; each sample's value is its monotonic time / 100, never 0."""

    def __init__(self) -> None:
        super().__init__(["true"])
        self.running = False
        self._pushing: asyncio.Task[None] | None = None

    async def hold(self) -> None:
        self.running = True
        self._pushing = asyncio.create_task(self._push_frames())

    async def release(self) -> None:
        self.running = False
        if self._pushing is not None:
            self._pushing.cancel()
        await super().release()

    async def _push_frames(self) -> None:
        began = time.monotonic()
        sent = 0
        while True:
            await asyncio.sleep(0.02)
            n = round((time.monotonic() - began) * RATE) - sent
            values = (began + (sent + np.arange(n)) / RATE) / 100
            sent += n
            self._hub.push(values.astype(np.float64).astype(np.float32), began + sent / RATE)


def monitor() -> tuple[SpeakerMonitor, list[PlayingSpeakers]]:
    made: list[PlayingSpeakers] = []

    def make() -> PlayingSpeakers:
        made.append(PlayingSpeakers())
        return made[-1]

    return SpeakerMonitor(make, keep_open=False), made


async def run(
    capture: WavCapture, backend: FakeBackend, speakers: SpeakerMonitor
) -> tuple[Session, list[SessionEvent]]:
    await capture.start()
    events: list[SessionEvent] = []
    session = Session(
        config=Config(), capture=capture, backend=backend,
        options=SessionOptions(OpenOptions(language="en")), emit=events.append,
        playback=speakers,
    )  # fmt: skip
    await session.start()
    return session, events


def taking_playback() -> FakeBackend:
    return FakeBackend(
        ["one", "two"], capabilities=replace(FAKE_CAPABILITIES, playback_reference=True)
    )


async def test_a_backend_that_takes_playback_gets_a_reference_per_commit() -> None:
    speakers, made = monitor()
    capture = WavCapture(FIXTURE)
    backend = taking_playback()
    try:
        session, _ = await run(capture, backend, speakers)
        await asyncio.to_thread(capture.wait_idle, 15)
        await session.stop()
    finally:
        await capture.stop()
    [conn] = backend.connections
    assert conn.options.mic
    [ref] = conn.references
    assert ref is not None
    # The pause was cut, and the tail held back at the stop follows a cut too.
    assert len(ref.spans) == 3
    assert ref.spans[0].offset == 0
    assert all(np.any(s.playback) for s in ref.spans)
    [speaker] = made
    assert not speaker.running


async def test_other_backends_get_no_reference_and_the_speakers_stay_off() -> None:
    speakers, made = monitor()
    capture = WavCapture(FIXTURE)
    backend = FakeBackend(["one"])
    try:
        session, _ = await run(capture, backend, speakers)
        await asyncio.to_thread(capture.wait_idle, 15)
        await session.stop()
    finally:
        await capture.stop()
    assert backend.connections[0].references == [None]
    assert not made


async def test_a_quiet_stretch_keeps_only_the_playback_it_may_still_send() -> None:
    speakers, _ = monitor()
    capture = WavCapture()  # silence
    backend = taking_playback()
    try:
        session, _ = await run(capture, backend, speakers)
        await asyncio.sleep(3.0)
        [track] = session._tracks  # pyright: ignore[reportPrivateUsage]
        playback = track._playback  # pyright: ignore[reportPrivateUsage]
        assert playback is not None
        now = time.monotonic()
        kept_from = now - PLAYBACK_LEAD_S - 1.0  # the pause gate holds well under 1 s
        assert not playback.read(now - 2.9, round((kept_from - (now - 2.9)) * RATE)).any()
        assert playback.read(now - PLAYBACK_LEAD_S - 0.1, 1600).all()
        await session.cancel()
    finally:
        await capture.stop()
