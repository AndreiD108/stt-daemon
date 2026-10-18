"""The daemon-wide VAD: speech events from all mic audio, and recent speech starts by id."""

import asyncio
import contextlib
import time
from collections.abc import Callable

from ..audio.capture import CaptureSource
from ..audio.vad import GAP_RESET_S, SpeechStarted, StreamingVAD, VadEvent
from ..config import VadConfig


class SpeechWatch:
    """Runs a `StreamingVAD` over every frame while the mic is open, sessions or not.

    Sessions run their own VAD on their own audio; this one feeds VAD events and `speech_id`
    starts. A speech in progress ends when frames stop for `GAP_RESET_S` (the mic closed).
    """

    def __init__(
        self,
        *,
        capture: CaptureSource,
        vad: VadConfig,
        buffer_ms: int,
        on_event: Callable[[VadEvent], None],
    ) -> None:
        self._capture = capture
        self._vad = StreamingVAD(
            sample_rate=capture.sample_rate,
            threshold=vad.threshold,
            min_speech_ms=vad.min_speech_ms,
            min_silence_ms=vad.min_silence_ms,
        )
        self._buffer_s = buffer_ms / 1000
        self._on_event = on_event
        self._starts: dict[str, float] = {}
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def start_of(self, speech_id: str) -> float | None:
        """Monotonic start of that speech while the ring still holds it, else None."""
        at = self._starts.get(speech_id)
        ring_start = self._capture.ring_start()
        if at is None or ring_start is None or ring_start > at:
            return None
        return at

    async def _run(self) -> None:
        attachment = self._capture.listen()
        try:
            while True:
                try:
                    async with asyncio.timeout(GAP_RESET_S):
                        frame = await attachment.queue.get()
                except TimeoutError:
                    if self._vad.speech_id is not None:
                        self._dispatch(self._vad.reset())
                    continue
                self._dispatch(self._vad.push(frame))
        finally:
            attachment.detach()

    def _dispatch(self, events: list[VadEvent]) -> None:
        for event in events:
            if isinstance(event, SpeechStarted):
                cutoff = time.monotonic() - self._buffer_s
                self._starts = {i: at for i, at in self._starts.items() if at >= cutoff}
                self._starts[event.speech_id] = event.at
            self._on_event(event)
