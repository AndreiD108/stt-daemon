"""What the speakers played behind a mic track, for backends that take a `Reference`.

The playback capture's frames are kept on one clock, as the built-in canceller keeps its far end:
each frame follows the last by sample count, and only a jump moves the clock to its stamp. Each
commit's reference holds, per stretch of its audio captured without a break, the playback from
`PLAYBACK_LEAD_S` before the stretch to `PLAYBACK_LOOKAHEAD_S` after it.
"""

import asyncio
import collections
import contextlib
import math
import time

import numpy as np

from ..audio.frames import Attachment, Frame, Samples
from ..backends.base import PLAYBACK_LEAD_S, PLAYBACK_LOOKAHEAD_S, PlaybackSpan, Reference
from .pause_gate import Outgoing
from .timeline import CAPTURE_GAP_S

JUMP_S = 0.2
"""A playback frame this far off the clock's count moves the clock to its stamp."""
COVER_WAIT_S = 0.3
STALE_S = 1.0
KEEP_S = 300.0
"""Playback older than this before the newest is dropped, needed or not (the mic may stop)."""


class PlaybackHistory:
    """Playback frames from an attachment, kept until `forget_before`."""

    def __init__(self, attachment: Attachment, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._attachment = attachment
        self._parts: collections.deque[tuple[float, Samples]] = collections.deque()
        """(start, samples) on the clock, in arrival order."""
        self._end = -math.inf
        """Where the newest part ends on the clock."""
        self._attached_at = time.monotonic()
        self._arrived = asyncio.Event()
        self._task = asyncio.create_task(self._take())

    def add(self, frame: Frame) -> None:
        start = frame.start
        if abs(start - self._end) <= JUMP_S:
            start = self._end
        self._parts.append((start, frame.samples))
        self._end = start + frame.samples.size / self.sample_rate
        self._arrived.set()
        self.forget_before(self._end - KEEP_S)

    async def cover(self, t: float) -> None:
        """Wait until the playback reaches `t`, or `COVER_WAIT_S` past it; not at all once the
        capture has delivered nothing for `STALE_S` (it delivers silence while nothing plays)."""
        while self._end < t and time.monotonic() - max(self._end, self._attached_at) < STALE_S:
            left = t + COVER_WAIT_S - time.monotonic()
            if left <= 0:
                return
            self._arrived.clear()
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(left):
                    await self._arrived.wait()

    def read(self, at: float, n: int) -> Samples:
        """`n` samples from time `at`; silence where there was no playback frame."""
        out = np.zeros(n, np.float32)
        rate = self.sample_rate
        for start, samples in self._parts:
            first = round((start - at) * rate)
            lo, hi = max(first, 0), min(first + samples.size, n)
            if lo < hi:
                out[lo:hi] = samples[lo - first : hi - first]
        return out

    def forget_before(self, t: float) -> None:
        while self._parts:
            start, samples = self._parts[0]
            if start + samples.size / self.sample_rate > t:
                return
            self._parts.popleft()

    async def close(self) -> None:
        self._attachment.detach()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def _take(self) -> None:
        queue = self._attachment.queue
        while True:
            self.add(await queue.get())


def reference(chunk: list[Outgoing], playback: PlaybackHistory) -> Reference:
    """The playback behind `chunk`, one commit's audio; a stretch ends where a pause was cut
    or the capture skipped (as `Timeline` splits runs)."""
    rate = playback.sample_rate
    lead, lookahead = round(PLAYBACK_LEAD_S * rate), round(PLAYBACK_LOOKAHEAD_S * rate)
    stretches: list[tuple[int, float, int]] = []
    """(offset in the chunk, capture start, samples)"""
    offset = 0
    for out in chunk:
        frame = out.frame
        size = frame.samples.size
        if stretches and not out.cut:
            first, start, samples = stretches[-1]
            if abs(frame.start - (start + samples / rate)) <= CAPTURE_GAP_S:
                stretches[-1] = (first, start, samples + size)
                offset += size
                continue
        stretches.append((offset, frame.start, size))
        offset += size
    spans = tuple(
        PlaybackSpan(
            first,
            samples,
            playback.read(start - lead / rate, lead + samples + lookahead),
            lead,
            lookahead,
        )
        for first, start, samples in stretches
    )
    return Reference(spans)
