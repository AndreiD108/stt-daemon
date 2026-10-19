"""The built-in echo canceller on a system session's mic track: mic frames in, with the system
track's audio for the same capture time as the reference; cancelled frames out, on the mic's
times. Both tracks' frames are stamped on arrival, so each is kept as a continuous stream and
paired by sample count; their stamps only place a stream where it starts or jumps."""

import asyncio
import logging
import time

import numpy as np

from .aec import EchoCanceller
from .capture import CaptureSource, CaptureUnavailable
from .frames import Attachment, Frame, Samples

log = logging.getLogger(__name__)

LOOKAHEAD_S = 0.1
"""The far end is read this far past the mic block's time: the system track's frames may arrive,
and be stamped, later than the mic's for the same sound. The canceller's delay search covers it."""
FAR_WAIT_S = 0.1
"""How long past that a mic block waits for far-end frames before it takes silence."""
JUMP_S = 0.2
"""A frame this far off its stream's count starts the stream anew there."""
FAR_HISTORY_S = 2.0
SETTLE_S = LOOKAHEAD_S + FAR_WAIT_S + 0.2
"""How long a stop waits for the mic audio still in the canceller; 0.2 s for the mic's frames."""


class _Stream:
    """One track's samples on a clock set by its first frame; a jump re-sets it."""

    def __init__(self, rate: int) -> None:
        self.rate = rate
        self.samples = np.zeros(0, dtype=np.float32)
        self.start: float | None = None
        """Time of `samples[0]`."""

    @property
    def end(self) -> float | None:
        return None if self.start is None else self.start + self.samples.size / self.rate

    def add(self, frame: Frame, *, fill_gaps: bool) -> bool:
        """Append `frame`; True when it jumped. With `fill_gaps`, a later frame gets silence
        before it, and only an earlier one moves the clock."""
        end = self.end
        if self.start is None or end is None:
            self.start, self.samples = frame.start, frame.samples.copy()
            return False
        gap = frame.start - end
        if abs(gap) <= JUMP_S:
            self.samples = np.concatenate((self.samples, frame.samples))
            return False
        if fill_gaps and gap > 0:
            silence = np.zeros(round(gap * self.rate), dtype=np.float32)
            self.samples = np.concatenate((self.samples, silence, frame.samples))
        else:
            self.start = frame.start - self.samples.size / self.rate
            self.samples = np.concatenate((self.samples, frame.samples))
        return True

    def read(self, at: float, n: int) -> Samples:
        """`n` samples from time `at`; silence where the stream has none."""
        out = np.zeros(n, dtype=np.float32)
        if self.start is None:
            return out
        first = round((at - self.start) * self.rate)
        lo, hi = max(first, 0), min(first + n, self.samples.size)
        if lo < hi:
            out[lo - first : hi - first] = self.samples[lo:hi]
        return out

    def drop_before(self, t: float) -> None:
        if self.start is None:
            return
        cut = min(max(0, round((t - self.start) * self.rate)), self.samples.size)
        self.samples = self.samples[cut:]
        self.start += cut / self.rate


class BuiltinEchoCancelledMic(CaptureSource):
    """A system session's mic track through `EchoCanceller`, with `far` (the system track's
    capture, held by the session) as the reference."""

    def __init__(self, mic: CaptureSource, far: CaptureSource) -> None:
        super().__init__(sample_rate=mic.sample_rate, buffer_ms=1000, keep_open=False)
        self._mic = mic
        self._far_source = far
        self._canceller = EchoCanceller(mic.sample_rate)
        self._mic_held = False
        self._feeds: list[Attachment] = []
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._done_until: float | None = None
        """Mic time up to which output is pushed."""
        self._progress = asyncio.Event()

    @property
    def is_open(self) -> bool:
        return not self._closed

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    async def hold(self) -> None:
        await self._mic.hold()
        self._mic_held = True

    async def release(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.wait([self._task])
        for feed in self._feeds:
            feed.detach()
        self._feeds = []
        if self._mic_held:
            self._mic_held = False
            await self._mic.release()

    def attach(self, since: float | None = None) -> Attachment:
        attachment = self._hub.attach(since)
        if self._task is None:
            near, far = _feed(self._mic, since), _feed(self._far_source, since)
            self._feeds = [near, far]
            self._task = asyncio.create_task(self._run(near.queue, far.queue))
            self._task.add_done_callback(_report_failure)
        return attachment

    def listen(self) -> Attachment:
        return self.attach(time.monotonic())

    async def settle(self, until: float) -> None:
        task = self._task
        if task is None:
            return
        deadline = time.monotonic() + SETTLE_S
        while not task.done() and (self._done_until is None or self._done_until < until):
            self._progress.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                await asyncio.wait_for(self._progress.wait(), remaining)
            except TimeoutError:
                return

    async def _run(self, near_queue: asyncio.Queue[Frame], far_queue: asyncio.Queue[Frame]) -> None:
        canceller, rate = self._canceller, self.sample_rate
        block, lag = canceller.block, canceller.latency / rate
        near, far = _Stream(rate), _Stream(rate)
        done = 0
        far_lost = False
        while True:
            frame = await near_queue.get()
            if near.add(frame, fill_gaps=False):
                log.info("echo cancellation: the mic's audio jumped; starting over")
                canceller.reset()
            # A far-end capture that restarted may face another device.
            lost = getattr(self._far_source, "lost_reason", None) is not None
            if far_lost and not lost:
                canceller.reset()
                far = _Stream(rate)
            far_lost = lost
            self._take_far(far_queue, far)
            while near.samples.size - done >= block:
                assert near.start is not None
                at = near.start + done / rate
                block_end = at + block / rate
                await self._wait_far(far_queue, far, block_end + LOOKAHEAD_S)
                out = canceller.process(
                    near.samples[done : done + block], far.read(at + LOOKAHEAD_S, block)
                )
                done += block
                self._hub.push(out.astype(np.float32), block_end - lag)
                self._done_until = block_end
                self._progress.set()
            if done:
                assert near.start is not None
                near.start += done / rate
                near.samples, done = near.samples[done:], 0
            if near.start is not None:
                far.drop_before(near.start - FAR_HISTORY_S)

    def _take_far(self, queue: asyncio.Queue[Frame], far: _Stream) -> None:
        while not queue.empty():
            far.add(queue.get_nowait(), fill_gaps=True)

    async def _wait_far(self, queue: asyncio.Queue[Frame], far: _Stream, until: float) -> None:
        while (end := far.end) is None or end < until:
            remaining = until + FAR_WAIT_S - time.monotonic()
            if remaining <= 0:
                return
            try:
                frame = await asyncio.wait_for(queue.get(), remaining)
            except TimeoutError:
                return
            far.add(frame, fill_gaps=True)


def _feed(source: CaptureSource, since: float | None) -> Attachment:
    try:
        return source.attach(since)
    except CaptureUnavailable:
        return source.listen()


def _report_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        log.error("echo cancellation stopped; the mic track gets no more audio", exc_info=error)
