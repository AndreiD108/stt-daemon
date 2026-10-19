"""Timestamped audio frames, the recent-audio ring, and session attachments."""

import asyncio
import collections
import contextlib
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

Samples = npt.NDArray[np.float32]


_WALL_OFFSET_S = time.time() - time.monotonic()


def epoch_ms(mono: float) -> int:
    """A monotonic time as epoch ms; one offset for the process, so wall clock jumps don't move
    one event against another."""
    return round((mono + _WALL_OFFSET_S) * 1000)


@dataclass(frozen=True)
class Frame:
    """Mono float32 samples; `end` is the monotonic time the last sample arrived."""

    samples: Samples
    end: float
    sample_rate: int

    @property
    def start(self) -> float:
        return self.end - self.samples.size / self.sample_rate

    @property
    def duration(self) -> float:
        return self.samples.size / self.sample_rate

    def since(self, t: float) -> "Frame | None":
        """The part of this frame from `t` on, or None when it all lies before `t`."""
        if self.end <= t:
            return None
        skip = round((t - self.start) * self.sample_rate)
        if skip <= 0:
            return self
        return Frame(self.samples[skip:], self.end, self.sample_rate)

    def pack(self) -> "PackedFrame":
        pcm = np.clip(np.rint(self.samples * 32768.0), -32768, 32767).astype(np.int16)
        return PackedFrame(pcm, self.end, self.sample_rate)


@dataclass(frozen=True)
class PackedFrame:
    """A frame as int16, half the memory, for audio that waits long."""

    pcm: npt.NDArray[np.int16]
    end: float
    sample_rate: int

    def unpack(self) -> Frame:
        return Frame(self.pcm.astype(np.float32) / 32768.0, self.end, self.sample_rate)


@dataclass(eq=False)
class Attachment:
    """A queue of frames: buffered audio since a point, then live frames until detached."""

    queue: asyncio.Queue[Frame]
    started_at: float
    """Monotonic time of the first sample queued."""
    prepended_samples: int
    _hub: "FrameHub" = field(repr=False)
    _loop: asyncio.AbstractEventLoop = field(repr=False)

    def detach(self) -> None:
        self._hub.detach(self)

    def _deliver(self, frame: Frame) -> None:
        with contextlib.suppress(RuntimeError):  # loop closed during shutdown
            self._loop.call_soon_threadsafe(self.queue.put_nowait, frame)


class FrameHub:
    """Keeps the last `buffer_ms` of audio and fans frames out to attachments.

    `push` runs on the capture thread; everything else on the event loop. One lock covers the
    ring and the attachment set, so an attach sees each frame once: in its snapshot or live.
    """

    def __init__(self, sample_rate: int, buffer_ms: int) -> None:
        self.sample_rate = sample_rate
        self._max_samples = sample_rate * buffer_ms // 1000
        self._lock = threading.Lock()
        self._ring: collections.deque[Frame] = collections.deque()
        self._ring_samples = 0
        self._attachments: set[Attachment] = set()

    def push(self, samples: Samples, end: float) -> None:
        samples.setflags(write=False)
        frame = Frame(samples, end, self.sample_rate)
        with self._lock:
            self._ring.append(frame)
            self._ring_samples += samples.size
            while self._ring_samples > self._max_samples and len(self._ring) > 1:
                self._ring_samples -= self._ring.popleft().samples.size
            targets = list(self._attachments)
        for attachment in targets:
            attachment._deliver(frame)  # pyright: ignore[reportPrivateUsage]

    def ring_since(self, since: float | None = None) -> list[Frame]:
        """A copy of the ring from `since` on (monotonic clock); None takes it all."""
        with self._lock:
            return self._since_locked(since)

    def ring_start(self) -> float | None:
        """Monotonic time of the oldest sample in the ring."""
        with self._lock:
            return self._ring[0].start if self._ring else None

    def attach(self, since: float | None = None) -> Attachment:
        """Queue the ring's audio from `since` on, then every live frame. Call on the loop."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Frame] = asyncio.Queue()
        with self._lock:
            frames = self._since_locked(since)
            for frame in frames:
                queue.put_nowait(frame)
            started_at = frames[0].start if frames else time.monotonic()
            attachment = Attachment(
                queue, started_at, sum(f.samples.size for f in frames), self, loop
            )
            self._attachments.add(attachment)
        return attachment

    def detach(self, attachment: Attachment) -> None:
        with self._lock:
            self._attachments.discard(attachment)

    def clear(self) -> None:
        """Drop the ring (the stream closed); attachments stay."""
        with self._lock:
            self._ring.clear()
            self._ring_samples = 0

    def _since_locked(self, since: float | None) -> list[Frame]:
        if since is None:
            return list(self._ring)
        kept: list[Frame] = []
        for frame in self._ring:
            part = frame.since(since)
            if part is not None:
                kept.append(part)
        return kept
