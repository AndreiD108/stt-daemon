"""Where each sample sent to a backend connection came from on the capture clock.

Each connection's sent audio is a list of runs: stretches of sent samples that are also
contiguous in capture time. A run ends where a pause was cut out or the capture skipped. Backend
times (seconds into a connection's sent audio) map through these runs to capture time, then to
session ms. At a cut, a start time maps to the run after it and an end time to the run before
it, so a word never lands inside the cut-out audio.
"""

import bisect
from dataclasses import dataclass
from typing import Literal

CAPTURE_GAP_S = 0.05
"""A frame starting this far from where the run ends starts a new run even without a cut."""


@dataclass
class _Run:
    offset: int
    """Index of the run's first sample in the connection's sent audio."""
    capture_start: float
    samples: int

    @property
    def end_offset(self) -> int:
        return self.offset + self.samples


class Timeline:
    def __init__(self, origin: float, sample_rate: int = 16000) -> None:
        self.origin = origin
        """Monotonic time of session ms 0."""
        self.sample_rate = sample_rate
        self._runs: dict[int, list[_Run]] = {}

    def sent(self, connection_id: int, capture_start: float, samples: int, *, cut: bool) -> None:
        """Record `samples` sent on the connection, the first captured at `capture_start`.

        `cut`: audio between the previous send and this one was held back.
        """
        if samples <= 0:
            return
        runs = self._runs.setdefault(connection_id, [])
        if runs and not cut:
            last = runs[-1]
            last_end = last.capture_start + last.samples / self.sample_rate
            if abs(capture_start - last_end) <= CAPTURE_GAP_S:
                last.samples += samples
                return
        offset = runs[-1].end_offset if runs else 0
        runs.append(_Run(offset, capture_start, samples))

    def sent_samples(self, connection_id: int) -> int:
        runs = self._runs.get(connection_id)
        return runs[-1].end_offset if runs else 0

    def to_capture(
        self, connection_id: int, seconds: float, edge: Literal["start", "end"]
    ) -> float:
        """Capture time of a backend time on the connection; clamped to the audio sent."""
        runs = self._runs.get(connection_id)
        if not runs:
            return self.origin
        offset = min(max(round(seconds * self.sample_rate), 0), runs[-1].end_offset)
        starts = [r.offset for r in runs]
        if edge == "start":
            i = max(bisect.bisect_right(starts, offset) - 1, 0)
        else:
            i = max(bisect.bisect_left(starts, offset) - 1, 0)
        run = runs[i]
        return run.capture_start + (offset - run.offset) / self.sample_rate

    def word_span(self, connection_id: int, start_s: float, end_s: float) -> tuple[float, float]:
        """Capture times of a word's backend start and end. A word that crosses a cut goes whole
        into the run holding more of it: decoders can start a word early, before the cut."""
        runs = self._runs.get(connection_id)
        if not runs:
            return self.origin, self.origin
        last = runs[-1].end_offset
        a = min(max(round(start_s * self.sample_rate), 0), last)
        b = min(max(round(end_s * self.sample_rate), a), last)
        starts = [r.offset for r in runs]
        first = max(bisect.bisect_right(starts, a) - 1, 0)
        final = max(bisect.bisect_left(starts, b) - 1, first)
        i = max(
            range(first, final + 1),
            key=lambda k: min(b, runs[k].end_offset) - max(a, runs[k].offset),
        )
        run = runs[i]
        a, b = max(a, run.offset), min(b, run.end_offset)
        rate = self.sample_rate
        return (
            run.capture_start + (a - run.offset) / rate,
            run.capture_start + (max(a, b) - run.offset) / rate,
        )

    def to_ms(self, t: float) -> int:
        """Session ms of a monotonic time; never negative."""
        return max(round((t - self.origin) * 1000), 0)
