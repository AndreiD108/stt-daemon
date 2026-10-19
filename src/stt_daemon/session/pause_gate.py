"""Which captured audio goes to the backend: speech and short pauses, not long silences.

Audio is sent while the VAD hears speech and until it confirms a pause (`vad.min_silence_ms`),
so short pauses stay whole. Then audio is held back, keeping only the newest `hold_s`. When the
VAD confirms the next speech, the held audio from `pad_s` before the speech's true start is sent
first. The VAD confirms a start late, so `hold_s` must cover that delay plus the pad.
A session starts holding: silence before the first speech isn't sent. The gate also keeps the
speech spans it saw, for word times estimated from the audio.
"""

import collections
from dataclasses import dataclass

from ..audio.frames import Frame
from ..audio.vad import SpeechEnded, SpeechStarted, VadEvent


class SpeechLog:
    """Speech spans in capture time, from the VAD's true edges; the newest may still be open."""

    def __init__(self) -> None:
        self._spans: collections.deque[tuple[float, float]] = collections.deque()
        self.open_since: float | None = None

    def started(self, at: float) -> None:
        self.open_since = at

    def ended(self, at: float) -> None:
        if self.open_since is not None:
            self._spans.append((self.open_since, at))
            self.open_since = None

    def between(self, start: float, end: float) -> list[tuple[float, float]]:
        """The spans' parts inside [start, end]."""
        spans = list(self._spans)
        if self.open_since is not None:
            spans.append((self.open_since, end))
        return [(max(s, start), min(e, end)) for s, e in spans if e > start and s < end]

    def forget_before(self, t: float) -> None:
        while self._spans and self._spans[0][1] <= t:
            self._spans.popleft()


@dataclass(frozen=True)
class Outgoing:
    frame: Frame
    cut: bool
    """Captured audio right before this frame was held back and dropped."""


class PauseGate:
    def __init__(self, *, hold_s: float, pad_s: float) -> None:
        self.hold_s = hold_s
        self.pad_s = pad_s
        self.speaking = False
        self._held: collections.deque[Frame] = collections.deque()
        self._dropped = False
        self.speech = SpeechLog()

    def push(self, frame: Frame, events: list[VadEvent]) -> list[Outgoing]:
        """Route a frame, given the VAD events its audio produced."""
        out: list[Outgoing] = []
        placed = False
        if not self.speaking:
            self._hold(frame)
            placed = True
        for event in events:
            if isinstance(event, SpeechStarted) and not self.speaking:
                if not placed:
                    self._hold(frame)
                    placed = True
                out += self._release(event.at - self.pad_s)
                self.speaking = True
                self.speech.started(event.at)
            elif isinstance(event, SpeechEnded) and self.speaking:
                self.speaking = False
                self.speech.ended(event.at)
        if not placed:
            if self.speaking:
                out.append(Outgoing(frame, cut=False))
            else:
                self._hold(frame)
        return out

    @property
    def held_since(self) -> float | None:
        """Start of the oldest audio held back, which may still be sent."""
        return self._held[0].start if self._held else None

    def flush(self) -> list[Outgoing]:
        """Everything still held (a word may have begun that the VAD hasn't confirmed yet)."""
        return self._release(None)

    def _hold(self, frame: Frame) -> None:
        self._held.append(frame)
        keep_from = frame.end - self.hold_s - frame.duration
        while len(self._held) > 1 and self._held[0].end <= keep_from:
            self._held.popleft()
            self._dropped = True

    def _release(self, since: float | None) -> list[Outgoing]:
        out: list[Outgoing] = []
        cut = self._dropped
        for frame in self._held:
            part = frame if since is None else frame.since(since)
            if part is None:
                cut = True
                continue
            if part is not frame:
                cut = True
            out.append(Outgoing(part, cut=cut and not out))
        self._held.clear()
        self._dropped = False
        return out
