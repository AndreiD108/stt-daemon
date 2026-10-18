"""Which captured audio goes to the backend: speech and short pauses, not long silences.

Audio is sent while the VAD hears speech and until it confirms a pause (`vad.min_silence_ms`),
so short pauses stay whole. Then audio is held back, keeping only the newest `hold_s`. When the
VAD confirms the next speech, the held audio from `pad_s` before the speech's true start is sent
first. The VAD confirms a start late, so `hold_s` must cover that delay plus the pad.
A session starts holding: silence before the first speech isn't sent.
"""

import collections
from dataclasses import dataclass

from ..audio.frames import Frame
from ..audio.vad import SpeechEnded, SpeechStarted, VadEvent


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
            elif isinstance(event, SpeechEnded) and self.speaking:
                self.speaking = False
        if not placed:
            if self.speaking:
                out.append(Outgoing(frame, cut=False))
            else:
                self._hold(frame)
        return out

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
