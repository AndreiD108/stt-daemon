"""Streaming speech detection: silero (ONNX, via sherpa-onnx) per window, debounced here.

sherpa-onnx's `VadModel.is_speech` is used with no minimum durations, as a per-window speech
flag. It reports a change one window late: it needs a second window over the threshold to start,
and a second window under it to stop. The debounce here keeps the window where each run began,
so events carry the true edge, not the time they were confirmed. Near the threshold, the flag can
rise a few windows into a speech, so a second model flags windows at `EDGE_THRESHOLD`, and a start
moves back to where that flag's run began (at most `EDGE_LOOKBACK_WINDOWS`).
"""

import collections
import itertools
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sherpa_onnx

from stt_daemon.audio.frames import Frame, Samples

log = logging.getLogger(__name__)
_sherpa: Any = sherpa_onnx  # untyped module

MODEL_PATH = Path(__file__).with_name("silero_vad.onnx")
WINDOW = 512
"""Samples per silero window at 16 kHz."""
FLAG_LAG_WINDOWS = 1
GAP_RESET_S = 0.5
EDGE_THRESHOLD = 0.15
EDGE_LOOKBACK_WINDOWS = 6
_TINY_DURATION_S = 0.001  # sherpa rejects 0; any value under one window acts as 0


@dataclass(frozen=True)
class SpeechStarted:
    speech_id: str
    at: float
    """Monotonic time of the speech's first window."""


@dataclass(frozen=True)
class SpeechEnded:
    speech_id: str
    at: float
    """Monotonic time the silence began."""


type VadEvent = SpeechStarted | SpeechEnded


def _create_model(threshold: float, sample_rate: int) -> Any:
    config = _sherpa.VadModelConfig()
    config.sample_rate = sample_rate
    config.num_threads = 1
    config.provider = "cpu"
    config.silero_vad.model = str(MODEL_PATH)
    config.silero_vad.threshold = threshold
    config.silero_vad.min_speech_duration = _TINY_DURATION_S
    config.silero_vad.min_silence_duration = _TINY_DURATION_S
    config.silero_vad.window_size = WINDOW
    if not config.validate():
        raise RuntimeError("sherpa-onnx rejected the VAD config")
    return _sherpa.VadModel.create(config)


class StreamingVAD:
    """Push frames of any size; get `SpeechStarted` / `SpeechEnded` events.

    A start is confirmed once speech has lasted `min_speech_ms`, an end once silence has lasted
    `min_silence_ms`, so both come late (`start_delay_s`, `end_delay_s`); their `at` is the edge.
    Speech ids run `s1`, `s2`, ... for the life of the object. A gap in frame times (the stream
    reopened) ends any speech at the last sample before the gap and starts fresh.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        threshold: float = 0.4,
        min_speech_ms: int = 250,
        min_silence_ms: int = 600,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError(f"the VAD needs 16 kHz audio, got {sample_rate} Hz")
        self.sample_rate = sample_rate
        self._model = _create_model(threshold, sample_rate)
        self._edge_model = _create_model(min(threshold, EDGE_THRESHOLD), sample_rate)
        # A run must span the lagging flag's own windows before it can be confirmed.
        min_run = (FLAG_LAG_WINDOWS + 1) * WINDOW
        self._min_speech = max(min_speech_ms * sample_rate // 1000, min_run)
        self._min_silence = max(min_silence_ms * sample_rate // 1000, min_run)
        self._ids = (f"s{n}" for n in itertools.count(1))
        self._pending: Samples = np.empty(0, dtype=np.float32)
        self._next_sample = 0  # stream sample index of the next window
        self._segment_start = 0
        # (stream sample index of a frame's first sample, its monotonic start time)
        self._anchors: collections.deque[tuple[int, float]] = collections.deque()
        self._last_end: float | None = None
        self._speech_id: str | None = None
        self._run_start: int | None = None
        self._low_run_start: int | None = None
        self._silence_start: int | None = None
        self._quiet_since: int | None = None

    @property
    def pause_s(self) -> float:
        """Silence since the last speech, past its `SpeechEnded` too; 0 while speaking or before
        any speech. A run too short to start speech doesn't end the pause."""
        if self._quiet_since is None:
            return 0.0
        return (self._next_sample - self._quiet_since) / self.sample_rate

    @property
    def start_delay_s(self) -> float:
        """Longest time from a start's `at` to its `SpeechStarted`, frame wait aside."""
        windows = math.ceil(self._min_speech / WINDOW) + EDGE_LOOKBACK_WINDOWS
        return windows * WINDOW / self.sample_rate

    @property
    def end_delay_s(self) -> float:
        return math.ceil(self._min_silence / WINDOW) * WINDOW / self.sample_rate

    @property
    def speech_id(self) -> str | None:
        """The speech in progress, if any."""
        return self._speech_id

    def push(self, frame: Frame) -> list[VadEvent]:
        if frame.samples.size == 0:
            return []
        events: list[VadEvent] = []
        if self._last_end is not None and frame.start - self._last_end > GAP_RESET_S:
            events += self.reset()
        self._last_end = frame.end
        first = self._next_sample + self._pending.size
        self._anchors.append((first, frame.start))
        buf = frame.samples
        if self._pending.size:
            buf = np.concatenate([self._pending, buf])
        offset = 0
        while buf.size - offset >= WINDOW:
            window = buf[offset : offset + WINDOW]
            offset += WINDOW
            event = self._step(
                bool(self._model.is_speech(window)), bool(self._edge_model.is_speech(window))
            )
            if event is not None:
                events.append(event)
        self._pending = buf[offset:].copy()
        self._trim_anchors()
        return events

    def reset(self) -> list[VadEvent]:
        """Forget all audio so far; ends speech in progress at the last sample seen."""
        events: list[VadEvent] = []
        if self._speech_id is not None and self._last_end is not None:
            events.append(SpeechEnded(self._speech_id, self._last_end))
        self._model.reset()
        self._edge_model.reset()
        self._next_sample += self._pending.size
        self._segment_start = self._next_sample
        self._pending = np.empty(0, dtype=np.float32)
        self._anchors.clear()
        self._last_end = None
        self._speech_id = self._run_start = self._low_run_start = self._silence_start = None
        self._quiet_since = None
        return events

    def _step(self, speech: bool, low: bool) -> VadEvent | None:
        window_start = self._next_sample
        self._next_sample += WINDOW
        flag_edge = max(window_start - FLAG_LAG_WINDOWS * WINDOW, self._segment_start)
        if not low or self._speech_id is not None:
            self._low_run_start = None
        elif self._low_run_start is None:
            self._low_run_start = flag_edge
        elif self._run_start is None:
            # Only the lookback before a start can use it; older history would never be trimmed.
            lookback_from = flag_edge - EDGE_LOOKBACK_WINDOWS * WINDOW
            self._low_run_start = max(self._low_run_start, lookback_from)
        if self._speech_id is None:
            if not speech:
                self._run_start = None
                return None
            if self._run_start is None:
                self._run_start = flag_edge
            if self._next_sample - self._run_start < self._min_speech:
                return None
            self._speech_id = next(self._ids)
            event = SpeechStarted(self._speech_id, self._time_of(self._start_edge(self._run_start)))
            self._run_start = self._silence_start = self._quiet_since = None
            return event
        if speech:
            self._silence_start = self._quiet_since = None
            return None
        if self._silence_start is None:
            self._silence_start = self._quiet_since = flag_edge
        if self._next_sample - self._silence_start < self._min_silence:
            return None
        event = SpeechEnded(self._speech_id, self._time_of(self._silence_start))
        self._speech_id = self._run_start = self._silence_start = None
        return event

    def _start_edge(self, run_start: int) -> int:
        low = self._low_run_start
        if low is None or low >= run_start:
            return run_start
        return max(low, run_start - EDGE_LOOKBACK_WINDOWS * WINDOW, self._segment_start)

    def _time_of(self, sample: int) -> float:
        first, start = self._anchors[0]
        for anchor_first, anchor_start in self._anchors:
            if anchor_first > sample:
                break
            first, start = anchor_first, anchor_start
        return start + (sample - first) / self.sample_rate

    def _trim_anchors(self) -> None:
        next_edge = self._next_sample - FLAG_LAG_WINDOWS * WINDOW
        oldest_needed = min(
            s
            for s in (self._run_start, self._low_run_start, self._silence_start, next_edge)
            if s is not None
        )
        while len(self._anchors) > 1 and self._anchors[1][0] <= oldest_needed:
            self._anchors.popleft()
