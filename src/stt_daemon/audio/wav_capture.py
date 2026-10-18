"""A capture source that plays WAV files in real time, silence in between."""

import collections
import threading
import time
import wave
from pathlib import Path

import numpy as np

from stt_daemon.audio.capture import SAMPLE_RATE, CaptureSource
from stt_daemon.audio.frames import Samples


def read_wav(path: Path, sample_rate: int = SAMPLE_RATE) -> Samples:
    """16-bit PCM at `sample_rate`, any channel count, averaged to mono float32."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        if w.getframerate() != sample_rate:
            raise ValueError(f"{path}: need {sample_rate} Hz, got {w.getframerate()} Hz")
        channels = w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    audio = pcm.reshape(-1, channels).mean(axis=1) / 32768.0
    return audio.astype(np.float32)


class WavCapture(CaptureSource):
    """Stands in for the mic in tests: frames paced by the real clock, timestamped like mic frames.

    Queued files play back to back; silence fills the time between them, so the source stays open
    like a mic would.
    """

    def __init__(
        self,
        *paths: Path,
        sample_rate: int = SAMPLE_RATE,
        buffer_ms: int = 5000,
        keep_open: bool = True,
        chunk_ms: int = 20,
    ) -> None:
        super().__init__(sample_rate=sample_rate, buffer_ms=buffer_ms, keep_open=keep_open)
        self._chunk = sample_rate * chunk_ms // 1000
        self._pending: collections.deque[Samples] = collections.deque(
            read_wav(p, sample_rate) for p in paths
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._idle = threading.Event()
        self._idle.set()

    @property
    def is_open(self) -> bool:
        return self._thread is not None

    def play(self, path: Path) -> None:
        """Queue a file after whatever is still playing."""
        audio = read_wav(path, self.sample_rate)
        with self._lock:
            self._pending.append(audio)
            self._idle.clear()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until every queued file has played."""
        return self._idle.wait(timeout)

    def _open(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        with self._lock:
            if self._pending:
                self._idle.clear()
        self._thread = threading.Thread(target=self._run, name="stt-wav-capture", daemon=True)
        self._thread.start()

    def _close(self) -> None:
        thread, self._thread = self._thread, None
        if thread is None:
            return
        self._stop.set()
        thread.join()
        self._hub.clear()

    def _next_chunk(self) -> Samples:
        with self._lock:
            out = np.zeros(self._chunk, dtype=np.float32)
            filled = 0
            while self._pending and filled < self._chunk:
                head = self._pending[0]
                take = min(head.size, self._chunk - filled)
                out[filled : filled + take] = head[:take]
                filled += take
                if take == head.size:
                    self._pending.popleft()
                else:
                    self._pending[0] = head[take:]
            return out

    def _run(self) -> None:
        origin = time.monotonic()
        sent = 0
        while not self._stop.is_set():
            chunk = self._next_chunk()
            sent += chunk.size
            due = origin + sent / self.sample_rate
            delay = due - time.monotonic()
            if delay > 0 and self._stop.wait(delay):
                return
            self._hub.push(chunk, due)
            with self._lock:
                if not self._pending:
                    self._idle.set()
