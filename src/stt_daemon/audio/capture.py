"""Mic capture: a PortAudio input stream feeding the frame hub, with stall recovery."""

import asyncio
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import sounddevice as sd

from stt_daemon.audio.frames import Attachment, Frame, FrameHub, Samples

log = logging.getLogger(__name__)
_sd: Any = sd  # untyped module

SAMPLE_RATE = 16000
STALL_TIMEOUT_S = 2.0
WATCHDOG_INTERVAL_S = 0.25
HEALTHY_RESET_S = 5.0
STREAM_STOP_TIMEOUT_S = 2.0
RECOVERY_BACKOFF_S = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
STATUS_RECOVERY_THRESHOLD = 3
CALLBACK_LOG_INTERVAL_S = 1.0
HUNG_STREAM_EXIT_CODE = 70
AUTO_DEVICE_NAMES = ("pipewire", "pulse")


class CaptureUnavailable(Exception):
    """The mic isn't open: not held, still opening, or recovering."""


def _fatal_exit(code: int) -> None:
    os._exit(code)


async def _run_to_end(job: Callable[[], None]) -> bool:
    """Run `job` in a thread and wait for it even if cancelled; True when cancelled meanwhile.

    A cancelled wait would leave the open or close running with the hold count already wrong.
    """
    task = asyncio.ensure_future(asyncio.to_thread(job))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    return cancelled


class CaptureSource(ABC):
    """An audio source behind the frame hub.

    With `keep_open`, `start` opens the source for the daemon's lifetime. Without it, the source
    opens on the first `hold` and closes on the last `release`, so the ring and the VAD only see
    audio while a session holds it.
    """

    def __init__(self, *, sample_rate: int, buffer_ms: int, keep_open: bool) -> None:
        self.sample_rate = sample_rate
        self.keep_open = keep_open
        self._hub = FrameHub(sample_rate, buffer_ms)
        self._holds = 0
        self._hold_lock = asyncio.Lock()

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Frames are flowing (opened and not recovering)."""

    @abstractmethod
    def _open(self) -> None:
        """Open the source; blocking. Raises when it can't."""

    @abstractmethod
    def _close(self) -> None:
        """Close the source; blocking, idempotent."""

    async def start(self) -> None:
        if self.keep_open:
            async with self._hold_lock:
                if await _run_to_end(self._open):
                    await _run_to_end(self._close)
                    raise asyncio.CancelledError

    async def stop(self) -> None:
        async with self._hold_lock:
            self._holds = 0
            if await _run_to_end(self._close):
                raise asyncio.CancelledError

    async def hold(self) -> None:
        """Keep the source open for a session; opens it when `keep_open` is off."""
        async with self._hold_lock:
            if not self.keep_open and self._holds == 0 and await _run_to_end(self._open):
                await _run_to_end(self._close)
                raise asyncio.CancelledError
            self._holds += 1

    async def release(self) -> None:
        async with self._hold_lock:
            if self._holds == 0:
                return
            self._holds -= 1
            if not self.keep_open and self._holds == 0 and await _run_to_end(self._close):
                raise asyncio.CancelledError

    def attach(self, since: float | None = None) -> Attachment:
        """Buffered audio from `since` (monotonic; None = whole ring), then live frames."""
        if not self.is_open:
            raise CaptureUnavailable("the mic is not open")
        return self._hub.attach(since)

    def listen(self) -> Attachment:
        """Live frames from now on, whenever the source is open; no hold, no open check."""
        return self._hub.attach(time.monotonic())

    def ring_since(self, since: float | None = None) -> list[Frame]:
        return self._hub.ring_since(since)

    def ring_start(self) -> float | None:
        return self._hub.ring_start()


def find_input_device(name: str) -> int:
    """The first shared input device whose name contains `name` (case-sensitive).

    Empty `name` takes the system default input, else a PipeWire or Pulse device. Raw ALSA
    `hw:` devices are refused: they take the card exclusively and block other apps.
    """
    devices: list[dict[str, Any]] = list(_sd.query_devices())

    def shared_input(index: int) -> bool:
        dev = devices[index]
        return dev.get("max_input_channels", 0) > 0 and "hw:" not in str(dev.get("name")).lower()

    if name:
        for i, dev in enumerate(devices):
            if name in str(dev.get("name")) and shared_input(i):
                return i
        raise RuntimeError(f"no shared input device matches {name!r} (raw hw: devices refused)")
    default = _sd.default.device[0]
    if isinstance(default, int) and 0 <= default < len(devices) and shared_input(default):
        return default
    for wanted in AUTO_DEVICE_NAMES:
        for i, dev in enumerate(devices):
            if dev.get("name") == wanted and shared_input(i):
                return i
    raise RuntimeError("no shared input device found; set audio.device in config.toml")


def device_channels(device: int) -> int:
    info: dict[str, Any] = _sd.query_devices(device, "input")
    return min(int(info.get("max_input_channels", 1)), 2)


class MicCapture(CaptureSource):
    """A long-lived `sd.InputStream` with a watchdog.

    The watchdog recovers the stream after `stall_timeout_s` without frames, when PortAudio
    finishes it, or after repeated callback status errors, reopening with backoff. A native stop
    that hangs past `stream_stop_timeout_s` exits the process so the service manager restarts it.
    """

    def __init__(
        self,
        *,
        device_name: str = "",
        sample_rate: int = SAMPLE_RATE,
        buffer_ms: int = 5000,
        keep_open: bool = True,
        stall_timeout_s: float = STALL_TIMEOUT_S,
        watchdog_interval_s: float = WATCHDOG_INTERVAL_S,
        healthy_reset_s: float = HEALTHY_RESET_S,
        stream_stop_timeout_s: float = STREAM_STOP_TIMEOUT_S,
        recovery_backoff_s: tuple[float, ...] = RECOVERY_BACKOFF_S,
    ) -> None:
        super().__init__(sample_rate=sample_rate, buffer_ms=buffer_ms, keep_open=keep_open)
        if min(stall_timeout_s, watchdog_interval_s, healthy_reset_s, stream_stop_timeout_s) <= 0:
            raise ValueError("capture recovery timings must be positive")
        if not recovery_backoff_s or min(recovery_backoff_s) < 0:
            raise ValueError("recovery_backoff_s must hold non-negative delays")
        self._device_name = device_name
        self._stall_timeout_s = stall_timeout_s
        self._watchdog_interval_s = watchdog_interval_s
        self._healthy_reset_s = healthy_reset_s
        self._stream_stop_timeout_s = stream_stop_timeout_s
        self._recovery_backoff_s = recovery_backoff_s

        self._stream: Any = None
        self._stream_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._recovery_requested = threading.Event()
        self._recovery_reason: str | None = None
        self._recovering = False
        self._generation = 0
        self._watchdog: threading.Thread | None = None
        self._last_frame_at = time.monotonic()
        self._healthy_since = self._last_frame_at
        self._recovery_attempt = 0
        self._status_errors = 0
        self._last_problem_log_at = 0.0

    @property
    def is_open(self) -> bool:
        with self._stream_lock:
            return self._stream is not None and not self._recovering

    def _open(self) -> None:
        with self._stream_lock:
            if self._stream is not None:
                return
            self._shutdown.clear()
            self._stream = self._open_stream_locked()
            self._last_frame_at = self._healthy_since = time.monotonic()
            self._recovering = False
            self._recovery_attempt = 0
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="stt-audio-watchdog", daemon=True
        )
        self._watchdog.start()
        log.info("mic open")

    def _close(self) -> None:
        self._shutdown.set()
        self._recovery_requested.set()
        with self._stream_lock:
            stream, self._stream = self._stream, None
            self._recovering = False
        self._stop_stream(stream, abort=False)
        watchdog, self._watchdog = self._watchdog, None
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=max(2.0, self._watchdog_interval_s * 4))
            if watchdog.is_alive():
                log.error("audio watchdog did not stop promptly")
        self._hub.clear()
        if stream is not None:
            log.info("mic closed")

    def _open_stream_locked(self) -> Any:
        device = find_input_device(self._device_name)
        channels = device_channels(device)
        self._generation += 1
        generation = self._generation
        stream = _sd.InputStream(
            callback=self._callback,
            finished_callback=lambda: self._request_recovery(
                "PortAudio stream finished", generation=generation
            ),
            channels=channels,
            samplerate=self.sample_rate,
            device=device,
        )
        try:
            stream.start()
        except Exception:
            self._stop_stream(stream, abort=True)
            raise
        log.info("input device %d, %d channel(s)", device, channels)
        return stream

    def _stop_stream(self, stream: Any, *, abort: bool) -> None:
        if stream is None:
            return
        done = threading.Event()

        def stop_native() -> None:
            try:
                try:
                    stream.abort() if abort else stream.stop()
                except Exception as exc:
                    log.debug("stream %s: %s", "abort" if abort else "stop", exc)
                try:
                    stream.close()
                except Exception as exc:
                    log.debug("stream close: %s", exc)
            finally:
                done.set()

        worker = threading.Thread(target=stop_native, name="stt-audio-stream-stop", daemon=True)
        worker.start()
        if not done.wait(self._stream_stop_timeout_s):
            log.critical(
                "PortAudio stream did not stop within %.1f s; exiting so the service restarts",
                self._stream_stop_timeout_s,
            )
            _fatal_exit(HUNG_STREAM_EXIT_CODE)
            return
        worker.join()

    def _request_recovery(self, reason: str, *, generation: int | None = None) -> None:
        with self._stream_lock:
            if (
                self._shutdown.is_set()
                or self._recovering
                or self._stream is None
                or (generation is not None and generation != self._generation)
            ):
                return
            if self._recovery_reason is None:
                self._recovery_reason = reason
        self._recovery_requested.set()

    def _watchdog_loop(self) -> None:
        while not self._shutdown.is_set():
            self._recovery_requested.wait(self._watchdog_interval_s)
            self._recovery_requested.clear()
            if self._shutdown.is_set():
                return
            with self._stream_lock:
                # Read even when the wait timed out: a request between the wait and the
                # clear() leaves only its reason behind.
                reason, self._recovery_reason = self._recovery_reason, None
                if (
                    reason is None
                    and self._stream is not None
                    and not self._recovering
                    and time.monotonic() - self._last_frame_at >= self._stall_timeout_s
                ):
                    reason = f"no audio frames for {self._stall_timeout_s:.1f} s"
            if reason is not None:
                self._recover_stream(reason)

    def _recover_stream(self, reason: str) -> None:
        with self._stream_lock:
            if self._shutdown.is_set() or self._recovering:
                return
            self._recovering = True
            stream, self._stream = self._stream, None
        log.error("audio capture unhealthy (%s); aborting the stream", reason)
        self._stop_stream(stream, abort=True)

        last = len(self._recovery_backoff_s) - 1
        while not self._shutdown.is_set():
            attempt = self._recovery_attempt
            delay = self._recovery_backoff_s[min(attempt, last)]
            if self._shutdown.wait(delay):
                break
            try:
                with self._stream_lock:
                    if self._shutdown.is_set():
                        break
                    self._stream = self._open_stream_locked()
                    self._last_frame_at = self._healthy_since = time.monotonic()
                    self._recovery_attempt = min(attempt + 1, last)
                    self._recovering = False
                log.info("audio capture recovered after %.1f s backoff", delay)
                return
            except Exception as exc:
                self._recovery_attempt = min(attempt + 1, last)
                log.warning("audio capture reopen failed after %.1f s backoff: %s", delay, exc)
        with self._stream_lock:
            self._recovering = False

    def _log_problem(self, level: int, message: str, *args: object) -> None:
        now = time.monotonic()
        if now - self._last_problem_log_at >= CALLBACK_LOG_INTERVAL_S:
            self._last_problem_log_at = now
            log.log(level, message, *args)

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self._status_errors += 1
            self._log_problem(logging.DEBUG, "audio status: %s", status)
            if self._status_errors >= STATUS_RECOVERY_THRESHOLD:
                self._request_recovery(f"{self._status_errors} callback errors in a row")
        else:
            self._status_errors = 0
        if indata is None or indata.size == 0:
            return
        if not np.issubdtype(indata.dtype, np.floating):
            self._log_problem(logging.ERROR, "non-float audio dtype: %s", indata.dtype)
            return
        mono: Samples = (
            indata.mean(axis=1) if indata.ndim > 1 and indata.shape[1] > 1 else indata.reshape(-1)
        ).astype(np.float32)
        if not np.all(np.isfinite(mono)):
            self._log_problem(logging.ERROR, "non-finite samples in an audio chunk; dropped")
            return
        now = time.monotonic()
        self._last_frame_at = now
        if self._recovery_attempt > 0 and now - self._healthy_since >= self._healthy_reset_s:
            self._recovery_attempt = 0
        self._hub.push(mono, now)
