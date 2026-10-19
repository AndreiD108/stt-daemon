"""System audio from a capture subprocess that writes s16 mono PCM to stdout (`pw-record` on
Linux, `audiotee` on macOS): what the speakers play, or one application's audio.

The process is restarted with backoff when it exits or goes quiet: both tools deliver silence
while nothing plays, so no frames at all means the capture is dead. An application's stream
ends with the stream; then the capture waits for that app to play again.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from .capture import SAMPLE_RATE, CaptureSource
from .stream_capture import STDERR_TAIL, ToolMissing

log = logging.getLogger(__name__)

STALL_S = 3.0
RESTART_BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 10.0)
READ_BYTES = 6400
"""At most 200 ms of s16 mono at 16 kHz per read."""
STDERR_WAIT_S = 0.5
BUFFER_MS = 1000


type Command = Callable[[str | None], Sequence[str]]
"""The command line for a target stream (None: everything the speakers play)."""


class AppTarget(Protocol):
    """The stream a system track follows: a PipeWire serial, or a macOS process id."""

    @property
    def serial(self) -> str: ...

    def current(self) -> str | None:
        """Its stream while it plays, else None."""
        ...

    async def appeared(self) -> str:
        """Wait until it plays again; its stream."""
        ...


class SystemCapture(CaptureSource):
    """One session's system track. `command` writes s16 mono at `sample_rate` to stdout; a fixed
    list ignores `target`. While the process runs, a target that plays on another stream
    restarts it there."""

    def __init__(
        self,
        command: Sequence[str] | Command,
        *,
        target: AppTarget | None = None,
        sample_rate: int = SAMPLE_RATE,
        stall_s: float = STALL_S,
        backoff_s: Sequence[float] = RESTART_BACKOFF_S,
        buffer_ms: int = BUFFER_MS,
    ) -> None:
        super().__init__(sample_rate=sample_rate, buffer_ms=buffer_ms, keep_open=False)
        if callable(command):
            self._command: Command = command
        else:
            fixed = list(command)
            self._command = lambda _serial: fixed
        self.target = target
        self._serial = None if target is None else target.serial
        self.command = list(self._command(self._serial))
        """The command line of the last start."""
        self.on_lost: Callable[[str], None] | None = None
        """Called once when frames stop, with why."""
        self.on_restored: Callable[[], None] | None = None
        """Called when frames come again after `on_lost`."""
        self.lost_reason: str | None = None
        """Why frames stopped, while they stay stopped."""
        self._stall_s = stall_s
        self._backoff_s = tuple(backoff_s)
        self._settled = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._carry = b""
        self._last_end = 0.0
        self._stderr = bytearray()

    @property
    def is_open(self) -> bool:
        return not self._closed

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    async def hold(self) -> None:
        """Start the process. Raises `ToolMissing` when it can't be started."""
        if self._task is not None:
            return
        proc = await self._spawn(self._serial)
        self._task = asyncio.create_task(self._run(proc))

    async def release(self) -> None:
        """Stop the process; restarts end too."""
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.wait([self._task])

    async def ready(self, timeout_s: float) -> None:
        """Wait for the first frame, or the first loss, at most `timeout_s`."""
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(timeout_s):
                await self._settled.wait()

    def stall_reason(self) -> str:
        return f"no audio from {self.command[0]} for {self._stall_s:g} s"

    def exit_reason(self, code: int) -> str:
        return f"{self.command[0]} exited with code {code}"

    def stderr_tail(self) -> str:
        """What the last start wrote to stderr (its end)."""
        return self._stderr.decode(errors="replace").strip()

    async def _spawn(self, serial: str | None) -> asyncio.subprocess.Process:
        self._serial = serial
        self.command = list(self._command(serial))
        self._stderr.clear()
        try:
            return await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            raise ToolMissing(self.command[0]) from None

    async def _run(self, proc: asyncio.subprocess.Process | None) -> None:
        failures = 0
        while True:
            moved = False
            if proc is not None:
                reason, produced = await self._read(proc)
                if produced:
                    failures = 0
                if reason is None:
                    moved = True
                else:
                    self._lose(reason)
            # Read after the teardown: a target never falls back to a capture of everything.
            serial = None if self.target is None else self.target.current()
            if self.target is not None and serial is None:
                if moved:
                    self._lose("the app's new stream ended at once")
                serial = await self.target.appeared()
                log.info("system capture: the app plays again, on stream %s", serial)
                failures = 0
            elif moved:
                log.info("system capture: the app plays on stream %s now", serial)
            else:
                await asyncio.sleep(self._backoff_s[min(failures, len(self._backoff_s) - 1)])
                failures += 1
            try:
                proc = await self._spawn(serial)
            except (ToolMissing, OSError) as exc:
                log.warning("system capture: %s could not be started: %s", self.command[0], exc)
                proc = None

    async def _read(self, proc: asyncio.subprocess.Process) -> tuple[str | None, bool]:
        """Push frames until the process exits or stalls, or its target plays on another stream;
        returns why (None: moved), and whether any frame came."""
        assert proc.stdout is not None
        stderr = asyncio.create_task(self._read_stderr(proc))
        produced = False
        try:
            while True:
                try:
                    async with asyncio.timeout(self._stall_s):
                        data = await proc.stdout.read(READ_BYTES)
                except TimeoutError:
                    return self.stall_reason(), produced
                if not data:
                    code = await proc.wait()
                    await asyncio.wait([stderr], timeout=STDERR_WAIT_S)
                    return self.exit_reason(code), produced
                produced = self._push(data) or produced
                if self._moved():
                    return None, produced
        finally:
            await self._end(proc, stderr)

    def _moved(self) -> bool:
        if self.target is None:
            return False
        now = self.target.current()
        return now is not None and now != self._serial

    async def _end(self, proc: asyncio.subprocess.Process, stderr: asyncio.Task[None]) -> None:
        # `proc.wait()` returns only once every pipe is closed, and a full pipe never closes:
        # read to EOF after the kill.
        assert proc.stdout is not None
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        while await proc.stdout.read(65536):
            pass
        await asyncio.wait([stderr])
        await proc.wait()
        self._carry = b""

    async def _read_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        while chunk := await proc.stderr.read(4096):
            self._stderr += chunk
            del self._stderr[:-STDERR_TAIL]

    def _push(self, data: bytes) -> bool:
        data = self._carry + data
        usable = len(data) - len(data) % 2
        self._carry = data[usable:]
        if not usable or self._closed:
            return False
        pcm = np.frombuffer(data[:usable], dtype="<i2")
        samples = (pcm / 32768.0).astype(np.float32)
        # Stamped on arrival like mic frames, but never overlapping the previous frame.
        end = max(time.monotonic(), self._last_end + samples.size / self.sample_rate)
        self._last_end = end
        self._hub.push(samples, end)
        self._settled.set()
        if self.lost_reason is not None:
            self.lost_reason = None
            log.info("system capture restored")
            if self.on_restored is not None:
                self.on_restored()
        return True

    def _lose(self, reason: str) -> None:
        self._settled.set()
        if self.lost_reason is not None:
            return
        self.lost_reason = reason
        tail = self.stderr_tail()
        log.warning("system capture lost: %s%s", reason, f": {tail}" if tail else "")
        if self.on_lost is not None:
            self.on_lost(reason)
