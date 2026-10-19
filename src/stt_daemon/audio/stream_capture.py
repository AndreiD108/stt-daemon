"""A capture source fed by a client's audio bytes: decoded by ffmpeg, or raw 16 kHz mono PCM.

Frames are stamped with their position in the audio from `origin`, not with when they came, so
a session on this source runs on audio time as fast as the bytes arrive. Its queue is bounded:
when the session falls behind, the reader waits, ffmpeg stops reading, and `write` waits.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable

import numpy as np

from .capture import SAMPLE_RATE, CaptureSource
from .frames import Attachment, Frame

log = logging.getLogger(__name__)

FRAME_SAMPLES = 512
QUEUE_FRAMES = 64
STDERR_TAIL = 2048


class DecodeFailed(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ToolMissing(Exception):
    """The decoder's executable could not be started."""


def ffmpeg_command(pcm: tuple[int, int] | None = None, sample_rate: int = SAMPLE_RATE) -> list[str]:
    """Any container ffmpeg reads → s16le mono at `sample_rate`; `pcm` (rate, channels) declares
    headerless s16le input."""
    raw = [] if pcm is None else ["-f", "s16le", "-ar", str(pcm[0]), "-ac", str(pcm[1])]
    out = ["-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"]
    return ["ffmpeg", "-nostdin", "-loglevel", "error", *raw, "-i", "pipe:0", *out]


class ByteStreamCapture(CaptureSource):
    """One session's input. `decoder` is the command that turns stdin bytes into s16le mono at
    `sample_rate` on stdout; None takes the bytes as that PCM already."""

    def __init__(self, decoder: list[str] | None, *, sample_rate: int = SAMPLE_RATE) -> None:
        super().__init__(sample_rate=sample_rate, buffer_ms=0, keep_open=False)
        self.origin = time.monotonic()
        """Monotonic time of the audio's first sample."""
        self.on_failed: Callable[[str], None] | None = None
        """Called when the decoder fails before the input ends."""
        self._decoder = decoder
        self._queue: asyncio.Queue[Frame] = asyncio.Queue(QUEUE_FRAMES)
        self._samples = 0
        self._carry: bytes = b""
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr = bytearray()
        self._ending = False
        self._closed = False

    @property
    def is_open(self) -> bool:
        return not self._closed

    @property
    def end(self) -> float:
        """Monotonic stamp of the end of the audio decoded so far."""
        return self.origin + self._samples / self.sample_rate

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    async def hold(self) -> None:
        """Start the decoder. Raises `ToolMissing`."""
        if self._decoder is None or self._proc is not None:
            return
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._decoder,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            raise ToolMissing(self._decoder[0]) from None
        self._reader = asyncio.create_task(self._read())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def release(self) -> None:
        """Stop the decoder; frames not yet taken are dropped."""
        self._closed = True
        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        # Nothing takes frames any more: emptying the queue wakes a reader or `write` waiting in
        # `put`; once closed, frames are dropped. The readers run on to EOF: `proc.wait()` only
        # returns once every pipe is closed, and a pipe left full never closes.
        while not self._queue.empty():
            self._queue.get_nowait()
        tasks = [t for t in (self._reader, self._stderr_task) if t is not None]
        if tasks:
            await asyncio.wait(tasks)
        if proc is not None:
            await proc.wait()

    def attach(self, since: float | None = None) -> Attachment:
        return Attachment(self._queue, self.origin, 0, self._hub, asyncio.get_running_loop())

    async def write(self, data: bytes) -> None:
        """Feed input bytes; waits while the session is behind. Ignored once the input ended."""
        if self._ending or self._closed:
            return
        proc = self._proc
        if proc is None:
            await self._put_pcm(data)
            return
        assert proc.stdin is not None
        if proc.stdin.is_closing():
            return  # the decoder ended on its own; its reader reports how
        proc.stdin.write(data)
        with contextlib.suppress(ConnectionError):
            await proc.stdin.drain()

    async def finish(self) -> None:
        """End of input: decode what is left and queue it. Raises `DecodeFailed`."""
        self._ending = True
        proc = self._proc
        if proc is not None:
            assert proc.stdin is not None
            assert self._reader is not None
            proc.stdin.close()
            with contextlib.suppress(ConnectionError):
                await proc.stdin.wait_closed()
            await asyncio.wait([self._reader])
            code = await proc.wait()
            if self._stderr_task is not None:
                await asyncio.wait([self._stderr_task])
            if code != 0:
                raise DecodeFailed(self._failure(f"ffmpeg exited with code {code}"))
        if self._samples == 0:
            raise DecodeFailed(self._failure("no audio could be decoded from the input"))

    async def _read(self) -> None:
        proc = self._proc
        assert proc is not None
        assert proc.stdout is not None
        size = FRAME_SAMPLES * 2
        while True:
            try:
                data = await proc.stdout.readexactly(size)
            except asyncio.IncompleteReadError as exc:
                await self._put_pcm(exc.partial)
                break
            await self._put_pcm(data)
        code = await proc.wait()
        if code != 0 and not self._ending and not self._closed:
            if self._stderr_task is not None:
                await asyncio.wait([self._stderr_task])
            if self.on_failed is not None:
                self.on_failed(self._failure(f"ffmpeg exited with code {code}"))

    async def _read_stderr(self) -> None:
        proc = self._proc
        assert proc is not None
        assert proc.stderr is not None
        while chunk := await proc.stderr.read(4096):
            self._stderr += chunk
            del self._stderr[:-STDERR_TAIL]

    async def _put_pcm(self, data: bytes) -> None:
        data = self._carry + data
        usable = len(data) - len(data) % 2
        self._carry = data[usable:]
        size = FRAME_SAMPLES * 2
        for at in range(0, usable, size):
            if self._closed:
                return
            pcm = np.frombuffer(data[at : min(at + size, usable)], dtype="<i2")
            samples = (pcm / 32768.0).astype(np.float32)
            samples.setflags(write=False)
            self._samples += samples.size
            await self._queue.put(Frame(samples, self.end, self.sample_rate))

    def _failure(self, fallback: str) -> str:
        tail = self._stderr.decode(errors="replace").strip()
        return f"{fallback}: {tail}" if tail else fallback
