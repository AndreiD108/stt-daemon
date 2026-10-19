"""The speakers' audio for the echo gate, recorded while the mic is: from daemon start to the end
with `keep_open`, so its ring holds the playback behind the mic's pre-start audio; otherwise from
the first session's hold to the last one's release."""

import asyncio
import logging
from collections.abc import Callable

from .capture import CaptureUnavailable
from .frames import Attachment
from .stream_capture import ToolMissing
from .system_capture import SystemCapture

log = logging.getLogger(__name__)


class SpeakerMonitor:
    def __init__(self, make: Callable[[], SystemCapture], *, keep_open: bool) -> None:
        """`make` gives a new capture of the default output's monitor; one capture runs at a
        time, and a released one is not reused."""
        self._make = make
        self.keep_open = keep_open
        self._capture: SystemCapture | None = None
        self._wanted = False
        self._holds = 0
        self._changing = asyncio.Lock()

    async def want(self, on: bool) -> None:
        """With `keep_open`, record from now on, or once no session holds it, stop; without,
        only sessions' holds count."""
        self._wanted = on and self.keep_open
        await self._follow()

    async def stop(self) -> None:
        self._wanted = False
        self._holds = 0
        await self._follow()

    async def hold(self) -> None:
        """Returns once the capture runs, or failed to start."""
        self._holds += 1
        await self._follow()

    async def release(self) -> None:
        if self._holds == 0:
            return
        self._holds -= 1
        await self._follow()

    def attach(self, since: float) -> Attachment:
        if self._capture is None:
            raise CaptureUnavailable("the speakers are not recorded")
        return self._capture.attach(since)

    async def _follow(self) -> None:
        async with self._changing:
            needed = self._wanted or self._holds > 0
            if needed and self._capture is None:
                await self._open()
            elif not needed and self._capture is not None:
                capture, self._capture = self._capture, None
                await capture.release()

    async def _open(self) -> None:
        capture = self._make()
        try:
            await capture.hold()
        except ToolMissing as exc:
            log.warning("can't record the speakers for the echo gate: %s is missing", exc)
            return
        self._capture = capture
