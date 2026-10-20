"""The shared mic on Linux with echo cancellation: PipeWire's echo-cancel source, recorded with
`pw-record`, or the plain mic while the module is unavailable."""

import asyncio
import contextlib
import itertools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..service.preflight import check_tool
from .aec_pipewire import AecUnavailable, PipewireAec
from .capture import CaptureSource, CaptureUnavailable
from .frames import Attachment, Frame
from .stream_capture import ToolMissing
from .system_capture import RESTART_BACKOFF_S, AppTarget, SystemCapture

log = logging.getLogger(__name__)

MIC_RETRY_S = RESTART_BACKOFF_S
AEC_RETRY_S = (5.0, 30.0, 60.0)
"""With the mic kept open, waits before each new try at loading the module."""
SETTLE_S = 0.3


@dataclass(eq=False)
class _Watch:
    on_lost: Callable[[str], None]
    on_restored: Callable[[], None]
    on_unavailable: Callable[[str], None]


class EchoCancelledMic(CaptureSource):
    """The shared mic: the echo-cancel module's source, recorded by a capture from `cleaned`; the
    plain `mic` while the module can't load, and from the moment its `pw-cli` ends. Frames of
    either go through this source's own hub. With `keep_open`, the module loads at `start` and a
    failed or ended module is loaded again, with backoff; without it, the module loads at the
    first `hold` and unloads at the last `release`."""

    def __init__(
        self,
        mic: CaptureSource,
        make_aec: Callable[[], PipewireAec],
        *,
        cleaned: Callable[[AppTarget], SystemCapture],
        buffer_ms: int,
        keep_open: bool,
    ) -> None:
        super().__init__(sample_rate=mic.sample_rate, buffer_ms=buffer_ms, keep_open=keep_open)
        self.lost_reason: str | None = None
        self.unavailable_reason: str | None = None
        """Why the plain mic is in use, until the module is back."""
        self._mic = mic
        self._make_aec = make_aec
        self._make_cleaned = cleaned
        self._watches: list[_Watch] = []
        self._aec: PipewireAec | None = None
        self._cleaned: SystemCapture | None = None
        self._mic_held = False
        self._feed: Attachment | None = None
        self._forward: asyncio.Task[None] | None = None
        self._switch: asyncio.Task[None] | None = None
        """Falling back to the plain mic, or loading the module again."""
        self._last_end: float | None = None
        self._progress = asyncio.Event()
        self._open_now = False

    @property
    def is_open(self) -> bool:
        return self._open_now

    @property
    def cleaned(self) -> bool:
        """Frames come from the echo-cancel source."""
        return self._cleaned is not None

    def watch(
        self,
        *,
        on_lost: Callable[[str], None],
        on_restored: Callable[[], None],
        on_unavailable: Callable[[str], None],
    ) -> Callable[[], None]:
        """Call these on loss of audio, its return, and a move to the plain mic; returns the
        unwatch."""
        entry = _Watch(on_lost, on_restored, on_unavailable)
        self._watches.append(entry)
        return lambda: self._watches.remove(entry) if entry in self._watches else None

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    async def start(self) -> None:
        if self.keep_open:
            async with self._hold_lock:
                await self._open_async()

    async def stop(self) -> None:
        async with self._hold_lock:
            self._holds = 0
            await self._close_async()

    async def hold(self) -> None:
        """Opens the source when `keep_open` is off. Raises what the mic's `hold` raises."""
        async with self._hold_lock:
            if not self.keep_open and self._holds == 0:
                await self._open_async()
            self._holds += 1

    async def release(self) -> None:
        async with self._hold_lock:
            if self._holds == 0:
                return
            self._holds -= 1
            if not self.keep_open and self._holds == 0:
                await self._close_async()

    def attach(self, since: float | None = None) -> Attachment:
        return self._hub.attach(since)

    def listen(self) -> Attachment:
        return self._hub.attach(time.monotonic())

    async def settle(self, until: float) -> None:
        deadline = time.monotonic() + SETTLE_S
        while self._feed is not None and (self._last_end is None or self._last_end < until):
            self._progress.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(remaining):
                    await self._progress.wait()

    async def _open_async(self) -> None:
        if self._open_now:
            return
        self._open_now = True
        try:
            loaded = await self._load()
            if isinstance(loaded, str):
                await self._use_plain(loaded)
            else:
                self._use_cleaned(*loaded)
        except BaseException:
            await self._close_async()
            raise

    async def _close_async(self) -> None:
        self._open_now = False
        if self._switch is not None:
            self._switch.cancel()
            await asyncio.wait([self._switch])
            self._switch = None
        await self._stop_feed()
        await self._release_inner()
        if not self.keep_open:
            self._hub.clear()
            self._last_end = None
        self.lost_reason = self.unavailable_reason = None

    async def _load(self) -> tuple[PipewireAec, SystemCapture] | str:
        """The module loaded and its source held, else why not, with nothing left running."""
        aec = self._make_aec()
        try:
            serial = await aec.load()
        except AecUnavailable as exc:
            return exc.message
        try:
            cleaned = self._make_cleaned(aec.target(serial))
            cleaned.on_lost, cleaned.on_restored = self._lost, self._restored
            await cleaned.hold()
        except (ToolMissing, OSError) as exc:
            await aec.unload()
            problem = check_tool(str(exc)) if isinstance(exc, ToolMissing) else None
            problem = problem or f"its source could not be recorded: {exc}"
            return f"echo cancellation is off for the mic: {problem}"
        except BaseException:
            await aec.unload()
            raise
        if (message := aec.exit_message) is not None:
            await cleaned.release()
            await aec.unload()
            return message
        return aec, cleaned

    def _use_cleaned(self, aec: PipewireAec, cleaned: SystemCapture) -> None:
        """Only after an `exit_message` check with no await since: an exit either was seen
        there or reaches `on_exit`."""
        self._aec, self._cleaned = aec, cleaned
        aec.on_exit = self._aec_exited
        self._feed_from(cleaned.attach(self._last_end))

    async def _use_plain(self, message: str) -> None:
        await self._mic.hold()
        self._mic_held = True
        self._feed_from(self._mic_feed())
        self._report(message)
        if self.keep_open:
            self._switch = asyncio.create_task(self._reload_module())

    def _mic_feed(self) -> Attachment:
        """The plain mic from where the last audio ended, out of its ring."""
        try:
            return self._mic.attach(self._last_end)
        except CaptureUnavailable:
            return self._mic.listen()

    def _feed_from(self, feed: Attachment) -> None:
        """The feed's buffered audio goes into the hub now: an attach right after must find it
        in the ring."""
        self._feed = feed
        while not feed.queue.empty():
            self._push(feed.queue.get_nowait())
        self._forward = asyncio.create_task(self._pass_on(feed.queue))

    async def _pass_on(self, queue: asyncio.Queue[Frame]) -> None:
        while True:
            self._push(await queue.get())

    def _push(self, frame: Frame) -> None:
        # A switch of feeds may repeat audio already pushed.
        part = frame if self._last_end is None else frame.since(self._last_end)
        if part is None:
            return
        self._last_end = part.end
        self._hub.push(part.samples, part.end)
        self._progress.set()

    async def _stop_feed(self) -> None:
        if self._feed is not None:
            self._feed.detach()
            self._feed = None
        if self._forward is not None:
            self._forward.cancel()
            await asyncio.wait([self._forward])
            self._forward = None

    async def _release_inner(self) -> None:
        cleaned, self._cleaned = self._cleaned, None
        if cleaned is not None:
            await cleaned.release()
        aec, self._aec = self._aec, None
        if aec is not None:
            await aec.unload()
        if self._mic_held:
            self._mic_held = False
            await self._mic.release()

    def _aec_exited(self, message: str) -> None:
        if self._open_now:
            self._switch = asyncio.create_task(self._fall_back(message))

    async def _fall_back(self, message: str) -> None:
        await self._stop_feed()
        await self._release_inner()
        failures = 0
        while True:
            try:
                await self._mic.hold()
                break
            except Exception as exc:
                self._report(message)
                if self.lost_reason is None:
                    self._lost(f"the mic did not open: {exc}")
                await asyncio.sleep(MIC_RETRY_S[min(failures, len(MIC_RETRY_S) - 1)])
                failures += 1
        self._mic_held = True
        self._feed_from(self._mic_feed())
        self._report(message)
        if self.lost_reason is not None:
            self._restored()
        if self.keep_open:
            await self._reload_module()

    async def _reload_module(self) -> None:
        """Load the module again, with backoff, and move the feed to its source."""
        for attempt in itertools.count():
            await asyncio.sleep(AEC_RETRY_S[min(attempt, len(AEC_RETRY_S) - 1)])
            loaded = await self._load()
            if isinstance(loaded, str):
                log.debug("echo cancellation: reload failed: %s", loaded)
                continue
            aec, cleaned = loaded
            try:
                await self._stop_feed()
            except BaseException:
                await cleaned.release()
                await aec.unload()
                raise
            if aec.exit_message is not None:
                await cleaned.release()
                await aec.unload()
                self._feed_from(self._mic_feed())
                continue
            self._use_cleaned(aec, cleaned)
            self._mic_held = False
            await self._mic.release()
            self.unavailable_reason = None
            log.info("echo cancellation is on for the mic again")
            return

    def _report(self, message: str) -> None:
        if self.unavailable_reason is not None:
            return
        self.unavailable_reason = message
        log.warning("%s", message)
        for entry in list(self._watches):
            entry.on_unavailable(message)

    def _lost(self, reason: str) -> None:
        self.lost_reason = reason
        for entry in list(self._watches):
            entry.on_lost(reason)

    def _restored(self) -> None:
        self.lost_reason = None
        for entry in list(self._watches):
            entry.on_restored()
