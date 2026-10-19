"""Applications playing audio on macOS, from Core Audio's process objects (macOS 14.2 or later):
a process is a source while its output runs. Read every 2 s, and at once when Core Audio reports
a change to its process list.

A source's `binary` is the process's bundle id, its stream (`Source.serial`) the process id that
`audiotee --include-processes` takes. Output running is not the same as audible: a paused player
can stay listed.
"""

import asyncio
import contextlib
import ctypes
import ctypes.util
import functools
import itertools
import logging
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath

from .peer import macos_cmdline
from .sources import Source, SourceAdded, SourceEvent, SourceRemoved, SourceScan

log = logging.getLogger(__name__)

POLL_S = 2.0


@dataclass(frozen=True)
class AudioProcess:
    pid: int
    bundle_id: str | None
    app_name: str | None
    """The app's localized name."""
    process_name: str | None


type ListProcesses = Callable[[], list[AudioProcess]]
"""The processes whose audio output runs now."""
type Listen = Callable[[Callable[[], None]], Callable[[], None]]
"""Calls its argument (on any thread) when the process list changes; returns the undo."""


class ProcessTable:
    """Process records → sources; a source keeps its id while its process stays listed, and ids
    are never reused. What a source says is fixed when it is first seen."""

    def __init__(self) -> None:
        self._by_pid: dict[int, Source] = {}
        self._ids = itertools.count(1)

    def sources(self) -> list[Source]:
        return list(self._by_pid.values())

    def update(self, processes: list[AudioProcess]) -> list[SourceEvent]:
        listed = {p.pid: p for p in processes}
        events: list[SourceEvent] = []
        for pid in [pid for pid in self._by_pid if pid not in listed]:
            events.append(SourceRemoved(self._by_pid.pop(pid)))
        for pid, process in listed.items():
            if pid in self._by_pid:
                continue
            source = Source(
                id=f"src_{next(self._ids)}",
                serial=str(pid),
                name=process.app_name or process.bundle_id or process.process_name or str(pid),
                binary=process.bundle_id,
                media_name=None,
            )
            self._by_pid[pid] = source
            events.append(SourceAdded(source))
        return events

    def clear(self) -> list[SourceEvent]:
        events: list[SourceEvent] = [SourceRemoved(s) for s in self._by_pid.values()]
        self._by_pid = {}
        return events


class CoreAudioScan(SourceScan):
    """Reads `processes` every `poll_s`, and soon after `listen` reports a change. A read that
    fails empties the sources until one works; it is logged once."""

    def __init__(
        self,
        processes: ListProcesses | None = None,
        listen: Listen | None = None,
        *,
        enabled: bool = sys.platform == "darwin",
        poll_s: float = POLL_S,
    ) -> None:
        super().__init__()
        self._processes = processes or list_output_processes
        self._listen = listen or listen_process_list
        self._enabled = enabled
        self._poll_s = poll_s
        self._table = ProcessTable()

    def start(self) -> None:
        if self._enabled:
            self._task = asyncio.create_task(self._run())

    def sources(self) -> list[Source]:
        return self._table.sources()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        changed = asyncio.Event()
        undo: Callable[[], None] | None = None

        def wake() -> None:
            loop.call_soon_threadsafe(changed.set)

        try:
            undo = self._listen(wake)
        except Exception as exc:
            log.warning("source scan: no Core Audio change listener: %s", exc)
        failing = False
        try:
            while True:
                changed.clear()
                try:
                    processes = await asyncio.to_thread(self._processes)
                except Exception as exc:
                    if not failing:
                        log.warning("source scan: Core Audio's process list: %s", exc)
                    failing = True
                    self._dispatch(self._table.clear())
                else:
                    failing = False
                    self._dispatch(self._table.update(processes))
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(self._poll_s):
                        await changed.wait()
        finally:
            if undo is not None:
                with contextlib.suppress(Exception):
                    undo()


def list_output_processes() -> list[AudioProcess]:
    if sys.platform != "darwin":
        raise OSError("Core Audio is only on macOS")
    import CoreAudio
    import objc
    from AppKit import NSRunningApplication

    with objc.autorelease_pool():
        system = CoreAudio.kAudioObjectSystemObject
        data = _property(system, CoreAudio.kAudioHardwarePropertyProcessObjectList)
        objects = struct.unpack(f"={len(data) // 4}I", data[: len(data) // 4 * 4])
        processes: list[AudioProcess] = []
        for obj in objects:
            try:
                running = _property(obj, CoreAudio.kAudioProcessPropertyIsRunningOutput)
                if not struct.unpack("=I", running[:4])[0]:
                    continue
                (pid,) = struct.unpack("=i", _property(obj, CoreAudio.kAudioProcessPropertyPID)[:4])
            except OSError:
                continue  # the process left between the list and its properties
            app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            bundle = _bundle_id(obj) or (app.bundleIdentifier() if app is not None else None)
            name = app.localizedName() if app is not None else None
            processes.append(
                AudioProcess(
                    pid=pid,
                    bundle_id=str(bundle) if bundle else None,
                    app_name=str(name) if name else None,
                    process_name=_process_name(pid),
                )
            )
        return processes


def listen_process_list(callback: Callable[[], None]) -> Callable[[], None]:
    if sys.platform != "darwin":
        raise OSError("Core Audio is only on macOS")
    import CoreAudio

    system = CoreAudio.kAudioObjectSystemObject
    address = _address(CoreAudio.kAudioHardwarePropertyProcessObjectList)

    def changed(_count: int, _addresses: object) -> None:
        callback()

    _check(CoreAudio.AudioObjectAddPropertyListenerBlock(system, address, None, changed))

    def undo() -> None:
        CoreAudio.AudioObjectRemovePropertyListenerBlock(system, address, None, changed)

    return undo


def _address(selector: int) -> object:
    if sys.platform != "darwin":
        raise OSError("Core Audio is only on macOS")
    import CoreAudio

    return CoreAudio.AudioObjectPropertyAddress(
        selector,
        CoreAudio.kAudioObjectPropertyScopeGlobal,
        CoreAudio.kAudioObjectPropertyElementMain,
    )


def _property(obj: int, selector: int) -> bytes:
    if sys.platform != "darwin":
        raise OSError("Core Audio is only on macOS")
    import CoreAudio

    address = _address(selector)
    status, size = CoreAudio.AudioObjectGetPropertyDataSize(obj, address, 0, None, None)
    _check(status)
    status, size, data = CoreAudio.AudioObjectGetPropertyData(obj, address, 0, None, size, None)
    _check(status)
    return bytes(data)[:size]


def _check(status: int) -> None:
    if status != 0:
        code = struct.pack(">i", status)
        shown = code.decode("ascii") if code.isascii() and code.decode().isprintable() else status
        raise OSError(f"Core Audio error {shown}")


def _bundle_id(obj: int) -> str | None:
    """`kAudioProcessPropertyBundleID`: a CFString the caller owns; helper processes have one
    even when no running application does."""
    if sys.platform != "darwin":
        raise OSError("Core Audio is only on macOS")
    import CoreAudio
    import objc

    try:
        data = _property(obj, CoreAudio.kAudioProcessPropertyBundleID)
    except OSError:
        return None
    if len(data) < 8:
        return None
    (pointer,) = struct.unpack("=Q", data[:8])
    if not pointer:
        return None
    try:
        text = str(objc.objc_object(c_void_p=pointer))
    finally:
        _core_foundation().CFRelease(ctypes.c_void_p(pointer))
    return text or None


@functools.cache
def _core_foundation() -> ctypes.CDLL:
    library = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    library.CFRelease.argtypes = [ctypes.c_void_p]
    library.CFRelease.restype = None
    return library


def _process_name(pid: int) -> str | None:
    cmdline = macos_cmdline(pid)
    return PurePosixPath(cmdline[0]).name if cmdline else None
