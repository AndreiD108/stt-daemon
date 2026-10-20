"""The macOS source scan on a stand-in Core Audio layer: process records → sources, ids, events,
the change listener, read failures, following an app by bundle id; the scan per platform."""

import asyncio
import logging
import threading
import time
from collections.abc import Callable

import pytest

from stt_daemon.server import sources_macos
from stt_daemon.server.sources import (
    PwDumpScan,
    SourceAdded,
    SourceEvent,
    SourceRemoved,
    SourceScan,
    default_scan,
)
from stt_daemon.server.sources_macos import AudioProcess, CoreAudioScan, ProcessTable

SAFARI = AudioProcess(11, "com.apple.Safari", "Safari", "Safari")


class FakeCoreAudio:
    """The processes whose output runs; `fail` makes reads raise; `changed` calls the listener
    from another thread, as Core Audio does."""

    def __init__(self, processes: list[AudioProcess] | None = None) -> None:
        self.processes = list(processes or [])
        self.fail: str | None = None
        self.reads = 0
        self.listener: Callable[[], None] | None = None
        self.undone = False

    def read(self) -> list[AudioProcess]:
        self.reads += 1
        if self.fail is not None:
            raise OSError(self.fail)
        return list(self.processes)

    def listen(self, callback: Callable[[], None]) -> Callable[[], None]:
        self.listener = callback

        def undo() -> None:
            self.undone = True

        return undo

    def changed(self) -> None:
        assert self.listener is not None
        thread = threading.Thread(target=self.listener)
        thread.start()
        thread.join()


async def until(check: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


def collect(scan: SourceScan) -> list[SourceEvent]:
    events: list[SourceEvent] = []
    scan.on_event = events.append
    return events


def test_a_process_becomes_a_source_named_by_its_app_and_keyed_by_pid() -> None:
    table = ProcessTable()
    [added] = table.update([SAFARI])
    assert isinstance(added, SourceAdded)
    source = added.source
    assert (source.id, source.serial, source.name, source.binary, source.media_name) == (
        "src_1",
        "11",
        "Safari",
        "com.apple.Safari",
        None,
    )


@pytest.mark.parametrize(
    ("process", "name"),
    [
        (AudioProcess(5, "com.example.player", None, "player"), "com.example.player"),
        (AudioProcess(5, None, None, "afplay"), "afplay"),
        (AudioProcess(5, None, None, None), "5"),
    ],
)
def test_a_name_falls_back_to_the_bundle_id_then_the_process_name(
    process: AudioProcess, name: str
) -> None:
    [added] = ProcessTable().update([process])
    assert isinstance(added, SourceAdded)
    assert added.source.name == name


def test_ids_last_while_the_process_is_listed_and_are_never_reused() -> None:
    table = ProcessTable()
    table.update([SAFARI])
    renamed = AudioProcess(11, "com.apple.Safari", "Safari (renamed)", "Safari")
    assert table.update([renamed]) == []  # what a source says is fixed when first seen
    [removed] = table.update([])
    assert isinstance(removed, SourceRemoved)
    assert removed.source.id == "src_1"
    [added] = table.update([SAFARI])
    assert isinstance(added, SourceAdded)
    assert added.source.id == "src_2"
    assert [s.id for s in table.sources()] == ["src_2"]


async def test_the_scan_reads_at_start_and_again_soon_after_a_change() -> None:
    audio = FakeCoreAudio([SAFARI])
    scan = CoreAudioScan(audio.read, audio.listen, enabled=True, poll_s=30.0)
    events = collect(scan)
    scan.start()
    try:
        await until(lambda: len(events) == 1)
        assert [s.name for s in scan.sources()] == ["Safari"]
        audio.processes = [AudioProcess(20, "com.spotify.client", "Spotify", "Spotify")]
        audio.changed()
        await until(lambda: len(events) == 3, timeout=1.0)
    finally:
        await scan.close()
    assert [type(e).__name__ for e in events] == ["SourceAdded", "SourceRemoved", "SourceAdded"]
    assert [s.name for s in scan.sources()] == ["Spotify"]
    assert audio.undone


async def test_the_scan_polls_without_a_change_report() -> None:
    audio = FakeCoreAudio()
    scan = CoreAudioScan(audio.read, audio.listen, enabled=True, poll_s=0.05)
    events = collect(scan)
    scan.start()
    try:
        await until(lambda: audio.reads >= 1)
        audio.processes = [SAFARI]  # output started: the list itself didn't change
        await until(lambda: len(events) == 1)
    finally:
        await scan.close()


async def test_a_scan_without_a_listener_still_polls(caplog: pytest.LogCaptureFixture) -> None:
    audio = FakeCoreAudio([SAFARI])

    def no_listener(_callback: Callable[[], None]) -> Callable[[], None]:
        raise OSError("Core Audio error !obj")

    scan = CoreAudioScan(audio.read, no_listener, enabled=True, poll_s=0.05)
    scan.start()
    try:
        await until(lambda: audio.reads >= 3)
    finally:
        await scan.close()
    assert [s.name for s in scan.sources()] == ["Safari"]
    assert "no Core Audio change listener" in caplog.text


async def test_a_failing_read_empties_the_sources_logs_once_and_recovers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    audio = FakeCoreAudio([SAFARI])
    scan = CoreAudioScan(audio.read, audio.listen, enabled=True, poll_s=0.02)
    events = collect(scan)
    scan.start()
    try:
        await until(lambda: len(events) == 1)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            audio.fail = "Core Audio error who?"
            reads = audio.reads
            await until(lambda: audio.reads >= reads + 4)
            assert scan.sources() == []
            assert caplog.text.count("Core Audio error who?") == 1
            audio.fail = None
            await until(lambda: len(scan.sources()) == 1)
    finally:
        await scan.close()
    assert [type(e).__name__ for e in events] == ["SourceAdded", "SourceRemoved", "SourceAdded"]


async def test_a_disabled_scan_never_reads() -> None:
    audio = FakeCoreAudio([SAFARI])
    scan = CoreAudioScan(audio.read, audio.listen, enabled=False)
    scan.start()
    await asyncio.sleep(0.05)
    await scan.close()
    assert audio.reads == 0
    assert scan.sources() == []


async def test_following_an_app_moves_to_its_new_pid_by_bundle_id() -> None:
    audio = FakeCoreAudio([SAFARI])
    scan = CoreAudioScan(audio.read, audio.listen, enabled=True, poll_s=30.0)
    scan.start()
    try:
        await until(lambda: len(scan.sources()) == 1)
        follow = scan.follow(scan.sources()[0])
        assert follow.current() == "11"
        audio.processes = []
        audio.changed()
        await until(lambda: scan.sources() == [])
        assert follow.current() is None
        waiting = asyncio.create_task(follow.appeared())
        audio.processes = [
            AudioProcess(30, "com.example.other", "Other", "Other"),
            AudioProcess(31, "com.apple.Safari", "Safari", "Safari"),
        ]
        audio.changed()
        assert await asyncio.wait_for(waiting, 1) == "31"
        assert follow.current() == "31"
    finally:
        await scan.close()


def test_the_default_scan_is_the_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    assert isinstance(default_scan(), PwDumpScan)
    monkeypatch.setattr("sys.platform", "darwin")
    assert isinstance(default_scan(), CoreAudioScan)
    monkeypatch.setattr("sys.platform", "win32")
    scan = default_scan()
    assert type(scan) is SourceScan
    assert scan.sources() == []


def test_the_core_audio_layer_refuses_off_macos() -> None:
    with pytest.raises(OSError, match="only on macOS"):
        sources_macos.list_output_processes()
    with pytest.raises(OSError, match="only on macOS"):
        sources_macos.listen_process_list(lambda: None)
