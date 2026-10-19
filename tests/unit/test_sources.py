"""Source scan: `pw-dump -m -N` output → sources and events; the subprocess; following an app."""

import asyncio
import json
import logging
from pathlib import Path

import pytest
from fake_elevenlabs import wait_until
from fake_pw_dump import FakePipeWire

from stt_daemon.server.sources import (
    ArraySplitter,
    Source,
    SourceAdded,
    SourceEvent,
    SourceScan,
    SourceTable,
)

RECORDED = Path(__file__).parents[1] / "fixtures" / "pw_dump_monitor.json"
"""`pw-dump -m -N` on PipeWire 1.4: the registry, `pw-play` starting and ending, then a
`pw-record` stream reusing the registry id `pw-play`'s node had."""


def recorded_arrays() -> list[list[object]]:
    text, decoder, at = RECORDED.read_text(), json.JSONDecoder(), 0
    arrays: list[list[object]] = []
    while (at := len(text) - len(text[at:].lstrip())) < len(text):
        array, at = decoder.raw_decode(text, at)
        arrays.append(array)
    return arrays


def short(events: list[SourceEvent]) -> list[tuple[str, str, str]]:
    return [
        ("added" if isinstance(e, SourceAdded) else "removed", e.source.id, e.source.name)
        for e in events
    ]


@pytest.mark.parametrize("chunk", [1, 7, 4096, 1 << 20])
def test_the_splitter_cuts_the_recording_into_its_arrays_at_any_chunk_size(chunk: int) -> None:
    text = RECORDED.read_text()
    splitter, found = ArraySplitter(), list[str]()
    for at in range(0, len(text), chunk):
        found += splitter.feed(text[at : at + chunk])
    assert [json.loads(t) for t in found] == recorded_arrays()
    assert len(found) == 26


def test_brackets_and_quotes_inside_strings_do_not_cut_an_array() -> None:
    text = '[{"media.name": "a ] b [ c \\" ] d"}]\n[{"x": "\\\\"}]'
    splitter = ArraySplitter()
    found = [part for char in text for part in splitter.feed(char)]
    assert [json.loads(t) for t in found] == [[{"media.name": 'a ] b [ c " ] d'}], [{"x": "\\"}]]


def test_the_recording_gives_one_event_per_stream_that_plays_and_ends() -> None:
    table = SourceTable()
    per_array = [table.update(array) for array in recorded_arrays()]
    assert short(per_array[0]) == [("added", "src_1", "speech-dispatcher-dummy")]
    assert short(per_array[1]) == [("added", "src_2", "pw-play")]
    removed = [i for i, events in enumerate(per_array) if events and i > 1]
    assert removed == [9]
    assert short(per_array[9]) == [("removed", "src_2", "pw-play")]
    assert [s.id for s in table.sources()] == ["src_1"]


def test_a_source_carries_its_binary_from_the_node_or_from_its_client() -> None:
    table = SourceTable()
    arrays = recorded_arrays()
    [dispatcher] = [e.source for e in table.update(arrays[0])]
    [player] = [e.source for e in table.update(arrays[1])]
    assert dispatcher == Source("src_1", "472", "speech-dispatcher-dummy", "sd_dummy", "playback")
    assert player == Source("src_2", "11811", "pw-play", "pw-cat", "silent.wav")


def test_updates_keep_the_id_and_a_reused_registry_id_is_a_new_source() -> None:
    table = SourceTable()

    def node(serial: int, media_class: str = "Stream/Output/Audio") -> list[object]:
        props = {"object.serial": serial, "media.class": media_class, "application.name": "Zoom"}
        return [{"id": 40, "type": "PipeWire:Interface:Node", "info": {"props": props}}]

    assert short(table.update(node(1))) == [("added", "src_1", "Zoom")]
    assert table.update(node(1)) == []
    assert table.update([{"id": 40, "type": "PipeWire:Interface:Node", "info": {}}]) == []
    assert short(table.update(node(2))) == [
        ("removed", "src_1", "Zoom"),
        ("added", "src_2", "Zoom"),
    ]
    assert short(table.update(node(2, "Stream/Input/Audio"))) == [("removed", "src_2", "Zoom")]
    assert table.update([{"id": 40, "info": None}]) == []


def test_the_name_falls_back_to_the_node_name() -> None:
    props = {"object.serial": 5, "media.class": "Stream/Output/Audio", "node.name": "mpv"}
    events = SourceTable().update(
        [{"id": 3, "type": "PipeWire:Interface:Node", "info": {"props": props}}]
    )
    assert [e.source for e in events] == [Source("src_1", "5", "mpv", None, None)]


async def scan_events(scan: SourceScan) -> list[SourceEvent]:
    events: list[SourceEvent] = []
    scan.on_event = events.append
    scan.start()
    return events


async def test_the_scan_follows_the_graph_and_restarts_the_subprocess(tmp_path: Path) -> None:
    graph = FakePipeWire(tmp_path / "pw")
    graph.play("Zoom", binary="zoom")
    log = tmp_path / "starts"
    scan = SourceScan(
        graph.command("--log", str(log), "--exit-after", "0.5"), enabled=True, retry_s=0.2
    )
    events = await scan_events(scan)
    try:
        await wait_until(lambda: len(events) >= 1, timeout=3)
        node, _ = graph.play("Spotify", binary="spotify")
        await wait_until(lambda: len(events) >= 2)
        graph.stop(node)
        await wait_until(lambda: len(events) >= 3)
        # the subprocess exits: its sources go, and come back with new ids after the restart
        await wait_until(lambda: len(events) >= 5, timeout=3)
        await wait_until(lambda: len(log.read_text().splitlines()) >= 2, timeout=3)
    finally:
        await scan.close()
    assert short(events[:5]) == [
        ("added", "src_1", "Zoom"),
        ("added", "src_2", "Spotify"),
        ("removed", "src_2", "Spotify"),
        ("removed", "src_1", "Zoom"),
        ("added", "src_3", "Zoom"),
    ]


async def test_a_missing_pw_dump_leaves_no_sources_and_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, "stt_daemon.server.sources")
    scan = SourceScan(["/nonexistent/pw-dump"], enabled=True, retry_s=0.05)
    events = await scan_events(scan)
    await asyncio.sleep(0.3)
    await scan.close()
    assert events == []
    assert scan.sources() == []
    assert len(caplog.records) == 1


async def test_off_linux_the_scan_never_runs(tmp_path: Path) -> None:
    graph = FakePipeWire(tmp_path)
    graph.play("Zoom")
    scan = SourceScan(graph.command(), enabled=False)
    events = await scan_events(scan)
    await asyncio.sleep(0.3)
    await scan.close()
    assert events == []
    assert scan.sources() == []


async def test_following_an_app_moves_to_its_next_stream(tmp_path: Path) -> None:
    graph = FakePipeWire(tmp_path)
    first, first_serial = graph.play("Zoom", binary="zoom")
    graph.play("Spotify", binary="spotify")
    scan = SourceScan(graph.command(), enabled=True)
    scan.start()
    try:
        await wait_until(lambda: len(scan.sources()) == 2, timeout=3)
        zoom = scan.find(lambda s: s.name == "Zoom")
        assert zoom is not None
        follow = scan.follow(zoom)
        assert follow.current() == first_serial
        graph.stop(first)
        await wait_until(lambda: len(scan.sources()) == 1)
        assert follow.current() is None
        waiting = asyncio.create_task(follow.appeared())
        await asyncio.sleep(0.1)
        graph.play("Spotify", binary="spotify")
        _, meeting_serial = graph.play("Zoom Meeting", binary="zoom")
        assert await asyncio.wait_for(waiting, 3) == meeting_serial
        assert follow.current() == meeting_serial
    finally:
        await scan.close()


async def test_an_app_without_a_binary_is_followed_by_name(tmp_path: Path) -> None:
    graph = FakePipeWire(tmp_path)
    first, _ = graph.play("mpv")
    scan = SourceScan(graph.command(), enabled=True)
    scan.start()
    try:
        await wait_until(lambda: len(scan.sources()) == 1, timeout=3)
        follow = scan.follow(scan.sources()[0])
        graph.stop(first)
        await wait_until(lambda: scan.sources() == [])
        graph.play("mpv-other")
        _, serial = graph.play("mpv")
        assert await asyncio.wait_for(follow.appeared(), 3) == serial
    finally:
        await scan.close()


async def test_a_node_is_found_by_name_once_it_appears_and_not_listed_as_a_source(
    tmp_path: Path,
) -> None:
    graph = FakePipeWire(tmp_path)
    scan = SourceScan(graph.command(), enabled=True)
    scan.start()
    try:
        waiting = asyncio.create_task(scan.wait_node("stt-daemon-aec-1"))
        await asyncio.sleep(0.2)
        graph.add_source("other")
        node, serial = graph.add_source("stt-daemon-aec-1")
        assert await asyncio.wait_for(waiting, 3) == serial
        assert scan.node_serial("stt-daemon-aec-1") == serial
        assert scan.sources() == []
        graph.remove(node)
        await wait_until(lambda: scan.node_serial("stt-daemon-aec-1") is None)
    finally:
        await scan.close()


def test_sources_go_on_the_wire_without_their_serial() -> None:
    source = Source("src_7", "123", "Zoom", "zoom", "Playback")
    assert source.to_wire() == {
        "id": "src_7",
        "name": "Zoom",
        "binary": "zoom",
        "media_name": "Playback",
    }
