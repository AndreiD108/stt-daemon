"""Sources over the socket: a stand-in `pw-dump` for the scan, a stand-in `pw-record` that ends
with its target stream."""

import os
import sys
from dataclasses import replace
from pathlib import Path

from fake_pw_dump import FakePipeWire
from harness import FIXTURE, Harness, TestClient

from stt_daemon.audio.pw_capture import AppTarget, PwRecordCapture
from stt_daemon.backends.base import CommitPolicy
from stt_daemon.backends.fake import FAKE_CAPABILITIES, FakeBackend
from stt_daemon.server.sources import SourceScan

FAKE_PW_RECORD = os.path.join(os.path.dirname(__file__), "..", "fake_pw_record.py")


class Graph:
    """A fake PipeWire graph, the scan reading it, and system captures that follow its streams."""

    def __init__(self, root: Path) -> None:
        self.pw = FakePipeWire(root / "pw")
        self.log = root / "pw-record-starts"
        self.scan = SourceScan(self.pw.command(), enabled=True)

    def capture(self, target: AppTarget | None) -> PwRecordCapture:
        def command(serial: str | None) -> list[str]:
            args = [sys.executable, FAKE_PW_RECORD, "--log", str(self.log), "--wav", str(FIXTURE)]
            if serial is not None:
                args += ["--target", serial, "--live", str(self.pw.live)]
            return args

        return PwRecordCapture(command, target=target, backoff_s=(0.05,))

    def starts(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []


async def start(daemon: Harness, graph: Graph, *, capacity: int | None = None) -> None:
    backend = FakeBackend(
        [], capabilities=replace(FAKE_CAPABILITIES, commit_policy=CommitPolicy(at_pauses=True))
    )
    if capacity is not None:
        backend.capacity = capacity
    await daemon.start(backend, system_capture=graph.capture, sources=graph.scan)


async def added(client: TestClient, name: str) -> str:
    event = await client.wait(
        lambda m: m["type"] == "source_added" and m["data"]["source"]["name"] == name
    )
    return event["data"]["source"]["id"]


async def test_welcome_lists_the_sources_and_only_opted_in_clients_get_their_events(
    daemon: Harness, tmp_path: Path
) -> None:
    graph = Graph(tmp_path)
    await start(daemon, graph)
    scribbly, welcome = await daemon.connect("Scribbly", sources=True)
    assert welcome["data"]["sources"] == []
    talkey, _ = await daemon.connect("Talkey")
    watch, _ = await daemon.connect("Watch", observe_all=True)
    node, _ = graph.pw.play("Zoom", binary="zoom", media_name="Meeting")
    event = await scribbly.event("source_added")
    source = event["data"]["source"]
    assert source == {
        "id": source["id"],
        "name": "Zoom",
        "binary": "zoom",
        "media_name": "Meeting",
    }
    _, late = await daemon.connect("Late")
    assert late["data"]["sources"] == [source]
    graph.pw.stop(node)
    removed = await scribbly.event("source_removed")
    assert removed["data"] == {"id": source["id"]}
    for other in (talkey, watch):
        assert other.of_type("source_added") == []
        assert other.of_type("source_removed") == []


async def test_a_system_session_on_an_app_records_that_apps_stream(
    daemon: Harness, tmp_path: Path
) -> None:
    graph = Graph(tmp_path)
    await start(daemon, graph)
    scribbly, _ = await daemon.connect("Scribbly", sources=True)
    graph.pw.play("Spotify", binary="spotify")
    _, serial = graph.pw.play("Zoom", binary="zoom")
    app = await added(scribbly, "Zoom")
    started = await scribbly.request("start_system_session", {"app": app, "with_mic": True})
    assert started["type"] == "session_started", started
    assert started["data"]["app"] == app
    assert started["data"]["app_name"] == "Zoom"
    assert started["data"]["tracks"] == ["system", "mic"]
    assert graph.starts() == [f"start {serial}"]
    assert (await scribbly.request("stop_session"))["data"]["reason"] == "stopped"


async def test_an_unknown_app_is_invalid(daemon: Harness, tmp_path: Path) -> None:
    graph = Graph(tmp_path)
    await start(daemon, graph)
    scribbly, _ = await daemon.connect("Scribbly")
    refused = await scribbly.request("start_system_session", {"app": "src_99"})
    assert refused["type"] == "error"
    assert refused["data"]["code"] == "invalid"
    assert refused["data"]["field"] == "app"
    assert graph.starts() == []


async def test_the_track_reattaches_when_the_app_plays_again(
    daemon: Harness, tmp_path: Path
) -> None:
    graph = Graph(tmp_path)
    await start(daemon, graph)
    scribbly, _ = await daemon.connect("Scribbly", sources=True)
    node, first = graph.pw.play("Zoom", binary="zoom")
    app = await added(scribbly, "Zoom")
    started = await scribbly.request("start_system_session", {"app": app})
    assert started["type"] == "session_started", started
    graph.pw.stop(node)
    lost = await scribbly.event("session_error", code="capture_lost")
    assert lost["data"]["track"] == "system"
    assert lost["data"]["fatal"] is False
    await scribbly.event("source_removed")
    _, second = graph.pw.play("Zoom", binary="zoom")
    restored = await scribbly.event("session_error", code="capture_restored")
    assert restored["data"]["track"] == "system"
    starts = graph.starts()
    assert starts[0] == f"start {first}"
    assert starts[-1] == f"start {second}"
    assert (await scribbly.request("stop_session"))["data"]["reason"] == "stopped"


async def test_a_suspended_session_reattaches_to_an_app_that_came_back(
    daemon: Harness, tmp_path: Path
) -> None:
    graph = Graph(tmp_path)
    await start(daemon, graph, capacity=1)
    scribbly, _ = await daemon.connect("Scribbly", sources=True)
    node, _ = graph.pw.play("Zoom", binary="zoom")
    app = await added(scribbly, "Zoom")
    assert (await scribbly.request("start_system_session", {"app": app}))["type"] == (
        "session_started"
    )
    talkey, _ = await daemon.connect("Talkey")
    assert (await talkey.request("start_session"))["type"] == "session_started"
    await scribbly.event("session_suspended")
    graph.pw.stop(node)
    await scribbly.event("session_error", code="capture_lost")
    _, second = graph.pw.play("Zoom", binary="zoom")
    assert (await talkey.request("cancel_session"))["data"]["reason"] == "cancelled"
    await scribbly.event("session_resumed")
    await scribbly.event("session_error", code="capture_restored")
    assert graph.starts()[-1] == f"start {second}"
    assert (await scribbly.request("stop_session"))["data"]["reason"] == "stopped"
