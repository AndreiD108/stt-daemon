"""The socket file, shutdown, the context store across sessions, marker times."""

import asyncio
import socket
import stat
import time

import pytest
from harness import FIXTURE, Harness, fake

from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.fake import FakeReply
from stt_daemon.config import Config
from stt_daemon.server.socket import Server, SocketInUse


async def test_the_socket_is_private(daemon: Harness) -> None:
    await daemon.start()
    assert stat.S_IMODE(daemon.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(daemon.path.parent.stat().st_mode) == 0o700


async def test_a_stale_socket_is_replaced(daemon: Harness) -> None:
    daemon.path.parent.mkdir()
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(daemon.path))
    stale.close()
    await daemon.start()
    await daemon.connect()


async def test_a_second_daemon_on_the_same_socket_refuses_to_start(daemon: Harness) -> None:
    first = await daemon.start()
    second = Server(
        path=daemon.path, capture=WavCapture(), setup=first.setup, reload_setup=lambda: first.setup
    )
    with pytest.raises(SocketInUse, match="already listening"):
        await second.start()
    await daemon.connect()


async def test_a_file_in_the_sockets_place_is_left_alone(daemon: Harness) -> None:
    daemon.path.parent.mkdir()
    daemon.path.write_text("mine")
    with pytest.raises(SocketInUse, match="not a socket"):
        await daemon.start()
    assert daemon.path.read_text() == "mine"


async def test_shutdown_stops_the_session_and_closes_clients(daemon: Harness) -> None:
    server = await daemon.start()
    observer, _ = await daemon.connect(observe_all=True)
    client, _ = await daemon.connect("Scribbly")
    await client.request("start_session")
    await client.request("add_marker", {"marker": "⟦A⟧"})
    await server.close()
    ended = await client.event("session_ended")
    assert ended["data"]["reason"] == "stopped"
    assert ended["data"]["final_text"] == "⟦A⟧"
    assert (await observer.event("session_ended"))["data"] == ended["data"]
    await asyncio.wait_for(client.eof.wait(), 2)
    await asyncio.wait_for(observer.eof.wait(), 2)
    assert not daemon.path.exists()


async def test_a_client_that_says_hello_during_shutdown_is_not_served(daemon: Harness) -> None:
    server = await daemon.start(fake([FakeReply("slow", delay_s=0.8)]), capture=WavCapture(FIXTURE))
    client, _ = await daemon.connect("Scribbly")
    started = await client.request("start_session")
    await asyncio.sleep(max(0.0, started["data"]["started_at"] / 1000 + 2.0 - time.time()))
    late = await daemon.raw()
    closing = asyncio.create_task(server.close())
    await asyncio.sleep(0.2)  # the session is still waiting for its last text
    assert not closing.done()
    late.send("hello", {"protocol": 1})
    late.send("start_session")
    await closing
    ended = await client.event("session_ended")
    assert ended["data"]["reason"] == "stopped"
    assert ended["data"]["final_text"] == "slow"
    await asyncio.wait_for(late.eof.wait(), 2)
    assert late.received == []


async def test_a_marker_keeps_its_arrival_time_while_a_start_is_slow(daemon: Harness) -> None:
    backend = fake()
    backend.open_delay_s = 1.0
    await daemon.start(backend)
    client, _ = await daemon.connect()
    await asyncio.sleep(1.2)  # the ring then holds the second before the command
    start_id = client.send("start_session")
    marker_id = client.send("add_marker", {"marker": "⟦A⟧"})
    started = await client.wait(lambda m: m.get("reply_to") == start_id)
    accepted = await client.wait(lambda m: m.get("reply_to") == marker_id)
    # The start's audio begins 1 s before the command; the marker came right after it.
    assert started["type"] == "session_started"
    assert 900 <= accepted["data"]["at_ms"] <= 1300


async def test_markers_before_the_session_are_refused(daemon: Harness) -> None:
    await daemon.start()
    client, _ = await daemon.connect()
    started = await client.request("start_session")
    early = started["data"]["started_at"] - 10
    reply = await client.request("add_marker", {"marker": "⟦A⟧", "at": early})
    assert reply["data"]["code"] == "out_of_range"
    late = round(time.time() * 1000) + 60_000
    accepted = await client.request("add_marker", {"marker": "⟦B⟧", "at": late})
    assert accepted["data"]["at_ms"] < 60_000  # clamped to the arrival time
    assert (await client.request("add_marker", {"marker": 3}))["data"]["code"] == "invalid"


async def test_context_key_carries_text_between_sessions_of_one_label(daemon: Harness) -> None:
    backend = fake([FakeReply("hello there")])
    await daemon.start(backend, capture=WavCapture(FIXTURE), config=Config())
    first, _ = await daemon.connect("Talkey")
    started = await first.request("start_session", {"context_key": "doc"})
    assert started["data"]["previous_text"] is None
    await asyncio.sleep(max(0.0, started["data"]["started_at"] / 1000 + 2.5 - time.time()))
    await first.request("add_marker", {"marker": "⟦A⟧"})
    ended = await first.request("stop_session")
    assert ended["data"]["final_text"] == "hello there ⟦A⟧"

    again, _ = await daemon.connect("Talkey")
    second = await again.request("start_session", {"context_key": "doc"})
    assert second["data"]["previous_text"] == "hello there"
    assert backend.connections[-1].options.previous_text == "hello there"
    await again.request("cancel_session")

    unlabelled, _ = await daemon.connect()
    third = await unlabelled.request("start_session", {"context_key": "doc"})
    assert third["data"]["previous_text"] is None
    await unlabelled.request("cancel_session")

    given = await again.request("start_session", {"context_key": "doc", "previous_text": "mine"})
    assert given["data"]["previous_text"] == "mine"
