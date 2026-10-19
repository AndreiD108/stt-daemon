"""Several clients over the socket: hello gate, busy and eviction, backend capacity, observers,
disconnects."""

import asyncio
import os

from harness import Harness, fake, setup_with

from stt_daemon.config import Config
from stt_daemon.server.backend_slots import BackendSession


async def test_the_first_message_must_be_hello(daemon: Harness) -> None:
    await daemon.start()
    client = await daemon.raw()
    reply = await client.request("start_session")
    assert reply["type"] == "error"
    assert reply["data"]["code"] == "protocol"
    await asyncio.wait_for(client.eof.wait(), 2)


async def test_another_protocol_is_refused(daemon: Harness) -> None:
    await daemon.start()
    client = await daemon.raw()
    reply = await client.request("hello", {"protocol": 2})
    assert reply["data"]["code"] == "protocol_unsupported"
    assert reply["data"]["supported"] == [1]
    await asyncio.wait_for(client.eof.wait(), 2)


async def test_observers_can_be_switched_off(daemon: Harness) -> None:
    config = Config()
    config.server.observers = False
    await daemon.start(config=config)
    client = await daemon.raw()
    reply = await client.request("hello", {"protocol": 1, "observe_all": True})
    assert reply["data"]["code"] == "observers_disabled"
    await asyncio.wait_for(client.eof.wait(), 2)
    _, welcome = await daemon.connect("Talkey")
    assert welcome["type"] == "welcome"


async def test_welcome_describes_the_daemon(daemon: Harness) -> None:
    await daemon.start()
    _, welcome = await daemon.connect("Talkey")
    data = welcome["data"]
    assert data["protocol"] == 1
    assert data["client_id"].startswith("c")
    assert data["backends"] == [
        {
            "name": "fake",
            "languages": ["auto", "en"],
            "word_timestamps": True,
            "partials": True,
            "hints": ["keyterms", "previous_text"],
        }
    ]
    assert data["default_backend"] == "fake"
    assert data["mic_open"] is True
    assert data["slot"] == {
        "state": "idle",
        "backends": {"fake": {"max_sessions": 2, "sessions": []}},
    }


async def test_without_a_backend_starts_fail_and_name_the_setup_commands(daemon: Harness) -> None:
    await daemon.start(backends={})
    client, welcome = await daemon.connect()
    assert welcome["data"]["backends"] == []
    assert welcome["data"]["default_backend"] is None
    reply = await client.request("start_session")
    assert reply["data"]["code"] == "no_backend"
    assert "stt-daemon key set elevenlabs" in reply["data"]["message"]
    assert "stt-daemon backend install parakeet" in reply["data"]["message"]


async def test_bad_commands_get_errors_and_the_connection_goes_on(daemon: Harness) -> None:
    await daemon.start()
    client, _ = await daemon.connect()
    unknown = await client.request("dance")
    assert unknown["data"]["code"] == "bad_request"
    bad = await client.request("start_session", {"partials": "yes"})
    assert bad["data"]["code"] == "bad_request"
    assert bad["data"]["field"] == "partials"
    assert (await client.request("stop_session"))["data"]["code"] == "no_session"
    assert (await client.request("add_marker", {"marker": "⟦A⟧"}))["data"]["code"] == "no_session"
    unsupported = await client.request("start_session", {"language": "xx"})
    assert unsupported["data"]["code"] == "unsupported_language"
    client.writer.write(b"not json\n")
    assert (await client.event("error"))["data"]["code"] == "bad_request"
    assert (await client.request("ping"))["type"] == "pong"


async def test_busy_then_eviction_end_to_end(daemon: Harness) -> None:
    config = Config()
    config.slot.evictable_after_s = 1
    await daemon.start(fake(), config=config)
    scribbly, _ = await daemon.connect("Scribbly")
    talkey, welcome = await daemon.connect("Talkey")
    started = await scribbly.request("start_session", {"label": "desks.html"})
    assert started["type"] == "session_started"
    assert started["data"]["client_label"] == "Scribbly"
    assert started["data"]["kind"] == "mic"
    busy = await talkey.event("slot", state="busy")
    holder = busy["data"]["holder"]
    assert holder["client_label"] == "Scribbly"
    assert holder["session_label"] == "desks.html"
    assert holder["pid"] == os.getpid()
    assert holder["since"] == started["data"]["started_at"]
    assert busy["data"]["backends"]["fake"]["sessions"] == [
        {
            "session_id": started["data"]["session_id"],
            "kind": "mic",
            "pid": os.getpid(),
            "client_label": "Scribbly",
        }
    ]
    refused = await talkey.request("start_session")
    assert refused["data"]["code"] == "busy"
    assert refused["data"]["slot"]["state"] == "busy"
    assert refused["data"]["full_slot"] == "mic"
    assert refused["data"]["holder"] == {
        "pid": os.getpid(),
        "client_label": "Scribbly",
        "kind": "mic",
    }
    await talkey.event("slot", state="evictable", timeout=3)
    took = await talkey.request("start_session")
    assert took["type"] == "session_started"
    ended = await scribbly.event("session_ended")
    assert ended["data"]["reason"] == "evicted"
    assert ended["data"]["evicted_by"] == {
        "client_id": welcome["data"]["client_id"],
        "pid": os.getpid(),
        "client_label": "Talkey",
    }
    states = [(m["data"]["state"], m["data"].get("holder", {}).get("client_label"))
              for m in scribbly.of_type("slot")]  # fmt: skip
    assert states == [("busy", "Scribbly"), ("evictable", "Scribbly"), ("busy", "Talkey")]
    order = [m["type"] for m in talkey.received if m["type"] in ("slot", "session_started")]
    assert order[-2:] == ["slot", "session_started"]


async def test_observers_get_copies_with_only_the_options_asked_for(daemon: Harness) -> None:
    await daemon.start()
    observer, _ = await daemon.connect("Log", observe_all=True)
    other, _ = await daemon.connect("Other")
    client, welcome = await daemon.connect("Scribbly")
    started = await client.request("start_session", {"label": "notes"})
    accepted = await client.request("add_marker", {"marker": "⟦A⟧"})
    assert accepted["type"] == "marker_accepted"
    assert accepted["data"]["marker"] == "⟦A⟧"
    ended = await client.request("stop_session")
    assert ended["data"]["final_text"] == "⟦A⟧"
    copy_end = await observer.event("session_ended")
    assert "reply_to" not in copy_end
    assert copy_end["data"] == ended["data"]
    copy_start = await observer.event("session_started")
    assert "reply_to" not in copy_start
    assert copy_start["data"] == started["data"]
    assert copy_start["data"]["client_id"] == welcome["data"]["client_id"]
    assert copy_start["data"]["session_label"] == "notes"
    [segment] = observer.of_type("segment")
    assert "words" not in segment["data"]
    assert segment["data"]["markers"] == [{"marker": "⟦A⟧", "at_ms": accepted["data"]["at_ms"]}]
    assert client.of_type("segment")[0]["data"] == segment["data"]
    assert not other.of_type("session_started")
    assert not other.of_type("segment")
    assert [m["data"]["state"] for m in other.of_type("slot")] == ["busy", "idle"]


async def test_a_holder_that_disconnects_ends_its_session(daemon: Harness) -> None:
    await daemon.start()
    observer, _ = await daemon.connect(observe_all=True)
    client, _ = await daemon.connect("Scribbly")
    await client.request("start_session")
    await client.close()
    ended = await observer.event("session_ended")
    assert ended["data"]["reason"] == "client_gone"
    await observer.event("slot", state="idle")


async def test_a_client_that_disconnects_while_its_start_waits_frees_the_slot(
    daemon: Harness,
) -> None:
    backend = fake()
    backend.open_delay_s = 5.0
    await daemon.start(backend)
    client, _ = await daemon.connect("Scribbly")
    other, _ = await daemon.connect("Talkey")
    client.send("start_session")
    await asyncio.sleep(0.2)
    await client.close()
    backend.open_delay_s = 0.0
    assert (await other.request("start_session"))["type"] == "session_started"


async def test_reload_replies_and_tells_every_other_client(daemon: Harness) -> None:
    await daemon.start()
    a, _ = await daemon.connect("A")
    b, _ = await daemon.connect("B")
    daemon.reloads = [setup_with(Config(), {}), ValueError("unused")]
    reply = await a.request("reload")
    assert reply["type"] == "backends"
    assert reply["data"] == {"backends": [], "default_backend": None}
    told = await b.event("backends")
    assert "reply_to" not in told
    assert told["data"] == reply["data"]
    assert not a.of_type("backends")[1:]
    assert (await a.request("start_session"))["data"]["code"] == "no_backend"


async def test_a_full_backend_answers_busy_naming_it(daemon: Harness) -> None:
    backend = fake()
    backend.capacity = 1
    server = await daemon.start(backend)
    held = BackendSession("held", "mic", "fake", "c0", 77, "Log")
    server._slot.backends.take(held)  # pyright: ignore[reportPrivateUsage]
    client, _ = await daemon.connect("Talkey")
    refused = await client.request("start_session")
    assert refused["data"]["code"] == "busy"
    assert refused["data"]["full_slot"] == "backend"
    assert refused["data"]["backend"] == "fake"
    assert refused["data"]["holder"] == {"pid": 77, "client_label": "Log", "kind": "mic"}
    assert refused["data"]["slot"]["state"] == "idle"
    assert not client.of_type("slot")


async def test_reload_picks_up_a_new_capacity(daemon: Harness) -> None:
    await daemon.start()
    a, _ = await daemon.connect("A")
    smaller = fake()
    smaller.capacity = 1
    daemon.reloads = [setup_with(Config(), {"fake": smaller})]
    await a.request("reload")
    slot = await a.event("slot")
    assert slot["data"] == {
        "state": "idle",
        "backends": {"fake": {"max_sessions": 1, "sessions": []}},
    }
    order = [m["type"] for m in a.received if m["type"] in ("backends", "slot")]
    assert order == ["backends", "slot"]
