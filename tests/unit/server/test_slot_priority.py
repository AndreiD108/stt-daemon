"""Dictation priority: which session a dictation suspends, resume order, refusals between kinds."""

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from stt_daemon.audio.stream_capture import ByteStreamCapture
from stt_daemon.backends.base import OpenFailed, OpenOptions
from stt_daemon.backends.fake import FakeBackend
from stt_daemon.config import Config
from stt_daemon.server.peer import Peer
from stt_daemon.server.slot import ClientGone, Emit
from stt_daemon.server.slot_state import Client, StartRefused
from stt_daemon.session.options import SessionOptions
from stt_daemon.session.session import Session
from unit.server.slot_harness import A, B, C, Harness, SpyCapture
from unit.waiting import wait_until

D = Client("c4", "Batch", Peer(900, ("batch",)))


def transcribe(backend: FakeBackend) -> Callable[[Emit], Session]:
    def make(emit: Emit) -> Session:
        return Session(
            config=Config(),
            capture=ByteStreamCapture(None),
            backend=backend,
            options=SessionOptions(OpenOptions(language="en")),
            emit=emit,
            kind="transcribe",
        )

    return make


def session(h: Harness, client: Client) -> Session:
    s = h.slot.session_of(client.client_id)
    assert isinstance(s, Session)
    return s


def suspended(h: Harness) -> list[str]:
    return [s.client_id for u in h.slot.state.backends for s in u.suspended]


async def test_a_dictation_suspends_the_most_recent_transcribe_and_resumes_it_after(
    silent: SpyCapture,
) -> None:
    h = Harness(silent, capacities={"fake": 2})
    backend = FakeBackend()
    await h.slot.start_input(B, make_session=transcribe(backend))
    await asyncio.sleep(0.01)
    await h.slot.start_input(C, make_session=transcribe(backend))
    await h.start(A, maker=h.maker(backend))
    assert suspended(h) == ["c3"]
    assert h.on("fake") == ["c2", "c1"]
    assert session(h, C).suspended
    assert not session(h, B).suspended
    wire: Any = h.changes[-1].to_wire()["backends"]
    flags = [(s["kind"], s["suspended"]) for s in wire["fake"]["sessions"]]
    assert flags == [("transcribe", False), ("mic", False), ("transcribe", True)]
    await session(h, A).stop()
    await wait_until(lambda: not session(h, C).suspended)
    assert suspended(h) == []
    assert h.on("fake") == ["c2", "c3"]
    for client in (B, C):
        await session(h, client).cancel()


async def test_the_longest_suspended_resumes_first(silent: SpyCapture) -> None:
    h = Harness(silent, evictable_after_s=0.1, capacities={"fake": 2})
    backend = FakeBackend()
    await h.slot.start_input(B, make_session=transcribe(backend))
    await h.slot.start_input(C, make_session=transcribe(backend))
    await h.start(A, maker=h.maker(backend))
    h.occupy(client=Client("c9"))
    await h.evictable()
    await h.start(D, maker=h.maker(backend))
    assert suspended(h) == ["c3", "c2"]
    h.capacities = {"fake": 3}
    h.slot.capacities_changed()
    await wait_until(lambda: not session(h, C).suspended)
    assert session(h, B).suspended
    for client in (B, C):
        await session(h, client).cancel()
    await session(h, D).stop()


async def test_only_a_dictation_suspends_and_only_other_kinds(silent: SpyCapture) -> None:
    h = Harness(silent, capacities={"fake": 1})
    backend = FakeBackend()
    await h.slot.start_input(B, make_session=transcribe(backend))
    with pytest.raises(StartRefused) as busy:
        await h.slot.start_input(D, make_session=transcribe(backend))
    assert busy.value.full_slot == "backend"
    await session(h, B).cancel()
    h.occupy()  # a mic session of another daemon client, holding the backend
    with pytest.raises(StartRefused):
        await h.start(A, maker=h.maker(backend))
    assert suspended(h) == []


async def test_a_suspended_session_whose_backend_is_gone_fails_with_config(
    silent: SpyCapture,
) -> None:
    h = Harness(silent, capacities={"fake": 1})
    backend = FakeBackend()
    await h.slot.start_input(B, make_session=transcribe(backend))
    await h.start(A, maker=h.maker(backend))
    h.capacities = {}
    h.slot.capacities_changed()
    await wait_until(lambda: "c2" in h.ended)
    assert h.ended["c2"].reason == "error"
    assert suspended(h) == []
    await session(h, A).stop()


async def test_a_cancelled_suspended_session_frees_its_place(silent: SpyCapture) -> None:
    h = Harness(silent, capacities={"fake": 1})
    backend = FakeBackend()
    await h.slot.start_input(B, make_session=transcribe(backend))
    await h.start(A, maker=h.maker(backend))
    await h.slot.client_gone("c2")
    assert h.ended["c2"].reason == "client_gone"
    assert suspended(h) == []
    await session(h, A).stop()


async def test_a_dictation_failing_to_start_gives_the_slot_back(silent: SpyCapture) -> None:
    h = Harness(silent, capacities={"fake": 1})
    backend = FakeBackend()
    await h.slot.start_input(B, make_session=transcribe(backend))
    refusing = FakeBackend(open_failures=[OpenFailed("auth", "bad key", retryable=False)])
    with pytest.raises(OpenFailed):
        await h.start(A, maker=h.maker(refusing))
    await wait_until(lambda: not session(h, B).suspended)
    assert h.on("fake") == ["c2"]
    await session(h, B).cancel()


async def test_a_provider_concurrency_refusal_suspends_a_session_and_retries_once(
    silent: SpyCapture,
) -> None:
    h = Harness(silent, capacities={"fake": 5})
    backend = FakeBackend()
    await h.slot.start_input(B, make_session=transcribe(backend))
    refusal = OpenFailed("concurrency_limit", "plan limit", retryable=False)
    backend.open_failures.append(refusal)
    await h.start(A, maker=h.maker(backend))
    assert session(h, B).suspended
    assert backend.open_calls == 3
    h.capacities = {"fake": 6}
    h.slot.capacities_changed()
    await asyncio.sleep(0.05)
    assert session(h, B).suspended, "it waits for the dictation, not for room in the count"
    await session(h, A).stop()
    await wait_until(lambda: not session(h, B).suspended)
    await session(h, B).cancel()


async def test_a_concurrency_refusal_with_no_one_to_suspend_fails_the_start(
    silent: SpyCapture,
) -> None:
    h = Harness(silent, capacities={"fake": 5})
    backend = FakeBackend(open_failures=[OpenFailed("concurrency_limit", "x", retryable=False)])
    with pytest.raises(OpenFailed) as failed:
        await h.start(A, maker=h.maker(backend))
    assert failed.value.code == "concurrency_limit"
    assert backend.open_calls == 1


async def test_a_dictation_gone_mid_suspension_resumes_its_victim_after_the_old_close(
    silent: SpyCapture,
) -> None:
    h = Harness(silent, capacities={"fake": 1})
    backend = FakeBackend()
    backend.live_limit = 1
    await h.slot.start_input(B, make_session=transcribe(backend))
    backend.close_gate = gate = asyncio.Event()
    starting = asyncio.create_task(h.start(A, maker=h.maker(backend)))
    await wait_until(lambda: backend.open_calls == 1 and session(h, B).suspendable is False)
    await h.slot.client_gone("c1")
    with pytest.raises(ClientGone):
        await starting
    await asyncio.sleep(0.05)
    assert backend.open_calls == 1, "no open while the old connection still counts"
    gate.set()
    await wait_until(lambda: backend.open_calls == 2 and not session(h, B).suspended)
    assert "c2" not in h.ended
    assert [c.ended for c in backend.connections] == [True, False]
    await session(h, B).cancel()
