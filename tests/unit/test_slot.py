"""The mic and backend slots: states, sticky evictable, takeover order and failures,
backend capacity, client_gone."""

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Literal

import pytest
from fake_elevenlabs import wait_until

from stt_daemon.audio.capture import CaptureSource
from stt_daemon.audio.frames import Attachment
from stt_daemon.audio.vad import SpeechEnded, SpeechStarted
from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.base import OpenFailed, OpenOptions
from stt_daemon.backends.fake import FakeBackend
from stt_daemon.config import Config
from stt_daemon.server.backend_slots import BackendSession
from stt_daemon.server.peer import Peer
from stt_daemon.server.slot import Client, ClientGone, Emit, Slot, SlotState, StartRefused
from stt_daemon.session.events import EvictedBy, SessionEnded, SessionEvent, SessionStarted
from stt_daemon.session.session import Session, SessionOptions

FIXTURE = Path(__file__).parents[1] / "fixtures" / "two_phrases.wav"

A = Client("c1", "Scribbly", Peer(41230, ("scribbly", "serve")))
B = Client("c2", "Talkey", Peer(5120, ("talkey",)))
C = Client("c3")


class SpyCapture(WavCapture):
    """Records when each attachment was made."""

    def __init__(self, *paths: Path, keep_open: bool = True) -> None:
        super().__init__(*paths, keep_open=keep_open)
        self.attached: list[float] = []

    def attach(self, since: float | None = None) -> Attachment:
        attachment = super().attach(since)
        self.attached.append(time.monotonic())
        return attachment


class SlowStop(Session):
    """Stops 0.3 s late; like `Session.stop`, a cancelled caller doesn't cancel the stop."""

    async def stop(self, reason: Literal["stopped", "evicted"] = "stopped") -> SessionEnded:
        async def later() -> SessionEnded:
            await asyncio.sleep(0.3)
            return await Session.stop(self, reason)

        self.stopping = asyncio.ensure_future(later())
        return await asyncio.shield(self.stopping)


class OtherFake(FakeBackend):
    name = "other"


class Harness:
    def __init__(
        self,
        capture: CaptureSource,
        evictable_after_s: float = 300.0,
        capacities: dict[str, int] | None = None,
    ) -> None:
        self.capture = capture
        self.capacities = {"fake": 2} if capacities is None else capacities
        self.log: list[tuple[str, ...]] = []
        self.changes: list[SlotState] = []
        self.ended_at: dict[str, float] = {}
        self.ended: dict[str, SessionEnded] = {}
        self.backends_at_end: dict[str, list[str]] = {}
        """Per ended client: who held backend slots as its `session_ended` went out."""
        self.gone_at_start: set[str] = set()
        """Clients that disconnect as their `session_started` goes out."""
        self.slot = Slot(
            capture=capture,
            pre_start_ms=1000,
            evictable_after_s=evictable_after_s,
            on_change=self._changed,
            on_event=self._event,
            capacities=lambda: self.capacities,
        )

    def _changed(self, state: SlotState) -> None:
        self.changes.append(state)
        self.log.append(("slot", state.state, state.holder.client_id if state.holder else ""))

    def _event(self, client: Client, event: SessionEvent) -> None:
        if isinstance(event, SessionStarted):
            self.log.append((client.client_id, "started"))
            if client.client_id in self.gone_at_start:
                asyncio.get_running_loop().create_task(self.slot.client_gone(client.client_id))
        elif isinstance(event, SessionEnded):
            self.log.append((client.client_id, event.reason))
            self.ended[client.client_id] = event
            held = [s.client_id for u in self.slot.state.backends for s in u.sessions]
            self.backends_at_end[client.client_id] = held
            self.ended_at[client.client_id] = time.monotonic()

    def maker(
        self, backend: FakeBackend | None = None, cls: type[Session] = Session
    ) -> Callable[[Emit], Session]:
        def make(emit: Emit) -> Session:
            return cls(
                config=Config(),
                capture=self.capture,
                backend=backend or FakeBackend(),
                options=SessionOptions(OpenOptions(language="en")),
                emit=emit,
            )

        return make

    async def start(
        self,
        client: Client,
        *,
        maker: Callable[[Emit], Session] | None = None,
        requested_at: float | None = None,
    ) -> SessionStarted:
        make = maker or self.maker()
        return await self.slot.start(client, make_session=make, requested_at=requested_at)

    async def evictable(self) -> None:
        await wait_until(lambda: self.slot.state.state == "evictable", 2.0)

    def occupy(self, backend: str = "fake", client: Client = C) -> BackendSession:
        """A session of another kind, holding a backend slot without the mic slot."""
        place = BackendSession(
            f"held-{client.client_id}", "mic", backend, client.client_id, 77, "Log"
        )
        self.slot.backends.take(place)
        return place

    def on(self, backend: str) -> list[str]:
        """Client ids of the sessions holding `backend`'s slots."""
        [usage] = [u for u in self.slot.state.backends if u.name == backend]
        return [s.client_id for s in usage.sessions]


@pytest.fixture
async def silent() -> AsyncIterator[SpyCapture]:
    capture = SpyCapture()
    await capture.start()
    await asyncio.sleep(0.1)
    yield capture
    await capture.stop()


async def test_start_announces_busy_before_started_and_idle_after_ended(
    silent: SpyCapture,
) -> None:
    h = Harness(silent)
    started = await h.slot.start(A, make_session=h.maker(), session_label="desks.html")
    [busy] = h.changes
    assert busy.holder is not None
    assert busy.to_wire() == {
        "state": "busy",
        "holder": {
            "client_id": "c1",
            "pid": 41230,
            "cmdline": ["scribbly", "serve"],
            "client_label": "Scribbly",
            "session_label": "desks.html",
            "session_id": started.session_id,
            "since": started.started_at,
            "last_speech_at": None,
        },
        "backends": {
            "fake": {
                "max_sessions": 2,
                "sessions": [
                    {
                        "session_id": started.session_id,
                        "kind": "mic",
                        "pid": 41230,
                        "client_label": "Scribbly",
                    }
                ],
            }
        },
    }
    assert started.kind == "mic"
    session = h.slot.session_of("c1")
    assert session is not None
    assert h.slot.session_of("c2") is None
    await session.stop()
    assert h.log == [
        ("slot", "busy", "c1"),
        ("c1", "started"),
        ("c1", "stopped"),
        ("slot", "idle", ""),
    ]
    assert h.slot.state.to_wire() == {
        "state": "idle",
        "backends": {"fake": {"max_sessions": 2, "sessions": []}},
    }


async def test_busy_and_session_active_refusals(silent: SpyCapture) -> None:
    h = Harness(silent)
    await h.start(A)
    with pytest.raises(StartRefused) as busy:
        await h.start(B)
    assert busy.value.code == "busy"
    assert busy.value.slot.holder is not None
    assert busy.value.slot.holder.client_id == "c1"
    wire = busy.value.to_wire()
    assert wire["full_slot"] == "mic"
    assert "backend" not in wire
    assert wire["holder"] == {"pid": 41230, "client_label": "Scribbly", "kind": "mic"}
    with pytest.raises(StartRefused) as active:
        await h.start(A)
    assert active.value.code == "session_active"
    assert active.value.to_wire().keys() == {"slot"}
    session = h.slot.session_of("c1")
    assert session is not None
    assert session.state == "running"
    await session.stop()


async def test_failed_start_from_idle_changes_nothing(silent: SpyCapture) -> None:
    h = Harness(silent)
    backend = FakeBackend(open_failures=[OpenFailed("auth", "bad key", retryable=False)])
    with pytest.raises(OpenFailed):
        await h.start(A, maker=h.maker(backend))
    assert h.changes == []
    assert h.slot.state.state == "idle"
    await h.start(B)
    assert h.log[0] == ("slot", "busy", "c2")


async def test_next_session_starts_where_the_previous_audio_ended(silent: SpyCapture) -> None:
    h = Harness(silent)
    first = await h.start(A)
    session = h.slot.session_of("c1")
    assert session is not None
    await asyncio.sleep(0.3)
    ended = await session.stop()
    second = await h.start(A)
    assert second.started_at >= first.started_at + ended.duration_ms - 25


async def test_quiet_session_turns_evictable_and_speech_does_not_undo_it(
    silent: SpyCapture,
) -> None:
    h = Harness(silent, evictable_after_s=0.3)
    await h.start(A)
    await h.evictable()
    assert h.slot.state.holder is not None
    assert h.slot.state.holder.last_speech_at is None
    h.slot.heard(SpeechStarted("s1", time.monotonic()))
    await asyncio.sleep(0.5)
    assert [c.state for c in h.changes] == ["busy", "evictable"]
    assert h.slot.state.state == "evictable"


async def test_speech_from_the_daemon_vad_keeps_the_slot_busy(silent: SpyCapture) -> None:
    h = Harness(silent, evictable_after_s=0.4)
    await h.start(A)
    h.slot.heard(SpeechStarted("s1", time.monotonic() - 1.0))  # began before the session
    await asyncio.sleep(0.6)
    assert h.slot.state.state == "busy"
    holder = h.slot.state.holder
    assert holder is not None
    assert holder.last_speech_at is not None
    assert abs(holder.last_speech_at / 1000 - time.time()) < 0.1  # now, while speech goes on
    ended_at = time.monotonic()
    h.slot.heard(SpeechEnded("s1", ended_at))
    await h.evictable()
    assert time.monotonic() - ended_at >= 0.4 - 0.01


async def test_takeover_order(silent: SpyCapture) -> None:
    h = Harness(silent, evictable_after_s=0.1)
    await h.start(A, maker=h.maker(cls=SlowStop))
    await h.evictable()
    await asyncio.sleep(1.0)  # the ring now reaches back past the pre-start second
    requested = time.monotonic()
    second = await h.start(B, requested_at=requested)
    assert h.log == [
        ("slot", "busy", "c1"),
        ("c1", "started"),
        ("slot", "evictable", "c1"),
        ("c1", "evicted"),
        ("slot", "busy", "c2"),
        ("c2", "started"),
    ]
    assert h.ended["c1"].evicted_by == EvictedBy("c2", 5120, "Talkey")
    # The new session's audio was taken before the holder's stop, from 1 s before the command.
    assert silent.attached[-1] < h.ended_at["c1"]
    wall_requested = (time.time() - (time.monotonic() - requested)) * 1000
    assert abs(second.started_at - (wall_requested - 1000)) < 30
    session = h.slot.session_of("c2")
    assert session is not None
    await session.stop()


async def test_bad_request_leaves_the_holder_alone(silent: SpyCapture) -> None:
    h = Harness(silent, evictable_after_s=0.1)
    await h.start(A)
    await h.evictable()

    def refuse(emit: Emit) -> Session:
        raise ValueError("unsupported_language")

    with pytest.raises(ValueError, match="unsupported_language"):
        await h.slot.start(B, make_session=refuse)
    session = h.slot.session_of("c1")
    assert session is not None
    assert session.state == "running"
    assert h.slot.state.state == "evictable"
    assert len(h.changes) == 2
    await session.stop()


async def test_start_failing_after_the_holder_stopped_leaves_the_slot_idle() -> None:
    capture = SpyCapture(keep_open=False)
    h = Harness(capture, evictable_after_s=0.1)
    await h.start(A)
    await h.evictable()
    backend = FakeBackend(open_failures=[OpenFailed("auth", "bad key", retryable=False)])
    with pytest.raises(OpenFailed):
        await h.start(B, maker=h.maker(backend))
    assert h.log[-2:] == [("c1", "evicted"), ("slot", "idle", "")]
    assert not capture.is_open  # every hold released, the reserved one too
    assert h.on("fake") == []
    await h.start(C)
    assert h.log[-2:] == [("slot", "busy", "c3"), ("c3", "started")]
    session = h.slot.session_of("c3")
    assert session is not None
    await session.stop()
    await capture.stop()


async def test_other_starts_get_busy_during_the_takeover(silent: SpyCapture) -> None:
    h = Harness(silent, evictable_after_s=0.1)
    await h.start(A, maker=h.maker(cls=SlowStop))
    await h.evictable()
    takeover = asyncio.create_task(h.start(B))
    await asyncio.sleep(0.1)
    with pytest.raises(StartRefused) as busy:
        await h.start(C)
    assert busy.value.code == "busy"
    assert busy.value.slot.state == "busy"
    assert busy.value.slot.holder is not None
    assert busy.value.slot.holder.client_id == "c2"
    with pytest.raises(StartRefused) as active:
        await h.start(B)
    assert active.value.code == "session_active"
    await takeover
    session = h.slot.session_of("c2")
    assert session is not None
    await session.stop()


async def test_new_client_gone_during_the_takeover() -> None:
    capture = SpyCapture(keep_open=False)
    h = Harness(capture, evictable_after_s=0.1)
    await h.start(A, maker=h.maker(cls=SlowStop))
    await h.evictable()
    takeover = asyncio.create_task(h.start(B))
    await asyncio.sleep(0.1)
    await h.slot.client_gone("c2")
    with pytest.raises(ClientGone):
        await takeover
    assert h.log[-2:] == [("c1", "evicted"), ("slot", "idle", "")]
    assert not capture.is_open
    assert h.on("fake") == []
    await capture.stop()


async def test_new_client_cancelled_during_the_takeover() -> None:
    capture = SpyCapture(keep_open=False)
    h = Harness(capture, evictable_after_s=0.1)
    await h.start(A, maker=h.maker(cls=SlowStop))
    await h.evictable()
    takeover = asyncio.create_task(h.start(B))
    await asyncio.sleep(0.1)
    takeover.cancel()
    with pytest.raises(asyncio.CancelledError):
        await takeover
    await wait_until(lambda: h.slot.state.state == "idle", 2.0)
    assert h.log[-2:] == [("c1", "evicted"), ("slot", "idle", "")]
    assert not capture.is_open
    await capture.stop()


async def test_holder_gone_cancels_its_session(silent: SpyCapture) -> None:
    h = Harness(silent)
    await h.start(A)
    await h.slot.client_gone("c1")
    assert h.log[-2:] == [("c1", "client_gone"), ("slot", "idle", "")]
    await h.slot.client_gone("c1")


async def test_client_gone_while_its_session_opens(silent: SpyCapture) -> None:
    h = Harness(silent)
    start = asyncio.create_task(h.start(A, maker=h.maker(FakeBackend(open_delay_s=10.0))))
    await asyncio.sleep(0.05)
    async with asyncio.timeout(1.0):  # not after the open's 10 s
        await h.slot.client_gone("c1")
        with pytest.raises(ClientGone):
            await start
    assert h.slot.state.state == "idle"
    assert h.log == []
    await h.start(B)
    assert h.log == [("slot", "busy", "c2"), ("c2", "started")]
    session = h.slot.session_of("c2")
    assert session is not None
    await session.stop()


async def test_client_gone_while_its_takeover_opens() -> None:
    capture = SpyCapture(keep_open=False)
    h = Harness(capture, evictable_after_s=0.1)
    await h.start(A)
    await h.evictable()
    takeover = asyncio.create_task(h.start(B, maker=h.maker(FakeBackend(open_delay_s=10.0))))
    await wait_until(lambda: h.slot.session_of("c1") is None and h.slot.state.state == "busy")
    async with asyncio.timeout(1.0):
        await h.slot.client_gone("c2")
        with pytest.raises(ClientGone):
            await takeover
    assert h.log[-2:] == [("c1", "evicted"), ("slot", "idle", "")]
    assert not capture.is_open
    await capture.stop()


async def test_client_gone_just_as_its_session_started(silent: SpyCapture) -> None:
    h = Harness(silent)
    h.gone_at_start.add("c1")
    await h.start(A)
    await wait_until(lambda: h.slot.state.state == "idle")
    assert h.log == [
        ("slot", "busy", "c1"),
        ("c1", "started"),
        ("c1", "client_gone"),
        ("slot", "idle", ""),
    ]


async def test_a_full_backend_refuses_the_start_and_changes_nothing(silent: SpyCapture) -> None:
    h = Harness(silent, capacities={"fake": 1})
    h.occupy()
    with pytest.raises(StartRefused) as busy:
        await h.start(A)
    assert busy.value.code == "busy"
    wire = busy.value.to_wire()
    assert wire["full_slot"] == "backend"
    assert wire["backend"] == "fake"
    assert wire["holder"] == {"pid": 77, "client_label": "Log", "kind": "mic"}
    assert h.changes == []
    assert h.slot.state.state == "idle"
    assert h.on("fake") == ["c3"]


async def test_capacity_two_takes_a_second_session(silent: SpyCapture) -> None:
    h = Harness(silent)
    h.occupy()
    await h.start(A)
    assert h.on("fake") == ["c3", "c1"]
    session = h.slot.session_of("c1")
    assert session is not None
    await session.stop()
    assert h.on("fake") == ["c3"]


async def test_eviction_frees_the_holders_backend_slot_as_its_session_ends(
    silent: SpyCapture,
) -> None:
    h = Harness(silent, evictable_after_s=0.1, capacities={"fake": 1})
    await h.start(A)
    await h.evictable()
    await h.start(B)
    assert h.backends_at_end["c1"] == ["c1", "c2"]
    assert h.on("fake") == ["c2"]
    assert h.changes[-1].to_wire()["backends"] == h.slot.state.to_wire()["backends"]
    session = h.slot.session_of("c2")
    assert session is not None
    await session.stop()


async def test_a_full_backend_elsewhere_leaves_the_evictable_holder_alone(
    silent: SpyCapture,
) -> None:
    h = Harness(silent, evictable_after_s=0.1, capacities={"fake": 1, "other": 1})
    h.occupy("other")
    await h.start(A)
    await h.evictable()
    with pytest.raises(StartRefused) as busy:
        await h.start(B, maker=h.maker(OtherFake()))
    assert busy.value.to_wire()["backend"] == "other"
    session = h.slot.session_of("c1")
    assert session is not None
    assert session.state == "running"
    assert h.slot.state.state == "evictable"
    assert h.on("fake") == ["c1"]
    await session.stop()


async def test_client_gone_and_failed_starts_free_the_backend_slot(silent: SpyCapture) -> None:
    h = Harness(silent, capacities={"fake": 1})
    await h.start(A)
    await h.slot.client_gone("c1")
    assert h.on("fake") == []
    backend = FakeBackend(open_failures=[OpenFailed("auth", "bad key", retryable=False)])
    with pytest.raises(OpenFailed):
        await h.start(B, maker=h.maker(backend))
    assert h.on("fake") == []
    await h.start(C)
    assert h.on("fake") == ["c3"]
    session = h.slot.session_of("c3")
    assert session is not None
    await session.stop()


async def test_a_capacity_change_sends_one_slot_event(silent: SpyCapture) -> None:
    h = Harness(silent)
    h.capacities = {"fake": 1}
    h.slot.capacities_changed()
    h.slot.capacities_changed()
    [change] = h.changes
    assert change.to_wire() == {
        "state": "idle",
        "backends": {"fake": {"max_sessions": 1, "sessions": []}},
    }


async def test_a_backend_gone_on_reload_keeps_its_sessions_until_they_end(
    silent: SpyCapture,
) -> None:
    h = Harness(silent, capacities={"fake": 1})
    await h.start(A)
    h.capacities = {}
    h.slot.capacities_changed()
    [usage] = h.slot.state.backends
    assert (usage.name, usage.max_sessions) == ("fake", 1)
    assert h.on("fake") == ["c1"]
    session = h.slot.session_of("c1")
    assert session is not None
    await session.stop()
    assert h.slot.state.to_wire() == {"state": "idle", "backends": {}}
