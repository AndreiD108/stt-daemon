"""A Slot with recorded events and slot changes, sessions on the WAV capture, test clients."""

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from stt_daemon.audio.capture import CaptureSource
from stt_daemon.audio.frames import Attachment
from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.base import OpenOptions
from stt_daemon.backends.fake import FakeBackend
from stt_daemon.config import Config
from stt_daemon.server.backend_slots import BackendSession
from stt_daemon.server.peer import Peer
from stt_daemon.server.slot import Emit, Slot
from stt_daemon.server.slot_state import Client, SlotState
from stt_daemon.session.events import SessionEnded, SessionEvent, SessionStarted
from stt_daemon.session.options import SessionOptions
from stt_daemon.session.session import Session
from unit.waiting import wait_until

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
