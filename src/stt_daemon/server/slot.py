"""The mic slot and the backend slots: who may start a session, takeover of an evictable one,
slot changes.

A `mic` session takes the mic slot and one slot on its backend; a `transcribe` session takes
only the backend slot and is never evictable. The slot sees every event of the
sessions it starts (it wraps their `emit`), so `slot` changes go out in protocol order: the new
holder's `slot` just before its `session_started`, and `idle` just after the last `session_ended`.
"""

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

from ..audio.capture import CaptureSource
from ..audio.frames import Attachment, epoch_ms
from ..audio.vad import SpeechStarted, VadEvent
from ..session.events import EvictedBy, SessionEnded, SessionEvent, SessionStarted
from ..session.session import Session, start_point
from .backend_slots import BackendSession, BackendSlots, BackendUsage, usage_to_wire
from .peer import Peer

type StateName = Literal["idle", "busy", "evictable"]
type Emit = Callable[[SessionEvent], None]


@dataclass(frozen=True)
class Client:
    client_id: str
    label: str | None = None
    peer: Peer = field(default_factory=Peer)


@dataclass(frozen=True)
class Holder:
    client_id: str
    pid: int | None
    cmdline: tuple[str, ...] | None
    client_label: str | None
    session_label: str | None
    session_id: str
    since: int | None
    """Epoch ms of the session's first sample; None while the session is still starting."""
    last_speech_at: int | None


@dataclass(frozen=True)
class SlotState:
    """The mic slot's `state` and `holder`, and every backend's slots."""

    state: StateName
    holder: Holder | None = None
    backends: tuple[BackendUsage, ...] = ()

    def to_wire(self) -> dict[str, object]:
        data: dict[str, object] = {"state": self.state}
        if self.holder is not None:
            holder = asdict(self.holder)
            if self.holder.cmdline is not None:
                holder["cmdline"] = list(self.holder.cmdline)
            data["holder"] = holder
        data["backends"] = usage_to_wire(self.backends)
        return data


class StartRefused(Exception):
    """`busy` or `session_active`; `slot` is the slot at the time. A `busy` names the slot that
    is full (`full_slot`), its backend for a backend slot, and the session in the way."""

    def __init__(
        self,
        code: Literal["busy", "session_active"],
        message: str,
        slot: SlotState,
        *,
        full_slot: Literal["mic", "backend"] | None = None,
        backend: str | None = None,
        blocker: BackendSession | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.slot = slot
        self.full_slot = full_slot
        self.backend = backend
        self.blocker = blocker

    def to_wire(self) -> dict[str, object]:
        """The `error` data next to `code` and `message`."""
        data: dict[str, object] = {"slot": self.slot.to_wire()}
        if self.full_slot is not None:
            data["full_slot"] = self.full_slot
        if self.backend is not None:
            data["backend"] = self.backend
        if self.blocker is not None:
            b = self.blocker
            data["holder"] = {"pid": b.pid, "client_label": b.client_label, "kind": b.kind}
        return data


class ClientGone(Exception):
    """The starting client disconnected before its session started."""


@dataclass(eq=False)
class _Claim:
    client: Client
    session_label: str | None
    session: Session | None = None
    started_at: int | None = None
    started_mono: float = 0.0
    evictable: bool = False
    gone: bool = False
    evicted_by: EvictedBy | None = None
    watch: asyncio.Task[None] | None = None
    opening: asyncio.Future[SessionStarted] | None = None
    place: BackendSession | None = None
    """Its slot on the session's backend."""
    mic: bool = True
    """Holds (or is taking) the mic slot."""


class Slot:
    def __init__(
        self,
        *,
        capture: CaptureSource,
        pre_start_ms: int,
        evictable_after_s: float,
        on_change: Callable[[SlotState], None],
        on_event: Callable[[Client, SessionEvent], None],
        capacities: Callable[[], Mapping[str, int]],
    ) -> None:
        """`capacities`: the backends set up now and their `max_sessions`."""
        self._capture = capture
        self._pre_start_ms = pre_start_ms
        self._evictable_after_s = evictable_after_s
        self._on_change = on_change
        self._on_event = on_event
        self._holder: _Claim | None = None
        self._incoming: _Claim | None = None
        """A takeover's new client while the holder stops."""
        self._previous_end: float | None = None
        self._inputs: dict[str, _Claim] = {}
        """Sessions on a backend slot only (`transcribe`), by client id."""
        self.backends = BackendSlots(capacities)
        self._announced = _change_key(SlotState("idle", backends=self.backends.usage()))
        self._speaking = False
        self._last_speech: float | None = None

    @property
    def state(self) -> SlotState:
        claim = self._incoming or self._holder
        backends = self.backends.usage()
        if claim is None:
            return SlotState("idle", backends=backends)
        state: StateName = "evictable" if claim.evictable else "busy"
        return SlotState(state, self._holder_info(claim), backends)

    def capacities_changed(self) -> None:
        """After a reload: backends or their `max_sessions` may differ."""
        self._announce()

    def heard(self, event: VadEvent) -> None:
        """A daemon-wide VAD event; feeds `last_speech_at` and the evictable timer."""
        self._speaking = isinstance(event, SpeechStarted)
        self._last_speech = event.at

    @property
    def last_speech_at(self) -> float | None:
        """Monotonic time speech was last heard; now while speech goes on."""
        return time.monotonic() if self._speaking else self._last_speech

    def session_of(self, client_id: str) -> Session | None:
        """The client's started session: the mic holder's, or one on a backend slot only."""
        claim = self._holder
        if claim is None or claim.client.client_id != client_id:
            claim = self._inputs.get(client_id)
        if claim is None or claim.started_at is None:
            return None
        return claim.session

    async def start(
        self,
        client: Client,
        *,
        make_session: Callable[[Emit], Session],
        session_label: str | None = None,
        since: float | None = None,
        requested_at: float | None = None,
    ) -> SessionStarted:
        """Start a session for `client`, taking an evictable slot over.

        `make_session` checks the request and builds the session with the given `emit`; what it
        raises fails the start with nothing changed. `since` (monotonic) overrides the command's
        start point, `pre_start_ms` before `requested_at` (default now). Raises `StartRefused`,
        `ClientGone`, or what `Session.start` raises.
        """
        requested_at = time.monotonic() if requested_at is None else requested_at
        self._check_evictable()
        if self._claim_of(client.client_id) is not None:
            raise StartRefused("session_active", "this client already holds a session", self.state)
        holder = self._holder
        if blocking := self._incoming or (holder if holder and not holder.evictable else None):
            raise StartRefused(
                "busy",
                "another client holds the mic slot",
                self.state,
                full_slot="mic",
                blocker=blocking.place,
            )
        claim = _Claim(client, session_label)
        claim.session = make_session(self._forward(claim))
        self._take_backend(claim, leaving=holder.place if holder else None)
        if since is None:
            since = start_point(requested_at, self._pre_start_ms, self._previous_end)
        if holder is None:
            self._holder = claim
            try:
                return await self._open(claim, since=since)
            except BaseException:
                self._drop(claim)
                raise
        return await self._take_over(holder, claim, since)

    async def start_input(
        self,
        client: Client,
        *,
        make_session: Callable[[Emit], Session],
        session_label: str | None = None,
    ) -> SessionStarted:
        """Start a session that takes a backend slot only. Raises `StartRefused`, `ClientGone`,
        or what `make_session` and `Session.start` raise."""
        if self._claim_of(client.client_id) is not None:
            raise StartRefused("session_active", "this client already holds a session", self.state)
        claim = _Claim(client, session_label, mic=False)
        claim.session = make_session(self._forward(claim))
        self._take_backend(claim, leaving=None)
        self._inputs[client.client_id] = claim
        try:
            return await self._open(claim)
        except BaseException:
            self._drop(claim)
            raise

    def _take_backend(self, claim: _Claim, *, leaving: BackendSession | None) -> None:
        session = claim.session
        assert session is not None
        backend = session.backend_name
        blocker = self.backends.full(backend, leaving=leaving)
        if blocker is not None:
            raise StartRefused(
                "busy",
                f"{backend} already runs as many sessions as its max_sessions allows",
                self.state,
                full_slot="backend",
                backend=backend,
                blocker=blocker,
            )
        client = claim.client
        claim.place = BackendSession(
            session.session_id,
            session.kind,
            backend,
            client.client_id,
            client.peer.pid,
            client.label,
        )
        self.backends.take(claim.place)

    async def client_gone(self, client_id: str) -> None:
        """The client disconnected: its running session is cancelled (`client_gone`)."""
        claim = self._claim_of(client_id)
        if claim is None:
            return
        claim.gone = True
        if claim.opening is not None and not claim.opening.done():
            claim.opening.cancel()
            await asyncio.wait([claim.opening])
        elif claim.session is not None and (
            claim.session.state == "running"
            or (claim.session.state == "stopping" and claim.session.kind != "mic")
        ):
            await claim.session.cancel("client_gone")

    async def shutdown(self) -> None:
        """Daemon exit, after pending starts are cancelled: the mic session stops as
        `stop_session` would; sessions on a backend slot only are cancelled (input unfinished)."""
        inputs = [c.session for c in self._inputs.values() if c.session is not None]
        await asyncio.gather(*(s.cancel() for s in inputs if s.state != "new"))
        claim = self._holder
        if claim is None or claim.session is None or claim.session.state == "new":
            return
        if claim.session.state == "running":
            await claim.session.stop()
        else:
            await claim.session.wait_ended()

    async def _open(
        self, claim: _Claim, *, since: float | None = None, attachment: Attachment | None = None
    ) -> SessionStarted:
        """`Session.start` in a task `client_gone` can cancel, so a gone client's start ends now."""
        assert claim.session is not None
        claim.opening = asyncio.ensure_future(
            claim.session.start(since=since, attachment=attachment)
        )
        try:
            return await claim.opening
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if claim.gone and (caller is None or not caller.cancelling()):
                raise ClientGone("the client disconnected while its session started") from None
            raise
        finally:
            claim.opening = None

    async def _take_over(self, holder: _Claim, claim: _Claim, since: float) -> SessionStarted:
        assert holder.session is not None
        assert claim.session is not None
        self._incoming = claim
        try:
            await self._capture.hold()
            try:
                # Copied now, so it can't leave the ring during the stop; live audio follows.
                attachment = self._capture.attach(since)
            except BaseException:
                await self._capture.release()
                raise
        except BaseException:
            self._drop(claim)
            raise
        handed = False
        try:
            client = claim.client
            holder.evicted_by = EvictedBy(client.client_id, client.peer.pid, client.label)
            await holder.session.stop("evicted")
            self._ended(holder)
            if claim.gone:
                raise ClientGone("the client disconnected before its session started")
            self._incoming, self._holder = None, claim
            handed = True
            return await self._open(claim, attachment=attachment)
        except BaseException:
            if not handed:
                await self._drop_attachment(attachment)
            self._drop(claim)
            raise

    async def _drop_attachment(self, attachment: Attachment) -> None:
        attachment.detach()
        await self._capture.release()

    def _forward(self, claim: _Claim) -> Emit:
        def emit(event: SessionEvent) -> None:
            if isinstance(event, SessionStarted):
                claim.started_at = event.started_at
                claim.started_mono = time.monotonic()
                if claim.mic:
                    claim.watch = asyncio.create_task(self._watch(claim))
                self._announce()
            elif isinstance(event, SessionEnded):
                if event.reason == "evicted" and claim.evicted_by is not None:
                    event = replace(event, evicted_by=claim.evicted_by)
                self._on_event(claim.client, event)
                self._ended(claim)
                return
            self._on_event(claim.client, event)

        return emit

    def _ended(self, claim: _Claim) -> None:
        if claim.watch is not None:
            claim.watch.cancel()
        self._release(claim)
        if self._inputs.get(claim.client.client_id) is claim:
            del self._inputs[claim.client.client_id]
        if self._holder is claim:
            assert claim.session is not None
            self._holder = None
            self._previous_end = claim.session.audio_end
        self._announce()

    def _drop(self, claim: _Claim) -> None:
        """A start that failed or was abandoned: the slot goes back to what is running."""
        if self._incoming is claim:
            self._incoming = None
        if self._holder is claim:
            self._holder = None
        if self._inputs.get(claim.client.client_id) is claim:
            del self._inputs[claim.client.client_id]
        self._release(claim)
        self._announce()

    def _release(self, claim: _Claim) -> None:
        if claim.place is not None:
            self.backends.release(claim.place)

    def _announce(self) -> None:
        holder = self._holder
        if holder is None and self._incoming is not None:
            return  # mid-takeover: the new holder announces itself when its session starts
        if holder is not None and holder.started_at is None:
            return
        state = self.state
        key = _change_key(state)
        if key != self._announced:
            self._announced = key
            self._on_change(state)

    async def _watch(self, claim: _Claim) -> None:
        while (wait := self._evictable_in(claim)) > 0:
            await asyncio.sleep(wait)
        claim.evictable = True
        self._announce()

    def _check_evictable(self) -> None:
        claim = self._holder
        if claim is None or claim.started_at is None or claim.evictable:
            return
        if self._evictable_in(claim) <= 0:
            claim.evictable = True
            self._announce()

    def _evictable_in(self, claim: _Claim) -> float:
        last = self.last_speech_at
        quiet_since = claim.started_mono if last is None else max(last, claim.started_mono)
        return quiet_since + self._evictable_after_s - time.monotonic()

    def _claim_of(self, client_id: str) -> _Claim | None:
        for claim in (self._incoming, self._holder):
            if claim is not None and claim.client.client_id == client_id:
                return claim
        return self._inputs.get(client_id)

    def _holder_info(self, claim: _Claim) -> Holder:
        assert claim.session is not None
        last = self.last_speech_at
        if last is not None and last < claim.started_mono:
            last = None
        return Holder(
            client_id=claim.client.client_id,
            pid=claim.client.peer.pid,
            cmdline=claim.client.peer.cmdline,
            client_label=claim.client.label,
            session_label=claim.session_label,
            session_id=claim.session.session_id,
            since=claim.started_at,
            last_speech_at=None if last is None else epoch_ms(last),
        )


def _change_key(state: SlotState) -> object:
    """What a `slot` event is sent for: not `last_speech_at`, which moves on its own."""
    held = state.holder
    backends = tuple(
        (u.name, u.max_sessions, tuple(s.session_id for s in u.sessions)) for u in state.backends
    )
    return (state.state, held and held.client_id, held and held.session_id, backends)
