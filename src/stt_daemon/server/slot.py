"""The one session slot: who may start a session, takeover of an evictable one, slot changes.

The slot sees every event of the sessions it starts (it wraps their `emit`), so `slot` changes go
out in protocol order: the new holder's `slot` just before its `session_started`, and `idle`
just after the last `session_ended`.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

from ..audio.capture import CaptureSource
from ..audio.frames import Attachment
from ..session.events import EvictedBy, SessionEnded, SessionEvent, SessionStarted
from ..session.session import Session, start_point
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
    state: StateName
    holder: Holder | None = None

    def to_wire(self) -> dict[str, object]:
        if self.holder is None:
            return {"state": self.state}
        holder = asdict(self.holder)
        if self.holder.cmdline is not None:
            holder["cmdline"] = list(self.holder.cmdline)
        return {"state": self.state, "holder": holder}


class StartRefused(Exception):
    """`busy` or `session_active`; `slot` is the slot at the time."""

    def __init__(self, code: Literal["busy", "session_active"], message: str, slot: SlotState):
        super().__init__(message)
        self.code = code
        self.message = message
        self.slot = slot


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


class Slot:
    def __init__(
        self,
        *,
        capture: CaptureSource,
        pre_start_ms: int,
        evictable_after_s: float,
        on_change: Callable[[SlotState], None],
        on_event: Callable[[Client, SessionEvent], None],
    ) -> None:
        self._capture = capture
        self._pre_start_ms = pre_start_ms
        self._evictable_after_s = evictable_after_s
        self._on_change = on_change
        self._on_event = on_event
        self._holder: _Claim | None = None
        self._incoming: _Claim | None = None
        """A takeover's new client while the holder stops."""
        self._previous_end: float | None = None
        self._announced: tuple[StateName, str | None, str | None] = ("idle", None, None)
        self._wall_offset = time.time() - time.monotonic()

    @property
    def state(self) -> SlotState:
        claim = self._incoming or self._holder
        if claim is None:
            return SlotState("idle")
        return SlotState("evictable" if claim.evictable else "busy", self._holder_info(claim))

    def session_of(self, client_id: str) -> Session | None:
        """The client's started session, if it holds the slot."""
        claim = self._holder
        if claim is None or claim.client.client_id != client_id or claim.started_at is None:
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
        if self._incoming is not None or (holder is not None and not holder.evictable):
            raise StartRefused("busy", "another client holds the session slot", self.state)
        claim = _Claim(client, session_label)
        claim.session = make_session(self._forward(claim))
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

    async def client_gone(self, client_id: str) -> None:
        """The client disconnected: its running session is cancelled (`client_gone`)."""
        claim = self._claim_of(client_id)
        if claim is None:
            return
        claim.gone = True
        if claim.opening is not None and not claim.opening.done():
            claim.opening.cancel()
            await asyncio.wait([claim.opening])
        elif claim.session is not None and claim.session.state == "running":
            await claim.session.cancel("client_gone")

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
        self._announce()

    def _announce(self) -> None:
        holder = self._holder
        if holder is None and self._incoming is not None:
            return  # mid-takeover: the new holder announces itself when its session starts
        if holder is not None and holder.started_at is None:
            return
        state = self.state
        held = state.holder
        key = (state.state, held and held.client_id, held and held.session_id)
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
        assert claim.session is not None
        last = claim.session.last_speech_at
        quiet_since = claim.started_mono if last is None else max(last, claim.started_mono)
        return quiet_since + self._evictable_after_s - time.monotonic()

    def _claim_of(self, client_id: str) -> _Claim | None:
        for claim in (self._incoming, self._holder):
            if claim is not None and claim.client.client_id == client_id:
                return claim
        return None

    def _holder_info(self, claim: _Claim) -> Holder:
        assert claim.session is not None
        last = claim.session.last_speech_at
        return Holder(
            client_id=claim.client.client_id,
            pid=claim.client.peer.pid,
            cmdline=claim.client.peer.cmdline,
            client_label=claim.client.label,
            session_label=claim.session_label,
            session_id=claim.session.session_id,
            since=claim.started_at,
            last_speech_at=None if last is None else round((last + self._wall_offset) * 1000),
        )
