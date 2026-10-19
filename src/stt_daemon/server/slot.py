"""The mic slot and the backend slots: who may start a session, takeover of an evictable one,
slot changes.

A `mic` session takes the mic slot and one slot on its backend; a `transcribe` session takes
only the backend slot and is never evictable. A dictation (`mic`) that finds its backend full
suspends the lowest-ranked other kind there; suspended sessions resume, longest suspended first,
as slots free up. The slot sees every event of the
sessions it starts (it wraps their `emit`), so `slot` changes go out in protocol order: the new
holder's `slot` just before its `session_started`, and `idle` just after the last `session_ended`.
"""

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from ..audio.capture import CaptureSource
from ..audio.frames import Attachment, epoch_ms
from ..audio.vad import SpeechStarted, VadEvent
from ..session.events import (
    KIND_RANK,
    EvictedBy,
    PreemptedBy,
    SessionEnded,
    SessionEvent,
    SessionStarted,
)
from ..session.options import start_point
from ..session.session import Session
from .backend_slots import BackendSession, BackendSlots
from .slot_state import Client, Holder, SlotState, StartRefused, StateName

type Emit = Callable[[SessionEvent], None]


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
    lent_to: "_Claim | None" = None
    """Suspended because the provider refused that session a connection: resumes after it."""
    released: bool = False


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
        self._resuming: set[asyncio.Task[bool]] = set()
        self._closing = False
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
        self._resume_waiting()
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
        claim.session = session = make_session(self._forward(claim))
        session.make_room = lambda: self._make_room(claim)
        victim = self._take_backend(claim, leaving=holder.place if holder else None)
        if since is None:
            since = start_point(requested_at, self._pre_start_ms, self._previous_end)
        if holder is None:
            self._holder = claim
            try:
                return await self._open(claim, since=since, victim=victim)
            except BaseException:
                self._drop(claim)
                raise
        return await self._take_over(holder, claim, since, victim)

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

    def _take_backend(self, claim: _Claim, *, leaving: BackendSession | None) -> _Claim | None:
        """Take a slot on the session's backend; a dictation may set another session aside for
        it, which is returned, to be suspended before the dictation starts."""
        session = claim.session
        assert session is not None
        backend = session.backend_name
        blocker = self.backends.full(backend, leaving=leaving)
        victim = None
        if blocker is not None and session.kind == "mic":
            victim = self._victim(backend)
        if victim is not None:
            assert victim.place is not None
            self.backends.suspend(victim.place)
        elif blocker is not None:
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
        return victim

    def _victim(self, backend: str) -> _Claim | None:
        """The suspendable session a dictation on `backend` sets aside: the lowest ranked kind,
        then the most recently started."""
        holders = {id(place) for place in self.backends.holders(backend)}
        candidates = [
            c
            for c in self._inputs.values()
            if id(c.place) in holders
            and c.started_at is not None
            and c.session is not None
            and c.session.suspendable
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda c: (_rank(c), c.started_mono))

    async def _suspend(self, victim: _Claim, by: _Claim) -> None:
        """Its slot is already given up."""
        assert victim.session is not None
        client = by.client
        preempted_by = PreemptedBy("mic", client.peer.pid, client.label)
        # Shielded: a cancelled start still leaves the victim fully suspended, then resumed.
        if not await asyncio.shield(victim.session.suspend(preempted_by)):
            return  # it is ending; its session_ended frees its place
        self._announce()

    async def _make_room(self, claim: _Claim) -> bool:
        """The provider refused one more connection: suspend a session on the backend, if any,
        though the daemon's own count had room."""
        assert claim.session is not None
        victim = self._victim(claim.session.backend_name)
        if victim is None:
            return False
        assert victim.place is not None
        self.backends.suspend(victim.place)
        victim.lent_to = claim
        await self._suspend(victim, claim)
        return True

    def _resume_waiting(self) -> None:
        """Give free slots to suspended sessions, longest suspended first; fail those whose
        backend is no longer set up."""
        if self._closing:
            return
        for place in self.backends.waiting():
            claim = self._inputs.get(place.client_id)
            if claim is None or claim.place is not place or claim.session is None:
                continue
            if not self.backends.set_up(place.backend):
                claim.session.give_up(
                    "config", f"{place.backend} is no longer set up, so the session can't resume"
                )
            elif (claim.lent_to is None or claim.lent_to.released) and self.backends.full(
                place.backend
            ) is None:
                claim.lent_to = None
                self.backends.resume(place)
                task = asyncio.create_task(claim.session.resume())
                self._resuming.add(task)
                task.add_done_callback(self._resume_done)

    def _resume_done(self, task: asyncio.Task[bool]) -> None:
        self._resuming.discard(task)
        if not task.cancelled() and task.exception() is None and task.result():
            self._announce()

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
        self._closing = True
        for task in self._resuming:
            task.cancel()
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
        self,
        claim: _Claim,
        *,
        since: float | None = None,
        attachment: Attachment | None = None,
        victim: _Claim | None = None,
    ) -> SessionStarted:
        """`Session.start` in a task `client_gone` can cancel, so a gone client's start ends now.
        A `victim` is suspended first."""
        session = claim.session
        assert session is not None

        async def start() -> SessionStarted:
            if victim is not None:
                await self._suspend(victim, claim)
            return await session.start(since=since, attachment=attachment)

        claim.opening = asyncio.ensure_future(start())
        try:
            return await claim.opening
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if claim.gone and (caller is None or not caller.cancelling()):
                raise ClientGone("the client disconnected while its session started") from None
            raise
        finally:
            claim.opening = None

    async def _take_over(
        self, holder: _Claim, claim: _Claim, since: float, victim: _Claim | None
    ) -> SessionStarted:
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
            return await self._open(claim, attachment=attachment, victim=victim)
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
        claim.released = True
        if claim.place is not None:
            self.backends.release(claim.place)
            self._resume_waiting()

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


def _rank(claim: _Claim) -> int:
    assert claim.session is not None
    return KIND_RANK[claim.session.kind]


def _change_key(state: SlotState) -> object:
    """What a `slot` event is sent for: not `last_speech_at`, which moves on its own."""
    held = state.holder
    backends = tuple(
        (
            u.name,
            u.max_sessions,
            tuple(s.session_id for s in u.sessions),
            tuple(s.session_id for s in u.suspended),
        )
        for u in state.backends
    )
    return (state.state, held and held.client_id, held and held.session_id, backends)
