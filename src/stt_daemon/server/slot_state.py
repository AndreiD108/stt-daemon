"""The slot's public shapes: clients, the mic slot's holder and state, start refusals."""

from dataclasses import asdict, dataclass, field
from typing import Literal

from .backend_slots import BackendSession, BackendUsage, usage_to_wire
from .peer import Peer

type StateName = Literal["idle", "busy", "evictable"]


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

    def change_key(self) -> object:
        """What a `slot` event is sent for: not `last_speech_at`, which moves on its own."""
        held = self.holder
        backends = tuple(
            (
                u.name,
                u.max_sessions,
                tuple(s.session_id for s in u.sessions),
                tuple(s.session_id for s in u.suspended),
            )
            for u in self.backends
        )
        return (self.state, held and held.client_id, held and held.session_id, backends)


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
