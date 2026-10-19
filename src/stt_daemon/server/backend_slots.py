"""Backend slots: how many sessions each backend may run at once, and which sessions hold them."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..config import DEFAULT_MAX_SESSIONS
from ..session.events import SessionKind


@dataclass(frozen=True, eq=False)
class BackendSession:
    """One session's place on its backend, from its start until its `session_ended`."""

    session_id: str
    kind: SessionKind
    backend: str
    client_id: str
    pid: int | None
    client_label: str | None

    def to_wire(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "kind": self.kind,
            "pid": self.pid,
            "client_label": self.client_label,
        }


@dataclass(frozen=True)
class BackendUsage:
    name: str
    max_sessions: int
    sessions: tuple[BackendSession, ...]


class BackendSlots:
    def __init__(self, capacities: Callable[[], Mapping[str, int]]) -> None:
        """`capacities`: the backends set up now and their `max_sessions`, read at each check."""
        self._capacities = capacities
        self._last_known: dict[str, int] = {}
        self._sessions: list[BackendSession] = []

    def usage(self) -> tuple[BackendUsage, ...]:
        """Every backend set up, then any no longer set up that still has sessions."""
        capacities = self._current()
        for held in self._sessions:
            if held.backend not in capacities:
                capacities[held.backend] = self._last_known.get(held.backend, DEFAULT_MAX_SESSIONS)
        return tuple(
            BackendUsage(name, cap, tuple(s for s in self._sessions if s.backend == name))
            for name, cap in capacities.items()
        )

    def full(self, backend: str, *, leaving: BackendSession | None = None) -> BackendSession | None:
        """The oldest holder when `backend` has no room, not counting `leaving`; else None."""
        holders = [s for s in self._sessions if s.backend == backend and s is not leaving]
        capacity = self._current().get(backend, DEFAULT_MAX_SESSIONS)
        return holders[0] if len(holders) >= capacity else None

    def take(self, session: BackendSession) -> None:
        self._sessions.append(session)

    def release(self, session: BackendSession) -> None:
        if session in self._sessions:
            self._sessions.remove(session)

    def _current(self) -> dict[str, int]:
        capacities = dict(self._capacities())
        self._last_known.update(capacities)
        return capacities


def usage_to_wire(usage: tuple[BackendUsage, ...]) -> dict[str, object]:
    return {
        u.name: {"max_sessions": u.max_sessions, "sessions": [s.to_wire() for s in u.sessions]}
        for u in usage
    }
