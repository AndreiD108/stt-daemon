"""Backend slots: how many sessions each backend may run at once, and which sessions hold them.

A suspended session still belongs to its backend but holds no slot; it waits for one."""

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

    def to_wire(self, *, suspended: bool = False) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "kind": self.kind,
            "pid": self.pid,
            "client_label": self.client_label,
            "suspended": suspended,
        }


@dataclass(frozen=True)
class BackendUsage:
    name: str
    max_sessions: int
    sessions: tuple[BackendSession, ...]
    suspended: tuple[BackendSession, ...] = ()


class BackendSlots:
    def __init__(self, capacities: Callable[[], Mapping[str, int]]) -> None:
        """`capacities`: the backends set up now and their `max_sessions`, read at each check."""
        self._capacities = capacities
        self._last_known: dict[str, int] = {}
        self._sessions: list[BackendSession] = []
        self._suspended: list[BackendSession] = []
        """In the order they were suspended."""

    def usage(self) -> tuple[BackendUsage, ...]:
        """Every backend set up, then any no longer set up that still has sessions."""
        capacities = self._current()
        for held in self._sessions + self._suspended:
            if held.backend not in capacities:
                capacities[held.backend] = self._last_known.get(held.backend, DEFAULT_MAX_SESSIONS)
        return tuple(
            BackendUsage(
                name,
                cap,
                tuple(s for s in self._sessions if s.backend == name),
                tuple(s for s in self._suspended if s.backend == name),
            )
            for name, cap in capacities.items()
        )

    def holders(self, backend: str) -> list[BackendSession]:
        """Those holding a slot on `backend`, oldest first."""
        return [s for s in self._sessions if s.backend == backend]

    def set_up(self, backend: str) -> bool:
        return backend in self._current()

    def full(self, backend: str, *, leaving: BackendSession | None = None) -> BackendSession | None:
        """The oldest holder when `backend` has no room, not counting `leaving`; else None."""
        holders = [s for s in self._sessions if s.backend == backend and s is not leaving]
        capacity = self._current().get(backend, DEFAULT_MAX_SESSIONS)
        return holders[0] if len(holders) >= capacity else None

    def take(self, session: BackendSession) -> None:
        self._sessions.append(session)

    def release(self, session: BackendSession) -> None:
        for held in (self._sessions, self._suspended):
            if session in held:
                held.remove(session)

    def suspend(self, session: BackendSession) -> None:
        """`session` gives its slot up and waits for one."""
        self._sessions.remove(session)
        self._suspended.append(session)

    def waiting(self) -> list[BackendSession]:
        """Suspended sessions, longest suspended first."""
        return list(self._suspended)

    def resume(self, session: BackendSession) -> None:
        self._suspended.remove(session)
        self._sessions.append(session)

    def _current(self) -> dict[str, int]:
        capacities = dict(self._capacities())
        self._last_known.update(capacities)
        return capacities


def usage_to_wire(usage: tuple[BackendUsage, ...]) -> dict[str, object]:
    return {
        u.name: {
            "max_sessions": u.max_sessions,
            "sessions": [s.to_wire() for s in u.sessions]
            + [s.to_wire(suspended=True) for s in u.suspended],
        }
        for u in usage
    }
