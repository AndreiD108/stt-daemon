"""`previous_text` by `context_key`: a session's last text, per client label, in memory."""

from collections import OrderedDict

_MAX_ENTRIES = 1000

type Scope = tuple[str, str]
"""("label", client label) for labelled clients; ("connection", client id) otherwise."""


def scope_of(client_id: str, label: str | None) -> Scope:
    return ("label", label) if label else ("connection", client_id)


class ContextStore:
    def __init__(self, max_entries: int = _MAX_ENTRIES) -> None:
        self._texts: OrderedDict[tuple[Scope, str], str] = OrderedDict()
        self._max_entries = max_entries

    def get(self, scope: Scope, key: str) -> str | None:
        return self._texts.get((scope, key))

    def put(self, scope: Scope, key: str, text: str) -> None:
        self._texts[(scope, key)] = text
        self._texts.move_to_end((scope, key))
        while len(self._texts) > self._max_entries:
            self._texts.popitem(last=False)

    def forget(self, scope: Scope) -> None:
        for entry in [entry for entry in self._texts if entry[0] == scope]:
            del self._texts[entry]
