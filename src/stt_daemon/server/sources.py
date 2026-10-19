"""Applications playing audio, from a `pw-dump -m -N` subprocess: sources, their ids, events.

`pw-dump -m` prints the whole registry as one JSON array, then one array per change. A removed
object comes as `{"id": n, "info": null}`; registry ids are reused, `object.serial` is not.
"""

import asyncio
import codecs
import contextlib
import itertools
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

log = logging.getLogger(__name__)

PW_DUMP_COMMAND = ("pw-dump", "-m", "-N")
RETRY_S = 30.0
SOURCE_CLASS = "Stream/Output/Audio"
_READ_BYTES = 65536


@dataclass(frozen=True)
class Source:
    id: str
    serial: str
    name: str
    binary: str | None
    media_name: str | None

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "binary": self.binary,
            "media_name": self.media_name,
        }


@dataclass(frozen=True)
class SourceAdded:
    source: Source


@dataclass(frozen=True)
class SourceRemoved:
    source: Source


type SourceEvent = SourceAdded | SourceRemoved


class ArraySplitter:
    """Cuts a stream of text into its top-level JSON arrays: bracket depth outside strings."""

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._depth = 0
        self._in_string = False
        self._escaped = False

    def feed(self, text: str) -> list[str]:
        done: list[str] = []
        start = 0
        for i, char in enumerate(text):
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._in_string = False
            elif char == '"':
                self._in_string = True
            elif char in "[{":
                if self._depth == 0:
                    start = i
                    self._buffer = []
                self._depth += 1
            elif char in "]}" and self._depth > 0:
                self._depth -= 1
                if self._depth == 0:
                    self._buffer.append(text[start : i + 1])
                    done.append("".join(self._buffer))
                    self._buffer = []
        if self._depth > 0:
            self._buffer.append(text[start:])
        return done


class SourceTable:
    """Registry objects → sources; `update` applies one printed array (the first is the whole
    registry) and returns the events. A source keeps its id while its serial lives, and ids are
    never reused."""

    def __init__(self) -> None:
        self._nodes: dict[int, Source] = {}
        """Registry id → source."""
        self._binaries: dict[int, str | None] = {}
        """Client registry id → its process binary."""
        self._named: dict[int, tuple[str, str]] = {}
        """Registry id → (node.name, serial), for every node."""
        self._ids = itertools.count(1)

    def sources(self) -> list[Source]:
        return list(self._nodes.values())

    def node_serial(self, name: str) -> str | None:
        return next((serial for n, serial in self._named.values() if n == name), None)

    def clear(self) -> list[SourceEvent]:
        events: list[SourceEvent] = [SourceRemoved(s) for s in self._nodes.values()]
        self._nodes, self._binaries, self._named = {}, {}, {}
        return events

    def update(self, objects: Sequence[object]) -> list[SourceEvent]:
        items = [cast(dict[str, Any], o) for o in objects if isinstance(o, dict)]
        events: list[SourceEvent] = []
        # Clients first: a node names its client by id, and both can come in one array.
        for item in items:
            if item.get("type") == "PipeWire:Interface:Client" and isinstance(item.get("id"), int):
                props = _props(item)
                if props is not None:
                    self._binaries[item["id"]] = _text(props, "application.process.binary")
        for item in items:
            key = item.get("id")
            if not isinstance(key, int):
                continue
            if "info" in item and item["info"] is None:
                self._binaries.pop(key, None)
                self._named.pop(key, None)
                if (gone := self._nodes.pop(key, None)) is not None:
                    events.append(SourceRemoved(gone))
                continue
            if item.get("type") != "PipeWire:Interface:Node":
                continue
            props = _props(item)
            if props is not None:
                name, serial = _text(props, "node.name"), props.get("object.serial")
                if name is not None and isinstance(serial, int | str):
                    self._named[key] = (name, str(serial))
            events += self._node(key, props)
        return events

    def _node(self, key: int, props: Mapping[str, Any] | None) -> list[SourceEvent]:
        if props is None:
            return []  # an update without props changes nothing we track
        old = self._nodes.get(key)
        serial = props.get("object.serial")
        if props.get("media.class") != SOURCE_CLASS or not isinstance(serial, int | str):
            if old is None:
                return []
            del self._nodes[key]
            return [SourceRemoved(old)]
        serial = str(serial)
        if old is not None and old.serial == serial:
            return []
        client = props.get("client.id")
        binary = _text(props, "application.process.binary") or (
            self._binaries.get(client) if isinstance(client, int) else None
        )
        source = Source(
            id=f"src_{next(self._ids)}",
            serial=serial,
            name=_text(props, "application.name") or _text(props, "node.name") or serial,
            binary=binary,
            media_name=_text(props, "media.name"),
        )
        self._nodes[key] = source
        events: list[SourceEvent] = [] if old is None else [SourceRemoved(old)]
        return [*events, SourceAdded(source)]


def _props(item: Mapping[str, Any]) -> Mapping[str, Any] | None:
    info = item.get("info")
    if not isinstance(info, dict):
        return None
    props = cast(dict[str, Any], info).get("props")
    return cast(dict[str, Any], props) if isinstance(props, dict) else None


def _text(props: Mapping[str, Any], key: str) -> str | None:
    value = props.get(key)
    return value if isinstance(value, str) and value else None


class SourceScan:
    """Runs `command` for the daemon's lifetime and keeps the table; restarts it `retry_s` after
    it exits or can't start (the sources are empty meanwhile). Off Linux it never runs."""

    def __init__(
        self,
        command: Sequence[str] = PW_DUMP_COMMAND,
        *,
        enabled: bool = sys.platform == "linux",
        retry_s: float = RETRY_S,
    ) -> None:
        self.command = list(command)
        self.on_event: Callable[[SourceEvent], None] | None = None
        self._enabled = enabled
        self._retry_s = retry_s
        self._table = SourceTable()
        self._waiters: list[tuple[Callable[[Source], bool], asyncio.Future[Source]]] = []
        self._node_waiters: list[tuple[str, asyncio.Future[str]]] = []
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._enabled:
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def sources(self) -> list[Source]:
        return self._table.sources()

    def get(self, source_id: str) -> Source | None:
        return next((s for s in self.sources() if s.id == source_id), None)

    def find(self, match: Callable[[Source], bool]) -> Source | None:
        return next((s for s in self.sources() if match(s)), None)

    async def wait_for(self, match: Callable[[Source], bool]) -> Source:
        """The first source that matches, now or once it appears."""
        if (found := self.find(match)) is not None:
            return found
        future: asyncio.Future[Source] = asyncio.get_running_loop().create_future()
        waiter = (match, future)
        self._waiters.append(waiter)
        try:
            return await future
        finally:
            self._waiters.remove(waiter)

    def follow(self, source: Source) -> "Follow":
        return Follow(self, source)

    def node_serial(self, name: str) -> str | None:
        """The serial of the node with this `node.name`, of any media class."""
        return self._table.node_serial(name)

    async def wait_node(self, name: str) -> str:
        """The serial of the node with this `node.name`, now or once it appears."""
        if (serial := self.node_serial(name)) is not None:
            return serial
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        waiter = (name, future)
        self._node_waiters.append(waiter)
        try:
            return await future
        finally:
            self._node_waiters.remove(waiter)

    async def _run(self) -> None:
        failed_before = False
        while True:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *self.command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except OSError as exc:
                if not failed_before:
                    log.warning("source scan: %s could not be started: %s", self.command[0], exc)
                failed_before = True
            else:
                printed, code = await self._read(proc)
                if printed or not failed_before:
                    log.warning("source scan: %s exited with code %s", self.command[0], code)
                failed_before = True
                self._dispatch(self._table.clear())
            await asyncio.sleep(self._retry_s)

    async def _read(self, proc: asyncio.subprocess.Process) -> tuple[bool, int | None]:
        """Apply what `proc` prints until it exits; whether it printed any array, its code."""
        assert proc.stdout is not None
        splitter = ArraySplitter()
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        printed = False
        try:
            while data := await proc.stdout.read(_READ_BYTES):
                for text in splitter.feed(decoder.decode(data)):
                    try:
                        objects = json.loads(text)
                    except json.JSONDecodeError:
                        log.warning(
                            "source scan: %s printed text that is not JSON", self.command[0]
                        )
                        continue
                    if isinstance(objects, list):
                        printed = True
                        self._dispatch(self._table.update(cast(list[object], objects)))
        finally:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            while await proc.stdout.read(_READ_BYTES):
                pass  # `wait()` returns only once the pipe is closed
            await proc.wait()
        return printed, proc.returncode

    def _dispatch(self, events: list[SourceEvent]) -> None:
        for name, future in self._node_waiters:
            if not future.done() and (serial := self.node_serial(name)) is not None:
                future.set_result(serial)
        for event in events:
            if isinstance(event, SourceAdded):
                for match, future in self._waiters:
                    if not future.done() and match(event.source):
                        future.set_result(event.source)
            if self.on_event is not None:
                self.on_event(event)


class Follow:
    """One application's stream for a system track: its serial while it plays; once it is gone,
    the next stream of the same app (same binary, else the same name)."""

    def __init__(self, scan: SourceScan, source: Source) -> None:
        self._scan = scan
        self._serial = source.serial
        self._binary = source.binary
        self._name = source.name

    @property
    def serial(self) -> str:
        return self._serial

    def current(self) -> str | None:
        found = self._scan.find(lambda s: s.serial == self._serial) or self._scan.find(self._same)
        if found is None:
            return None
        self._serial = found.serial
        return found.serial

    async def appeared(self) -> str:
        found = await self._scan.wait_for(self._same)
        self._serial = found.serial
        return found.serial

    def _same(self, source: Source) -> bool:
        if self._binary is not None:
            return source.binary == self._binary
        return source.name == self._name
