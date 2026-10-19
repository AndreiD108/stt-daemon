"""What a link's log holds: queued audio (packed to int16 while suspended) and commit points."""

import itertools
from dataclasses import dataclass

from ..audio.frames import PackedFrame
from ..backends.base import Reference
from .pause_gate import Outgoing


@dataclass(eq=False)
class Point:
    """A commit."""

    final: bool
    reference: Reference | None = None
    """Built when first sent, so a replay sends the same."""


@dataclass(frozen=True)
class Kept:
    """Audio queued while suspended."""

    frame: PackedFrame
    cut: bool


type Audio = Outgoing | Kept
type Item = Audio | Point


def size(item: Audio) -> int:
    return item.frame.pcm.size if isinstance(item, Kept) else item.frame.samples.size


def unpacked(items: list[Audio]) -> list[Outgoing]:
    return [i if isinstance(i, Outgoing) else Outgoing(i.frame.unpack(), i.cut) for i in items]


class LinkLog:
    """Audio and commits no connection has answered yet, in order; the current connection got
    the first `sent` items."""

    def __init__(self, sample_rate: int) -> None:
        self.items: list[Item] = []
        self.samples = 0
        self.sent = 0
        self.acked_samples = 0
        """Samples of the current connection answered and dropped."""
        self._rate = sample_rate

    def __bool__(self) -> bool:
        return bool(self.items)

    def add(self, out: Outgoing, *, pack: bool) -> bool:
        """False for an empty frame, which is not kept."""
        if not (n := out.frame.samples.size):
            return False
        self.items.append(Kept(out.frame.pack(), out.cut) if pack else out)
        self.samples += n
        return True

    def commit(self, final: bool) -> bool:
        """True when a new point was added; after a point, `final` marks that one instead."""
        if not self.items:
            return False
        last = self.items[-1]
        if isinstance(last, Point):
            last.final = last.final or final
            return False
        self.items.append(Point(final))
        return True

    @property
    def points(self) -> bool:
        return any(isinstance(item, Point) for item in self.items)

    def last_point(self) -> Point | None:
        return next((item for item in reversed(self.items) if isinstance(item, Point)), None)

    def unsent(self, point: Point) -> bool:
        return point in self.items[self.sent :]

    def restart(self) -> None:
        """A new connection: nothing sent on it yet."""
        self.sent = 0
        self.acked_samples = 0

    def ready(self, chunk: int) -> tuple[list[Outgoing], Point | None] | None:
        """Unsent audio up to `chunk` samples or to the next commit, when there is that much."""
        audio: list[Audio] = []
        samples = 0
        for item in itertools.islice(self.items, self.sent, None):
            if isinstance(item, Point):
                return unpacked(audio), item
            audio.append(item)
            samples += size(item)
            if samples >= chunk:
                return unpacked(audio), None
        return None

    def chunk_before(self, index: int) -> list[Outgoing]:
        """The audio of the commit at `index`: back to the commit before it."""
        audio: list[Audio] = []
        for item in reversed(self.items[:index]):
            if isinstance(item, Point):
                break
            audio.append(item)
        return unpacked(audio[::-1])

    @property
    def queued_since(self) -> float | None:
        """Start of the oldest audio kept, which may still be sent or replayed."""
        first = next((item for item in self.items if not isinstance(item, Point)), None)
        if first is None:
            return None
        return first.frame.end - size(first) / self._rate

    def answer_point(self) -> tuple[bool, list[Audio]] | None:
        """The first point was answered, if sent: drop through it; its `final` and audio."""
        index = next((i for i, item in enumerate(self.items) if isinstance(item, Point)), None)
        if index is None or index >= self.sent:
            return None
        point = self.items[index]
        assert isinstance(point, Point)
        return point.final, self._drop(index + 1)

    def answer_through(self, offset: int) -> list[Audio]:
        """Drop sent audio a provider commit covered (`offset` samples into the connection), up
        to the first unanswered commit."""
        count = 0
        acked = self.acked_samples
        for item in self.items[: self.sent]:
            if isinstance(item, Point) or acked + size(item) > offset:
                break
            acked += size(item)
            count += 1
        return self._drop(count)

    def _drop(self, count: int) -> list[Audio]:
        audio = [item for item in self.items[:count] if not isinstance(item, Point)]
        for item in audio:
            self.acked_samples += size(item)
            self.samples -= size(item)
        del self.items[:count]
        self.sent -= count
        return audio
