"""What a link's log holds: queued audio (packed to int16 while suspended) and commit points."""

from dataclasses import dataclass

from ..audio.frames import PackedFrame
from .pause_gate import Outgoing


@dataclass(eq=False)
class Point:
    """A commit."""

    final: bool


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
