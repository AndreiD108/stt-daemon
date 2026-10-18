"""Markers: waiting for a segment, then placed into its text between words."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True)
class TimedWord:
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class PendingMarker:
    marker: str
    at_ms: int


@dataclass(frozen=True)
class PlacedMarker:
    marker: str
    at_ms: int
    after_word: int
    """Index of the word the marker follows; -1 before the first."""


class MarkerQueue:
    """Accepted markers not yet in a segment, in arrival order."""

    def __init__(self) -> None:
        self._pending: list[PendingMarker] = []

    def __len__(self) -> int:
        return len(self._pending)

    def add(self, marker: str, at_ms: int) -> None:
        self._pending.append(PendingMarker(marker, at_ms))

    def take(self, until_ms: int | None = None) -> list[PendingMarker]:
        """Those at or before `until_ms` (all when None); a segment's markers end at its span."""
        if until_ms is None:
            taken, self._pending = self._pending, []
            return taken
        taken = [m for m in self._pending if m.at_ms <= until_ms]
        self._pending = [m for m in self._pending if m.at_ms > until_ms]
        return taken

    def clear(self) -> None:
        self._pending.clear()


def place(
    text: str,
    words: Sequence[TimedWord] | None,
    span: tuple[int, int],
    markers: Sequence[PendingMarker],
) -> tuple[str, list[PlacedMarker]]:
    """Put each marker in the word gap nearest its time; returns the text with markers inline.

    With word times, a marker inside a word goes on the nearer side of it. Without them, the
    segment's span is spread over its text by character count.
    """
    if not markers:
        return text, []
    tokens = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
    if words is not None:
        mids = [(w.start_ms + w.end_ms) / 2 for w in words]
        ends = _word_ends(text, words, tokens)
    else:
        start_ms, end_ms = span
        length = len(text)
        scale = length / (end_ms - start_ms) if end_ms > start_ms else 0.0
        mids = [start_ms + (s + e) / 2 / scale if scale else start_ms for s, e in tokens]
        ends = [e for _, e in tokens]
    placed = [
        PlacedMarker(m.marker, m.at_ms, sum(1 for mid in mids if mid <= m.at_ms) - 1)
        for m in markers
    ]
    by_position: dict[int, list[str]] = {}
    for p in placed:
        position = ends[p.after_word] if p.after_word >= 0 else 0
        by_position.setdefault(position, []).append(p.marker)
    for position in sorted(by_position, reverse=True):
        text = _insert(text, position, " ".join(by_position[position]))
    return text, placed


def _word_ends(text: str, words: Sequence[TimedWord], tokens: list[tuple[int, int]]) -> list[int]:
    """Where each word ends in `text`, including punctuation stuck to it."""
    ends: list[int] = []
    cursor = 0
    for word in words:
        found = text.find(word.text, cursor) if word.text else -1
        if found < 0:
            return _spread(len(words), tokens, len(text))
        end = found + len(word.text)
        while end < len(text) and not text[end].isspace():
            end += 1
        ends.append(end)
        cursor = end
    return ends


def _spread(count: int, tokens: list[tuple[int, int]], length: int) -> list[int]:
    """Word ends when the words can't be found in the text: tokens shared out in order."""
    if not tokens:
        return [length] * count
    return [
        tokens[min(len(tokens) - 1, (i + 1) * len(tokens) // count - 1)][1] for i in range(count)
    ]


def _insert(text: str, position: int, insert: str) -> str:
    before, after = text[:position], text[position:]
    if before and not before[-1].isspace():
        insert = " " + insert
    if after and not after[0].isspace():
        insert += " "
    return before + insert + after
