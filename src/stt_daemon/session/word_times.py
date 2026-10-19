"""Word times estimated from the audio, for answers with text but no times.

Speech spans (the pause gate's, in chunk seconds) anchor the text: each word goes to one span, so
none straddles a pause, and a span's time is split over its words by syllable count (character
count for text without Latin vowels). Which span a word goes to follows the spans' durations, with
a pause after a sentence's end preferred.
"""

import itertools
import re
from collections.abc import Callable, Iterable
from dataclasses import replace

import numpy as np

from ..audio.frames import Frame, Samples
from ..backends.base import Committed, Word

type Span = tuple[float, float]

SENTENCE_END = (".", "?", "!", "…", "。", "\uff1f", "\uff01")
SENTENCE_BONUS = 2.0
"""A pause after a sentence's end outweighs this much syllable-count mismatch."""
TAIL_DB = 30.0
"""A span ends at its last 10 ms this close to its loud level (the VAD reports ends late)."""
TAIL_PAD_S = 0.03

VOWELS = frozenset("aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿœāēīōūăĕĭŏŭąęįųőű")
_PART = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z]?[a-zÀ-ɏ]+|[A-ZÀ-ɏ]|\d")


def syllables(word: str) -> int:
    """Syllables by vowel groups; acronyms count one per letter."""
    count = 0
    for part in _PART.findall(word):
        if part.isdigit():
            count += 1
        elif part.isupper() and 2 <= len(part) <= 5:
            count += sum(3 if ch == "W" else 1 for ch in part)
        else:
            count += _vowel_groups(part.casefold())
    return max(count, 1)


def _vowel_groups(part: str) -> int:
    groups = 0
    previous = False
    for ch in part:
        vowel = ch in VOWELS
        groups += vowel and not previous
        previous = vowel
    silent_e = len(part) > 2 and part[-1] == "e" and part[-2] not in VOWELS
    if groups > 1 and silent_e and not (part[-2] == "l" and part[-3] not in VOWELS):
        groups -= 1  # "table", "little": the e is heard
    if groups > 1 and part.endswith("ea") and len(part) > 2 and part[-3] not in VOWELS:
        groups += 1  # idea, area: the final a is its own syllable
    return groups


type Speech = Callable[[float, float], list[Span]]
"""The speech inside a capture-time range, as spans."""


def with_times(event: Committed, frames: list[Frame], speech: Speech, offset: int) -> Committed:
    """`event` with word times estimated when it has text but none. `frames`: the audio it
    answered, in order, starting `offset` samples into the connection."""
    texts = event.text.split()
    if event.words or not texts or not frames:
        return event
    rate = frames[0].sample_rate
    spans = speech_spans([(f.start, f.samples.size) for f in frames], speech, rate)
    times = estimate(texts, np.concatenate([f.samples for f in frames]), spans, rate)
    base = offset / rate
    words = tuple(Word(t, base + s, base + e) for t, (s, e) in zip(texts, times, strict=True))
    return replace(event, words=words)


def speech_spans(pieces: Iterable[tuple[float, int]], speech: Speech, rate: int) -> list[Span]:
    """Speech spans in seconds into the audio of `pieces`."""
    spans: list[Span] = []
    at = 0.0
    for start, samples in pieces:
        for s, e in speech(start, start + samples / rate):
            lo, hi = at + s - start, at + e - start
            if spans and lo - spans[-1][1] < 1 / rate:
                spans[-1] = (spans[-1][0], hi)
            else:
                spans.append((lo, hi))
        at += samples / rate
    return spans


def estimate(words: list[str], audio: Samples, speech: list[Span], rate: int) -> list[Span]:
    """(start_s, end_s) per word, seconds into `audio`. `speech`: speech spans in it, in order;
    with none, the whole audio counts as one."""
    if not words:
        return []
    duration = audio.size / rate
    spans = [(max(s, 0.0), min(e, duration)) for s, e in speech if e > s] or [(0.0, duration)]
    spans = [_tail_trimmed(span, audio, rate) for span in spans]
    latin = any(ch in VOWELS for w in words for ch in w.casefold())
    weights = [syllables(w) if latin else max(len(w), 1) for w in words]
    ends = [w.rstrip("\"')»”").endswith(SENTENCE_END) for w in words]
    times: list[Span] = []
    at = 0
    for (lo, hi), size in zip(spans, _distribute(weights, ends, spans), strict=True):
        group = weights[at : at + size]
        at += size
        scale = (hi - lo) / max(sum(group), 1)
        for weight in group:
            times.append((lo, lo + weight * scale))
            lo += weight * scale
    return times


def _tail_trimmed(span: Span, audio: Samples, rate: int) -> Span:
    lo, hi = span
    hop = rate // 100
    first = round(lo * rate)
    count = (round(hi * rate) - first) // hop
    if count < 2:
        return span
    frames = audio[first : first + count * hop].astype(np.float64).reshape(count, hop)
    level = np.sqrt(np.mean(frames**2, axis=1))
    loud = np.flatnonzero(level >= np.percentile(level, 95) * 10 ** (-TAIL_DB / 20))
    if not loud.size:
        return span
    return lo, min(hi, lo + (int(loud[-1]) + 1) * hop / rate + TAIL_PAD_S)


def _distribute(weights: list[int], ends: list[bool], spans: list[Span]) -> list[int]:
    """How many words, in order, go to each span: the split whose weight per span best matches
    the span's share of the duration, preferring pauses after a sentence's end."""
    total = sum(weights)
    span_s = sum(e - s for s, e in spans)
    want = [total * (e - s) / span_s for s, e in spans]
    n = len(weights)
    prefix = list(itertools.accumulate(weights, initial=0))
    inf = float("inf")
    cost = [[inf] * (n + 1) for _ in range(len(spans) + 1)]
    back = [[0] * (n + 1) for _ in range(len(spans) + 1)]
    cost[0][0] = 0.0
    for k, target in enumerate(want, start=1):
        last = k == len(spans)
        for j in range(n + 1) if not last else (n,):
            for i in range(j + 1):
                c = cost[k - 1][i] + (prefix[j] - prefix[i] - target) ** 2 / max(target, 1.0)
                if not last and j > i and ends[j - 1]:
                    c -= SENTENCE_BONUS
                if c < cost[k][j]:
                    cost[k][j] = c
                    back[k][j] = i
    sizes: list[int] = []
    j = n
    for k in range(len(spans), 0, -1):
        i = back[k][j]
        sizes.append(j - i)
        j = i
    return sizes[::-1]
