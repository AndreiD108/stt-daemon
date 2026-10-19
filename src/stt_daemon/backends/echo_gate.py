"""A level gate for mic audio in front of a local decoder: per 20 ms frame, audio far under the
chunk's voice level is cut, so neither background talk nor echo of the speakers gets transcribed.

Frames the speakers' playback explains (echo) are cut harder than the rest: within 25 dB of the
voice level in the body, 6 dB before the first loud frame; others within 35 and 10 dB, so a quiet
start or a fading tail of the user's own speech stays in. Kept stretches are joined by 100 ms of
near-silence; `GateMap` maps decoder times back to the chunk.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..audio.frames import Samples
from .base import SAMPLE_RATE, PlaybackSpan, Reference

FRAME = SAMPLE_RATE // 50
"""20 ms."""
ECHO_DB = (25.0, 6.0)
"""(body, start) for frames the playback explains."""
OTHER_DB = (35.0, 10.0)
LOUD_DB = 6.0
"""Frames this close to the voice level start the body."""
KEEP_AROUND = 8
"""Frames kept on each side of a kept frame."""
FILL_UNDER = 10
"""Cut stretches shorter than this between kept ones are kept."""
JOINT_S = 0.1
JOINT_DB = -70.0

MAX_LAG = 25
"""Frames the echo may come after the playback's time stamp (500 ms)."""
MIN_LAG = -5
"""Frames it may come before it: playback frames are stamped on arrival, maybe after the mic's."""
TAIL = 15
"""Frames of room reverb after each playback frame."""
DECAY_DB = 4.0
"""Per frame along that tail."""
MARGIN_DB = 6.0
"""A frame up to this much over the predicted echo counts as echo."""
ACTIVE_DB = -60.0
"""Playback frames under this are silence."""
MIN_CORR = 0.3
"""Below this, the mic doesn't follow the playback: nothing leaks, nothing is echo."""
MIN_ACTIVE = 10
LEAD = (MAX_LAG + TAIL) * FRAME
"""Playback samples a span needs before its first mic sample."""
LOOKAHEAD = -MIN_LAG * FRAME
"""And after its last."""

type Mask = npt.NDArray[np.bool_]
type Levels = npt.NDArray[np.float64]


def frame_db(x: Samples) -> Levels:
    n = len(x) // FRAME
    power = (np.asarray(x[: n * FRAME], np.float64).reshape(n, FRAME) ** 2).mean(axis=1)
    return 10 * np.log10(power + 1e-12)


def _smear(db: Levels) -> Levels:
    """Each frame's level, or the decaying tail of a louder frame before it."""
    out = db.copy()
    for k in range(1, TAIL + 1):
        out[k:] = np.maximum(out[k:], db[:-k] - DECAY_DB * k)
    return out


def _window(span: PlaybackSpan, frames: int) -> Samples:
    """The span's playback from `LEAD` before its first mic sample to `LOOKAHEAD` after its
    last frame; silence where the span has less."""
    out = np.zeros(LEAD + frames * FRAME + LOOKAHEAD, np.float32)
    first = span.lead - LEAD  # index in span.playback of out[0]
    lo, hi = max(first, 0), min(first + out.size, span.playback.size)
    if lo < hi:
        out[lo - first : hi - first] = span.playback[lo:hi]
    return out


def echo_frames(n: int, mic: Samples, spans: Sequence[PlaybackSpan]) -> Mask:
    """Per 20 ms frame of a chunk of `n` frames: True where the playback, delayed and scaled,
    explains the mic's level. One delay and one coupling for the whole chunk, found from its
    spans; each span is matched only to the playback around its own capture time."""
    parts: list[tuple[int, Levels, Levels]] = []
    for span in spans:
        frames = span.samples // FRAME
        if frames == 0:
            continue
        mic_db = frame_db(mic[span.offset : span.offset + frames * FRAME])
        parts.append((span.offset, mic_db, frame_db(_window(span, frames))))
    none = np.zeros(n, bool)
    if not parts:
        return none

    def at_lag(lag: int, levels: list[Levels]) -> Levels:
        lead = LEAD // FRAME
        pairs = zip(parts, levels, strict=True)
        return np.concatenate([db[lead - lag : lead - lag + len(m)] for (_, m, _), db in pairs])

    raw = [p for _, _, p in parts]
    mic_db = np.concatenate([m for _, m, _ in parts])
    if (at_lag(0, raw) > ACTIVE_DB).sum() < MIN_ACTIVE:
        return none
    best, lag = -1.0, 0
    for k in range(MIN_LAG, MAX_LAG + 1):
        ref = at_lag(k, raw)
        on = ref > ACTIVE_DB
        if on.sum() < MIN_ACTIVE:
            continue
        c = float(np.corrcoef(mic_db[on], ref[on])[0, 1])
        if c > best:
            best, lag = c, k
    if best < MIN_CORR:
        return none
    pred = at_lag(lag, [_smear(p) for p in raw])
    on = pred > ACTIVE_DB
    coupling = float(np.percentile(mic_db[on] - pred[on], 30))
    echo = on & (mic_db <= pred + coupling + MARGIN_DB)
    per_sample = np.zeros(n * FRAME, bool)
    at = 0
    for offset, m, _ in parts:
        end = min(offset + len(m) * FRAME, n * FRAME)
        per_sample[offset:end] = np.repeat(echo[at : at + len(m)], FRAME)[: end - offset]
        at += len(m)
    return per_sample.reshape(n, FRAME).mean(axis=1) > 0.5


@dataclass(frozen=True)
class GateMap:
    runs: tuple[tuple[int, int, int], ...]
    """(gated offset, chunk offset, samples) of each kept stretch."""

    def word_span(self, start_s: float, end_s: float) -> tuple[float, float]:
        """Chunk seconds of a word's gated start and end; a word across a joint goes whole
        into the stretch holding more of it."""
        if not self.runs:
            return start_s, end_s
        a, b = round(start_s * SAMPLE_RATE), round(end_s * SAMPLE_RATE)
        b = max(a, b)

        def overlap(run: tuple[int, int, int]) -> int:
            return min(b, run[0] + run[2]) - max(a, run[0])

        def distance(run: tuple[int, int, int]) -> int:
            return max(run[0] - b, a - (run[0] + run[2]), 0)

        run = max(self.runs, key=lambda r: (overlap(r), -distance(r)))
        gated, chunk, size = run
        lo = min(max(a, gated), gated + size)
        hi = min(max(b, lo), gated + size)
        return (chunk + lo - gated) / SAMPLE_RATE, (chunk + hi - gated) / SAMPLE_RATE


def _dilate(mask: Mask, k: int) -> Mask:
    out = mask.copy()
    for s in range(1, k + 1):
        out[s:] |= mask[:-s]
        out[:-s] |= mask[s:]
    return out


def _runs(mask: Mask) -> list[tuple[int, int]]:
    steps = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    edges = [int(e) for e in np.flatnonzero(steps)]
    return list(zip(edges[::2], edges[1::2], strict=True))


def gate(audio: Samples, echo: Mask) -> tuple[Samples, GateMap]:
    """The kept stretches of `audio`, joined; `echo` per 20 ms frame (`echo_frames`)."""
    db = frame_db(audio)
    if not len(db):
        return audio, GateMap(((0, 0, len(audio)),))
    level = float(np.percentile(db, 95))
    body = np.where(echo, ECHO_DB[0], OTHER_DB[0])
    start = np.where(echo, ECHO_DB[1], OTHER_DB[1])
    keep = db > level - body
    loud = np.flatnonzero(db > level - LOUD_DB)
    first = int(loud[0]) if loud.size else len(db)
    keep[:first] &= db[:first] > level - start[:first]
    keep = _dilate(keep, KEEP_AROUND)
    for a, b in _runs(~keep):
        if a > 0 and b < len(keep) and b - a < FILL_UNDER:
            keep[a:b] = True
    rng = np.random.default_rng(0)
    joint = rng.normal(0, 10 ** (JOINT_DB / 20), round(JOINT_S * SAMPLE_RATE)).astype(np.float32)
    parts: list[Samples] = []
    runs: list[tuple[int, int, int]] = []
    at = 0
    for a, b in _runs(keep):
        if parts:
            parts.append(joint)
            at += joint.size
        part = audio[a * FRAME : b * FRAME]
        runs.append((at, a * FRAME, part.size))
        parts.append(part)
        at += part.size
    if not parts:
        return audio[:0], GateMap(())
    return np.concatenate(parts).astype(np.float32), GateMap(tuple(runs))


def apply(audio: Samples, reference: Reference | None) -> tuple[Samples, GateMap]:
    """Gate a chunk of mic audio; without `reference`, no frame counts as echo."""
    n = len(audio) // FRAME
    spans = reference.spans if reference is not None else ()
    return gate(audio, echo_frames(n, audio, spans))
