"""Parakeet TDT decoding on onnxruntime: features as sherpa-onnx makes them, then our beam search.

sherpa-onnx's NeMo beam search returns empty text for whole sentences (it finds the "nothing
said" path: with no token emitted yet, the model gives blank at almost every frame after the
speech's onset). This search keeps hypotheses in buckets by the frame they stand at, so only
hypotheses that took in the same audio are compared; equal token sequences in a bucket merge
(logaddexp). Each emitted token earns `TOKEN_BONUS`, which keeps the empty path from winning.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kaldi_native_fbank
import numpy as np
import onnxruntime

from ..audio.frames import Samples

_knf: Any = kaldi_native_fbank  # untyped modules
_ort: Any = onnxruntime

BLANK = 8192
VOCAB = 8193
FRAME_S = 0.08
MAX_SYMBOLS = 5
"""Tokens emitted at one frame at most, as in sherpa."""
BEAM = 4
TOKEN_BONUS = 0.5
MARGIN = 5.0
"""Hypotheses this far below their bucket's best, or continuations this far below a
hypothesis's best one, are dropped."""
WORD_START = "▁"


@dataclass(frozen=True)
class Tokens:
    ids: tuple[int, ...]
    frames: tuple[int, ...]
    durations: tuple[int, ...]
    """In frames, as the model predicted them."""


class TdtModel:
    def __init__(self, model_dir: Path, num_threads: int) -> None:
        def session(name: str, threads: int) -> Any:
            options = _ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            return _ort.InferenceSession(
                str(model_dir / name), options, providers=["CPUExecutionProvider"]
            )

        self.encoder = session("encoder.int8.onnx", num_threads)
        self.decoder = session("decoder.int8.onnx", 1)
        self.joiner = session("joiner.int8.onnx", 1)
        pieces: dict[int, str] = {}
        for line in (model_dir / "tokens.txt").read_text(encoding="utf-8").splitlines():
            piece, i = line.rsplit(" ", 1)
            pieces[int(i)] = piece
        self.pieces = pieces
        self._ids = {piece: i for i, piece in pieces.items() if i < BLANK}
        opts = _knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.remove_dc_offset = False
        opts.frame_opts.window_type = "povey"
        opts.frame_opts.preemph_coeff = 0.97
        opts.mel_opts.num_bins = 128
        opts.mel_opts.low_freq = 0
        opts.mel_opts.high_freq = -400
        opts.mel_opts.is_librosa = True
        self._fbank = opts

    def encode(self, audio: Samples) -> np.ndarray:
        """(encoder frames, 1024), one per `FRAME_S`."""
        fbank = _knf.OnlineFbank(self._fbank)
        fbank.accept_waveform(16000, np.asarray(audio, np.float32).tolist())
        fbank.input_finished()
        x = np.array([fbank.get_frame(i) for i in range(fbank.num_frames_ready)], np.float32)
        mean = x.mean(axis=0)
        x = (x - mean) / (np.sqrt(((x - mean) ** 2).mean(axis=0)) + 1e-5)
        out, lengths = self.encoder.run(
            None, {"audio_signal": x.T[None].copy(), "length": np.array([len(x)], np.int64)}
        )
        return np.ascontiguousarray(out[0, :, : int(lengths[0])].T)

    def step(
        self, token: int, state: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Decoder output (640,) after `token`, and the next state."""
        out, _, h, c = self.decoder.run(
            None,
            {
                "targets": np.array([[token]], np.int32),
                "target_length": np.array([1], np.int32),
                "states.1": state[0],
                "onnx::Slice_3": state[1],
            },
        )
        return out[0, :, 0], (h, c)

    def joint(self, enc: np.ndarray, dec: np.ndarray) -> np.ndarray:
        """One frame (1024,) and one decoder output (640,) -> logits (8198,). One call per
        hypothesis: the int8 model shares quantization scales across a batch, which would make
        a hypothesis's scores depend on the others in it."""
        (out,) = self.joiner.run(
            None, {"encoder_outputs": enc[None, :, None], "decoder_outputs": dec[None, :, None]}
        )
        return out[0, 0, 0, :]

    def text(self, ids: tuple[int, ...]) -> str:
        return "".join(self.pieces[t] for t in ids).replace(WORD_START, " ").strip()

    def tokenize(self, phrase: str) -> tuple[int, ...] | None:
        return tokenize(self._ids, phrase)


def tokenize(ids: dict[str, int], phrase: str) -> tuple[int, ...] | None:
    """BPE from single characters, merging the pair whose piece has the lowest id first, as the
    model's own tokenizer splits text; None when a character has no piece."""
    parts = list(WORD_START + phrase.replace(" ", WORD_START))
    if any(p not in ids for p in parts):
        return None
    while True:
        merges = [
            (i, at)
            for at, pair in enumerate(itertools.pairwise(parts))
            if (i := ids.get("".join(pair))) is not None
        ]
        if not merges:
            return tuple(ids[p] for p in parts)
        _, at = min(merges)
        parts[at : at + 2] = [parts[at] + parts[at + 1]]


class Keyterms:
    """A trie of keyterm token sequences. Each token that goes deeper earns `score`; a match
    that breaks off gives back what it earned past its last whole keyterm, and so does one still
    open at the end. A broken match restarts only from the root, so a keyterm that begins inside
    another's match ("b c" in "a b c" with "a b") isn't found."""

    def __init__(self, sequences: list[tuple[int, ...]], score: float) -> None:
        self.score = score
        self._children: list[dict[int, int]] = [{}]
        self._depth = [0]
        self._kept = [0]
        """Depth of the deepest whole keyterm on the way to each node."""
        for seq in sequences:
            node = 0
            for token in seq:
                if token not in self._children[node]:
                    self._children.append({})
                    self._depth.append(self._depth[node] + 1)
                    self._kept.append(self._kept[node])
                    self._children[node][token] = len(self._children) - 1
                node = self._children[node][token]
            self._kept[node] = self._depth[node]
        for node in range(len(self._kept)):  # a short keyterm can be added after a longer one
            for child in self._children[node].values():
                self._kept[child] = max(self._kept[child], self._kept[node])

    def advance(self, node: int, token: int) -> tuple[int, float]:
        """The node after `token`, and the score change."""
        child = self._children[node].get(token)
        if child is not None:
            return (0 if not self._children[child] else child), self.score
        change = -self.open_score(node)
        child = self._children[0].get(token) if node else None
        if child is None:
            return 0, change
        return (0 if not self._children[child] else child), change + self.score

    def next_tokens(self, node: int) -> set[int]:
        """Tokens that go on from `node`, or start a keyterm afresh."""
        return set(self._children[node]) | set(self._children[0])

    def open_score(self, node: int) -> float:
        return (self._depth[node] - self._kept[node]) * self.score


def keyterms(model: TdtModel, phrases: list[str], score: float) -> Keyterms | None:
    sequences = [ids for p in phrases if (ids := model.tokenize(p))]
    return Keyterms(sequences, score) if sequences else None


@dataclass(frozen=True)
class _Hyp:
    ids: tuple[int, ...]
    score: float
    frames: tuple[int, ...]
    durations: tuple[int, ...]
    best: float
    """The best single alignment's score; picks which alignment's times a merged hypothesis
    keeps."""
    node: int = 0
    """Where the hypothesis stands in the keyterm trie."""


def _add(bucket: dict[tuple[int, ...], _Hyp], h: _Hyp) -> None:
    old = bucket.get(h.ids)
    if old is None:
        bucket[h.ids] = h
        return
    keep = h if h.best > old.best else old
    score = float(np.logaddexp(old.score, h.score))
    bucket[h.ids] = _Hyp(h.ids, score, keep.frames, keep.durations, keep.best, h.node)


def _prune(bucket: dict[tuple[int, ...], _Hyp]) -> list[_Hyp]:
    kept = sorted(bucket.values(), key=lambda h: h.score, reverse=True)[:BEAM]
    return [h for h in kept if h.score >= kept[0].score - MARGIN] if kept else kept


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def beam_search(model: TdtModel, enc: np.ndarray, terms: Keyterms | None = None) -> Tokens:
    """Per hypothesis and frame: the `BEAM` best tokens, each with every duration, and blank
    with every duration from 1. A hypothesis lands in the bucket of frame t + duration; one with
    duration 0 is expanded again at t, at most `MAX_SYMBOLS` tokens per frame. Scores are
    log P(token) + log P(duration) per step, blanks included."""
    n = len(enc)
    zero = np.zeros((2, 1, 640), np.float32)
    cache: dict[tuple[int, ...], tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]] = {
        (): model.step(BLANK, (zero, zero))
    }

    def dec_out(ids: tuple[int, ...]) -> np.ndarray:
        if ids not in cache:
            cache[ids] = model.step(ids[-1], cache[ids[:-1]][1])
        return cache[ids][0]

    buckets: list[dict[tuple[int, ...], _Hyp]] = [{} for _ in range(n + 1)]
    buckets[0][()] = _Hyp((), 0.0, (), (), 0.0)
    for t in range(n):
        pool = _prune(buckets[t])
        buckets[t] = {}
        for emitted in range(MAX_SYMBOLS):
            if not pool:
                break
            logits = np.stack([model.joint(enc[t], dec_out(h.ids)) for h in pool])
            tok_lp = _log_softmax(logits[:, :VOCAB])
            dur_lp = _log_softmax(logits[:, VOCAB:])
            top = np.argpartition(tok_lp[:, :BLANK], -BEAM, axis=1)[:, -BEAM:]
            blank_inc = tok_lp[:, BLANK, None] + dur_lp[:, 1:]
            first_d = 1 if emitted + 1 == MAX_SYMBOLS else 0
            stay: dict[tuple[int, ...], _Hyp] = {}
            for b, h in enumerate(pool):
                tokens = {int(x) for x in top[b]}
                if terms is not None:
                    tokens |= terms.next_tokens(h.node)
                steps: list[tuple[int, int, float, int]] = []
                for token in tokens:
                    node, bonus = terms.advance(h.node, token) if terms else (0, 0.0)
                    gain = float(tok_lp[b, token]) + TOKEN_BONUS + bonus
                    steps += [
                        (token, d, gain + float(dur_lp[b, d]), node)
                        for d in range(first_d, dur_lp.shape[1])
                    ]
                floor = max(float(blank_inc[b].max()), max(s for *_, s, _ in steps)) - MARGIN
                for i in np.flatnonzero(blank_inc[b] >= floor):
                    s = float(blank_inc[b, i])
                    nh = _Hyp(h.ids, h.score + s, h.frames, h.durations, h.best + s, h.node)
                    _add(buckets[min(t + 1 + int(i), n)], nh)
                for token, d, s, node in steps:
                    if s < floor:
                        continue
                    nh = _Hyp(
                        (*h.ids, token), h.score + s, (*h.frames, t), (*h.durations, d),
                        h.best + s, node,
                    )  # fmt: skip
                    _add(stay if d == 0 else buckets[min(t + d, n)], nh)
            pool = _prune(stay)

    def final(h: _Hyp) -> float:
        return h.score - (terms.open_score(h.node) if terms is not None else 0.0)

    best = max(buckets[n].values(), key=final)
    return Tokens(best.ids, best.frames, best.durations)
