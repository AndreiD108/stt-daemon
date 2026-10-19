"""A built-in acoustic echo canceller, in NumPy, for a system session's mic track where PipeWire's
echo-cancel module is not there.

Per 10 ms block, with the far end (what the machine plays) and the near end (the mic) for the
same capture time: a GCC-PHAT delay estimate shifts the far end; a partitioned-block
frequency-domain NLMS filter predicts the echo and subtracts it; a double-talk guard freezes
adaptation while the near end speaks over it; a per-bin gain suppresses what echo remains. After
1 s of a quiet far end the near end passes unchanged.
"""

import collections
import math

import numpy as np
import numpy.typing as npt

type Block = npt.NDArray[np.float64]
type Audio = npt.NDArray[np.float32] | Block

TAIL_S = 0.26
MAX_DELAY_S = 0.5
DELAY_WINDOW_S = 2.0
DELAY_EVERY_S = 10.0
FIRST_DELAY_EVERY_S = 0.5
DELAY_MARGIN_BLOCKS = 2
"""The far end is shifted by the delay estimate less this, so an echo a little early still fits."""
DELAY_MIN_ACTIVE = 0.5
"""Share of the estimate window's blocks that must have far-end audio and no double talk."""
DELAY_BAND_HZ = (200.0, 4000.0)
DELAY_MIN_PEAK = 6.0
"""The correlation peak over the search range's standard deviation, below which no estimate."""
FAR_FLOOR_DB = -60.0
BYPASS_S = 1.0
STEP = 0.5
TRAINED_ERLE_DB = 6.0
"""The guard acts once the filter takes this much echo out; before, its echo estimate is too low
and the guard would freeze it for good."""
ERLE_SMOOTHING = 0.95
DOUBLE_TALK_DB = 6.0
STUCK_BLOCKS = 300
"""Frozen far-active blocks in a row after which the filter adapts again, slower: an echo path
that changed looks like double talk to the guard."""
STUCK_STEP = STEP / 4
SUPPRESS_FLOOR_DB = -20.0
SUPPRESS_OVER = 3.0
SUPPRESS_SMOOTHING = 0.5
LEAK_SMOOTHING = 0.95
PICK_SMOOTHING = 0.9
PICK_BETTER = 0.8
PICK_WORSE = 2.0


class _History:
    """The last `size` samples; every window up to `size` long is one contiguous slice."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._buf = np.zeros(2 * size)
        self._at = size

    def push(self, block: Block) -> None:
        n = block.size
        if self._at + n > 2 * self._size:
            keep = self._buf[self._at - self._size : self._at].copy()
            self._buf[: self._size] = keep
            self._at = self._size
        self._buf[self._at : self._at + n] = block
        self._at += n

    def last(self, n: int, offset: int = 0) -> Block:
        """`n` samples ending `offset` samples before the newest."""
        end = self._at - offset
        return self._buf[end - n : end]


class EchoCanceller:
    """`process(near, far)` per block of `block` samples (10 ms); the output lags the input by
    `latency` samples."""

    def __init__(self, rate: int = 16000) -> None:
        self.rate = rate
        self.block = rate // 100
        self.latency = self.block
        self._partitions = math.ceil(TAIL_S * rate / self.block)
        self._max_delay = round(MAX_DELAY_S * rate)
        self._window = round(DELAY_WINDOW_S * rate)
        bins = self.block + 1
        self._bin_power_floor = 2 * self.block * 10 ** (FAR_FLOOR_DB / 10)
        self._far_floor = 10 ** (FAR_FLOOR_DB / 10)
        self._double_talk = 10 ** (DOUBLE_TALK_DB / 10)
        self._trained = 10 ** (TRAINED_ERLE_DB / 10)
        self._suppress_floor = 10 ** (SUPPRESS_FLOOR_DB / 20)
        self._window_fn = np.sqrt(
            0.5 - 0.5 * np.cos(2 * np.pi * np.arange(2 * self.block) / (2 * self.block))
        )
        self._overlap = self._window_fn[self.block :] ** 2
        self._bins = bins
        self._delay_fft = 1 << (2 * self._window - 1).bit_length()
        delay_freqs = np.fft.rfftfreq(self._delay_fft, 1 / rate)
        self._delay_band = (delay_freqs >= DELAY_BAND_HZ[0]) & (delay_freqs <= DELAY_BAND_HZ[1])
        self.reset()

    def reset(self) -> None:
        """Forget the echo path and the delay (the capture restarted)."""
        b, p, bins = self.block, self._partitions, self._bins
        history = max(self._window, self._max_delay + (p + 2) * b)
        self._far = _History(history)
        self._near = _History(self._window)
        self._x = np.zeros((p, bins), dtype=np.complex128)
        self._w = np.zeros((p, bins), dtype=np.complex128)
        """The filter that adapts."""
        self._fg = self._w.copy()
        """The filter that cancels: the adapting one's last state that did better than it."""
        self._adapting_error = 0.0
        self._cancelling_error = 0.0
        self._shift = 0
        self._near_level = 0.0
        self._error_level = 0.0
        self._frozen = 0
        self._quiet_blocks = 0
        self._estimable: collections.deque[bool] = collections.deque(maxlen=self._window // b)
        self._seen = 0
        self._since_estimate = 0
        self._pending: int | None = None
        self.delay: int | None = None
        """The echo delay estimate in samples, once there is one."""
        self._e_prev = np.zeros(b)
        self._y_prev = np.zeros(b)
        self._near_prev = np.zeros(b)
        self._tail = np.zeros(b)
        self._gain = np.ones(bins)
        self._leak = 1.0

    def process(self, near: Audio, far: Audio) -> Block:
        b = self.block
        near, far = near.astype(np.float64), far.astype(np.float64)
        self._far.push(far)
        self._near.push(near)
        self._seen += 1
        far_active = float(np.mean(far**2)) > self._far_floor
        self._quiet_blocks = 0 if far_active else self._quiet_blocks + 1

        self._x[1:] = self._x[:-1]
        self._x[0] = np.fft.rfft(self._far.last(2 * b, self._shift))
        y = np.fft.irfft(np.sum(self._fg * self._x, axis=0))[b:]
        e = near - y
        e_adapting = near - np.fft.irfft(np.sum(self._w * self._x, axis=0))[b:]
        near_power, echo_power = float(np.mean(near**2)), float(np.mean(y**2))
        trained = self._near_level > self._trained * self._error_level
        double_talk = trained and near_power > self._double_talk * echo_power
        step = 0.0
        if far_active:
            if not double_talk:
                step, self._frozen = STEP, 0
                a = ERLE_SMOOTHING
                self._near_level = a * self._near_level + (1 - a) * near_power
                self._error_level = a * self._error_level + (1 - a) * float(np.mean(e**2))
            else:
                self._frozen += 1
                if self._frozen > STUCK_BLOCKS:
                    step = STUCK_STEP
        if step:
            self._adapt(e_adapting, step)
        if far_active:
            self._pick_filter(e_adapting, e)
        self._estimable.append(far_active and not double_talk)
        self._maybe_estimate_delay()

        if self._quiet_blocks * b >= BYPASS_S * self.rate:
            out = self._near_prev
            self._tail = near * self._overlap
            self._e_prev, self._y_prev, self._near_prev = near, y, near
            return out
        out = self._suppress(e, y, update_leak=step == STEP)
        self._e_prev, self._y_prev, self._near_prev = e, y, near
        return out

    def _adapt(self, e: Block, step: float) -> None:
        b = self.block
        err = np.fft.rfft(np.concatenate((np.zeros(b), e)))
        # The partitions' own power, not a smoothed one: at a far-end onset a smoothed power lags
        # and the step blows up.
        norm = np.sum(np.abs(self._x) ** 2, axis=0) + self._bin_power_floor
        self._w += step * np.conj(self._x) * (err / norm)
        taps = np.fft.irfft(self._w, axis=1)
        taps[:, b:] = 0.0
        self._w = np.fft.rfft(taps, axis=1)

    def _pick_filter(self, e_adapting: Block, e: Block) -> None:
        """Take the adapting filter when it cancels better; go back to the cancelling one when
        it does much worse, as after adapting on double talk the guard missed."""
        a = PICK_SMOOTHING
        self._adapting_error = a * self._adapting_error + (1 - a) * float(np.mean(e_adapting**2))
        self._cancelling_error = a * self._cancelling_error + (1 - a) * float(np.mean(e**2))
        if self._adapting_error < PICK_BETTER * self._cancelling_error:
            self._fg = self._w.copy()
            self._cancelling_error = self._adapting_error
        elif self._adapting_error > PICK_WORSE * self._cancelling_error:
            self._w = self._fg.copy()
            self._adapting_error = self._cancelling_error

    def _suppress(self, e: Block, y: Block, *, update_leak: bool) -> Block:
        b, win = self.block, self._window_fn
        err = np.fft.rfft(np.concatenate((self._e_prev, e)) * win)
        echo = np.fft.rfft(np.concatenate((self._y_prev, y)) * win)
        err_power, echo_power = np.abs(err) ** 2, np.abs(echo) ** 2
        if update_leak and (total := float(np.sum(echo_power))) > 0:
            a = LEAK_SMOOTHING
            ratio = min(1.0, float(np.sum(err_power)) / total)
            self._leak = a * self._leak + (1 - a) * ratio
        gain = 1.0 - SUPPRESS_OVER * self._leak * echo_power / (err_power + 1e-12)
        gain = np.clip(gain, self._suppress_floor, 1.0)
        s = SUPPRESS_SMOOTHING
        self._gain = s * self._gain + (1 - s) * gain
        frame = np.fft.irfft(err * self._gain) * win
        out = self._tail + frame[:b]
        self._tail = frame[b:]
        return out

    def _maybe_estimate_delay(self) -> None:
        self._since_estimate += 1
        every = FIRST_DELAY_EVERY_S if self.delay is None else DELAY_EVERY_S
        if self._since_estimate * self.block < every * self.rate:
            return
        if self._seen * self.block < self._window:
            return
        if sum(self._estimable) < DELAY_MIN_ACTIVE * len(self._estimable):
            return
        self._since_estimate = 0
        found = self._estimate_delay()
        if found is None:
            return
        if self.delay is None:
            self._apply_delay(found)
        elif abs(found - self.delay) <= self.block:
            self._pending = None
        elif self._pending is not None and abs(found - self._pending) <= self.block:
            self._pending = None
            self._apply_delay(found)
        else:
            self._pending = found

    def _estimate_delay(self) -> int | None:
        n = self._delay_fft
        far = np.fft.rfft(self._far.last(self._window), n)
        near = np.fft.rfft(self._near.last(self._window), n)
        cross = np.conj(far) * near
        cross = np.where(self._delay_band, cross / (np.abs(cross) + 1e-12), 0.0)
        corr = np.fft.irfft(cross, n)[: self._max_delay + 1]
        lag = int(np.argmax(corr))
        spread = float(np.std(corr))
        if spread <= 0 or corr[lag] / spread < DELAY_MIN_PEAK:
            return None
        return lag

    def _apply_delay(self, delay: int) -> None:
        """Move the filter's taps with the far end's new shift, so it stays adapted."""
        b, p = self.block, self._partitions
        self.delay = delay
        shift = max(0, delay - DELAY_MARGIN_BLOCKS * b)
        move = shift - self._shift
        if move == 0:
            return
        self._w = self._moved(self._w, move)
        self._fg = self._moved(self._fg, move)
        self._shift = shift
        self._x = np.stack([np.fft.rfft(self._far.last(2 * b, shift + i * b)) for i in range(p)])

    def _moved(self, filter: npt.NDArray[np.complex128], move: int) -> npt.NDArray[np.complex128]:
        """`filter` with its taps `move` samples earlier (later when negative)."""
        b, p = self.block, self._partitions
        taps = np.fft.irfft(filter, axis=1)[:, :b].reshape(-1)
        moved = np.zeros_like(taps)
        if abs(move) >= taps.size:
            pass  # the whole echo path moved past the filter: start from zero
        elif move > 0:
            moved[: taps.size - move] = taps[move:]
        else:
            moved[-move:] = taps[: taps.size + move]
        padded = np.zeros((p, 2 * b))
        padded[:, :b] = moved.reshape(p, b)
        return np.fft.rfft(padded, axis=1)
