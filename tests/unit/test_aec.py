"""The built-in echo canceller on synthetic echo: TTS far end through a synthetic room into TTS
near end (`tests/aec_fixture.py`)."""

import time

import numpy as np
import pytest
from aec_fixture import FIXTURES, NEAR_PHRASES, RATE, Audio, Fixture, build, echo_of, read, room_ir

from stt_daemon.audio.aec import EchoCanceller
from stt_daemon.backends.install import is_installed
from stt_daemon.backends.parakeet import RecognizerKey, load_recognizer, model_dir


def level_db(audio: Audio) -> float:
    return float(10 * np.log10(np.mean(audio**2) + 1e-20))


def span(seconds: tuple[float, float]) -> slice:
    return slice(round(seconds[0] * RATE), round(seconds[1] * RATE))


def cancel(canceller: EchoCanceller, near: Audio, far: Audio) -> tuple[Audio, list[int | None]]:
    """The output, moved back by the canceller's latency, and the delay estimate per block."""
    b = canceller.block
    out = np.zeros(near.size)
    delays: list[int | None] = []
    for i in range(near.size // b):
        part = slice(i * b, (i + 1) * b)
        out[part] = canceller.process(near[part], far[part])
        delays.append(canceller.delay)
    return np.concatenate((out[canceller.latency :], np.zeros(canceller.latency))), delays


@pytest.fixture(scope="module")
def fixture() -> Fixture:
    return build()


@pytest.fixture(scope="module")
def cancelled(fixture: Fixture) -> tuple[Audio, list[int | None]]:
    return cancel(EchoCanceller(), fixture.mixed, fixture.far)


def test_far_end_only_audio_comes_out_at_least_12_db_lower(
    fixture: Fixture, cancelled: tuple[Audio, list[int | None]]
) -> None:
    out, _ = cancelled
    for region in fixture.far_only:
        drop = level_db(fixture.mixed[span(region)]) - level_db(out[span(region)])
        assert drop >= 12, f"{region}: {drop:.1f} dB"


def test_near_end_only_audio_keeps_its_level_within_1_db(
    fixture: Fixture, cancelled: tuple[Audio, list[int | None]]
) -> None:
    out, _ = cancelled
    for region in fixture.near_only:
        change = level_db(out[span(region)]) - level_db(fixture.near[span(region)])
        assert abs(change) < 1, f"{region}: {change:.2f} dB"


def test_the_delay_estimate_lands_within_one_block_of_the_echo_delay(
    cancelled: tuple[Audio, list[int | None]],
) -> None:
    _, delays = cancelled
    assert delays[-1] is not None
    assert abs(delays[-1] - round(0.08 * RATE)) <= RATE // 100


def test_a_delay_change_is_picked_up_within_30_s() -> None:
    phrases = read(FIXTURES / "two_phrases.wav")
    far = np.tile(phrases, 7)[: 60 * RATE]
    jump = 20 * RATE
    near = np.concatenate((echo_of(far, room_ir(0.08))[:jump], echo_of(far, room_ir(0.14))[jump:]))
    canceller = EchoCanceller()
    out, delays = cancel(canceller, near, far)
    block = canceller.block
    after = delays[jump // block :]
    new = round(0.14 * RATE)
    picked = next(i for i, d in enumerate(after) if d is not None and abs(d - new) <= block)
    assert picked * block / RATE < 30
    assert all(d is not None and abs(d - new) <= block for d in after[picked:])
    tail = slice(55 * RATE, 60 * RATE)
    assert level_db(near[tail]) - level_db(out[tail]) >= 12


@pytest.mark.parametrize(("first_s", "then_s"), [(0.4, 0.4), (0.03, 0.45), (0.45, 0.03)])
def test_delays_across_the_search_range_cancel_and_moves_past_the_filter_restart_it(
    first_s: float, then_s: float
) -> None:
    phrases = read(FIXTURES / "two_phrases.wav")
    far = np.tile(phrases, 7)[: 60 * RATE]
    jump = 20 * RATE
    near = np.concatenate(
        (echo_of(far, room_ir(first_s))[:jump], echo_of(far, room_ir(then_s))[jump:])
    )
    canceller = EchoCanceller()
    out, delays = cancel(canceller, near, far)
    assert np.isfinite(out).all()
    assert delays[-1] is not None
    assert abs(delays[-1] - round(then_s * RATE)) <= canceller.block
    for tail in (slice(15 * RATE, 20 * RATE), slice(55 * RATE, 60 * RATE)):
        assert level_db(near[tail]) - level_db(out[tail]) >= 12, tail


def test_after_1_s_of_a_quiet_far_end_the_near_end_passes_unchanged(fixture: Fixture) -> None:
    canceller = EchoCanceller()
    b = canceller.block
    warm = span((0.0, 4.0))
    cancel(canceller, fixture.mixed[warm], fixture.far[warm])
    rng = np.random.default_rng(1)
    near = rng.standard_normal(2 * RATE) * 0.05
    out, _ = cancel(canceller, near, np.zeros(near.size))
    quiet = slice(RATE + b, 2 * RATE - b)
    np.testing.assert_array_equal(out[quiet], near[quiet])


def test_reset_forgets_the_delay(fixture: Fixture) -> None:
    canceller = EchoCanceller()
    cancel(canceller, fixture.mixed[: 4 * RATE], fixture.far[: 4 * RATE])
    assert canceller.delay is not None
    canceller.reset()
    assert canceller.delay is None


def test_a_block_takes_well_under_a_millisecond(fixture: Fixture) -> None:
    canceller = EchoCanceller()
    b = canceller.block
    began = time.perf_counter()
    for i in range(1000):
        part = slice(i * b, (i + 1) * b)
        canceller.process(fixture.mixed[part], fixture.far[part])
    assert time.perf_counter() - began < 1.0


MODEL = model_dir()
EXPECTED = (
    "Restart the stt-daemon, then open kitty and check the Huion tablet. "
    "Scribbly sends pen marks while Talkey holds the push to talk key."
)
"""What the fixture's near end says: jargon.wav's first two phrases."""


def words(text: str) -> set[str]:
    return {w.strip(".,!?").lower() for w in text.split()} - {""}


@pytest.mark.slow
@pytest.mark.skipif(not is_installed(MODEL), reason="the Parakeet model is not installed")
def test_parakeet_hears_every_near_end_word_in_the_cancelled_audio_that_it_hears_in_the_clean(
    fixture: Fixture, cancelled: tuple[Audio, list[int | None]]
) -> None:
    recognizer = load_recognizer(RecognizerKey(MODEL, 2, 1.5))

    def transcribe(audio: Audio) -> set[str]:
        heard: set[str] = set()
        # Each phrase on its own, padded as the pause gate sends it.
        for start, end in NEAR_PHRASES:
            stream = recognizer.create_stream()
            stream.accept_waveform(RATE, audio[span((start - 0.1, end + 0.2))].astype(np.float32))
            recognizer.decode_stream(stream)
            heard |= words(stream.result.text)
        return heard

    clean = transcribe(fixture.near) & words(EXPECTED)
    assert len(clean) >= len(words(EXPECTED)) // 2, clean
    out, _ = cancelled
    assert clean - transcribe(out) == set()
    assert transcribe(fixture.mixed) & clean != clean
