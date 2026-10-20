"""Playback for the echo gate on one clock: jitter and jumps, forgetting, waiting for coverage,
and each commit's reference spans."""

import asyncio
import time

import numpy as np
import pytest

from stt_daemon.audio.frames import Frame, FrameHub
from stt_daemon.backends.base import PLAYBACK_LEAD_S, PLAYBACK_LOOKAHEAD_S
from stt_daemon.session import reference
from stt_daemon.session.pause_gate import Outgoing
from stt_daemon.session.reference import PlaybackHistory

RATE = 16000
FRAME = 320


def history(*frames: tuple[float, int, float]) -> PlaybackHistory:
    """(start, samples, value) frames, already delivered."""
    hub = FrameHub(RATE, 60_000)
    for start, n, value in frames:
        hub.push(np.full(n, value, np.float32), start + n / RATE)
    return PlaybackHistory(hub.attach(0.0), RATE)


async def taken(playback: PlaybackHistory) -> PlaybackHistory:
    await asyncio.sleep(0.01)
    return playback


def mic(start: float, n: int, *, cut: bool = False) -> Outgoing:
    return Outgoing(Frame(np.zeros(n, np.float32), start + n / RATE, RATE), cut)


async def test_frames_off_the_count_by_jitter_follow_it() -> None:
    playback = await taken(history((10.0, 1600, 1.0), (10.13, 1600, 2.0)))  # 30 ms late
    out = playback.read(10.0, 3200)
    assert (out[:1600] == 1).all()
    assert (out[1600:] == 2).all()
    await playback.close()


async def test_a_jump_moves_the_clock_to_the_stamp() -> None:
    playback = await taken(history((10.0, 1600, 1.0), (10.5, 1600, 2.0)))
    out = playback.read(10.0, 16000 * 7 // 10)
    assert (out[1600:6400] == 0).all()
    assert (out[8000:9600] == 2).all()
    await playback.close()


async def test_forget_drops_only_parts_that_end_before() -> None:
    playback = await taken(history((10.0, 1600, 1.0), (10.1, 1600, 2.0), (10.2, 1600, 3.0)))
    playback.forget_before(10.15)
    out = playback.read(10.0, 4800)
    assert (out[:1600] == 0).all()
    assert (out[1600:3200] == 2).all()
    assert (out[3200:] == 3).all()
    await playback.close()


async def test_cover_waits_for_playback_to_reach_the_time() -> None:
    hub = FrameHub(RATE, 1000)
    now = time.monotonic()
    hub.push(np.zeros(1600, np.float32), now)
    playback = await taken(PlaybackHistory(hub.attach(0.0), RATE))
    waiting = asyncio.ensure_future(playback.cover(now + 0.15))
    await asyncio.sleep(0.05)
    assert not waiting.done()
    hub.push(np.zeros(3200, np.float32), now + 0.2)
    await asyncio.wait_for(waiting, 0.1)
    await playback.close()


async def test_cover_gives_up_shortly_after_the_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reference, "COVER_WAIT_S", 0.1)
    hub = FrameHub(RATE, 1000)
    now = time.monotonic()
    hub.push(np.zeros(1600, np.float32), now)
    playback = await taken(PlaybackHistory(hub.attach(0.0), RATE))
    began = time.monotonic()
    await playback.cover(now + 0.1)
    assert 0.15 <= time.monotonic() - began < 0.3
    await playback.close()


async def test_cover_does_not_wait_on_a_capture_that_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference, "STALE_S", 0.1)
    hub = FrameHub(RATE, 1000)
    hub.push(np.zeros(1600, np.float32), time.monotonic() - 2.0)
    playback = await taken(PlaybackHistory(hub.attach(0.0), RATE))
    await asyncio.sleep(0.1)
    began = time.monotonic()
    await playback.cover(time.monotonic())
    assert time.monotonic() - began < 0.05
    await playback.close()


async def test_cover_waits_for_a_new_capture_s_first_frame() -> None:
    hub = FrameHub(RATE, 1000)
    playback = PlaybackHistory(hub.attach(0.0), RATE)
    now = time.monotonic()
    waiting = asyncio.ensure_future(playback.cover(now + 0.05))
    await asyncio.sleep(0.03)
    assert not waiting.done()
    hub.push(np.zeros(1600, np.float32), now + 0.1)
    await asyncio.wait_for(waiting, 0.1)
    await playback.close()


async def test_old_playback_is_dropped_even_when_no_one_forgets_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference, "KEEP_S", 1.0)
    playback = await taken(history((10.0, 1600, 1.0), (10.1, 16000, 2.0), (11.1, 16000, 3.0)))
    out = playback.read(10.0, 3 * 16000)
    assert not out[:1600].any()
    assert (out[1600 + 16000 : 1600 + 32000] == 3).all()
    await playback.close()


async def test_a_reference_has_a_span_per_stretch_with_the_playback_of_its_time() -> None:
    """Playback values are their clock time; a cut and a capture skip each start a stretch."""
    t = np.arange(round(20.0 * RATE)) / RATE
    hub = FrameHub(RATE, 60_000)
    hub.push(t.astype(np.float32), 20.0)
    playback = await taken(PlaybackHistory(hub.attach(0.0), RATE))
    chunk = [
        mic(5.0, 3200),
        mic(5.2, 3200),  # follows on
        mic(8.0, 3200, cut=True),
        mic(8.2, 1600),
        mic(9.0, 1600),  # the capture skipped
    ]
    ref = reference.reference(chunk, playback)
    assert [(s.offset, s.samples) for s in ref.spans] == [(0, 6400), (6400, 4800), (11200, 1600)]
    for span, start in zip(ref.spans, (5.0, 8.0, 9.0), strict=True):
        assert span.lead == round(PLAYBACK_LEAD_S * RATE)
        assert span.lookahead == round(PLAYBACK_LOOKAHEAD_S * RATE)
        assert span.playback.size == span.lead + span.samples + span.lookahead
        assert span.playback[span.lead] == pytest.approx(start, abs=1 / RATE)
        assert span.playback[0] == pytest.approx(start - PLAYBACK_LEAD_S, abs=1 / RATE)
    await playback.close()
