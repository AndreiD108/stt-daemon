import asyncio
import threading
import time

import numpy as np
import pytest

from stt_daemon.audio.frames import Attachment, Frame, FrameHub

RATE = 16000
CHUNK = 512
CHUNK_S = CHUNK / RATE


def _push(hub: FrameHub, value: float, end: float, n: int = CHUNK) -> None:
    hub.push(np.full(n, value, dtype=np.float32), end)


async def _drain(queue: asyncio.Queue[Frame]) -> list[Frame]:
    await asyncio.sleep(0)  # let call_soon_threadsafe deliveries land
    out: list[Frame] = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def _values(frames: list[Frame]) -> list[float]:
    return [round(float(f.samples[0]), 3) for f in frames]


async def test_attach_prepends_ring_in_order_then_live() -> None:
    hub = FrameHub(RATE, 1000)
    for i, value in enumerate((0.1, 0.2, 0.3)):
        _push(hub, value, 10.0 + (i + 1) * CHUNK_S)
    attachment = hub.attach()
    assert attachment.prepended_samples == 3 * CHUNK
    assert attachment.started_at == pytest.approx(10.0)
    _push(hub, 0.4, 10.0 + 4 * CHUNK_S)
    assert _values(await _drain(attachment.queue)) == [0.1, 0.2, 0.3, 0.4]


async def test_attach_since_trims_the_boundary_frame() -> None:
    hub = FrameHub(RATE, 1000)
    for i, value in enumerate((0.1, 0.2, 0.3)):
        _push(hub, value, 10.0 + (i + 1) * CHUNK_S)
    since = 10.0 + 1.5 * CHUNK_S
    attachment = hub.attach(since)
    frames = await _drain(attachment.queue)
    assert [f.samples.size for f in frames] == [CHUNK // 2, CHUNK]
    assert _values(frames) == [0.2, 0.3]
    assert attachment.started_at == pytest.approx(since)
    assert attachment.prepended_samples == CHUNK // 2 + CHUNK


async def test_since_before_the_ring_keeps_everything() -> None:
    hub = FrameHub(RATE, 1000)
    _push(hub, 0.1, 10.0)
    _push(hub, 0.2, 10.0 + CHUNK_S)
    assert hub.attach(since=0.0).prepended_samples == 2 * CHUNK


async def test_ring_keeps_filling_while_attached() -> None:
    hub = FrameHub(RATE, 1000)
    holder = hub.attach()
    _push(hub, 0.5, 10.0)
    _push(hub, 0.6, 10.0 + CHUNK_S)
    evicting = hub.attach(since=10.0 - CHUNK_S)
    assert _values(await _drain(evicting.queue)) == [0.5, 0.6]
    assert _values(await _drain(holder.queue)) == [0.5, 0.6]


async def test_every_attachment_gets_live_frames_until_detached() -> None:
    hub = FrameHub(RATE, 1000)
    first, second = hub.attach(), hub.attach()
    _push(hub, 0.1, 10.0)
    first.detach()
    _push(hub, 0.2, 10.0 + CHUNK_S)
    assert _values(await _drain(first.queue)) == [0.1]
    assert _values(await _drain(second.queue)) == [0.1, 0.2]


def test_ring_drops_the_oldest_audio_past_its_length() -> None:
    hub = FrameHub(RATE, 1000)
    count = 2 * RATE // CHUNK + 4
    for i in range(count):
        _push(hub, (i + 1) / 1000, 10.0 + (i + 1) * CHUNK_S)
    ring = hub.ring_since()
    kept = sum(f.samples.size for f in ring)
    assert RATE - CHUNK <= kept <= RATE
    assert _values(ring)[-1] == round(count / 1000, 3)
    assert hub.ring_start() == pytest.approx(ring[0].start)


def test_ring_since_is_a_copy() -> None:
    hub = FrameHub(RATE, 1000)
    _push(hub, 0.1, 10.0)
    copy = hub.ring_since()
    hub.clear()
    assert _values(copy) == [0.1]
    assert hub.ring_since() == []
    assert hub.ring_start() is None


def test_pushed_samples_are_read_only() -> None:
    hub = FrameHub(RATE, 1000)
    _push(hub, 0.1, 10.0)
    with pytest.raises(ValueError, match="read-only"):
        hub.ring_since()[0].samples[0] = 1.0


async def test_attach_while_pushing_sees_each_frame_once() -> None:
    hub = FrameHub(RATE, 60_000)
    stop = threading.Event()

    def produce() -> None:
        i = 0
        while not stop.is_set():
            i += 1
            hub.push(np.full(16, float(i), dtype=np.float32), 10.0 + i)
            time.sleep(0.0001)

    producer = threading.Thread(target=produce)
    producer.start()
    try:
        attachments: list[Attachment] = []
        for _ in range(50):
            attachments.append(hub.attach())
            await asyncio.sleep(0.001)
    finally:
        stop.set()
        producer.join()
    await asyncio.sleep(0.05)
    for attachment in attachments:
        seen = [int(f.samples[0]) for f in await _drain(attachment.queue)]
        assert seen == list(range(seen[0], seen[0] + len(seen)))
