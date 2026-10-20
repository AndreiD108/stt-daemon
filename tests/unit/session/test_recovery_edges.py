"""Recovery boundaries: close-time text, unknown cuts, a failure after the last answer, and the
reference a replay sends."""

import asyncio

import pytest

from stt_daemon.backends.base import Committed, OpenOptions, Reference
from stt_daemon.backends.fake import FakeBackend, FakeReply
from stt_daemon.session import recovery
from stt_daemon.session.recovery import Link
from stt_daemon.session.timeline import Timeline
from unit.session.test_recovery import Rig, limited, numbers, span, started
from unit.session.test_reference import FRAME, RATE, history, mic, taken


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recovery, "RECONNECT_COOLDOWN_S", 0.0)
    monkeypatch.setattr(recovery, "ROLLOVER_ACK_WAIT_S", 0.05)
    monkeypatch.setattr(recovery, "SOFT_ROLLOVER_MARGIN_S", 30.0)


async def test_fatal_after_final_answer_is_seen_before_settle_returns() -> None:
    rig = await started(FakeBackend([FakeReply("kept", then_fail=("quota", False))]))
    rig.speak(200)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    assert rig.fatal == ["quota"]
    assert rig.committed == [("kept", True)]
    await rig.link.close()


async def test_close_time_ack_is_kept_before_later_replay() -> None:
    backend = limited([FakeReply("trailing", delay_s=10), "tail"])
    backend.answer_on_close = True
    rig = await started(backend)
    rig.speak(200)
    rig.link.tick(speaking=False)
    await rig.until(lambda: len(backend.connections) == 2)
    assert rig.committed == [("trailing", False)]
    assert rig.conn(1).options.previous_text == "before trailing"
    assert rig.conn(1).audio.size == 0
    rig.speak(200)
    await rig.until(lambda: rig.conn(1).sent_s >= 0.2)
    rig.conn(1).fail("network", True)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    assert numbers(rig.conn(2)) == span(3200, 6400)
    assert rig.committed == [("trailing", False), ("tail", True)]
    await rig.link.close()


async def unknown_cut(rig: Rig) -> None:
    rig.speak(400)
    await rig.until(lambda: rig.conn(0).sent_s >= 0.4)
    rig.conn(0)._sink(Committed(1, None, "first", None, (), 0.0))  # pyright: ignore[reportPrivateUsage]
    assert rig.committed == []


async def test_unknown_cut_is_not_published_twice_after_reconnect() -> None:
    rig = await started(FakeBackend(["first rest"]))
    await unknown_cut(rig)
    rig.conn(0).fail("network", True)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    assert numbers(rig.conn(1)) == span(0, 6400)
    assert rig.committed == [("first rest", True)]
    await rig.link.close()


async def test_known_ack_releases_unknown_cut_text_in_order() -> None:
    rig = await started(FakeBackend(["rest"]))
    await unknown_cut(rig)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    assert rig.committed == [("first", False), ("rest", True)]
    rig.conn(0).fail("network", True)
    await asyncio.sleep(0.05)
    assert len(rig.backend.connections) == 1
    await rig.link.close()


async def test_final_coalesces_with_an_inflight_periodic_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = await started(FakeBackend(["once"]))
    entered, release = asyncio.Event(), asyncio.Event()
    commit = rig.conn(0).commit

    async def blocked_commit(reference: Reference | None = None) -> int:
        entered.set()
        await release.wait()
        return await commit(reference)

    monkeypatch.setattr(rig.conn(0), "commit", blocked_commit)
    rig.speak(200)
    rig.link.commit()
    await asyncio.wait_for(entered.wait(), 2)
    rig.link.commit(final=True)
    release.set()
    await asyncio.wait_for(rig.link.settle(), 2)
    assert rig.conn(0).commit_ids == [1]
    assert rig.committed == [("once", True)]
    await rig.link.close()


async def test_a_replay_sends_the_reference_built_at_the_first_send() -> None:
    playback = await taken(history((0.0, RATE * 2, 0.5)))
    backend = FakeBackend([FakeReply(fail=("network", True)), "after"])
    committed: list[str] = []
    link = Link(
        backend=backend,
        options=OpenOptions(),
        timeline=Timeline(0.0, RATE),
        chunk_samples=RATE // 10,
        on_committed=lambda event, _: committed.append(event.text),
        on_partial=lambda _: None,
        on_error=lambda _code, _message: None,
        on_fatal=lambda _code, _message: None,
        playback=playback,
    )
    await link.open()
    for i in range(15):
        link.queue(mic(1.0 + i * FRAME / RATE, FRAME))
    link.commit()
    async with asyncio.timeout(3):
        while not committed:
            await asyncio.sleep(0.01)
    first, second = backend.connections
    [sent] = first.references
    [replayed] = second.references
    assert sent is not None
    assert replayed is sent
    assert sent.spans[0].playback[0] == 0.5
    await link.close()
    await playback.close()
