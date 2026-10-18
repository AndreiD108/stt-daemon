"""The backend link on its own: replay, reconnect, rollover and commits, with numbered samples."""

import asyncio
import time
from dataclasses import replace

import numpy as np
import pytest

from stt_daemon.audio.frames import Frame
from stt_daemon.backends.base import Committed, OpenFailed, OpenOptions
from stt_daemon.backends.fake import FAKE_CAPABILITIES, FakeBackend, FakeConnection, FakeReply
from stt_daemon.session import recovery
from stt_daemon.session.pause_gate import Outgoing
from stt_daemon.session.recovery import Link
from stt_daemon.session.timeline import Timeline

RATE = 16000
FRAME = 320  # 20 ms


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recovery, "RECONNECT_COOLDOWN_S", 0.0)
    monkeypatch.setattr(recovery, "ROLLOVER_ACK_WAIT_S", 0.5)
    monkeypatch.setattr(recovery, "SEND_TIMEOUT_S", 0.3)


class Rig:
    """A link whose audio is sample numbers 0, 1, 2…: a connection's audio shows what it got."""

    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend
        self.timeline = Timeline(0.0, RATE)
        self.committed: list[tuple[str, bool]] = []
        self.errors: list[str] = []
        self.fatal: list[str] = []
        self.queued = 0
        self.link = Link(
            backend=backend,
            options=OpenOptions(previous_text="before"),
            timeline=self.timeline,
            chunk_samples=RATE // 10,
            on_committed=self._committed,
            on_partial=lambda _: None,
            on_error=lambda code, _: self.errors.append(code),
            on_fatal=lambda code, _: self.fatal.append(code),
        )

    def _committed(self, event: Committed, final: bool) -> None:
        self.committed.append((event.text, final))

    def speak(self, ms: int) -> None:
        for _ in range(ms * RATE // 1000 // FRAME):
            samples = np.arange(self.queued, self.queued + FRAME, dtype=np.float32)
            self.queued += FRAME
            self.link.queue(Outgoing(Frame(samples, self.queued / RATE, RATE), cut=False))

    def conn(self, i: int) -> FakeConnection:
        return self.backend.connections[i]

    async def until(self, check: object, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while not (check() if callable(check) else check):
            assert time.monotonic() < deadline, "timed out"
            await asyncio.sleep(0.01)


def numbers(conn: FakeConnection) -> list[int]:
    return [int(x) for x in conn.audio]


def span(first: int, end: int) -> list[int]:
    return list(range(first, end))


async def started(backend: FakeBackend) -> Rig:
    rig = Rig(backend)
    await rig.link.open()
    return rig


async def test_replay_sends_only_the_unanswered_tail_paced() -> None:
    rig = await started(FakeBackend(["one", "two"]))
    rig.speak(300)
    rig.link.commit()
    await rig.until(lambda: rig.committed)
    rig.speak(1000)
    await rig.until(lambda: rig.conn(0).sent_s >= 1.3)
    began = time.monotonic()
    rig.conn(0).fail("network", True)
    await rig.until(lambda: len(rig.backend.connections) == 2 and rig.conn(1).sent_s >= 1.0)
    assert time.monotonic() - began >= 0.45  # 1 s of audio at 2x real time
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    await rig.link.close()
    assert numbers(rig.conn(1)) == span(4800, 20800)
    assert rig.committed == [("one", False), ("two", True)]
    assert rig.conn(1).options.previous_text == "before one"
    assert rig.errors == ["network"]
    assert rig.fatal == []


async def test_audio_during_a_reconnect_waits_for_the_new_connection() -> None:
    rig = await started(FakeBackend(["all of it"]))
    rig.speak(200)
    await rig.until(lambda: rig.conn(0).sent_s >= 0.2)
    rig.backend.open_delay_s = 0.3
    rig.conn(0).fail("network", True)
    for _ in range(10):
        rig.speak(40)
        await asyncio.sleep(0.04)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 3)
    await rig.link.close()
    assert numbers(rig.conn(1)) == span(0, rig.queued)
    assert rig.committed == [("all of it", True)]


async def test_a_commit_answered_keeps_the_audio_queued_after_it() -> None:
    rig = await started(FakeBackend([FakeReply("first", delay_s=0.2), "rest"]))
    rig.speak(200)
    rig.link.commit()
    rig.speak(60)  # under a chunk: stays unsent
    await rig.until(lambda: rig.committed)
    rig.conn(0).fail("network", True)
    await rig.until(lambda: len(rig.backend.connections) == 2)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    await rig.link.close()
    assert numbers(rig.conn(1)) == span(3200, 4160)
    assert rig.committed == [("first", False), ("rest", True)]


async def test_a_failed_replay_is_tried_again() -> None:
    backend = FakeBackend(
        [FakeReply(fail=("network", True)), FakeReply(fail=("rate_limited", True)), "recovered"]
    )
    rig = await started(backend)
    rig.speak(300)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 3)
    await rig.link.close()
    assert len(backend.connections) == 3
    assert numbers(rig.conn(2)) == span(0, 4800)
    assert rig.committed == [("recovered", True)]
    assert rig.errors == ["network", "rate_limited"]
    assert rig.fatal == []


async def test_gives_up_after_three_reconnects() -> None:
    rig = await started(FakeBackend([FakeReply(fail=("network", True))] * 5))
    rig.speak(300)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 3)
    await rig.link.close()
    assert len(rig.backend.connections) == 4
    assert rig.fatal == ["network"]
    assert rig.committed == []


async def test_an_answer_resets_the_attempt_count() -> None:
    fail = FakeReply(fail=("network", True))
    rig = await started(FakeBackend([fail, fail, fail, "a", fail, fail, fail, "b"]))
    rig.speak(300)
    rig.link.commit()
    await rig.until(lambda: rig.committed, 5)
    rig.speak(300)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 5)
    await rig.link.close()
    assert rig.committed == [("a", False), ("b", True)]
    assert rig.fatal == []


async def test_a_failure_that_cant_be_retried_is_fatal_at_once() -> None:
    rig = await started(FakeBackend([FakeReply(fail=("auth", False))]))
    rig.speak(300)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    await rig.link.close()
    assert len(rig.backend.connections) == 1
    assert rig.fatal == ["auth"]
    assert rig.errors == []


async def test_reconnect_open_failures() -> None:
    rig = await started(FakeBackend(["back"]))
    rig.speak(200)
    retry = OpenFailed("network", "down", retryable=True)
    rig.backend.open_failures.extend([retry, retry])
    rig.conn(0).fail("network", True)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    assert rig.committed == [("back", True)]
    assert rig.backend.open_calls == 4

    rig.speak(200)
    rig.backend.open_failures.append(OpenFailed("auth", "bad key", retryable=False))
    rig.conn(1).fail("network", True)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    await rig.link.close()
    assert rig.fatal == ["auth"]


async def test_idle_drop_with_nothing_unanswered_reconnects_at_the_next_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recovery, "RECONNECT_COOLDOWN_S", 0.3)
    rig = await started(FakeBackend(["one", "two"]))
    opened = time.monotonic()
    rig.speak(200)
    rig.link.commit()
    await rig.until(lambda: rig.committed)
    rig.conn(0).fail("idle_drop", True)
    await asyncio.sleep(0.1)
    assert len(rig.backend.connections) == 1
    rig.speak(200)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    await rig.link.close()
    assert time.monotonic() - opened >= 0.3
    assert numbers(rig.conn(1)) == span(3200, 6400)
    assert rig.committed == [("one", False), ("two", True)]
    assert rig.errors == []


async def test_a_send_that_hangs_counts_as_a_lost_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recovery, "RECONNECT_COOLDOWN_S", 0.6)
    rig = await started(FakeBackend(["through"]))
    rig.backend.send_delay_s = 10.0
    rig.speak(200)
    await rig.until(lambda: rig.errors)
    rig.backend.send_delay_s = 0.0
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    await rig.link.close()
    assert rig.errors == ["network"]
    assert numbers(rig.conn(1)) == span(0, 3200)
    assert rig.committed == [("through", True)]


async def test_final_with_nothing_new_marks_the_pending_commit() -> None:
    rig = await started(FakeBackend([FakeReply("slow", delay_s=0.3)]))
    rig.speak(200)
    rig.link.commit()
    await rig.until(lambda: rig.conn(0).commit_ids)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    await rig.link.close()
    assert rig.conn(0).commit_ids == [1]
    assert rig.committed == [("slow", True)]


async def test_no_commit_without_new_audio() -> None:
    rig = await started(FakeBackend(["one"]))
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 1)
    assert rig.conn(0).commit_ids == []
    rig.speak(200)
    rig.link.commit()
    await rig.until(lambda: rig.committed)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 1)
    await rig.link.close()
    assert rig.conn(0).commit_ids == [1]


async def test_settle_waits_for_every_commit_in_order() -> None:
    rig = await started(FakeBackend([FakeReply("first", delay_s=0.3), "second"]))
    rig.speak(200)
    rig.link.commit()
    rig.speak(200)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    await rig.link.close()
    assert rig.committed == [("first", False), ("second", True)]


async def test_a_warning_does_not_answer_a_commit() -> None:
    rig = await started(FakeBackend([FakeReply("late", delay_s=0.4)]))
    rig.speak(200)
    rig.link.commit(final=True)
    await rig.until(lambda: rig.conn(0).commit_ids)
    rig.conn(0).warn("rate_limited")
    settling = asyncio.create_task(rig.link.settle())
    await asyncio.sleep(0.1)
    assert not settling.done()
    await asyncio.wait_for(settling, 2)
    await rig.link.close()
    assert rig.errors == ["rate_limited"]
    assert rig.committed == [("late", True)]


def limited(replies: list[FakeReply | str], limit_s: float = 30.0) -> FakeBackend:
    return FakeBackend(replies, capabilities=replace(FAKE_CAPABILITIES, max_connection_s=limit_s))


@pytest.fixture
def rollover_soon(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a 30 s limit: soft rollover after 0.3 s, hard after 0.8 s."""
    monkeypatch.setattr(recovery, "SOFT_ROLLOVER_MARGIN_S", 29.7)
    monkeypatch.setattr(recovery, "HARD_ROLLOVER_MARGIN_S", 29.2)


@pytest.mark.usefixtures("rollover_soon")
async def test_soft_rollover_waits_for_a_pause_and_the_answer() -> None:
    rig = await started(limited([FakeReply("alpha", delay_s=0.1), "bravo"]))
    rig.speak(200)
    await asyncio.sleep(0.4)
    rig.link.tick(speaking=True)
    await asyncio.sleep(0.05)
    assert len(rig.backend.connections) == 1
    began = time.monotonic()
    rig.link.tick(speaking=False)
    await rig.until(lambda: rig.conn(0).commit_ids)
    rig.speak(100)  # during the swap
    await rig.until(lambda: len(rig.backend.connections) == 2)
    assert time.monotonic() - began < 0.4  # the answer ended the wait early
    rig.speak(100)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    await rig.link.close()
    assert rig.conn(0).ended
    assert numbers(rig.conn(0)) == span(0, 3200)
    assert numbers(rig.conn(1)) == span(3200, 6400)
    assert rig.conn(1).options.previous_text == "before alpha"
    assert rig.committed == [("alpha", False), ("bravo", True)]
    assert rig.errors == []


@pytest.mark.usefixtures("rollover_soon")
async def test_hard_rollover_goes_mid_speech() -> None:
    rig = await started(limited(["alpha", "bravo"]))
    rig.speak(200)
    await asyncio.sleep(0.85)
    rig.link.tick(speaking=True)
    await rig.until(lambda: len(rig.backend.connections) == 2)
    assert rig.committed == [("alpha", False)]


@pytest.mark.usefixtures("rollover_soon")
async def test_rollover_without_an_answer_replays_the_audio() -> None:
    rig = await started(limited([FakeReply("never", delay_s=5.0), "replayed"]))
    rig.speak(200)
    await asyncio.sleep(0.4)
    rig.link.tick(speaking=False)
    await rig.until(lambda: rig.conn(0).commit_ids)
    rig.speak(100)
    while len(rig.backend.connections) < 2:
        rig.link.tick(speaking=False)  # still due while the old connection is waited on
        await asyncio.sleep(0.02)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    await rig.link.close()
    assert len(rig.backend.connections) == 2
    assert numbers(rig.conn(1)) == span(0, 4800)
    assert rig.conn(1).commit_ids == [1, 2]  # the rollover's commit, replayed, then the final one
    assert rig.committed == [("replayed", False), ("", True)]


@pytest.mark.usefixtures("rollover_soon")
async def test_rollover_then_replay_keeps_the_text_before_the_seam() -> None:
    rig = await started(limited(["alpha", FakeReply(fail=("network", True)), "bravo"]))
    rig.speak(200)
    await asyncio.sleep(0.4)
    rig.link.tick(speaking=False)
    await rig.until(lambda: len(rig.backend.connections) == 2)
    rig.speak(200)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 3)
    await rig.link.close()
    assert numbers(rig.conn(2)) == span(3200, 6400)
    assert rig.committed == [("alpha", False), ("bravo", True)]
    assert rig.conn(2).options.previous_text == "before alpha"


async def test_the_backend_time_limit_is_a_retried_failure() -> None:
    rig = await started(limited(["after"], limit_s=0.3))
    rig.speak(200)
    await rig.until(lambda: rig.conn(0).ended)
    rig.link.commit(final=True)
    await asyncio.wait_for(rig.link.settle(), 2)
    await rig.link.close()
    assert rig.errors == ["session_limit"]
    assert rig.committed == [("after", True)]
