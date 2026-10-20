"""The shared mic on the echo-cancel source, with stand-ins: when the module loads and unloads, the
plain-mic fallback and the switch back, watchers, settle."""

import asyncio
import itertools
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from stt_daemon.audio import aec_mic
from stt_daemon.audio.aec_mic import EchoCancelledMic
from stt_daemon.audio.aec_pipewire import PipewireAec
from stt_daemon.audio.frames import Frame
from stt_daemon.audio.pw_capture import PwRecordCapture
from stt_daemon.audio.system_capture import AppTarget
from unit.audio.aec_stand_ins import (
    MIC_LEVEL,
    NAME,
    FakeNodes,
    FlakyMic,
    PushedMic,
    cleaned_capture,
    levels,
    pw_cli,
    pw_cli_log,
    until,
)
from unit.paths import FAKE_PW_RECORD


def shared(
    mic: PushedMic,
    tmp_path: Path,
    *options: str,
    nodes: FakeNodes | None = None,
    keep_open: bool = False,
    targets: list[str] | None = None,
    cleaned: Callable[[AppTarget], PwRecordCapture] | None = None,
) -> EchoCancelledMic:
    if nodes is None:
        nodes = FakeNodes()
        nodes.serials[NAME] = "777"
    found = nodes

    def make_aec() -> PipewireAec:
        return PipewireAec(NAME, nodes=found, command=pw_cli(tmp_path, *options))

    return EchoCancelledMic(
        mic,
        make_aec,
        cleaned=cleaned or cleaned_capture(tmp_path, [] if targets is None else targets),
        buffer_ms=5000,
        keep_open=keep_open,
    )


class Watched:
    def __init__(self, source: EchoCancelledMic) -> None:
        self.events: list[str] = []
        self.unwatch = source.watch(
            on_lost=lambda reason: self.events.append(f"lost: {reason}"),
            on_restored=lambda: self.events.append("restored"),
            on_unavailable=lambda message: self.events.append(f"unavailable: {message}"),
        )

    @property
    def unavailable(self) -> list[str]:
        return [e for e in self.events if e.startswith("unavailable")]


async def frames(source: EchoCancelledMic, count: int) -> list[Frame]:
    attachment = source.attach(time.monotonic())
    out: list[Frame] = []
    async with asyncio.timeout(5):
        while len(out) < count:
            out.append(await attachment.queue.get())
    attachment.detach()
    return out


async def test_the_source_records_the_cleaned_source(tmp_path: Path) -> None:
    mic = PushedMic()
    mic.run()
    targets: list[str] = []
    source = shared(mic, tmp_path, targets=targets)
    watched = Watched(source)
    await source.hold()
    attachment = source.attach(time.monotonic())
    assert await levels(attachment, 5) == [0.0] * 5
    assert targets == ["777"]
    assert source.cleaned
    assert mic.holds == 0
    assert watched.events == []
    await source.release()
    mic.stop_running()
    assert pw_cli_log(tmp_path) == [f"start {NAME}", f"end {NAME}"]


async def test_kept_open_the_module_loads_once_at_start_and_unloads_at_stop(tmp_path: Path) -> None:
    mic = PushedMic()
    mic.run()
    source = shared(mic, tmp_path, keep_open=True)
    await source.start()
    await until(lambda: pw_cli_log(tmp_path) == [f"start {NAME}"])
    for _ in range(2):
        await source.hold()
        assert source.cleaned
        assert set(await levels(source.attach(time.monotonic()), 3)) == {0.0}
        await source.release()
    assert source.cleaned
    assert pw_cli_log(tmp_path) == [f"start {NAME}"]
    await source.stop()
    mic.stop_running()
    assert pw_cli_log(tmp_path) == [f"start {NAME}", f"end {NAME}"]


async def test_not_kept_open_the_module_loads_per_open(tmp_path: Path) -> None:
    mic = PushedMic()
    mic.run()
    source = shared(mic, tmp_path)
    await source.start()
    assert pw_cli_log(tmp_path) == []
    await source.hold()
    await source.hold()
    # Before any release: a pw-cli ended before it logs its start leaves no line.
    await until(lambda: len(pw_cli_log(tmp_path)) == 1)
    await source.release()
    assert source.cleaned
    await source.release()
    assert not source.cleaned
    await source.hold()
    await until(lambda: len(pw_cli_log(tmp_path)) == 3)
    await source.release()
    mic.stop_running()
    assert pw_cli_log(tmp_path) == [f"start {NAME}", f"end {NAME}"] * 2


async def test_without_the_module_the_plain_mic_is_used_and_said_once(tmp_path: Path) -> None:
    mic = PushedMic()
    mic.run()
    await asyncio.sleep(0.1)
    source = shared(mic, tmp_path, "--fail", nodes=FakeNodes())
    watched = Watched(source)
    await source.hold()
    origin = time.monotonic() - 0.05
    attachment = source.attach(origin)
    first = await attachment.queue.get()
    assert first.start >= origin - 0.021  # the ring, from the asked point on
    assert set(await levels(attachment, 5)) == {MIC_LEVEL}
    assert not source.cleaned
    assert mic.holds == 1
    assert len(watched.unavailable) == 1
    assert "Could not load module" in watched.unavailable[0]
    assert source.unavailable_reason is not None
    await source.release()
    assert source.unavailable_reason is None
    mic.stop_running()
    assert mic.holds == 0


async def test_pw_cli_ending_switches_to_the_plain_mic_without_going_back_in_time(
    tmp_path: Path,
) -> None:
    mic = PushedMic()
    mic.run()
    source = shared(mic, tmp_path, "--exit-after", "0.5")
    watched, gone = Watched(source), Watched(source)
    gone.unwatch()
    await source.hold()
    attachment = source.attach(time.monotonic())
    assert await levels(attachment, 3) == [0.0] * 3
    await until(lambda: bool(watched.unavailable))
    seen: list[Frame] = []
    async with asyncio.timeout(5):
        while len(seen) < 40:
            seen.append(await attachment.queue.get())
    assert [float(f.samples.max()) for f in seen[-5:]] == [MIC_LEVEL] * 5
    for before, after in itertools.pairwise(seen):
        assert after.start >= before.end - 1 / 16000  # trimmed to whole samples
    assert not source.cleaned
    assert mic.holds == 1
    assert len(watched.unavailable) == 1
    assert "pw-cli exited" in watched.unavailable[0]
    assert gone.events == []
    await source.release()
    mic.stop_running()
    assert mic.holds == 0


async def test_kept_open_a_module_that_could_not_load_is_loaded_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aec_mic, "AEC_RETRY_S", (0.2,))
    mic = PushedMic()
    mic.run()
    nodes = FakeNodes()
    nodes.serials[NAME] = "777"
    tries: list[str] = []

    def make_aec() -> PipewireAec:
        tries.append("try")
        if len(tries) == 1:
            return PipewireAec(NAME, nodes=FakeNodes(), command=pw_cli(tmp_path, "--fail"))
        return PipewireAec(NAME, nodes=nodes, command=pw_cli(tmp_path))

    source = EchoCancelledMic(
        mic, make_aec, cleaned=cleaned_capture(tmp_path, []), buffer_ms=5000, keep_open=True
    )
    watched = Watched(source)
    await source.start()
    assert not source.cleaned
    assert mic.holds == 1
    assert len(watched.unavailable) == 1
    await until(lambda: source.cleaned)
    assert mic.holds == 0
    assert source.unavailable_reason is None
    assert {float(f.samples.max()) for f in await frames(source, 3)} == {0.0}
    await source.stop()
    mic.stop_running()
    assert len(tries) == 2


async def test_kept_open_a_module_that_ended_is_loaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aec_mic, "AEC_RETRY_S", (0.2,))
    mic = PushedMic()
    mic.run()
    nodes = FakeNodes()
    nodes.serials[NAME] = "777"
    loads: list[str] = []

    def make_aec() -> PipewireAec:
        loads.append("load")
        options = ("--exit-after", "0.3") if len(loads) == 1 else ()
        return PipewireAec(NAME, nodes=nodes, command=pw_cli(tmp_path, *options))

    source = EchoCancelledMic(
        mic, make_aec, cleaned=cleaned_capture(tmp_path, []), buffer_ms=5000, keep_open=True
    )
    watched = Watched(source)
    await source.start()
    await until(lambda: bool(watched.unavailable))
    assert not source.cleaned
    await until(lambda: source.cleaned)
    assert mic.holds == 0
    await source.stop()
    mic.stop_running()
    assert len(loads) == 2


async def test_stop_cancels_a_pending_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aec_mic, "AEC_RETRY_S", (0.1,))
    mic = PushedMic()
    mic.run()
    source = shared(mic, tmp_path, "--fail", nodes=FakeNodes(), keep_open=True)
    await source.start()
    await asyncio.sleep(0.3)
    await source.stop()
    mic.stop_running()
    assert mic.holds == 0
    lines = pw_cli_log(tmp_path)
    await asyncio.sleep(0.3)
    assert pw_cli_log(tmp_path) == lines  # no try after the stop


class SlowStart(PwRecordCapture):
    async def hold(self) -> None:
        await asyncio.sleep(0.5)
        await super().hold()


async def test_pw_cli_ending_while_the_cleaned_capture_starts_falls_back(tmp_path: Path) -> None:
    mic = PushedMic()
    mic.run()

    def slow(target: AppTarget) -> PwRecordCapture:
        return SlowStart(lambda s: [sys.executable, str(FAKE_PW_RECORD), "--target", str(s)])

    source = shared(mic, tmp_path, "--exit-after", "0.05", cleaned=slow)
    watched = Watched(source)
    await source.hold()
    attachment = source.attach(time.monotonic())
    await asyncio.sleep(0.3)
    assert set(await levels(attachment, 5)) == {MIC_LEVEL}
    assert not source.cleaned
    assert mic.holds == 1
    assert len(watched.unavailable) == 1
    assert "pw-cli exited" in watched.unavailable[0]
    await source.release()
    mic.stop_running()


async def test_a_mic_that_fails_to_open_after_the_switch_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aec_mic, "MIC_RETRY_S", (0.05,))
    mic = FlakyMic(failures=2)
    mic.run()
    source = shared(mic, tmp_path, "--exit-after", "0.3")
    watched = Watched(source)
    await source.hold()
    attachment = source.attach(time.monotonic())
    await until(lambda: "restored" in watched.events)
    lost_and_back = [e for e in watched.events if not e.startswith("unavailable")]
    assert lost_and_back == ["lost: the mic did not open: the mic is busy", "restored"]
    assert mic.tries == 3
    seen = await levels(attachment, 40)
    assert seen[-5:] == [MIC_LEVEL] * 5
    await source.release()
    mic.stop_running()
    assert mic.holds == 0


async def test_settle_waits_for_audio_up_to_the_point_and_no_longer_than_its_bound(
    tmp_path: Path,
) -> None:
    mic = PushedMic()
    mic.run()
    source = shared(mic, tmp_path, "--fail", nodes=FakeNodes())
    await source.hold()
    attachment = source.attach(time.monotonic())
    until_now = time.monotonic()
    await source.settle(until_now)
    pushed = [attachment.queue.get_nowait() for _ in range(attachment.queue.qsize())]
    assert pushed[-1].end >= until_now
    mic.stop_running()
    began = time.monotonic()
    await source.settle(time.monotonic() + 1.0)
    assert aec_mic.SETTLE_S - 0.05 <= time.monotonic() - began < aec_mic.SETTLE_S + 0.2
    await source.release()
