"""Echo cancellation over the socket: a stand-in `pw-cli` announces the module's source through the
stand-in `pw-dump`; the shared mic records it with the stand-in `pw-record`."""

import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

import aec_fixture
import fake_pw_cli
import numpy as np
from fake_pw_dump import FakePipeWire
from harness import FIXTURE, Harness, Msg, TestClient

from stt_daemon.audio.aec_pipewire import Command, pw_cli_command
from stt_daemon.audio.pw_capture import PwRecordCapture
from stt_daemon.audio.system_capture import AppTarget
from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.base import CommitPolicy
from stt_daemon.backends.fake import FAKE_CAPABILITIES, FakeBackend
from stt_daemon.config import AudioConfig, Config
from stt_daemon.server.sources import PwDumpScan

FAKE_PW_RECORD = str(Path(__file__).parents[1] / "fake_pw_record.py")
PAUSES = replace(FAKE_CAPABILITIES, commit_policy=CommitPolicy(pause_steps=((0, 0.6),)))
MIC_NOISE_DB = -60.0


class Setup:
    def __init__(self, root: Path) -> None:
        self.graph = FakePipeWire(root / "pw")
        self.cli_log = root / "pw-cli.log"
        self.record_log = root / "pw-record.log"

    def system_capture(self, target: AppTarget | None) -> PwRecordCapture:
        """The system track plays the fixture; the echo-cancelled source plays it over and over."""
        if target is None:
            return PwRecordCapture([sys.executable, FAKE_PW_RECORD, "--wav", str(FIXTURE)])
        live, log = str(self.graph.live), str(self.record_log)
        return PwRecordCapture(
            lambda serial: [
                *(sys.executable, FAKE_PW_RECORD, "--wav", str(FIXTURE), "--loop", "--live", live),
                *("--log", log, "--target", str(serial)),
            ],
            target=target,
        )

    def aec_command(self, *options: str) -> Command:
        def command(name: str) -> list[str]:
            prefix = fake_pw_cli.command(
                "--dir", str(self.graph.root), "--log", str(self.cli_log), *options
            )
            return [*prefix, *pw_cli_command(name)[1:]]

        return command

    def cli_lines(self) -> list[str]:
        return self.cli_log.read_text().splitlines() if self.cli_log.exists() else []


async def start(
    daemon: Harness, setup: Setup, *options: str, mic: WavCapture | None = None
) -> None:
    sources = PwDumpScan(setup.graph.command(), enabled=True)
    await daemon.start(
        FakeBackend([f"text {i}" for i in range(8)], capabilities=PAUSES),
        capture=mic,
        system_capture=setup.system_capture,
        sources=sources,
        aec_command=setup.aec_command(*options),
    )


def track_segments(client: TestClient, track: str) -> list[Msg]:
    return [m for m in client.of_type("segment") if m["data"]["track"] == track]


def longest_gap(spans: list[tuple[int, int]], start: int, end: int) -> int:
    """The longest stretch of [start, end] ms no span covers."""
    longest, at = 0, start
    for s, e in sorted(spans):
        longest = max(longest, min(s, end) - at)
        at = max(at, e)
    return max(longest, end - at)


def aec_errors(client: TestClient) -> list[Msg]:
    return [m for m in client.of_type("session_error") if m["data"]["code"] == "aec_unavailable"]


def module_name() -> str:
    return f"stt-daemon-aec-{os.getpid()}"


async def test_the_module_loads_at_start_and_every_mic_consumer_takes_its_source(
    daemon: Harness, tmp_path: Path
) -> None:
    setup = Setup(tmp_path)
    await start(daemon, setup)
    name = module_name()
    assert setup.cli_lines() == [f"start {name}"]
    talkey, _ = await daemon.connect("Talkey")
    await talkey.request("start_session")
    # The plain mic plays nothing: segments come from the cleaned source only.
    await talkey.wait(lambda _m: len(talkey.of_type("segment")) >= 2, timeout=15)
    await talkey.request("stop_session")
    scribbly, _ = await daemon.connect("Scribbly")
    started = await scribbly.request("start_system_session", {"with_mic": True})
    assert started["data"]["tracks"] == ["system", "mic"], started
    await scribbly.wait(lambda _m: len(track_segments(scribbly, "mic")) >= 1, timeout=15)
    await scribbly.request("stop_session")
    assert aec_errors(talkey) == aec_errors(scribbly) == []
    serial = setup.graph.live.read_text().split()
    assert setup.record_log.read_text().splitlines() == [f"start {serial[0]}"]
    assert setup.cli_lines() == [f"start {name}"]  # no second module
    await daemon.stop()
    assert setup.cli_lines() == [f"start {name}", f"end {name}"]


async def test_with_the_mic_opened_per_session_each_start_loads_the_module(
    daemon: Harness, tmp_path: Path
) -> None:
    setup = Setup(tmp_path)
    await start(daemon, setup, mic=WavCapture(keep_open=False))
    assert setup.cli_lines() == []
    talkey, _ = await daemon.connect("Talkey")
    name = module_name()
    for round_ in (1, 2):
        await talkey.request("start_session")
        assert setup.cli_lines()[-1] == f"start {name}"
        await talkey.wait(lambda _m, n=round_: len(talkey.of_type("segment")) >= n, timeout=15)
        await talkey.request("stop_session")
        assert setup.cli_lines() == [f"start {name}", f"end {name}"] * round_
    assert aec_errors(talkey) == []


async def test_without_the_module_sessions_run_on_the_plain_mic_and_say_so(
    daemon: Harness, tmp_path: Path
) -> None:
    setup = Setup(tmp_path)
    mic = WavCapture()
    await start(daemon, setup, "--fail", mic=mic)
    talkey, _ = await daemon.connect("Talkey")
    await talkey.request("start_session")
    mic.play(FIXTURE)
    await talkey.wait(lambda _m: len(talkey.of_type("segment")) == 2, timeout=15)
    await talkey.request("stop_session")
    errors = aec_errors(talkey)
    assert len(errors) == 1, errors
    assert errors[0]["data"]["fatal"] is False
    assert errors[0]["data"].get("track") is None
    assert "Could not load module" in errors[0]["data"]["message"]
    scribbly, _ = await daemon.connect("Scribbly")
    await scribbly.request("start_system_session", {"with_mic": True})
    await scribbly.request("stop_session")
    assert [e["data"]["track"] for e in aec_errors(scribbly)] == ["mic"]
    assert not setup.record_log.exists()  # the cleaned source was never recorded


async def test_pw_cli_ending_mid_session_moves_the_mic_to_the_plain_mic(
    daemon: Harness, tmp_path: Path
) -> None:
    setup = Setup(tmp_path)
    mic = WavCapture()
    await start(daemon, setup, "--exit-after", "2", mic=mic)
    talkey, _ = await daemon.connect("Talkey")
    await talkey.request("start_session")
    error = await talkey.event("session_error", code="aec_unavailable")
    assert "pw-cli exited with code 1" in error["data"]["message"]
    before = len(talkey.of_type("segment"))
    mic.play(FIXTURE)
    await talkey.wait(lambda _m: len(talkey.of_type("segment")) >= before + 2, timeout=15)
    await talkey.request("stop_session")
    assert len(aec_errors(talkey)) == 1


async def test_a_mic_named_in_audio_device_keeps_the_plain_mic(
    daemon: Harness, tmp_path: Path
) -> None:
    setup = Setup(tmp_path)
    mic = WavCapture()
    await daemon.start(
        FakeBackend([f"text {i}" for i in range(8)], capabilities=PAUSES),
        capture=mic,
        config=Config(audio=AudioConfig(device="Kiyo")),
        system_capture=setup.system_capture,
        sources=PwDumpScan(setup.graph.command(), enabled=True),
        aec_command=setup.aec_command(),
    )
    talkey, _ = await daemon.connect("Talkey")
    await talkey.request("start_session")
    mic.play(FIXTURE)
    await talkey.wait(lambda _m: len(talkey.of_type("segment")) == 2, timeout=15)
    await talkey.request("stop_session")
    assert setup.cli_lines() == []


async def test_with_echo_cancellation_off_the_module_never_loads(
    daemon: Harness, tmp_path: Path
) -> None:
    setup = Setup(tmp_path)
    mic = WavCapture()
    await daemon.start(
        FakeBackend([f"text {i}" for i in range(8)], capabilities=PAUSES),
        capture=mic,
        config=Config(audio=AudioConfig(echo_cancel="off")),
        system_capture=setup.system_capture,
        sources=PwDumpScan(setup.graph.command(), enabled=True),
        aec_command=setup.aec_command(),
    )
    talkey, _ = await daemon.connect("Talkey")
    await talkey.request("start_session")
    mic.play(FIXTURE)
    await talkey.wait(lambda _m: len(talkey.of_type("segment")) == 2, timeout=15)
    await talkey.request("stop_session")
    assert setup.cli_lines() == []
    assert aec_errors(talkey) == []


async def test_the_built_in_canceller_keeps_the_speakers_out_of_the_mic_track(
    daemon: Harness, tmp_path: Path
) -> None:
    clip = slice(0, round(14.0 * aec_fixture.RATE))
    fixture = aec_fixture.build()
    far, mixed = tmp_path / "far.wav", tmp_path / "mixed.wav"
    aec_fixture.write(far, fixture.far[clip])
    # A mic's noise floor: without one, the VAD wavers on the faint residual echo.
    noise = np.random.default_rng(1).standard_normal(clip.stop) * 10 ** (MIC_NOISE_DB / 20)
    aec_fixture.write(mixed, fixture.mixed[clip] + noise)
    mic = WavCapture()
    config = Config(audio=AudioConfig(echo_cancel="builtin"))
    await daemon.start(
        FakeBackend([f"text {i}" for i in range(20)], capabilities=PAUSES),
        capture=mic,
        config=config,
        system_capture=lambda _target: PwRecordCapture(
            [sys.executable, FAKE_PW_RECORD, "--wav", str(far)]
        ),
    )
    scribbly, _ = await daemon.connect("Scribbly")
    started = await scribbly.request("start_system_session", {"with_mic": True})
    assert started["data"]["tracks"] == ["system", "mic"], started
    mic.play(mixed)
    await asyncio.sleep(14.5)
    await scribbly.request("stop_session")
    spans = [(m["data"]["start_ms"], m["data"]["end_ms"]) for m in track_segments(scribbly, "mic")]
    near_start = round(aec_fixture.NEAR_PHRASES[0][0] * 1000)
    # The speakers play with no pause the VAD would hear from 3.1 s to the near end's phrase; the
    # filter converges in the first seconds, so the mic track falls silent before that phrase.
    assert longest_gap(spans, 5000, near_start) >= 1000, spans
    # The phrase's segment may start with the speakers' phrase it overlaps, at 8.8 s.
    assert [s for s in spans if 8700 <= s[0] <= near_start + 300], spans
    assert aec_errors(scribbly) == []
