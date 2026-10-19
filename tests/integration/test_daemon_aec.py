"""Echo cancellation over the socket: a stand-in `pw-cli` announces the module's source through the
stand-in `pw-dump`; the mic track records it with the stand-in `pw-record`."""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import fake_pw_cli
from fake_pw_dump import FakePipeWire
from harness import FIXTURE, Harness, Msg, TestClient

from stt_daemon.audio.aec_pipewire import Command, pw_cli_command
from stt_daemon.audio.pw_capture import AppTarget, PwRecordCapture
from stt_daemon.audio.wav_capture import WavCapture
from stt_daemon.backends.base import CommitPolicy
from stt_daemon.backends.fake import FAKE_CAPABILITIES, FakeBackend
from stt_daemon.server.sources import SourceScan

FAKE_PW_RECORD = str(Path(__file__).parents[1] / "fake_pw_record.py")
PAUSES = replace(FAKE_CAPABILITIES, commit_policy=CommitPolicy(at_pauses=True))


class Setup:
    def __init__(self, root: Path) -> None:
        self.graph = FakePipeWire(root / "pw")
        self.cli_log = root / "pw-cli.log"
        self.record_log = root / "pw-record.log"

    def system_capture(self, target: AppTarget | None) -> PwRecordCapture:
        """The system track plays the fixture; so does the echo-cancelled source."""
        if target is None:
            return PwRecordCapture([sys.executable, FAKE_PW_RECORD, "--wav", str(FIXTURE)])
        live, log = str(self.graph.live), str(self.record_log)
        return PwRecordCapture(
            lambda serial: [
                *(sys.executable, FAKE_PW_RECORD, "--wav", str(FIXTURE), "--live", live),
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
    sources = SourceScan(setup.graph.command(), enabled=True)
    await daemon.start(
        FakeBackend([f"text {i}" for i in range(8)], capabilities=PAUSES),
        capture=mic,
        system_capture=setup.system_capture,
        sources=sources,
        aec_command=setup.aec_command(*options),
    )


def track_segments(client: TestClient, track: str) -> list[Msg]:
    return [m for m in client.of_type("segment") if m["data"]["track"] == track]


def aec_errors(client: TestClient) -> list[Msg]:
    return [m for m in client.of_type("session_error") if m["data"]["code"] == "aec_unavailable"]


async def test_the_mic_track_records_the_echo_cancelled_source(
    daemon: Harness, tmp_path: Path
) -> None:
    setup = Setup(tmp_path)
    await start(daemon, setup)
    scribbly, _ = await daemon.connect("Scribbly")
    started = await scribbly.request("start_system_session", {"with_mic": True})
    assert started["data"]["tracks"] == ["system", "mic"], started
    name = f"stt-daemon-aec-{started['data']['session_id']}"
    assert setup.cli_lines() == [f"start {name}"]
    # The plain mic plays nothing: mic segments come from the cleaned source only.
    await scribbly.wait(lambda _m: len(track_segments(scribbly, "mic")) == 2, timeout=15)
    serial = setup.graph.live.read_text().split()
    assert setup.record_log.read_text().splitlines() == [f"start {serial[0]}"]
    ended = await scribbly.request("stop_session")
    assert ended["data"]["reason"] == "stopped", ended
    assert aec_errors(scribbly) == []
    assert setup.cli_lines() == [f"start {name}", f"end {name}"]
    assert setup.graph.live.read_text() == ""


async def test_without_the_module_the_session_runs_on_the_plain_mic_and_says_so_once(
    daemon: Harness, tmp_path: Path
) -> None:
    setup = Setup(tmp_path)
    mic = WavCapture()
    await start(daemon, setup, "--fail", mic=mic)
    scribbly, _ = await daemon.connect("Scribbly")
    started = await scribbly.request("start_system_session", {"with_mic": True})
    assert started["data"]["tracks"] == ["system", "mic"], started
    mic.play(FIXTURE)
    await scribbly.wait(lambda _m: len(track_segments(scribbly, "mic")) == 2, timeout=15)
    await scribbly.request("stop_session")
    errors = aec_errors(scribbly)
    assert len(errors) == 1, errors
    assert errors[0]["data"]["track"] == "mic"
    assert errors[0]["data"]["fatal"] is False
    assert "Could not load module" in errors[0]["data"]["message"]
    assert not setup.record_log.exists()  # the cleaned source was never recorded


async def test_pw_cli_ending_mid_session_moves_the_mic_track_to_the_plain_mic(
    daemon: Harness, tmp_path: Path
) -> None:
    setup = Setup(tmp_path)
    mic = WavCapture()
    await start(daemon, setup, "--exit-after", "1", mic=mic)
    scribbly, _ = await daemon.connect("Scribbly")
    await scribbly.request("start_system_session", {"with_mic": True})
    error = await scribbly.event("session_error", code="aec_unavailable")
    assert error["data"]["track"] == "mic"
    assert "pw-cli exited with code 1" in error["data"]["message"]
    mic.play(FIXTURE)
    # Both cleaned phrases were cut off at 1 s; the mic's two phrases come after the switch.
    await scribbly.wait(lambda _m: len(track_segments(scribbly, "mic")) >= 2, timeout=15)
    await asyncio.sleep(0.2)
    await scribbly.request("stop_session")
    assert len(aec_errors(scribbly)) == 1
