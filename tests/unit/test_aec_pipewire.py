"""PipeWire echo cancellation with stand-ins: loading the module through `pw-cli`, the wait for its
node, fallback to the plain mic, termination; the mode per platform; the install check."""

import asyncio
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from stt_daemon.audio import aec_pipewire
from stt_daemon.audio.aec_pipewire import (
    AecUnavailable,
    EchoCancelledMic,
    PipewireAec,
    pw_cli_command,
)
from stt_daemon.audio.capture import CaptureSource, CaptureUnavailable
from stt_daemon.audio.frames import Attachment
from stt_daemon.audio.pw_capture import AppTarget, PwRecordCapture
from stt_daemon.config import AudioConfig, ConfigError, echo_cancel_mode, load_config
from stt_daemon.service.preflight import check_pipewire_aec

FAKE_PW_CLI = Path(__file__).parents[1] / "fake_pw_cli.py"
FAKE_PW_RECORD = Path(__file__).parents[1] / "fake_pw_record.py"
NAME = "stt-daemon-aec-s1"
MIC_LEVEL = 0.25


class FakeNodes:
    def __init__(self) -> None:
        self.serials: dict[str, str] = {}

    def node_serial(self, name: str) -> str | None:
        return self.serials.get(name)

    async def wait_node(self, name: str) -> str:
        while name not in self.serials:
            await asyncio.sleep(0.01)
        return self.serials[name]

    def appear_after(self, delay_s: float, serial: str = "777") -> None:
        async def later() -> None:
            await asyncio.sleep(delay_s)
            self.serials[NAME] = serial

        asyncio.get_running_loop().create_task(later())


class PushedMic(CaptureSource):
    """Frames at `MIC_LEVEL` every 20 ms while running; counts holds."""

    def __init__(self) -> None:
        super().__init__(sample_rate=16000, buffer_ms=5000, keep_open=True)
        self.holds = 0
        self._task: asyncio.Task[None] | None = None

    @property
    def is_open(self) -> bool:
        return True

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    async def hold(self) -> None:
        self.holds += 1

    async def release(self) -> None:
        self.holds -= 1

    def run(self) -> None:
        async def loop() -> None:
            while True:
                self._hub.push(np.full(320, MIC_LEVEL, dtype=np.float32), time.monotonic())
                await asyncio.sleep(0.02)

        self._task = asyncio.get_running_loop().create_task(loop())

    def stop_running(self) -> None:
        assert self._task is not None
        self._task.cancel()


def pw_cli(tmp_path: Path, *options: str) -> Callable[[str], list[str]]:
    log = tmp_path / "pw-cli.log"

    def command(name: str) -> list[str]:
        real = pw_cli_command(name)
        return [sys.executable, str(FAKE_PW_CLI), "--log", str(log), *options, "--", *real[1:]]

    return command


def pw_cli_log(tmp_path: Path) -> list[str]:
    path = tmp_path / "pw-cli.log"
    return path.read_text().splitlines() if path.exists() else []


async def until(check: Callable[[], bool], timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.01)


def cleaned_capture(tmp_path: Path, targets: list[str]) -> Callable[[AppTarget], PwRecordCapture]:
    """A `pw-record` stand-in writing silence; records each target serial."""

    def make(target: AppTarget) -> PwRecordCapture:
        targets.append(target.serial)
        return PwRecordCapture(
            lambda serial: [sys.executable, str(FAKE_PW_RECORD), "--target", str(serial)],
            target=target,
        )

    return make


async def levels(attachment: Attachment, count: int) -> list[float]:
    out: list[float] = []
    async with asyncio.timeout(5):
        while len(out) < count:
            frame = await attachment.queue.get()
            out.append(float(np.max(np.abs(frame.samples))))
    return out


def test_the_load_command_asks_for_monitor_mode_and_names_the_source() -> None:
    assert pw_cli_command(NAME) == [
        "pw-cli",
        "-m",
        "load-module",
        "libpipewire-module-echo-cancel",
        '{ monitor.mode = true source.props = { node.name = "stt-daemon-aec-s1" } }',
    ]


@pytest.mark.parametrize(
    ("value", "platform", "mode"),
    [
        ("auto", "linux", "pipewire"),
        ("auto", "darwin", "builtin"),
        ("off", "linux", "off"),
        ("builtin", "linux", "builtin"),
        ("pipewire", "darwin", "pipewire"),
    ],
)
def test_the_mode_resolves_per_platform(value: str, platform: str, mode: str) -> None:
    assert echo_cancel_mode(AudioConfig(echo_cancel=value), platform) == mode


def test_an_unknown_mode_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[audio]\necho_cancel = "speex"\n')
    with pytest.raises(ConfigError, match=r"audio\.echo_cancel"):
        load_config(path)


async def test_the_load_returns_the_nodes_serial_and_unload_ends_pw_cli(tmp_path: Path) -> None:
    nodes = FakeNodes()
    nodes.appear_after(0.2)
    aec = PipewireAec(NAME, nodes=nodes, command=pw_cli(tmp_path))
    assert await aec.load() == "777"
    exits: list[str] = []
    aec.on_exit = exits.append
    await aec.unload()
    assert pw_cli_log(tmp_path) == [f"start {NAME}", f"end {NAME}"]
    assert exits == []


async def test_a_node_that_never_appears_fails_the_load_and_ends_pw_cli(tmp_path: Path) -> None:
    aec = PipewireAec(NAME, nodes=FakeNodes(), command=pw_cli(tmp_path), wait_s=0.5)
    began = time.monotonic()
    with pytest.raises(AecUnavailable, match=r"did not appear within 0\.5 s"):
        await aec.load()
    assert time.monotonic() - began < 2.0
    assert pw_cli_log(tmp_path) == [f"start {NAME}"]  # killed: no end line


async def test_a_load_error_fails_at_once_with_pw_clis_message(tmp_path: Path) -> None:
    aec = PipewireAec(NAME, nodes=FakeNodes(), command=pw_cli(tmp_path, "--fail"), wait_s=10)
    began = time.monotonic()
    with pytest.raises(AecUnavailable, match="Could not load module"):
        await aec.load()
    assert time.monotonic() - began < 5.0
    assert pw_cli_log(tmp_path) == [f"start {NAME}"]


async def test_pw_cli_exiting_during_the_load_fails_it(tmp_path: Path) -> None:
    command = pw_cli(tmp_path, "--exit-after", "0.1")
    aec = PipewireAec(NAME, nodes=FakeNodes(), command=command, wait_s=10)
    with pytest.raises(AecUnavailable, match="exited with code 1"):
        await aec.load()


async def test_a_missing_pw_cli_fails_the_load() -> None:
    aec = PipewireAec(NAME, nodes=FakeNodes(), command=lambda _name: ["/nonexistent/pw-cli"])
    with pytest.raises(AecUnavailable, match="not installed"):
        await aec.load()


async def test_pw_cli_ending_after_the_load_is_reported_once(tmp_path: Path) -> None:
    nodes = FakeNodes()
    nodes.serials[NAME] = "5"
    aec = PipewireAec(NAME, nodes=nodes, command=pw_cli(tmp_path, "--exit-after", "0.3"))
    await aec.load()
    exits: list[str] = []
    aec.on_exit = exits.append
    await until(lambda: bool(exits))
    await asyncio.sleep(0.1)
    assert len(exits) == 1
    assert "pw-cli exited with code 1" in exits[0]
    await aec.unload()


async def test_a_pw_cli_that_ignores_term_is_killed(tmp_path: Path) -> None:
    nodes = FakeNodes()
    nodes.serials[NAME] = "5"
    command = pw_cli(tmp_path, "--ignore-term")
    aec = PipewireAec(NAME, nodes=nodes, command=command, term_wait_s=0.3)
    await aec.load()
    await until(lambda: bool(pw_cli_log(tmp_path)))
    began = time.monotonic()
    await aec.unload()
    assert 0.3 <= time.monotonic() - began < 2.0
    assert pw_cli_log(tmp_path) == [f"start {NAME}"]  # killed: no end line


async def test_two_sessions_load_their_own_instances(tmp_path: Path) -> None:
    nodes = FakeNodes()
    nodes.serials.update({"stt-daemon-aec-a": "1", "stt-daemon-aec-b": "2"})
    first = PipewireAec("stt-daemon-aec-a", nodes=nodes, command=pw_cli(tmp_path))
    second = PipewireAec("stt-daemon-aec-b", nodes=nodes, command=pw_cli(tmp_path))
    assert (await first.load(), await second.load()) == ("1", "2")
    await until(lambda: len(pw_cli_log(tmp_path)) == 2)
    await first.unload()
    await second.unload()
    assert sorted(pw_cli_log(tmp_path)) == [
        "end stt-daemon-aec-a",
        "end stt-daemon-aec-b",
        "start stt-daemon-aec-a",
        "start stt-daemon-aec-b",
    ]


async def test_the_track_records_the_cleaned_source(tmp_path: Path) -> None:
    mic = PushedMic()
    mic.run()
    nodes = FakeNodes()
    nodes.serials[NAME] = "777"
    targets: list[str] = []
    source = EchoCancelledMic(
        mic,
        PipewireAec(NAME, nodes=nodes, command=pw_cli(tmp_path)),
        cleaned=cleaned_capture(tmp_path, targets),
    )
    reports: list[str] = []
    source.on_unavailable = reports.append
    await source.hold()
    attachment = source.attach(time.monotonic())
    assert await levels(attachment, 5) == [0.0] * 5
    assert targets == ["777"]
    assert source.cleaned
    assert mic.holds == 0
    assert reports == []
    await source.release()
    mic.stop_running()
    assert pw_cli_log(tmp_path) == [f"start {NAME}", f"end {NAME}"]


async def test_without_the_module_the_track_takes_the_plain_mic_and_says_so_once(
    tmp_path: Path,
) -> None:
    mic = PushedMic()
    mic.run()
    await asyncio.sleep(0.1)
    source = EchoCancelledMic(
        mic,
        PipewireAec(NAME, nodes=FakeNodes(), command=pw_cli(tmp_path, "--fail")),
        cleaned=cleaned_capture(tmp_path, []),
    )
    reports: list[str] = []
    source.on_unavailable = reports.append
    await source.hold()
    origin = time.monotonic() - 0.05
    attachment = source.attach(origin)
    first = await attachment.queue.get()
    assert first.start >= origin - 0.021  # from the mic's ring, from the asked point on
    assert set(await levels(attachment, 5)) == {MIC_LEVEL}
    assert not source.cleaned
    assert mic.holds == 1
    assert len(reports) == 1
    assert "Could not load module" in reports[0]
    await source.release()
    mic.stop_running()
    assert mic.holds == 0


async def test_pw_cli_ending_mid_session_switches_to_the_plain_mic(tmp_path: Path) -> None:
    mic = PushedMic()
    mic.run()
    nodes = FakeNodes()
    nodes.serials[NAME] = "777"
    source = EchoCancelledMic(
        mic,
        PipewireAec(NAME, nodes=nodes, command=pw_cli(tmp_path, "--exit-after", "0.5")),
        cleaned=cleaned_capture(tmp_path, []),
    )
    reports: list[str] = []
    source.on_unavailable = reports.append
    await source.hold()
    attachment = source.attach(time.monotonic())
    assert await levels(attachment, 3) == [0.0] * 3
    await until(lambda: bool(reports))
    seen = await levels(attachment, 40)
    assert seen[-5:] == [MIC_LEVEL] * 5
    assert not source.cleaned
    assert mic.holds == 1
    assert len(reports) == 1
    assert "pw-cli exited" in reports[0]
    await source.release()
    mic.stop_running()
    assert mic.holds == 0


def test_the_install_check_finds_the_files_under_the_usual_directories(tmp_path: Path) -> None:
    module_dir = tmp_path / "usr/lib/x86_64-linux-gnu/pipewire-0.3"
    plugin_dir = tmp_path / "usr/lib/x86_64-linux-gnu/spa-0.2/aec"
    module_dir.mkdir(parents=True)
    plugin_dir.mkdir(parents=True)
    (module_dir / "libpipewire-module-echo-cancel.so").touch()
    (plugin_dir / "libspa-aec-webrtc.so").touch()
    assert check_pipewire_aec(lambda _var, _pkg: None, root=tmp_path) is None


def test_the_install_check_uses_pkg_config_dirs(tmp_path: Path) -> None:
    (tmp_path / "mods").mkdir()
    (tmp_path / "plugins").mkdir()
    (tmp_path / "mods/libpipewire-module-echo-cancel.so").touch()
    (tmp_path / "plugins/libspa-aec-webrtc.so").touch()
    dirs = {"moduledir": str(tmp_path / "mods"), "plugindir": str(tmp_path / "plugins")}
    assert check_pipewire_aec(lambda var, _pkg: dirs[var], root=tmp_path / "empty") is None


def test_the_install_check_names_whats_missing_and_the_install_command(tmp_path: Path) -> None:
    module_dir = tmp_path / "usr/lib64/pipewire-0.3"
    module_dir.mkdir(parents=True)
    (module_dir / "libpipewire-module-echo-cancel.so").touch()
    os_release = tmp_path / "os-release"
    os_release.write_text("ID=fedora\n")
    problem = check_pipewire_aec(
        lambda _var, _pkg: None, root=tmp_path, platform="linux", os_release=os_release
    )
    assert problem is not None
    assert "libspa-aec-webrtc.so" in problem
    assert "libpipewire-module-echo-cancel.so" not in problem
    assert "sudo dnf install pipewire-libs" in problem


class SlowStart(PwRecordCapture):
    async def hold(self) -> None:
        await asyncio.sleep(0.5)
        await super().hold()


async def test_pw_cli_ending_while_the_cleaned_capture_starts_falls_back(tmp_path: Path) -> None:
    mic = PushedMic()
    mic.run()
    nodes = FakeNodes()
    nodes.serials[NAME] = "777"

    def slow(target: AppTarget) -> PwRecordCapture:
        return SlowStart(lambda s: [sys.executable, str(FAKE_PW_RECORD), "--target", str(s)])

    source = EchoCancelledMic(
        mic,
        PipewireAec(NAME, nodes=nodes, command=pw_cli(tmp_path, "--exit-after", "0.05")),
        cleaned=slow,
    )
    reports: list[str] = []
    source.on_unavailable = reports.append
    await source.hold()
    attachment = source.attach(time.monotonic())
    await asyncio.sleep(0.3)
    assert set(await levels(attachment, 5)) == {MIC_LEVEL}
    assert not source.cleaned
    assert mic.holds == 1
    assert len(reports) == 1
    assert "pw-cli exited" in reports[0]
    await source.release()
    mic.stop_running()


class FlakyMic(PushedMic):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures
        self.tries = 0

    async def hold(self) -> None:
        self.tries += 1
        if self.tries <= self.failures:
            raise CaptureUnavailable("the mic is busy")
        await super().hold()


async def test_a_mic_that_fails_to_open_after_the_switch_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aec_pipewire, "MIC_RETRY_S", (0.05,))
    mic = FlakyMic(failures=2)
    mic.run()
    nodes = FakeNodes()
    nodes.serials[NAME] = "777"
    source = EchoCancelledMic(
        mic,
        PipewireAec(NAME, nodes=nodes, command=pw_cli(tmp_path, "--exit-after", "0.3")),
        cleaned=cleaned_capture(tmp_path, []),
    )
    events: list[str] = []
    source.on_lost = lambda reason: events.append(f"lost: {reason}")
    source.on_restored = lambda: events.append("restored")
    await source.hold()
    attachment = source.attach(time.monotonic())
    await until(lambda: "restored" in events)
    assert events == ["lost: the mic did not open: the mic is busy", "restored"]
    assert mic.tries == 3
    seen = await levels(attachment, 40)
    assert seen[-5:] == [MIC_LEVEL] * 5
    await source.release()
    mic.stop_running()
    assert mic.holds == 0


async def test_a_failed_load_stays_within_the_wait_even_when_the_install_check_is_slow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow_check() -> str | None:
        time.sleep(3)
        return "missing"

    monkeypatch.setattr(aec_pipewire, "check_pipewire_aec", slow_check)
    command = pw_cli(tmp_path, "--ignore-term")
    aec = PipewireAec(NAME, nodes=FakeNodes(), command=command, wait_s=0.5, term_wait_s=5)
    began = time.monotonic()
    with pytest.raises(AecUnavailable, match="did not appear"):
        await aec.load()
    assert time.monotonic() - began < 1.0
