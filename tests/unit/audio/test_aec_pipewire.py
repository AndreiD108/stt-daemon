"""The echo-cancel module with a `pw-cli` stand-in: loading it, the wait for its node, load errors,
termination; the mode per platform; the install check."""

import asyncio
import time
from pathlib import Path

import pytest

from stt_daemon.audio import aec_pipewire
from stt_daemon.audio.aec_pipewire import AecUnavailable, PipewireAec, pw_cli_command
from stt_daemon.config import AudioConfig, ConfigError, echo_cancel_mode, load_config
from stt_daemon.service.preflight import check_pipewire_aec
from unit.audio.aec_stand_ins import NAME, FakeNodes, pw_cli, pw_cli_log, until


def test_the_load_command_asks_for_monitor_mode_and_names_the_source() -> None:
    assert pw_cli_command(NAME) == [
        "pw-cli",
        "-m",
        "load-module",
        "libpipewire-module-echo-cancel",
        "{ monitor.mode = true aec.args = { webrtc.noise_suppression = true"
        " webrtc.gain_control = true webrtc.high_pass_filter = true } source.props = {"
        ' node.name = "stt-daemon-aec-s1" node.description = "Echo-cancelled mic (stt-daemon)" } }',
    ]


@pytest.mark.parametrize(
    ("value", "platform", "mode"),
    [
        ("auto", "linux", "pipewire"),
        ("auto", "darwin", "builtin"),
        ("off", "linux", "off"),
        ("builtin", "linux", "builtin"),
        ("pipewire", "linux", "pipewire"),
        ("pipewire", "darwin", "off"),
        ("builtin", "darwin", "builtin"),
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


async def test_two_instances_load_and_unload_on_their_own(tmp_path: Path) -> None:
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
