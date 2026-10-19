import plistlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from stt_daemon.service.common import ServiceError
from stt_daemon.service.launchd import LABEL, Launchd, render_plist
from stt_daemon.service.preflight import check_portaudio, check_tool, install_hint
from stt_daemon.service.systemd import LIMITS, Systemd, exec_arg, render_unit


class Recorder:
    def __init__(self, fail: str | None = None, code: int = 1) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail
        self.code = code

    def __call__(self, cmd: Sequence[str]) -> int:
        self.calls.append(list(cmd))
        return self.code if self.fail and self.fail in cmd else 0


def test_unit_runs_the_installed_script_and_restarts_it() -> None:
    lines = render_unit(Path("/opt/tools/bin/stt-daemon")).splitlines()
    assert "ExecStart=/opt/tools/bin/stt-daemon run" in lines
    assert "Restart=always" in lines
    assert "RestartSec=1" in lines
    assert "StartLimitIntervalSec=0" in lines
    assert "WantedBy=default.target" in lines


def test_limits_drop_in_leaves_exec_start_to_the_users_own_drop_ins() -> None:
    assert "ExecStart" not in LIMITS
    assert "MemoryMax=2G" in LIMITS.splitlines()


@pytest.mark.parametrize(
    ("arg", "word"),
    [
        ("/a b/stt-daemon", '"/a b/stt-daemon"'),
        ("/x/100%/stt", "/x/100%%/stt"),
        ("/x/$HOME/stt", "/x/$$HOME/stt"),
        ('/x/"q"/stt', '"/x/\\"q\\"/stt"'),
    ],
)
def test_exec_start_words_escape_what_systemd_would_expand(arg: str, word: str) -> None:
    assert exec_arg(arg) == word


def test_install_writes_both_files_then_enables_and_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    run = Recorder()
    written = Systemd(run).install(Path("/bin/stt-daemon"))
    unit_dir = tmp_path / "systemd" / "user"
    unit = unit_dir / "stt-daemon.service"
    assert written == [unit, unit_dir / "stt-daemon.service.d" / "limits.conf"]
    assert [c[2:] for c in run.calls] == [
        ["daemon-reload"],
        ["enable", "stt-daemon.service"],
        ["restart", "stt-daemon.service"],
    ]


def test_install_fails_when_systemctl_does(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    with pytest.raises(ServiceError, match="enable"):
        Systemd(Recorder(fail="enable")).install(Path("/bin/stt-daemon"))


def test_remove_deletes_its_files_and_keeps_other_drop_ins(tmp_path: Path) -> None:
    systemd = Systemd(Recorder(), unit_dir=tmp_path)
    systemd.install(Path("/bin/stt-daemon"))
    own = tmp_path / "stt-daemon.service.d" / "keys.conf"
    own.write_text("[Service]\nExecStart=\nExecStart=/usr/local/bin/wrapper\n")
    assert systemd.remove() == [systemd.unit_file, systemd.limits_file]
    assert not systemd.unit_file.exists()
    assert own.exists()


def test_remove_refuses_a_unit_it_did_not_write(tmp_path: Path) -> None:
    unit = tmp_path / "stt-daemon.service"
    unit.write_text("[Service]\nExecStart=/somewhere/else\n")
    with pytest.raises(ServiceError, match="by hand"):
        Systemd(Recorder(), unit_dir=tmp_path).remove()
    assert unit.exists()


def test_with_a_unit_dir_systemctl_is_never_called(tmp_path: Path) -> None:
    run = Recorder()
    systemd = Systemd(run, unit_dir=tmp_path)
    systemd.install(Path("/bin/stt-daemon"))
    systemd.remove()
    assert run.calls == []


def test_plist_runs_the_script_at_load_and_keeps_tmpdir() -> None:
    raw = render_plist(Path("/bin/stt-daemon"), Path("/logs/stt.log"), {"TMPDIR": "/var/t/"})
    plist = plistlib.loads(raw)
    assert plist["Label"] == LABEL
    assert plist["ProgramArguments"] == ["/bin/stt-daemon", "run"]
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True
    assert plist["EnvironmentVariables"] == {"TMPDIR": "/var/t/"}


@pytest.mark.parametrize(
    ("os_release", "command"),
    [
        ("ID=ubuntu\nID_LIKE=debian\n", "sudo apt install libportaudio2"),
        ('ID=linuxmint\nID_LIKE="ubuntu debian"\n', "sudo apt install libportaudio2"),
        ("ID=fedora\n", "sudo dnf install portaudio"),
        ('ID="rocky"\nID_LIKE="rhel centos fedora"\n', "sudo dnf install portaudio"),
        ("ID=arch\n", "sudo pacman -S portaudio"),
        ('ID="opensuse-tumbleweed"\nID_LIKE="suse"\n', "sudo zypper install libportaudio2"),
    ],
)
def test_install_hint_follows_os_release(tmp_path: Path, os_release: str, command: str) -> None:
    path = tmp_path / "os-release"
    path.write_text(os_release)
    assert install_hint("linux", path) == command


def test_install_hint_without_a_known_distro_names_the_library(tmp_path: Path) -> None:
    assert "libportaudio.so.2" in install_hint("linux", tmp_path / "missing")
    assert install_hint("darwin") == "brew install portaudio"


def test_check_portaudio_names_the_install_command_when_loading_fails(tmp_path: Path) -> None:
    path = tmp_path / "os-release"
    path.write_text("ID=debian\n")

    def missing() -> None:
        raise OSError("PortAudio library not found")

    problem = check_portaudio(missing, "linux", path)
    assert problem is not None
    assert "sudo apt install libportaudio2" in problem
    assert check_portaudio(lambda: None, "linux", path) is None


def test_remove_keeps_the_files_when_systemctl_cannot_stop_the_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    systemd = Systemd(Recorder(fail="disable"))
    systemd.install(Path("/bin/stt-daemon"))
    with pytest.raises(ServiceError, match="disable"):
        systemd.remove()
    assert systemd.unit_file.exists()


def test_launchd_remove_keeps_the_plist_when_bootout_fails(tmp_path: Path) -> None:
    launchd = Launchd(Recorder(fail="bootout"), env={})
    launchd.agents_dir = tmp_path
    launchd.plist_file.write_bytes(b"")
    with pytest.raises(ServiceError, match="bootout"):
        launchd.remove()
    assert launchd.plist_file.exists()


def test_launchd_remove_of_an_agent_not_loaded_deletes_the_plist(tmp_path: Path) -> None:
    launchd = Launchd(Recorder(fail="print", code=113), env={})
    launchd.agents_dir = tmp_path
    launchd.plist_file.write_bytes(b"")
    assert launchd.remove() == [launchd.plist_file]


def test_launchd_remove_keeps_the_plist_when_launchctl_print_fails_otherwise(
    tmp_path: Path,
) -> None:
    launchd = Launchd(Recorder(fail="print", code=5), env={})
    launchd.agents_dir = tmp_path
    launchd.plist_file.write_bytes(b"")
    with pytest.raises(ServiceError, match="print"):
        launchd.remove()
    assert launchd.plist_file.exists()


@pytest.mark.parametrize(
    ("os_release", "command"),
    [
        ("ID=debian\n", "sudo apt install ffmpeg"),
        ("ID=fedora\n", "sudo dnf install ffmpeg-free"),
        ("ID=arch\n", "sudo pacman -S ffmpeg"),
    ],
)
def test_check_tool_names_the_install_command_when_missing(
    tmp_path: Path, os_release: str, command: str
) -> None:
    path = tmp_path / "os-release"
    path.write_text(os_release)
    problem = check_tool("ffmpeg", lambda _name: None, "linux", path)
    assert problem is not None
    assert problem.endswith(command)
    assert check_tool("ffmpeg", lambda name: f"/usr/bin/{name}", "linux", path) is None
    assert check_tool("ffmpeg", lambda _name: None, "darwin") == (
        "ffmpeg is not installed or not on PATH. Install it: brew install ffmpeg"
    )


@pytest.mark.parametrize(
    ("os_release", "command"),
    [
        ("ID=debian\n", "sudo apt install pipewire-bin"),
        ("ID=fedora\n", "sudo dnf install pipewire-utils"),
        ("ID=arch\n", "sudo pacman -S pipewire"),
        ('ID="opensuse-tumbleweed"\nID_LIKE="suse"\n', "sudo zypper install pipewire-tools"),
    ],
)
def test_pw_record_hint_names_the_distro_package(
    tmp_path: Path, os_release: str, command: str
) -> None:
    path = tmp_path / "os-release"
    path.write_text(os_release)
    assert install_hint("linux", path, package="pw-record") == command
    assert install_hint("linux", path, package="pw-dump") == command


def test_pw_record_hint_without_a_known_distro_or_on_macos_names_the_tool(tmp_path: Path) -> None:
    assert "pw-record" in install_hint("linux", tmp_path / "missing", package="pw-record")
    assert "pw-record" in install_hint("darwin", package="pw-record")
