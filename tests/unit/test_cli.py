import io
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from stt_daemon import cli, control, daemon
from stt_daemon.backends.install import InstallError
from stt_daemon.backends.parakeet import MODEL_FILES, model_dir
from stt_daemon.cli import main
from stt_daemon.credentials import load_credentials, write_key


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config dir of its own and no daemon listening."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("STT_DAEMON_SOCKET", str(tmp_path / "no-daemon.sock"))
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(cli, "check_portaudio", lambda: None)
    return tmp_path / "stt-daemon"


def _stdin(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(text))


def _default_backend(home: Path) -> str | None:
    return tomllib.loads((home / "config.toml").read_text())["backends"].get("default")


def test_run_with_a_broken_config_exits_naming_the_file(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = home / "config.toml"
    config.parent.mkdir()
    config.write_text("[audio]\nbuffer_ms = 'long'\n")
    assert main(["run"]) == 1
    assert str(config) in capsys.readouterr().err


def test_run_keys_stdin_win_over_env_and_file(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_key("elevenlabs", "from-file")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "from-env")
    _stdin(monkeypatch, "ELEVENLABS_API_KEY=from-stdin\n")
    passed: list[Mapping[str, str] | None] = []

    def fake_run(keys: Mapping[str, str] | None) -> int:
        passed.append(keys)
        return 0

    monkeypatch.setattr(daemon, "run", fake_run)
    assert main(["run", "--keys-stdin"]) == 0
    assert load_credentials(passed[0]).get("elevenlabs") == "from-stdin"


def test_run_keys_stdin_errors_hide_the_value(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stdin(monkeypatch, "ELEVENLABS_API_KEY=secret-1\nELEVENLABS_API_KEY=secret-2\n")

    def fake_run(keys: Mapping[str, str] | None) -> int:
        pytest.fail("the daemon started")

    monkeypatch.setattr(daemon, "run", fake_run)
    assert main(["run", "--keys-stdin"]) == 1
    err = capsys.readouterr().err
    assert "line 2" in err
    assert "secret" not in err


def test_key_set_stores_the_key_and_makes_the_first_backend_the_default(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stdin(monkeypatch, "sk-abc123\n")
    assert main(["key", "set", "elevenlabs"]) == 0
    assert load_credentials().get("elevenlabs") == "sk-abc123"
    assert (home / "credentials.toml").stat().st_mode & 0o777 == 0o600
    assert _default_backend(home) == "elevenlabs"
    out = capsys.readouterr()
    assert "not running" in out.out
    assert "sk-abc123" not in out.out + out.err


def test_key_set_keeps_a_default_the_user_chose(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home.mkdir()
    (home / "config.toml").write_text('[backends]\ndefault = "parakeet"\n')
    _stdin(monkeypatch, "sk-abc123\n")
    assert main(["key", "set", "elevenlabs"]) == 0
    assert _default_backend(home) == "parakeet"


def test_set_default_refuses_a_backend_that_is_not_set_up(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["backend", "set-default", "elevenlabs"]) == 1
    assert "key set elevenlabs" in capsys.readouterr().err
    assert not (home / "config.toml").exists()


def test_set_default_refuses_the_batch_backend(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_key("elevenlabs", "sk-abc123")
    assert main(["backend", "set-default", "elevenlabs-batch"]) == 1
    assert "transcribe sessions only" in capsys.readouterr().err
    assert not (home / "config.toml").exists()


def _fake_install(monkeypatch: pytest.MonkeyPatch) -> None:
    def install() -> tuple[Path, bool]:
        model = model_dir()
        model.mkdir(parents=True)
        for name in MODEL_FILES:
            (model / name).write_text("x")
        return model, True

    monkeypatch.setattr(cli, "install_parakeet", install)


def test_backend_install_makes_the_first_backend_the_default(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_install(monkeypatch)
    assert main(["backend", "install", "parakeet"]) == 0
    assert _default_backend(home) == "parakeet"
    assert "not running" in capsys.readouterr().out
    assert main(["backend", "set-default", "parakeet"]) == 0


def test_backend_install_keeps_the_backend_already_in_use(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_key("elevenlabs", "sk-abc123")
    _fake_install(monkeypatch)
    assert main(["backend", "install", "parakeet"]) == 0
    assert _default_backend(home) == "elevenlabs"


def test_backend_install_failure_exits_1(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> tuple[Path, bool]:
        raise InstallError("the download failed: connection refused")

    monkeypatch.setattr(cli, "install_parakeet", fail)
    assert main(["backend", "install", "parakeet"]) == 1
    assert "connection refused" in capsys.readouterr().err
    assert not (home / "config.toml").exists()


def _installed() -> Path:
    model = model_dir()
    model.mkdir(parents=True)
    for name in MODEL_FILES:
        (model / name).write_text("x")
    return model


def test_backend_remove_moves_the_default_to_the_other_backend(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = _installed()
    write_key("elevenlabs", "sk-abc123")
    home.joinpath("config.toml").write_text('[backends]\ndefault = "parakeet"\n')
    requests: list[str] = []

    def request(kind: str) -> dict[str, object]:
        requests.append(kind)
        return {"backends": [{"name": "elevenlabs"}], "default_backend": "elevenlabs"}

    monkeypatch.setattr(control, "request", request)
    assert main(["backend", "remove", "parakeet"]) == 0
    assert not model.exists()
    assert _default_backend(home) == "elevenlabs"
    assert requests == ["reload"]
    out = capsys.readouterr().out
    assert str(model) in out
    assert "elevenlabs is the default backend" in out


def test_backend_remove_of_the_only_backend_leaves_no_default(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _installed()
    home.mkdir()
    home.joinpath("config.toml").write_text('[backends]\ndefault = "parakeet"\n')
    assert main(["backend", "remove", "parakeet"]) == 0
    assert _default_backend(home) is None
    out = capsys.readouterr().out
    assert "no default backend" in out
    assert "not running" in out


def test_backend_remove_keeps_another_default(home: Path) -> None:
    _installed()
    write_key("elevenlabs", "sk-abc123")
    home.joinpath("config.toml").write_text('[backends]\ndefault = "elevenlabs"\n')
    assert main(["backend", "remove", "parakeet"]) == 0
    assert _default_backend(home) == "elevenlabs"


def test_backend_remove_keeps_the_model_when_the_config_cannot_be_edited(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = _installed()
    home.mkdir()
    home.joinpath("config.toml").write_text('backends.default = "parakeet"\n')
    assert main(["backend", "remove", "parakeet"]) == 1
    assert "edit the file by hand" in capsys.readouterr().err
    assert model.exists()


def test_backend_remove_refuses_a_model_that_is_not_installed(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["backend", "remove", "parakeet"]) == 1
    assert str(model_dir()) in capsys.readouterr().err


def test_backend_remove_names_parakeet_as_the_only_removable_backend(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["backend", "remove", "elevenlabs"])
    assert exited.value.code == 2
    assert "parakeet" in capsys.readouterr().err


def test_service_install_with_a_unit_dir_writes_files_without_systemctl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_commands(cmd: Sequence[str]) -> int:
        raise AssertionError(f"ran {cmd}")

    monkeypatch.setattr(cli, "run_command", no_commands)
    monkeypatch.setattr(cli, "check_portaudio", lambda: None)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(cli, "daemon_executable", lambda: Path("/tools/bin/stt-daemon"))
    assert main(["service", "install", "--unit-dir", str(tmp_path)]) == 0
    unit = (tmp_path / "stt-daemon.service").read_text()
    assert "ExecStart=/tools/bin/stt-daemon run" in unit
    assert (tmp_path / "stt-daemon.service.d" / "limits.conf").exists()
    out = capsys.readouterr().out
    assert "stt-daemon key set elevenlabs" in out
    assert "stt-daemon backend install parakeet" in out


def test_service_install_stops_when_portaudio_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "check_portaudio", lambda: "PortAudio did not load. Install it: X")
    assert main(["service", "install", "--unit-dir", str(tmp_path)]) == 1
    assert "Install it: X" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def ffmpeg_missing(tool: str) -> str:
    return f"{tool} is missing. Install it: Y"


def test_service_install_names_ffmpeg_when_missing_and_still_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "check_portaudio", lambda: None)
    monkeypatch.setattr(cli, "check_tool", ffmpeg_missing)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(cli, "daemon_executable", lambda: Path("/tools/bin/stt-daemon"))
    assert main(["service", "install", "--unit-dir", str(tmp_path)]) == 0
    assert "ffmpeg is missing. Install it: Y" in capsys.readouterr().out
    assert (tmp_path / "stt-daemon.service").exists()


def test_service_install_on_macos_names_a_missing_audiotee(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "check_portaudio", lambda: None)

    def only_audiotee_missing(tool: str) -> str | None:
        return ffmpeg_missing(tool) if tool == "audiotee" else None

    monkeypatch.setattr(cli, "check_tool", only_audiotee_missing)
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(cli, "daemon_executable", lambda: Path("/tools/bin/stt-daemon"))
    assert main(["service", "install", "--unit-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "system audio sessions need audiotee: audiotee is missing" in out
    assert "pw-record" not in out
