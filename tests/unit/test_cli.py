import io
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from stt_daemon import cli, daemon
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
