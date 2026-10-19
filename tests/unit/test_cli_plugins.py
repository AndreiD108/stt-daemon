import io
import shutil
import tomllib
from pathlib import Path

import pytest

from stt_daemon import cli
from stt_daemon.cli import main
from stt_daemon.config import data_dir
from stt_daemon.credentials import load_credentials

FIXTURES = Path(__file__).parents[1] / "fixtures" / "plugins"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config dir with the fixture plugins, and no daemon listening."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("STT_DAEMON_SOCKET", str(tmp_path / "no-daemon.sock"))
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "check_portaudio", lambda: None)
    home = tmp_path / "stt-daemon"
    shutil.copytree(FIXTURES, home / "backends", ignore=shutil.ignore_patterns("__pycache__"))
    return home


def _default_backend(home: Path) -> str | None:
    return tomllib.loads((home / "config.toml").read_text())["backends"].get("default")


def test_backend_list_shows_every_backend_and_failed_file_from_the_files(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["backend", "list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    rows = {line.split()[0]: line.split(maxsplit=1)[1] for line in lines[:-1]}
    assert rows == {
        "elevenlabs": "no key",
        "elevenlabs-batch": "no key",
        "parakeet": "not installed",
        "local-model": "not installed",
        "openai": "no key",
        "raising-setup": "failed: RuntimeError: no device found",
        "builtin_name.py": "failed: 'parakeet' is a built-in backend's name; rename the plugin",
        "two_classes.py": "failed: defines 2 Backend subclasses (First, Second); keep one per file",
    }
    assert lines[-1] == "(the daemon is not running; this is what it would set up)"


def test_key_set_takes_a_plugin_name_and_makes_it_ready_and_the_default(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-plugin\n"))
    assert main(["key", "set", "openai"]) == 0
    assert load_credentials().get("openai") == "sk-plugin"
    assert _default_backend(home) == "openai"
    capsys.readouterr()
    assert main(["backend", "list"]) == 0
    assert "openai            ready, default" in capsys.readouterr().out


def test_key_set_refuses_a_name_no_backend_has(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-plugin\n"))
    assert main(["key", "set", "nosuch"]) == 1
    assert "no backend named 'nosuch' takes a key" in capsys.readouterr().err
    assert not (home / "credentials.toml").exists()


def test_backend_install_runs_the_plugins_install_into_its_data_dir(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["backend", "install", "local-model"]) == 0
    assert (data_dir() / "models" / "local-model" / "model.bin").read_bytes() == b"weights"
    assert _default_backend(home) == "local-model"
    assert "local-model is installed in" in capsys.readouterr().out


def test_backend_install_refuses_a_plugin_without_install(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["backend", "install", "openai"]) == 1
    assert "openai has nothing to install" in capsys.readouterr().err


def test_set_default_takes_a_plugin_and_refuses_a_failed_one(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["backend", "install", "local-model"]) == 0
    (home / "credentials.toml").unlink(missing_ok=True)
    assert main(["backend", "set-default", "parakeet"]) == 1
    assert main(["backend", "set-default", "local-model"]) == 0
    assert _default_backend(home) == "local-model"
    capsys.readouterr()
    assert main(["backend", "set-default", "raising-setup"]) == 1
    err = capsys.readouterr().err
    assert "raising-setup is not set up: RuntimeError: no device found" in err
    assert main(["backend", "set-default", "nosuch"]) == 1
    assert "there is no backend named 'nosuch'" in capsys.readouterr().err
