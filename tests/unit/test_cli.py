from pathlib import Path

import pytest

from stt_daemon.cli import main


def test_run_with_a_broken_config_exits_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "stt-daemon" / "config.toml"
    config.parent.mkdir()
    config.write_text("[audio]\nbuffer_ms = 'long'\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert main(["run"]) == 1
    assert str(config) in capsys.readouterr().err
