import logging
import os
import tomllib
from pathlib import Path

import pytest

from stt_daemon.config import (
    Config,
    ConfigError,
    config_dir,
    default_num_threads,
    load_config,
    remove_config_key,
    set_config_string,
    socket_path,
)


def test_socket_path_env_override_wins_on_every_platform() -> None:
    env = {"STT_DAEMON_SOCKET": "/x/s.sock", "XDG_RUNTIME_DIR": "/run/user/5", "TMPDIR": "/t"}
    assert socket_path(env, "linux") == Path("/x/s.sock")
    assert socket_path(env, "darwin") == Path("/x/s.sock")


def test_socket_path_linux_uses_runtime_dir() -> None:
    env = {"XDG_RUNTIME_DIR": "/run/user/5", "TMPDIR": "/t"}
    assert socket_path(env, "linux") == Path("/run/user/5/stt-daemon/stt.sock")


def test_socket_path_linux_without_runtime_dir_falls_back_to_run_user() -> None:
    assert socket_path({}, "linux") == Path(f"/run/user/{os.getuid()}/stt-daemon/stt.sock")


def test_socket_path_macos_uses_tmpdir() -> None:
    env = {"XDG_RUNTIME_DIR": "/run/user/5", "TMPDIR": "/var/folders/ab/T/"}
    assert socket_path(env, "darwin") == Path("/var/folders/ab/T/stt-daemon/stt.sock")


def test_config_dir_honors_absolute_xdg_config_home_only() -> None:
    assert config_dir({"XDG_CONFIG_HOME": "/cfg"}) == Path("/cfg/stt-daemon")
    assert config_dir({"XDG_CONFIG_HOME": "rel"}) == Path.home() / ".config/stt-daemon"
    assert config_dir({}) == Path.home() / ".config/stt-daemon"


def test_missing_file_gives_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path / "none.toml")
    assert config == Config()
    assert config.audio.keep_mic_open is True
    assert config.audio.buffer_ms == 5000
    assert config.audio.pre_start_ms == 1000
    assert config.vad.min_silence_ms == 600
    assert config.vad.pause_keep_ms == 100
    assert config.slot.evictable_after_s == 300
    assert config.backends.default is None


def test_values_override_defaults_and_ints_pass_as_floats(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[audio]\nkeep_mic_open = false\n[vad]\nthreshold = 1\n"
        '[backends]\ndefault = "elevenlabs"\n'
        '[backends.elevenlabs]\nkeyterms = ["Scribbly"]\n'
    )
    config = load_config(path)
    assert config.audio.keep_mic_open is False
    assert config.audio.buffer_ms == 5000
    assert config.vad.threshold == 1.0
    assert isinstance(config.vad.threshold, float)
    assert config.backends.default == "elevenlabs"
    assert config.backends.elevenlabs.keyterms == ["Scribbly"]
    assert config.backends.elevenlabs.language == "auto"


@pytest.mark.parametrize(
    ("toml", "key"),
    [
        ('[audio]\nbuffer_ms = "5000"\n', "audio.buffer_ms"),
        ("[audio]\nkeep_mic_open = 1\n", "audio.keep_mic_open"),
        ("[vad]\nmin_speech_ms = true\n", "vad.min_speech_ms"),
        ("[backends.elevenlabs]\nkeyterms = [1]\n", "backends.elevenlabs.keyterms"),
        ("[backends]\ndefault = 3\n", "backends.default"),
        ("[backends.parakeet]\nmax_sessions = 1.5\n", "backends.parakeet.max_sessions"),
        ("audio = 3\n", "audio"),
    ],
)
def test_bad_type_names_the_key_path(tmp_path: Path, toml: str, key: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(toml)
    with pytest.raises(ConfigError, match=rf"{key} must be"):
        load_config(path)


def test_unknown_keys_warn_and_load(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[audio]\nbuffer_ms = 4000\nvolume = 3\n[extra]\nx = 1\n")
    with caplog.at_level(logging.WARNING):
        config = load_config(path)
    assert config.audio.buffer_ms == 4000
    assert "audio.volume" in caplog.text
    assert "extra" in caplog.text


def test_other_backend_tables_are_kept_for_plugins(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[backends]\nlimit = 3\n[backends.openai]\nmodel = "whisper-1"\n')
    with caplog.at_level(logging.WARNING):
        config = load_config(path)
    assert config.backends.plugins == {"openai": {"model": "whisper-1"}}
    assert "backends.limit" in caplog.text
    assert "openai" not in caplog.text


def test_out_of_range_value_fails(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[audio]\nbuffer_ms = 500\npre_start_ms = 1000\n")
    with pytest.raises(ConfigError, match=r"audio\.pre_start_ms"):
        load_config(path)


@pytest.mark.parametrize("backend", ["elevenlabs", "parakeet"])
def test_max_sessions_must_be_at_least_one(tmp_path: Path, backend: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"[backends.{backend}]\nmax_sessions = 0\n")
    with pytest.raises(ConfigError, match=rf"backends\.{backend}\.max_sessions"):
        load_config(path)
    path.write_text(f"[backends.{backend}]\nmax_sessions = 1\n")
    assert getattr(load_config(path).backends, backend).max_sessions == 1


@pytest.mark.parametrize(
    ("key", "bad", "good"),
    [
        ("num_threads", "0", "1"),
        ("max_chunk_seconds", "4.5", "5"),
        ("unload_after_minutes", "-1", "0"),
    ],
)
def test_parakeet_lifecycle_limits(tmp_path: Path, key: str, bad: str, good: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"[backends.parakeet]\n{key} = {bad}\n")
    with pytest.raises(ConfigError, match=rf"backends\.parakeet\.{key}"):
        load_config(path)
    path.write_text(f"[backends.parakeet]\n{key} = {good}\n")
    assert getattr(load_config(path).backends.parakeet, key) == float(good)


@pytest.mark.parametrize(("cpus", "threads"), [(None, 1), (1, 1), (3, 1), (4, 2), (6, 4), (32, 4)])
def test_parakeet_threads_leave_two_cores_free(
    monkeypatch: pytest.MonkeyPatch, cpus: int | None, threads: int
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    assert default_num_threads() == threads
    assert Config().backends.parakeet.num_threads == threads


def test_invalid_toml_fails_with_path(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[audio\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_set_string_replaces_value_and_keeps_other_text(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = (
        "# my settings\n[audio]\nbuffer_ms = 4000  # longer\n\n"
        '[backends]\n# which one\ndefault = "elevenlabs"\n\n'
        '[backends.elevenlabs]\nlanguage = "en"\n'
    )
    path.write_text(original)
    set_config_string("backends", "default", "parakeet", path)
    assert path.read_text() == original.replace('default = "elevenlabs"', 'default = "parakeet"')


def test_set_string_inserts_into_existing_section_without_touching_subtables(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[backends]\n\n[backends.elevenlabs]\ndefault = "nope"\n')
    set_config_string("backends", "default", "parakeet", path)
    data = tomllib.loads(path.read_text())
    assert data["backends"]["default"] == "parakeet"
    assert data["backends"]["elevenlabs"]["default"] == "nope"


def test_set_string_appends_missing_section(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[audio]\nbuffer_ms = 4000")
    set_config_string("backends", "default", "parakeet", path)
    assert tomllib.loads(path.read_text()) == {
        "audio": {"buffer_ms": 4000},
        "backends": {"default": "parakeet"},
    }


def test_set_string_creates_missing_file_and_escapes(tmp_path: Path) -> None:
    path = tmp_path / "new" / "config.toml"
    value = 'a "quoted" \\ value\twith tab'
    set_config_string("backends", "default", value, path)
    assert load_config(path).backends.default == value


def test_set_string_keeps_file_mode(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("")
    path.chmod(0o640)
    set_config_string("backends", "default", "x", path)
    assert path.stat().st_mode & 0o777 == 0o640


def test_set_string_never_edits_text_inside_a_multiline_string(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = (
        '[notes]\ntext = """\n[backends]\ndefault = "kept"\n"""\n\n'
        '[backends]\ndefault = "parakeet"\n'
    )
    path.write_text(original)
    with pytest.raises(ConfigError, match="edit the file by hand"):
        set_config_string("backends", "default", "parakeet", path)
    assert path.read_text() == original


def test_set_string_refuses_invalid_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[backends\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        set_config_string("backends", "default", "x", path)
    assert path.read_text() == "[backends\n"


def test_set_string_refuses_what_it_cannot_express(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = 'backends = { default = "elevenlabs" }\n'
    path.write_text(original)
    with pytest.raises(ConfigError, match="edit the file by hand"):
        set_config_string("backends", "default", "parakeet", path)
    assert path.read_text() == original


def test_remove_key_keeps_other_text(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = (
        '# mine\n[backends]\n# which one\ndefault = "parakeet"  # local\n\n'
        '[backends.parakeet]\ndefault = "x"\n'
    )
    path.write_text(original)
    remove_config_key("backends", "default", path)
    assert path.read_text() == original.replace('default = "parakeet"  # local\n', "")


def test_remove_key_without_the_key_or_the_file_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    remove_config_key("backends", "default", path)
    assert not path.exists()
    path.write_text("[audio]\nbuffer_ms = 4000\n")
    remove_config_key("backends", "default", path)
    assert path.read_text() == "[audio]\nbuffer_ms = 4000\n"


def test_remove_key_refuses_what_it_cannot_express(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('backends.default = "parakeet"\n')
    with pytest.raises(ConfigError, match="edit the file by hand"):
        remove_config_key("backends", "default", path)
    assert path.read_text() == 'backends.default = "parakeet"\n'
