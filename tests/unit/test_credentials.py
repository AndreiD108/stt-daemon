import os
import tomllib
from pathlib import Path

import pytest

from stt_daemon import credentials
from stt_daemon.credentials import (
    CredentialsError,
    load_credentials,
    parse_keys_stdin,
    write_key,
)

SECRET = "sk_secret_value_123"


def _write(path: Path, text: str, mode: int = 0o600) -> Path:
    path.write_text(text)
    path.chmod(mode)
    return path


def test_missing_file_means_no_keys(tmp_path: Path) -> None:
    creds = load_credentials(path=tmp_path / "credentials.toml", env={})
    assert creds.get("elevenlabs") is None


def test_reads_key_per_backend(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.toml", f'[elevenlabs]\napi_key = "{SECRET}"\n')
    creds = load_credentials(path=path, env={})
    assert creds.get("elevenlabs") == SECRET
    assert creds.get("parakeet") is None


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o400, 0o700])
def test_refuses_any_mode_but_600(tmp_path: Path, mode: int) -> None:
    path = _write(tmp_path / "c.toml", f'[elevenlabs]\napi_key = "{SECRET}"\n', mode)
    with pytest.raises(CredentialsError, match=r"chmod 600") as info:
        load_credentials(path=path, env={})
    assert SECRET not in str(info.value)


def test_refuses_file_owned_by_someone_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path / "c.toml", f'[elevenlabs]\napi_key = "{SECRET}"\n')
    real_uid = os.getuid()
    monkeypatch.setattr(credentials.os, "getuid", lambda: real_uid + 1)
    with pytest.raises(CredentialsError, match="owned by"):
        load_credentials(path=path, env={})


@pytest.mark.parametrize(
    "text",
    [
        f'[elevenlabs]\napi_key = "{SECRET}\n',
        f"[elevenlabs]\napi_key = {SECRET}\n",
        f'elevenlabs = "{SECRET}"\n',
        f'[elevenlabs]\napi_key = "{SECRET}"\nkey = "{SECRET}"\n',
    ],
)
def test_bad_file_errors_never_hold_the_key(tmp_path: Path, text: str) -> None:
    path = _write(tmp_path / "c.toml", text)
    with pytest.raises(CredentialsError) as info:
        load_credentials(path=path, env={})
    assert SECRET not in str(info.value)


def test_env_overrides_file_and_stdin_overrides_env(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.toml", '[elevenlabs]\napi_key = "from-file"\n')
    env = {"ELEVENLABS_API_KEY": "from-env"}
    assert load_credentials(path=path, env=env).get("elevenlabs") == "from-env"
    stdin = {"ELEVENLABS_API_KEY": "from-stdin"}
    assert load_credentials(stdin, path=path, env=env).get("elevenlabs") == "from-stdin"


def test_empty_env_var_does_not_hide_file_key(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.toml", '[elevenlabs]\napi_key = "from-file"\n')
    assert load_credentials(path=path, env={"ELEVENLABS_API_KEY": ""}).get("elevenlabs") == (
        "from-file"
    )


def test_repr_masks_values(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.toml", f'[elevenlabs]\napi_key = "{SECRET}"\n')
    creds = load_credentials(
        {"PARAKEET_API_KEY": SECRET}, path=path, env={"ELEVENLABS_API_KEY": SECRET}
    )
    assert SECRET not in repr(creds)
    assert SECRET not in str(creds)
    assert "elevenlabs" in repr(creds)


def test_write_key_creates_private_dir_and_file(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "credentials.toml"
    write_key("elevenlabs", f"  {SECRET}\n", path=path)
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_credentials(path=path, env={}).get("elevenlabs") == SECRET


def test_write_key_keeps_other_backends(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "c.toml",
        '[elevenlabs]\napi_key = "old"\n\n[parakeet]\napi_key = "other"\n',
    )
    write_key("elevenlabs", "new", path=path)
    assert tomllib.loads(path.read_text()) == {
        "elevenlabs": {"api_key": "new"},
        "parakeet": {"api_key": "other"},
    }
    assert [p.name for p in tmp_path.iterdir()] == ["c.toml"]


@pytest.mark.parametrize("key", ["", "   ", f"{SECRET} more", f"{SECRET}\x00"])
def test_write_key_rejects_bad_keys_without_echoing(tmp_path: Path, key: str) -> None:
    with pytest.raises(CredentialsError) as info:
        write_key("elevenlabs", key, path=tmp_path / "c.toml")
    assert SECRET not in str(info.value)
    assert not (tmp_path / "c.toml").exists()


def test_write_key_rejects_bad_backend_name(tmp_path: Path) -> None:
    with pytest.raises(CredentialsError, match="backend name"):
        write_key("../x", "k", path=tmp_path / "c.toml")


def test_parse_keys_stdin_reads_lines_and_skips_blanks() -> None:
    text = f"\nELEVENLABS_API_KEY={SECRET}\r\n  \nPARAKEET_API_KEY = x=y \n"
    assert parse_keys_stdin(text) == {"ELEVENLABS_API_KEY": SECRET, "PARAKEET_API_KEY": "x=y"}


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (SECRET, "expected NAME=value"),
        (f"export ELEVENLABS_API_KEY={SECRET}", "must look like"),
        (f"{SECRET}=x", "must look like"),
        ("ELEVENLABS_API_KEY=", "no value"),
        (f"ELEVENLABS_API_KEY={SECRET}\nELEVENLABS_API_KEY={SECRET}", "set twice"),
        (f"# comment {SECRET}", "expected NAME=value"),
    ],
)
def test_parse_keys_stdin_errors_never_hold_the_value(text: str, error: str) -> None:
    with pytest.raises(CredentialsError, match=error) as info:
        parse_keys_stdin(text)
    assert SECRET not in str(info.value)
