"""API keys: credentials.toml, `<BACKEND>_API_KEY` env vars, and keys read from stdin.

Lookup order: stdin keys, then env, then the file. No error message ever holds a key.
"""

import os
import re
import stat
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from stt_daemon.config import atomic_write, config_dir, toml_string

_KEY_NAME = re.compile(r"[A-Z][A-Z0-9_]*_API_KEY")
_BACKEND_NAME = re.compile(r"[a-z][a-z0-9_]*")


class CredentialsError(Exception):
    pass


def credentials_path(env: Mapping[str, str] | None = None) -> Path:
    return config_dir(env) / "credentials.toml"


def key_env_name(backend: str) -> str:
    return f"{backend.upper()}_API_KEY"


@dataclass(frozen=True, repr=False)
class Credentials:
    file_keys: Mapping[str, str] = field(default_factory=dict[str, str])
    env_keys: Mapping[str, str] = field(default_factory=dict[str, str])
    stdin_keys: Mapping[str, str] = field(default_factory=dict[str, str])

    def get(self, backend: str) -> str | None:
        name = key_env_name(backend)
        return self.stdin_keys.get(name) or self.env_keys.get(name) or self.file_keys.get(backend)

    def __repr__(self) -> str:
        return (
            f"Credentials(file={sorted(self.file_keys)}, env={sorted(self.env_keys)}, "
            f"stdin={sorted(self.stdin_keys)})"
        )


def load_credentials(
    stdin_keys: Mapping[str, str] | None = None,
    *,
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Credentials:
    """Reads the file and env now; `stdin_keys` (from `parse_keys_stdin`) win over both."""
    env = os.environ if env is None else env
    return Credentials(
        file_keys=_read_file(credentials_path(env) if path is None else path),
        env_keys={k: v for k, v in env.items() if _KEY_NAME.fullmatch(k) and v},
        stdin_keys=dict(stdin_keys or {}),
    )


def _read_file(path: Path) -> dict[str, str]:
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise CredentialsError(f"{path}: {exc.strerror}") from None
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if st.st_uid != os.getuid():
            raise CredentialsError(
                f"{path} is owned by uid {st.st_uid}, not by you; refusing to read it"
            )
        mode = stat.S_IMODE(st.st_mode)
        if mode != 0o600:
            raise CredentialsError(
                f"{path} has mode {mode:03o}, must be 600; run: chmod 600 {path}"
            )
        try:
            raw = tomllib.load(f)
        except tomllib.TOMLDecodeError:
            raise CredentialsError(f"{path} is not valid TOML") from None
    keys: dict[str, str] = {}
    for backend, table in raw.items():
        if not isinstance(table, dict):
            raise CredentialsError(f"{path}: [{backend}] must be a table with api_key")
        entries = cast(dict[str, Any], table)
        extra = sorted(set(entries) - {"api_key"})
        if extra:
            raise CredentialsError(f"{path}: [{backend}] has unknown keys: {', '.join(extra)}")
        key = entries.get("api_key")
        if not isinstance(key, str) or not key:
            raise CredentialsError(f"{path}: [{backend}] api_key must be a non-empty string")
        keys[backend] = key
    return keys


def write_key(backend: str, key: str, *, path: Path | None = None) -> Path:
    """Stores `key` for `backend`, keeping the other backends' keys. Returns the file's path."""
    if not _BACKEND_NAME.fullmatch(backend):
        raise CredentialsError(f"not a backend name: {backend!r}")
    key = key.strip()
    if not key:
        raise CredentialsError("the key is empty")
    if any(ch.isspace() or not ch.isprintable() for ch in key):
        raise CredentialsError("the key has spaces or control characters in it")
    path = credentials_path() if path is None else path
    keys = _read_file(path)
    keys[backend] = key
    text = "".join(
        f"[{name}]\napi_key = {toml_string(value)}\n\n" for name, value in sorted(keys.items())
    )
    atomic_write(path, text.rstrip("\n") + "\n", mode=0o600)
    return path


def parse_keys_stdin(text: str) -> dict[str, str]:
    """Parses `NAME=value` lines, like `ELEVENLABS_API_KEY=...`; blank lines are skipped."""
    keys: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        name, sep, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not sep:
            raise CredentialsError(f"keys on stdin, line {number}: expected NAME=value")
        if not _KEY_NAME.fullmatch(name):
            raise CredentialsError(
                f"keys on stdin, line {number}: the name must look like ELEVENLABS_API_KEY"
            )
        if not value:
            raise CredentialsError(f"keys on stdin, line {number}: {name} has no value")
        if name in keys:
            raise CredentialsError(f"keys on stdin, line {number}: {name} is set twice")
        keys[name] = value
    return keys
