"""Config file, defaults, and per-OS paths."""

import dataclasses
import logging
import os
import re
import sys
import tempfile
import tomllib
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

log = logging.getLogger(__name__)

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
DEFAULT_MAX_SESSIONS = 2
ECHO_CANCEL = ("auto", "pipewire", "builtin", "off")

type EchoCancel = typing.Literal["pipewire", "builtin", "off"]


class ConfigError(Exception):
    pass


@dataclass
class AudioConfig:
    device: str = ""
    """Substring of the input device name; empty picks automatically."""
    keep_mic_open: bool = True
    buffer_ms: int = 5000
    pre_start_ms: int = 1000
    chunk_ms: int = 200
    echo_cancel: str = "auto"
    """For a system session's mic track: auto, pipewire, builtin or off."""


@dataclass
class VadConfig:
    threshold: float = 0.4
    min_speech_ms: int = 250
    min_silence_ms: int = 600
    """A pause is sent whole until it lasts this long; after that it is held back."""
    pause_keep_ms: int = 100
    """How much of a held-back pause is still sent, from its end."""


@dataclass
class SessionConfig:
    stop_wait_ms: int = 5000
    """How long a stop waits for the backend's last text before ending without it."""


@dataclass
class SlotConfig:
    evictable_after_s: int = 300


@dataclass
class ServerConfig:
    observers: bool = True
    """Whether a client may say `observe_all: true` and get every session's events."""


@dataclass
class ElevenLabsConfig:
    model_id: str = "scribe_v2_realtime"
    endpoint: str = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    language: str = "auto"
    keyterms: list[str] = field(default_factory=list[str])
    max_sessions: int = DEFAULT_MAX_SESSIONS


@dataclass
class ElevenLabsBatchConfig:
    """`language` and `keyterms` come from `[backends.elevenlabs]`."""

    model_id: str = "scribe_v2"
    endpoint: str = "https://api.elevenlabs.io/v1/speech-to-text"
    max_sessions: int = DEFAULT_MAX_SESSIONS


def default_num_threads() -> int:
    """Leaves two cores to the rest of the machine; at most 4."""
    return min(4, max(1, (os.cpu_count() or 1) - 2))


@dataclass
class ParakeetConfig:
    keyterms: list[str] = field(default_factory=list[str])
    hotwords_score: float = 1.5
    """Bias per keyterm token in beam search; too high and keyterms replace similar words."""
    max_sessions: int = DEFAULT_MAX_SESSIONS
    num_threads: int = field(default_factory=default_num_threads)
    max_chunk_seconds: float = 30.0
    """Audio sent without a pause is committed once it reaches this length."""
    unload_after_minutes: float = 10.0
    """The model leaves memory this long after the last Parakeet session; 0 keeps it."""


@dataclass
class BackendsConfig:
    default: str | None = None
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)
    elevenlabs_batch: ElevenLabsBatchConfig = field(
        default_factory=ElevenLabsBatchConfig, metadata={"key": "elevenlabs-batch"}
    )
    parakeet: ParakeetConfig = field(default_factory=ParakeetConfig)


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    slot: SlotConfig = field(default_factory=SlotConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    backends: BackendsConfig = field(default_factory=BackendsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def echo_cancel_mode(audio: AudioConfig, platform: str | None = None) -> EchoCancel:
    """`auto` is the PipeWire module on Linux, the built-in canceller elsewhere."""
    platform = sys.platform if platform is None else platform
    if audio.echo_cancel == "auto":
        return "pipewire" if platform == "linux" else "builtin"
    return typing.cast(EchoCancel, audio.echo_cancel)


def config_dir(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    base = env.get("XDG_CONFIG_HOME", "")
    root = Path(base) if os.path.isabs(base) else Path.home() / ".config"
    return root / "stt-daemon"


def data_dir(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    base = env.get("XDG_DATA_HOME", "")
    root = Path(base) if os.path.isabs(base) else Path.home() / ".local" / "share"
    return root / "stt-daemon"


def config_path(env: Mapping[str, str] | None = None) -> Path:
    return config_dir(env) / "config.toml"


def socket_path(env: Mapping[str, str] | None = None, platform: str | None = None) -> Path:
    """The daemon's socket; the daemon and its clients both use this."""
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if override := env.get("STT_DAEMON_SOCKET"):
        return Path(override)
    if platform == "darwin":
        base = env.get("TMPDIR") or tempfile.gettempdir()
    else:
        base = env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(base) / "stt-daemon" / "stt.sock"


def load_config(path: Path | None = None) -> Config:
    """Reads the config file; a missing file gives all defaults."""
    path = config_path() if path is None else path
    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        return Config()
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from None
    except OSError as exc:
        raise ConfigError(f"{path}: {exc.strerror}") from None
    config = _build(Config, raw, "", path)
    _validate(config, path)
    return config


def _build[T: DataclassInstance](cls: type[T], raw: dict[str, Any], prefix: str, path: Path) -> T:
    hints = typing.get_type_hints(cls)
    keys = {f.metadata.get("key", f.name): f.name for f in dataclasses.fields(cls)}
    for key in raw:
        if key not in keys:
            log.warning("%s: unknown key %s%s, ignored", path, prefix, key)
    values = {
        name: _convert(hints[name], raw[key], f"{prefix}{key}", path)
        for key, name in keys.items()
        if key in raw
    }
    return cls(**values)


def _convert(hint: Any, value: Any, key: str, path: Path) -> Any:
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, dict):
            raise _type_error(path, key, "a table", value)
        table = typing.cast(dict[str, Any], value)
        return _build(typing.cast("type[DataclassInstance]", hint), table, f"{key}.", path)
    if isinstance(hint, types.UnionType):
        hint = next(arg for arg in typing.get_args(hint) if arg is not type(None))
    if hint is bool:
        if not isinstance(value, bool):
            raise _type_error(path, key, "true or false", value)
        return value
    if hint is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise _type_error(path, key, "an integer", value)
        return value
    if hint is float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise _type_error(path, key, "a number", value)
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise _type_error(path, key, "a string", value)
        return value
    if typing.get_origin(hint) is list:
        items = typing.cast(list[Any], value) if isinstance(value, list) else None
        if items is None or not all(isinstance(item, str) for item in items):
            raise _type_error(path, key, "a list of strings", value)
        return list(items)
    raise TypeError(f"no config conversion for {hint!r}")


def _type_error(path: Path, key: str, expected: str, value: Any) -> ConfigError:
    return ConfigError(f"{path}: {key} must be {expected}, got {type(value).__name__}")


def _validate(config: Config, path: Path) -> None:
    def check(ok: bool, key: str, rule: str) -> None:
        if not ok:
            raise ConfigError(f"{path}: {key} must be {rule}")

    audio, vad = config.audio, config.vad
    check(audio.buffer_ms > 0, "audio.buffer_ms", "positive")
    check(0 <= audio.pre_start_ms <= audio.buffer_ms, "audio.pre_start_ms", "0 to audio.buffer_ms")
    check(50 <= audio.chunk_ms <= 1000, "audio.chunk_ms", "50 to 1000")
    check(audio.echo_cancel in ECHO_CANCEL, "audio.echo_cancel", f"one of {', '.join(ECHO_CANCEL)}")
    check(0.0 <= vad.threshold <= 1.0, "vad.threshold", "0 to 1")
    check(vad.min_speech_ms >= 0, "vad.min_speech_ms", "0 or more")
    check(vad.min_silence_ms > 0, "vad.min_silence_ms", "positive")
    check(
        0 <= vad.pause_keep_ms <= vad.min_silence_ms, "vad.pause_keep_ms", "0 to vad.min_silence_ms"
    )
    check(config.session.stop_wait_ms > 0, "session.stop_wait_ms", "positive")
    parakeet = config.backends.parakeet
    check(
        0 < parakeet.hotwords_score <= 10, "backends.parakeet.hotwords_score", "above 0, at most 10"
    )
    check(parakeet.num_threads >= 1, "backends.parakeet.num_threads", "1 or more")
    check(parakeet.max_chunk_seconds >= 5, "backends.parakeet.max_chunk_seconds", "5 or more")
    check(parakeet.unload_after_minutes >= 0, "backends.parakeet.unload_after_minutes", "0 or more")
    for name, section in (
        ("elevenlabs", config.backends.elevenlabs),
        ("elevenlabs-batch", config.backends.elevenlabs_batch),
        ("parakeet", config.backends.parakeet),
    ):
        check(section.max_sessions >= 1, f"backends.{name}.max_sessions", "1 or more")
    check(config.slot.evictable_after_s > 0, "slot.evictable_after_s", "positive")
    level = config.logging.level.upper()
    check(level in LOG_LEVELS, "logging.level", f"one of {', '.join(LOG_LEVELS)}")
    config.logging.level = level


_HEADER = re.compile(r"^\s*\[\s*([^\[\]]+?)\s*\]\s*(#.*)?$")


def set_config_string(section: str, key: str, value: str, path: Path | None = None) -> None:
    """Sets `[section] key = "value"` in the config file, keeping the rest of its text."""
    path = config_path() if path is None else path
    original, expected = _read_toml(path)
    _table(expected, section, path)[key] = value
    lines = original.splitlines(keepends=True)
    new_line = f"{key} = {toml_string(value)}\n"
    found = _find_key(lines, section, key)
    if found is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines += ([] if not lines else ["\n"]) + [f"[{section}]\n", new_line]
    elif (existing := found[1]) is None:
        lines.insert(found[0] + 1, new_line)
    else:
        lines[existing] = new_line
    _write_checked(path, "".join(lines), expected, f"set {section}.{key}")


def remove_config_key(section: str, key: str, path: Path | None = None) -> None:
    """Deletes `[section] key` from the config file, if there, keeping the rest of its text."""
    path = config_path() if path is None else path
    if not path.exists():
        return
    original, expected = _read_toml(path)
    if _table(expected, section, path).pop(key, None) is None:
        return
    lines = original.splitlines(keepends=True)
    found = _find_key(lines, section, key)
    if found is not None and found[1] is not None:
        del lines[found[1]]
    _write_checked(path, "".join(lines), expected, f"remove {section}.{key}")


def _read_toml(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        original = path.read_text()
    except FileNotFoundError:
        original = ""
    try:
        return original, tomllib.loads(original)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from None


def _table(root: dict[str, Any], section: str, path: Path) -> dict[str, Any]:
    node = root
    for part in section.split("."):
        child: Any = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"{path}: {section} is not a table; edit the file by hand")
        node = typing.cast(dict[str, Any], child)
    return node


def _find_key(lines: list[str], section: str, key: str) -> tuple[int, int | None] | None:
    """(header line, key line or None) of `[section]`; None without that header."""
    header_at = next(
        (i for i, line in enumerate(lines) if (m := _HEADER.match(line)) and m[1] == section),
        None,
    )
    if header_at is None:
        return None
    end = next(
        (
            i
            for i in range(header_at + 1, len(lines))
            if _HEADER.match(lines[i]) or lines[i].lstrip().startswith("[[")
        ),
        len(lines),
    )
    key_line = re.compile(rf"^\s*{re.escape(key)}\s*=")
    return header_at, next((i for i in range(header_at + 1, end) if key_line.match(lines[i])), None)


def _write_checked(path: Path, text: str, expected: dict[str, Any], change: str) -> None:
    # The line edit is textual; it is kept only if it made exactly this one change.
    try:
        written = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        written = None
    if written != expected:
        raise ConfigError(f"{path}: could not {change}; edit the file by hand")
    atomic_write(path, text)


def atomic_write(path: Path, text: str, mode: int | None = None) -> None:
    """Replaces `path` in one step. `mode` None keeps the file's mode (new files: 644)."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if mode is None:
        try:
            mode = path.stat().st_mode & 0o777
        except FileNotFoundError:
            mode = 0o644
    # mkstemp creates the file as 600, so it is never readable by others before the chmod
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            os.fchmod(f.fileno(), mode)
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def toml_string(value: str) -> str:
    """A TOML basic string literal for `value`."""
    out = ['"']
    for ch in value:
        if ch in '"\\':
            out.append("\\" + ch)
        elif ch < " " or ch == "\x7f":
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)
