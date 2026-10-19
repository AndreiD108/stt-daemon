"""Backend plugins: one `stt_daemon.plugin.Backend` subclass per file in the plugin directory,
imported by path and run through the chunk-per-commit connection."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
import sys
import traceback
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from .. import plugin
from ..config import DEFAULT_MAX_SESSIONS, Config, config_dir
from ..credentials import Credentials
from .base import (
    Backend,
    BackendUnavailable,
    Capabilities,
    CommitPolicy,
    EventSink,
    OpenFailed,
    OpenOptions,
    Word,
)
from .chunked import Chunk, ChunkedConnection, Transcribed, Turns

NAME = re.compile(r"[a-z][a-z0-9-]*")
TRANSCRIBE_TIMEOUT_S = 120.0
DEFAULT_MAX_CHUNK_S = 30.0
MIN_MAX_CHUNK_S = 5.0

log = logging.getLogger(__name__)

type Factory = Callable[[Config, Credentials], Backend]


def plugins_dir(env: Mapping[str, str] | None = None) -> Path:
    return config_dir(env) / "backends"


@dataclass(frozen=True)
class Plugins:
    classes: Mapping[str, type[plugin.Backend]]
    """Name → class, in file name order."""
    failed: Mapping[str, BackendUnavailable]
    """File name → why it did not load (code "failed")."""

    def factories(self) -> dict[str, Factory]:
        return {name: _factory(cls) for name, cls in self.classes.items()}


NO_PLUGINS = Plugins({}, {})


def load_plugins(directory: Path, reserved: Collection[str]) -> Plugins:
    """Imports every `*.py` in `directory`; `reserved` are the built-in names. A missing directory
    has no plugins. Never raises for a plugin's fault."""
    classes: dict[str, type[plugin.Backend]] = {}
    files: dict[str, str] = {}
    failed: dict[str, BackendUnavailable] = {}
    for path in sorted(directory.glob("*.py")) if directory.is_dir() else ():
        try:
            cls = _load_file(path)
            name = cls.name
            if name in reserved:
                raise _PluginError(f"{name!r} is a built-in backend's name; rename the plugin")
            if name in files:
                raise _PluginError(f"{name!r} is also the name of the plugin in {files[name]}")
        except _PluginError as exc:
            failed[path.name] = BackendUnavailable("failed", str(exc))
        except (Exception, SystemExit) as exc:
            failed[path.name] = _failure(exc)
        else:
            classes[name] = cls
            files[name] = path.name
    return Plugins(classes, failed)


class _PluginError(Exception):
    pass


def scrub(text: str, secret: str | None) -> str:
    """`text` without the key in it."""
    return text.replace(secret, "[key]") if secret else text


def error_line(exc: BaseException, secret: str | None = None) -> str:
    """`Type: first line of the message`, without the key."""
    first = (str(exc).strip().splitlines() or [""])[0]
    return scrub(f"{type(exc).__name__}: {first}" if first else type(exc).__name__, secret)


def _failure(exc: BaseException, secret: str | None = None) -> BackendUnavailable:
    detail = scrub("".join(traceback.format_exception(exc)), secret)
    return BackendUnavailable("failed", error_line(exc, secret), detail=detail)


def _load_file(path: Path) -> type[plugin.Backend]:
    module_name = f"stt_daemon_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise _PluginError(f"can't import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses in the plugin look their module up here
    writes_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # no __pycache__ in the plugin directory
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.dont_write_bytecode = writes_bytecode
    found = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, plugin.Backend)
        and value is not plugin.Backend
        and value.__module__ == module_name
    ]
    if not found:
        raise _PluginError("defines no stt_daemon.plugin.Backend subclass")
    if len(found) > 1:
        names = ", ".join(c.__name__ for c in found)
        raise _PluginError(f"defines {len(found)} Backend subclasses ({names}); keep one per file")
    cls = found[0]
    name: object = vars(cls).get("name") or getattr(cls, "name", None)
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise _PluginError(
            f"{cls.__name__}.name must be lowercase letters, digits and '-', "
            f"starting with a letter; got {name!r}"
        )
    capabilities = getattr(cls, "capabilities", None)
    if not isinstance(capabilities, plugin.Capabilities):
        raise _PluginError(f"{cls.__name__}.capabilities must be an stt_daemon.plugin.Capabilities")
    if cls.transcribe is plugin.Backend.transcribe:
        raise _PluginError(f"{cls.__name__} does not implement transcribe")
    return cls


def has_install(cls: type[plugin.Backend]) -> bool:
    return cls.install is not plugin.Backend.install


@dataclass(frozen=True)
class Limits:
    max_sessions: int
    max_chunk_s: float


def limits(name: str, section: Mapping[str, Any]) -> Limits:
    """`max_sessions` and `max_chunk_seconds` of `[backends.<name>]`; raises BackendUnavailable."""
    sessions = section.get("max_sessions", DEFAULT_MAX_SESSIONS)
    chunk = section.get("max_chunk_seconds", DEFAULT_MAX_CHUNK_S)
    if not isinstance(sessions, int) or isinstance(sessions, bool) or sessions < 1:
        raise BackendUnavailable("failed", f"backends.{name}.max_sessions must be 1 or more")
    if not isinstance(chunk, int | float) or isinstance(chunk, bool) or chunk < MIN_MAX_CHUNK_S:
        raise BackendUnavailable(
            "failed", f"backends.{name}.max_chunk_seconds must be {MIN_MAX_CHUNK_S:g} or more"
        )
    return Limits(sessions, float(chunk))


def _factory(cls: type[plugin.Backend]) -> Factory:
    name = cls.name

    def create(config: Config, credentials: Credentials) -> Backend:
        section = config.backends.plugins.get(name, {})
        bounds = limits(name, section)
        key = credentials.get(name)
        if cls.needs_key and not key:
            raise BackendUnavailable(
                "no_key", f"no API key; set one with `stt-daemon key set {name}`"
            )
        try:
            instance = cls()
            instance.setup(key, dict(section))
            ready = instance.available()
        except Exception as exc:
            raise _failure(exc, key) from None
        if not ready:
            hint = f"; run `stt-daemon backend install {name}`" if has_install(cls) else ""
            raise BackendUnavailable("not_installed", f"{name} is not installed{hint}")
        return PluginBackend(instance, bounds, secret=key)

    return create


class PluginBackend(Backend):
    """A plugin as a backend: each commit's chunk goes to `transcribe`, one call at a time,
    dictation's first."""

    def __init__(
        self,
        instance: plugin.Backend,
        bounds: Limits,
        *,
        secret: str | None = None,
        timeout_s: float = TRANSCRIBE_TIMEOUT_S,
    ) -> None:
        """`secret`: the plugin's key, kept out of error messages and logs."""
        self.name = type(instance).name
        self.plugin = instance
        self._secret = secret
        self._max_sessions = bounds.max_sessions
        self._timeout_s = timeout_s
        self._turns = Turns()
        caps = instance.capabilities
        hints = {"keyterms": caps.keyterms, "previous_text": caps.previous_text}
        self.capabilities = Capabilities(
            languages=caps.languages,
            word_timestamps=caps.word_times,
            partials=False,
            hints=frozenset(h for h, on in hints.items() if on),
            max_connection_s=None,
            commit_policy=CommitPolicy(at_pauses=True, max_chunk_s=bounds.max_chunk_s),
            paced_replay=False,
            diarize=caps.diarize,
        )

    def __repr__(self) -> str:
        return f"PluginBackend({self.name!r})"

    @property
    def max_sessions(self) -> int:
        return self._max_sessions

    @classmethod
    def create(cls, config: Config, credentials: Credentials) -> Self:
        raise BackendUnavailable("failed", "plugins are set up by the plugin loader")

    async def open(
        self, options: OpenOptions, connection_id: int, sink: EventSink
    ) -> ChunkedConnection:
        language = None if options.language in (None, "auto") else options.language
        wanted = plugin.Options(
            language=language,
            keyterms=tuple(options.keyterms),
            previous_text=options.previous_text,
            diarize=options.diarize,
            num_speakers=options.num_speakers,
        )
        first = options.dictation

        async def transcribe(chunk: Chunk) -> Transcribed:
            pcm = np.clip(np.rint(chunk.audio * 32768.0), -32768, 32767).astype("<i2").tobytes()
            return to_transcribed(await self._call(pcm, wanted, first=first), chunk)

        return ChunkedConnection(connection_id, sink, transcribe, label=self.name)

    async def _call(self, pcm: bytes, options: plugin.Options, *, first: bool) -> plugin.Result:
        """The plugin's answer. A call holds the plugin's turn until it ends, even when its caller
        stops waiting (cancelled, or past the timeout): it may still be running in a thread."""
        await self._turns.acquire(first=first)
        try:
            call = asyncio.ensure_future(self.plugin.transcribe(pcm, options))
        except BaseException:
            self._turns.release()
            raise
        call.add_done_callback(self._call_done)
        try:
            result = await asyncio.wait_for(asyncio.shield(call), self._timeout_s)
        except TimeoutError:
            if call.done():
                raise self._failed(call.exception() or TimeoutError()) from None
            log.error("%s gave no answer within %g s", self.name, self._timeout_s)
            raise OpenFailed(
                "backend_error",
                f"{self.name} gave no answer within {self._timeout_s:g} s",
                retryable=False,
            ) from None
        except OpenFailed:
            raise
        except Exception as exc:
            raise self._failed(exc) from None
        if not isinstance(result, plugin.Result):  # pyright: ignore[reportUnnecessaryIsInstance]
            wrong = type(result).__name__
            raise self._failed(TypeError(f"transcribe returned {wrong}, not a Result"))
        return result

    def _call_done(self, call: asyncio.Future[plugin.Result]) -> None:
        self._turns.release()
        if not call.cancelled():
            call.exception()  # retrieved, so a call nobody waits for any more logs nothing

    def _failed(self, exc: BaseException) -> OpenFailed:
        detail = scrub("".join(traceback.format_exception(exc)), self._secret)
        log.error("%s failed:\n%s", self.name, detail.rstrip())
        message = scrub(f"{self.name} failed: {exc}", self._secret)
        return OpenFailed("backend_error", message, retryable=False)


def to_transcribed(result: plugin.Result, chunk: Chunk) -> Transcribed:
    """Word times move to connection seconds, kept inside the chunk; words without times give
    None, so the daemon estimates them from the audio."""
    words = [w for w in result.words if w.text.strip()]
    text = result.text.strip() or " ".join(w.text.strip() for w in words)
    length = chunk.end_s - chunk.start_s
    timed: list[Word] = []
    for w in words:
        if w.start is None or w.end is None:
            return Transcribed(text, None, _speakers(words, text))
        start = min(max(w.start, 0.0), length)
        end = min(max(w.end, start), length)
        timed.append(Word(w.text.strip(), chunk.start_s + start, chunk.start_s + end, w.speaker))
    if not timed:
        return Transcribed(text, None if text else ())
    return Transcribed(text, tuple(timed))


def _speakers(words: list[plugin.Word], text: str) -> tuple[str | None, ...] | None:
    """Each word's speaker per whitespace-split word of `text`; None without any, or when `text`
    is not the words."""
    if not any(w.speaker for w in words):
        return None
    speakers = tuple(w.speaker for w in words for _ in w.text.split())
    return speakers if len(speakers) == len(text.split()) else None
