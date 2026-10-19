"""The backends set up on this machine, and a `start_session` request checked against them."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..backends import BACKENDS
from ..backends.base import Backend, BackendUnavailable, FileBackend, OpenOptions
from ..backends.plugins import NO_PLUGINS, Plugins, load_plugins, plugins_dir
from ..config import Config
from ..credentials import Credentials
from ..session.events import SessionKind
from ..session.options import SessionOptions
from .messages import RequestError, StartRequest

log = logging.getLogger(__name__)

type Factory = Callable[[Config, Credentials], Backend]

FACTORIES: Mapping[str, Factory] = {name: cls.create for name, cls in BACKENDS.items()}

NO_BACKEND = (
    "no backend is set up; run `stt-daemon key set elevenlabs` or "
    "`stt-daemon backend install parakeet`, then start again"
)


@dataclass(frozen=True)
class Setup:
    config: Config
    backends: Mapping[str, Backend]
    """Set up here, in registry order, plugins last."""
    unavailable: Mapping[str, BackendUnavailable]
    """Known but not set up: name → what is missing."""
    default: str | None
    failed_files: Mapping[str, BackendUnavailable] = field(
        default_factory=dict[str, BackendUnavailable]
    )
    """Plugin files that did not load: file name → why."""

    def to_wire(self) -> dict[str, object]:
        return {"backends": self.backends_wire(), "default_backend": self.default}

    def capacities(self) -> dict[str, int]:
        return {name: b.max_sessions for name, b in self.backends.items()}

    def backends_wire(self) -> list[dict[str, object]]:
        return [
            {
                "name": name,
                "languages": list(b.capabilities.languages),
                "word_timestamps": gives_word_times(b),
                "partials": b.capabilities.partials,
                "hints": sorted(b.capabilities.hints),
                "diarize": b.capabilities.diarize,
                "kinds": list(kinds_of(b)),
            }
            for name, b in self.backends.items()
        ]

    def status_wire(self) -> dict[str, list[dict[str, Any]]]:
        """Every backend known here with its status; plugin files that did not load."""
        statuses = {name: "ready" for name in self.backends} | {
            name: status_of(problem) for name, problem in self.unavailable.items()
        }
        return {
            "backends": [
                {"name": name, "status": status, "default": name == self.default}
                for name, status in statuses.items()
            ],
            "failed_files": [
                {"file": file, "status": status_of(problem)}
                for file, problem in self.failed_files.items()
            ],
        }

    def log_failures(self) -> None:
        """Plugins that failed, one log line each with the traceback."""
        named = {f"backend {name}": p for name, p in self.unavailable.items()}
        files = {f"plugin file {file}": p for file, p in self.failed_files.items()}
        for where, problem in (named | files).items():
            if problem.code == "failed":
                detail = f"\n{problem.detail.rstrip()}" if problem.detail else ""
                log.error("%s failed: %s%s", where, problem.message, detail)


def status_of(problem: BackendUnavailable) -> str:
    match problem.code:
        case "no_key":
            return "no key"
        case "no_model" | "not_installed":
            return "not installed"
        case "failed":
            return f"failed: {problem.message}"
        case _:
            return problem.message


def installed_plugins() -> Plugins:
    """The plugins in the user's plugin directory."""
    return load_plugins(plugins_dir(), BACKENDS)


def load_setup(
    config: Config,
    credentials: Credentials,
    factories: Mapping[str, Factory] = FACTORIES,
    plugins: Plugins = NO_PLUGINS,
) -> Setup:
    backends: dict[str, Backend] = {}
    unavailable: dict[str, BackendUnavailable] = {}
    for name, create in {**factories, **plugins.factories()}.items():
        try:
            backends[name] = create(config, credentials)
        except BackendUnavailable as exc:
            unavailable[name] = exc
    for name in config.backends.plugins:
        if name not in backends and name not in unavailable:
            log.warning("config: [backends.%s] names no backend; ignored", name)
    default = config.backends.default
    if default in backends and isinstance(backends[default], FileBackend):
        log.warning("default backend %s takes transcribe sessions only; ignoring it", default)
        default = None
    elif default is not None and default not in backends:
        log.warning("default backend %s is not set up; using the first one set up", default)
        default = None
    if default is None:
        default = next((n for n, b in backends.items() if not isinstance(b, FileBackend)), None)
    return Setup(config, backends, unavailable, default, plugins.failed)


def kinds_of(backend: Backend) -> tuple[SessionKind, ...]:
    """The session kinds the backend takes."""
    if isinstance(backend, FileBackend):
        return ("transcribe",)
    return ("mic", "system", "transcribe")


def gives_word_times(backend: Backend) -> bool:
    """Its own, or estimated from the audio for answers without them (not for whole files)."""
    return backend.capabilities.word_timestamps or not isinstance(backend, FileBackend)


@dataclass(frozen=True)
class Resolved:
    backend: Backend
    options: SessionOptions
    context_key: str | None
    """None when the request had none or the backend takes no `previous_text`."""


def resolve(
    setup: Setup,
    req: StartRequest,
    lookup: Callable[[str], str | None],
    *,
    kind: SessionKind = "mic",
) -> Resolved:
    """Checks the request; `lookup` gives the stored text for a `context_key`."""
    backend = _backend(setup, req.backend, kind)
    caps = backend.capabilities
    name = backend.name
    if kind not in kinds_of(backend):
        raise RequestError(
            "unsupported",
            f"{name} takes transcribe sessions only; use another backend for {kind} sessions",
        )
    if req.diarize and not caps.diarize:
        raise RequestError(
            "unsupported_option",
            f"{name} can't tell speakers apart; send without diarize and num_speakers",
            field="diarize",
        )
    if req.partials and not caps.partials:
        raise RequestError("unsupported", f"{name} has no partials; start without partials")
    if req.word_timestamps and not gives_word_times(backend):
        raise RequestError(
            "unsupported", f"{name} has no word times; start without word_timestamps"
        )
    language = _language(backend, req.language)
    keyterms = req.keyterms if "keyterms" in caps.hints else ()
    previous_text, context_key = None, None
    if "previous_text" in caps.hints:
        previous_text, context_key = req.previous_text, req.context_key
        if previous_text is None and context_key is not None:
            previous_text = lookup(context_key)
    open_options = OpenOptions(
        language=language,
        keyterms=keyterms,
        previous_text=previous_text,
        partials=req.partials,
        diarize=req.diarize,
        num_speakers=req.num_speakers,
    )
    return Resolved(
        backend, SessionOptions(open_options, word_timestamps=req.word_timestamps), context_key
    )


def _backend(setup: Setup, requested: str | None, kind: SessionKind) -> Backend:
    if not setup.backends or (requested is None and setup.default is None):
        raise RequestError("no_backend", NO_BACKEND)
    name = requested or setup.default
    assert name is not None
    if requested is None and kind == "transcribe":
        name = next(
            (
                n
                for n, b in setup.backends.items()
                if isinstance(b, FileBackend) and b.default_for == name
            ),
            name,
        )
    if (backend := setup.backends.get(name)) is not None:
        return backend
    ready = ", ".join(setup.backends)
    if name in setup.unavailable:
        reason = setup.unavailable[name].message
        message = f"{name} is not set up: {reason}; or use one that is: {ready}"
    else:
        message = f"there is no backend named {name!r}; the backends set up here: {ready}"
    raise RequestError("unknown_backend", message)


def _language(backend: Backend, requested: str | None) -> str | None:
    languages = backend.capabilities.languages
    if requested is not None:
        if not languages:
            raise RequestError(
                "unsupported", f"{backend.name} takes no language; start without language"
            )
        if requested not in languages:
            raise RequestError(
                "unsupported_language",
                f"{backend.name} has no language {requested!r}; "
                "pick one from its languages in welcome",
            )
        return requested
    language = backend.default_language or ("auto" if "auto" in languages else None)
    if language is not None and languages and language not in languages:
        raise RequestError(
            "unsupported_language",
            f"the config sets {backend.name}'s language to {language!r}, which it doesn't have; "
            f"fix backends.{backend.name}.language in config.toml or send a language",
        )
    return language
