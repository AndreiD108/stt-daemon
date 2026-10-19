"""The backends set up on this machine, and a `start_session` request checked against them."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..backends import BACKENDS
from ..backends.base import Backend, BackendUnavailable, OpenOptions
from ..config import Config
from ..credentials import Credentials
from ..session.session import SessionOptions
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
    """Set up here, in registry order."""
    unavailable: Mapping[str, str]
    """Known but not set up: name → what is missing."""
    default: str | None

    def to_wire(self) -> dict[str, object]:
        return {"backends": self.backends_wire(), "default_backend": self.default}

    def capacities(self) -> dict[str, int]:
        return {name: b.max_sessions for name, b in self.backends.items()}

    def backends_wire(self) -> list[dict[str, object]]:
        return [
            {
                "name": name,
                "languages": list(b.capabilities.languages),
                "word_timestamps": b.capabilities.word_timestamps,
                "partials": b.capabilities.partials,
                "hints": sorted(b.capabilities.hints),
            }
            for name, b in self.backends.items()
        ]


def load_setup(
    config: Config, credentials: Credentials, factories: Mapping[str, Factory] = FACTORIES
) -> Setup:
    backends: dict[str, Backend] = {}
    unavailable: dict[str, str] = {}
    for name, create in factories.items():
        try:
            backends[name] = create(config, credentials)
        except BackendUnavailable as exc:
            unavailable[name] = exc.message
    default = config.backends.default
    if default not in backends:
        if default is not None:
            log.warning("default backend %s is not set up; using the first one set up", default)
        default = next(iter(backends), None)
    return Setup(config, backends, unavailable, default)


@dataclass(frozen=True)
class Resolved:
    backend: Backend
    options: SessionOptions
    context_key: str | None
    """None when the request had none or the backend takes no `previous_text`."""


def resolve(setup: Setup, req: StartRequest, lookup: Callable[[str], str | None]) -> Resolved:
    """Checks the request; `lookup` gives the stored text for a `context_key`."""
    backend = _backend(setup, req.backend)
    caps = backend.capabilities
    name = backend.name
    if req.partials and not caps.partials:
        raise RequestError("unsupported", f"{name} has no partials; start without partials")
    if req.word_timestamps and not caps.word_timestamps:
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
        language=language, keyterms=keyterms, previous_text=previous_text, partials=req.partials
    )
    return Resolved(
        backend, SessionOptions(open_options, word_timestamps=req.word_timestamps), context_key
    )


def _backend(setup: Setup, requested: str | None) -> Backend:
    if not setup.backends:
        raise RequestError("no_backend", NO_BACKEND)
    name = requested or setup.default
    assert name is not None
    if (backend := setup.backends.get(name)) is not None:
        return backend
    ready = ", ".join(setup.backends)
    if name in setup.unavailable:
        reason = setup.unavailable[name]
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
