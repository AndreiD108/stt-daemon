from collections.abc import Callable
from dataclasses import replace

import pytest

from stt_daemon.backends.base import Backend, BackendUnavailable
from stt_daemon.backends.fake import FAKE_CAPABILITIES, FakeBackend
from stt_daemon.config import Config
from stt_daemon.credentials import Credentials
from stt_daemon.server.backend_set import Setup, load_setup, resolve
from stt_daemon.server.messages import RequestError, StartRequest


class ConfiguredFake(FakeBackend):
    def __init__(self, language: str | None) -> None:
        super().__init__()
        self._language = language

    @property
    def default_language(self) -> str | None:
        return self._language


def unavailable(_config: Config, _credentials: Credentials) -> Backend:
    raise BackendUnavailable("no_key", "no key; set one with `stt-daemon key set other`")


def _returning(backend: Backend) -> Callable[[Config, Credentials], Backend]:
    return lambda _config, _credentials: backend


def setup_of(*backends: Backend, default: str | None = None) -> Setup:
    config = Config()
    config.backends.default = default
    factories = {f"b{i}": _returning(b) for i, b in enumerate(backends)}
    return load_setup(config, Credentials(), {**factories, "other": unavailable})


def no_lookup(key: str) -> str | None:
    raise AssertionError("no lookup expected")


def refused(setup: Setup, **request: object) -> RequestError:
    with pytest.raises(RequestError) as caught:
        resolve(setup, StartRequest(**request), no_lookup)  # type: ignore[arg-type]
    return caught.value


def test_load_setup_skips_backends_not_set_up_and_picks_a_default() -> None:
    setup = setup_of(FakeBackend(), FakeBackend())
    assert list(setup.backends) == ["b0", "b1"]
    assert setup.unavailable == {"other": "no key; set one with `stt-daemon key set other`"}
    assert setup.default == "b0"
    assert setup_of(FakeBackend(), FakeBackend(), default="b1").default == "b1"
    assert setup_of(FakeBackend(), default="other").default == "b0"
    assert setup_of().default is None


def test_language_defaults_to_the_backends_config_then_auto() -> None:
    assert resolve(setup_of(FakeBackend()), StartRequest(), no_lookup).options.open.language == (
        "auto"
    )
    configured = setup_of(ConfiguredFake("en"))
    assert resolve(configured, StartRequest(), no_lookup).options.open.language == "en"
    request = StartRequest(language="auto")
    assert resolve(configured, request, no_lookup).options.open.language == "auto"
    no_auto = FakeBackend(capabilities=replace(FAKE_CAPABILITIES, languages=("en",)))
    assert resolve(setup_of(no_auto), StartRequest(), no_lookup).options.open.language is None


def test_language_checks() -> None:
    assert refused(setup_of(FakeBackend()), language="ro").code == "unsupported_language"
    assert refused(setup_of(ConfiguredFake("ro"))).code == "unsupported_language"
    none = FakeBackend(capabilities=replace(FAKE_CAPABILITIES, languages=()))
    assert refused(setup_of(none), language="en").code == "unsupported"


def test_features_the_backend_lacks_fail_the_start() -> None:
    plain = FakeBackend(
        capabilities=replace(FAKE_CAPABILITIES, partials=False, word_timestamps=False)
    )
    assert refused(setup_of(plain), partials=True).code == "unsupported"
    assert refused(setup_of(plain), word_timestamps=True).code == "unsupported"


def test_hints_the_backend_cant_use_are_dropped() -> None:
    blind = FakeBackend(capabilities=replace(FAKE_CAPABILITIES, hints=frozenset()))
    request = StartRequest(keyterms=("Scribbly",), previous_text="hi", context_key="doc")
    resolved = resolve(setup_of(blind), request, no_lookup)
    assert resolved.options.open.keyterms == ()
    assert resolved.options.open.previous_text is None
    assert resolved.context_key is None


def test_context_key_looks_up_only_when_previous_text_is_missing() -> None:
    setup = setup_of(FakeBackend())
    stored = {"doc": "last time"}
    resolved = resolve(setup, StartRequest(context_key="doc"), stored.get)
    assert resolved.options.open.previous_text == "last time"
    assert resolved.context_key == "doc"
    given = resolve(setup, StartRequest(context_key="doc", previous_text=""), no_lookup)
    assert given.options.open.previous_text == ""


def test_backend_choice_errors() -> None:
    assert refused(setup_of()).code == "no_backend"
    setup = setup_of(FakeBackend())
    error = refused(setup, backend="other")
    assert error.code == "unknown_backend"
    assert "stt-daemon key set other" in error.message
    assert refused(setup, backend="nope").code == "unknown_backend"
    assert resolve(setup, StartRequest(backend="b0"), no_lookup).backend is setup.backends["b0"]
