"""The daemon process: mic capture, slot and socket server, until SIGTERM or SIGINT."""

import asyncio
import logging
import signal
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from .audio.capture import CaptureSource, MicCapture
from .backends import parakeet
from .config import Config, ConfigError, load_config, socket_path
from .credentials import CredentialsError, load_credentials
from .server.backend_set import Setup, load_setup
from .server.socket import Server, SocketInUse

log = logging.getLogger(__name__)


class StartupFailed(Exception):
    pass


async def serve(
    *,
    path: Path,
    capture: CaptureSource,
    setup: Setup,
    reload_setup: Callable[[], Setup],
    stop: asyncio.Event,
) -> None:
    """Runs until `stop` is set. Raises `StartupFailed` or `SocketInUse`."""
    try:
        await capture.start()
    except Exception as exc:
        raise StartupFailed(f"the mic did not open: {exc}") from exc
    try:
        server = Server(path=path, capture=capture, setup=setup, reload_setup=reload_setup)
        await server.start()
        try:
            await stop.wait()
            log.info("shutting down")
        finally:
            await server.close()
    finally:
        await capture.stop()


def reload(
    started: Config, config_file: Path | None, stdin_keys: Mapping[str, str] | None
) -> Setup:
    """The setup after `reload`; `started` is the config the daemon started with."""
    fresh = load_config(config_file)
    if fresh.audio != started.audio or fresh.slot != started.slot:
        log.warning("changes to [audio] and [slot] take effect when the daemon restarts")
    setup = load_setup(fresh, load_credentials(stdin_keys))
    if parakeet.ParakeetBackend.name not in setup.backends:
        parakeet.ENGINE.unload_when_idle()
    return setup


def run(stdin_keys: Mapping[str, str] | None = None, *, config_file: Path | None = None) -> int:
    """`stt-daemon run`; `stdin_keys` are API keys read at startup, kept until restart."""
    try:
        config = load_config(config_file)
        setup = load_setup(config, load_credentials(stdin_keys))
    except (ConfigError, CredentialsError) as exc:
        print(f"stt-daemon: {exc}", file=sys.stderr)
        return 1
    logging.basicConfig(
        level=config.logging.level, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s"
    )

    def reload_setup() -> Setup:
        return reload(config, config_file, stdin_keys)

    capture = MicCapture(
        device_name=config.audio.device,
        buffer_ms=config.audio.buffer_ms,
        keep_open=config.audio.keep_mic_open,
    )
    try:
        asyncio.run(_main(capture, setup, reload_setup))
    except (StartupFailed, SocketInUse) as exc:
        print(f"stt-daemon: {exc}", file=sys.stderr)
        return 1
    return 0


async def _main(capture: CaptureSource, setup: Setup, reload_setup: Callable[[], Setup]) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    await serve(
        path=socket_path(),
        capture=capture,
        setup=setup,
        reload_setup=reload_setup,
        stop=stop,
    )
