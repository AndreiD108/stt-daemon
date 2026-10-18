import argparse
import asyncio
import getpass
import sys
from collections.abc import Mapping
from pathlib import Path

from stt_daemon import __version__, control
from stt_daemon.backends import BACKENDS
from stt_daemon.config import ConfigError, load_config, set_config_string, socket_path
from stt_daemon.credentials import (
    CredentialsError,
    load_credentials,
    parse_keys_stdin,
    write_key,
)
from stt_daemon.service.common import ServiceError, daemon_executable, run_command
from stt_daemon.service.launchd import Launchd
from stt_daemon.service.preflight import check_portaudio
from stt_daemon.service.systemd import Systemd

KEY_BACKENDS = [name for name, cls in BACKENDS.items() if cls.takes_key]

BACKEND_SETUP = """\
Next, set up a backend:
  stt-daemon key set elevenlabs        ElevenLabs Scribe, in the cloud; reads an API key
  stt-daemon backend install parakeet  Parakeet, on this machine; downloads its model"""


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        match args.command:
            case "run":
                return _run(args.keys_stdin)
            case "service":
                return _service(args)
            case "key":
                return _key_set(args.backend)
            case "backend":
                return _set_default(args.backend)
            case "listen":
                return _listen(args)
            case _:
                _parser().print_help()
                return 0
    except (ConfigError, CredentialsError, ServiceError) as exc:
        return _fail(str(exc))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stt-daemon",
        description="Speech-to-text daemon: one mic, many clients, over a Unix socket.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="run the daemon in the foreground; the service runs this")
    run.add_argument(
        "--keys-stdin",
        action="store_true",
        help="read API keys as NAME=value lines from stdin at startup; they win over "
        "credentials.toml and the environment until the daemon restarts",
    )

    service = commands.add_parser("service", help="the user service that runs the daemon at login")
    actions = service.add_subparsers(dest="action", required=True)
    for name, text in (("install", "write, enable and start it"), ("remove", "stop and delete it")):
        action = actions.add_parser(name, help=text)
        action.add_argument(
            "--unit-dir",
            type=Path,
            help="write or delete the service files here without calling systemctl or launchctl",
        )
    actions.add_parser("status", help="show whether it runs")
    logs = actions.add_parser("logs", help="show its log")
    logs.add_argument("-f", "--follow", action="store_true", help="keep printing new lines")
    logs.add_argument("-n", "--lines", type=int, default=100, help="lines to show (default 100)")

    key = commands.add_parser("key", help="API keys")
    key_actions = key.add_subparsers(dest="action", required=True)
    key_set = key_actions.add_parser(
        "set", help="store a backend's API key (read from stdin) and tell the daemon"
    )
    key_set.add_argument("backend", choices=KEY_BACKENDS)

    backend = commands.add_parser("backend", help="transcription backends")
    backend_actions = backend.add_subparsers(dest="action", required=True)
    default = backend_actions.add_parser("set-default", help="the backend sessions use by default")
    default.add_argument("backend", choices=list(BACKENDS))

    listen = commands.add_parser("listen", help="start a session and print what it hears")
    listen.add_argument("--backend", help="default: the daemon's default backend")
    listen.add_argument("--words", action="store_true", help="print word times")
    listen.add_argument("--partials", action="store_true", help="print partial text")
    listen.add_argument("--marker-key", default="m", help="key that adds a marker (default m)")
    return parser


def _fail(message: str) -> int:
    print(f"stt-daemon: {message}", file=sys.stderr)
    return 1


def _run(keys_stdin: bool) -> int:
    keys: Mapping[str, str] | None = None
    if keys_stdin:
        if sys.stdin.isatty():
            return _fail("--keys-stdin reads NAME=value lines from a pipe, not a terminal")
        keys = parse_keys_stdin(sys.stdin.read())
    if problem := check_portaudio():
        return _fail(problem)
    from stt_daemon import daemon

    return daemon.run(keys)


def _service(args: argparse.Namespace) -> int:
    unit_dir: Path | None = getattr(args, "unit_dir", None)
    manager = (
        Launchd(run_command, agents_dir=unit_dir)
        if sys.platform == "darwin"
        else Systemd(run_command, unit_dir=unit_dir)
    )
    match args.action:
        case "install":
            if problem := check_portaudio():
                return _fail(problem)
            for path in manager.install(daemon_executable()):
                print(f"wrote {path}")
            if unit_dir is None:
                print("the service is enabled and running; it starts at login")
            print(BACKEND_SETUP)
        case "remove":
            removed = manager.remove()
            for path in removed:
                print(f"deleted {path}")
            if not removed:
                print("the service is not installed")
        case "status":
            return manager.status()
        case _:
            return manager.logs(follow=args.follow, lines=args.lines)
    return 0


def _effective_default() -> str | None:
    """The default backend the daemon would pick now."""
    from stt_daemon.server.backend_set import load_setup

    return load_setup(load_config(), load_credentials()).default


def _pin_default(before: str | None, added: str) -> None:
    """Keeps the default as it was, or makes `added` it when there was none."""
    if load_config().backends.default is not None:
        return
    chosen = before or added
    set_config_string("backends", "default", chosen)
    print(f"{chosen} is the default backend")


def _key_set(backend: str) -> int:
    before = _effective_default()
    if sys.stdin.isatty():
        key = getpass.getpass(f"{backend} API key (not shown): ")
    else:
        key = sys.stdin.readline()
    path = write_key(backend, key)
    print(f"stored the {backend} key in {path}")
    _pin_default(before, backend)
    return _reload_daemon()


def _set_default(backend: str) -> int:
    from stt_daemon.server.backend_set import load_setup

    setup = load_setup(load_config(), load_credentials())
    if backend not in setup.backends:
        reason = setup.unavailable.get(backend, "not set up")
        return _fail(f"{backend} is not set up: {reason}")
    set_config_string("backends", "default", backend)
    print(f"{backend} is the default backend")
    return _reload_daemon()


def _reload_daemon() -> int:
    try:
        data = control.request("reload")
    except control.NotRunning:
        print("the daemon is not running; it reads this when it starts")
        return 0
    except control.DaemonError as exc:
        return _fail(f"the daemon could not reload: {exc.message}")
    except OSError as exc:
        return _fail(f"could not reach the daemon at {socket_path()}: {exc}")
    names = ", ".join(b["name"] for b in data.get("backends", [])) or "none"
    print(f"the daemon reloaded; backends: {names}; default: {data.get('default_backend')}")
    return 0


def _listen(args: argparse.Namespace) -> int:
    from stt_daemon.listen import ListenOptions, listen, terminal_keys

    if len(args.marker_key) != 1 or args.marker_key.isspace():
        return _fail("--marker-key must be one visible character")
    options = ListenOptions(backend=args.backend, words=args.words, partials=args.partials)

    async def go() -> int:
        keys = terminal_keys(args.marker_key)
        return await listen(socket_path(), options, keys, sys.stdout, args.marker_key)

    try:
        return asyncio.run(go())
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        return _fail(f"lost the daemon: {exc}")
