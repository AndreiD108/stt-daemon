import argparse
import asyncio
import getpass
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from stt_daemon import __version__, control
from stt_daemon.backends import BACKENDS
from stt_daemon.backends.base import FileBackend
from stt_daemon.backends.install import CREDIT, InstallError, install_parakeet
from stt_daemon.backends.parakeet import model_dir
from stt_daemon.config import (
    ConfigError,
    data_dir,
    load_config,
    remove_config_key,
    set_config_string,
    socket_path,
)
from stt_daemon.credentials import (
    CredentialsError,
    load_credentials,
    parse_keys_stdin,
    write_key,
)
from stt_daemon.service.common import ServiceError, daemon_executable, run_command
from stt_daemon.service.launchd import Launchd
from stt_daemon.service.preflight import check_pipewire_aec, check_portaudio, check_tool
from stt_daemon.service.systemd import Systemd

if TYPE_CHECKING:
    from stt_daemon.server.backend_set import Setup

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
            case "backend" if args.action == "install":
                return _install(args.backend)
            case "backend" if args.action == "remove":
                return _remove(args.backend)
            case "backend" if args.action == "list":
                return _list()
            case "backend":
                return _set_default(args.backend)
            case "listen":
                return _listen(args)
            case "transcribe":
                return _transcribe(args)
            case _:
                _parser().print_help()
                return 0
    except (ConfigError, CredentialsError, ServiceError, InstallError) as exc:
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
    key_set.add_argument("backend", help=f"{', '.join(KEY_BACKENDS)} or a plugin's name")

    backend = commands.add_parser("backend", help="transcription backends")
    backend_actions = backend.add_subparsers(dest="action", required=True)
    default = backend_actions.add_parser("set-default", help="the backend sessions use by default")
    default.add_argument("backend")
    install = backend_actions.add_parser(
        "install", help="download a local backend's model and tell the daemon"
    )
    install.add_argument("backend", help="parakeet or a plugin's name")
    remove = backend_actions.add_parser(
        "remove", help="delete a local backend's model and tell the daemon"
    )
    remove.add_argument("backend", choices=["parakeet"])
    backend_actions.add_parser("list", help="every backend, its status, and the default")

    listen = commands.add_parser("listen", help="start a session and print what it hears")
    listen.add_argument("--backend", help="default: the daemon's default backend")
    listen.add_argument("--words", action="store_true", help="print word times")
    listen.add_argument("--partials", action="store_true", help="print partial text")
    listen.add_argument("--marker-key", default="m", help="key that adds a marker (default m)")

    transcribe = commands.add_parser(
        "transcribe", help="transcribe audio or video files (anything ffmpeg reads)"
    )
    transcribe.add_argument("files", nargs="+", type=Path, metavar="FILE")
    transcribe.add_argument(
        "--backend",
        help="default: the daemon's default backend (elevenlabs-batch when that is elevenlabs)",
    )
    transcribe.add_argument("--language", help="language code; default: the backend's")
    transcribe.add_argument(
        "--json", action="store_true", help="one JSON object per file: file, text, words, segments"
    )
    transcribe.add_argument(
        "--diarize", action="store_true", help="name the speaker of each segment (elevenlabs-batch)"
    )
    transcribe.add_argument(
        "--speakers", type=int, metavar="N", help="with --diarize: at most N speakers"
    )
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
            if problem := check_tool("ffmpeg"):
                print(f"`stt-daemon transcribe` needs ffmpeg: {problem}")
            if sys.platform == "linux" and (problem := check_tool("pw-record")):
                print(f"system audio sessions need pw-record: {problem}")
            if sys.platform == "linux" and (problem := check_tool("pw-dump")):
                print(f"picking one app's audio needs pw-dump: {problem}")
            if sys.platform == "linux" and (
                problem := check_tool("pw-cli") or check_pipewire_aec()
            ):
                print(f"echo cancellation on a system session's mic track: {problem}")
            if sys.platform == "darwin" and (problem := check_tool("audiotee")):
                print(f"system audio sessions need audiotee: {problem}")
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


def _setup() -> "Setup":
    """What the daemon would set up now, plugins included."""
    from stt_daemon.server.backend_set import installed_plugins, load_setup

    return load_setup(load_config(), load_credentials(), plugins=installed_plugins())


def _effective_default() -> str | None:
    return _setup().default


def _pin_default(before: str | None, added: str) -> None:
    """Keeps the default as it was, or makes `added` it when there was none."""
    if load_config().backends.default is not None:
        return
    chosen = before or added
    set_config_string("backends", "default", chosen)
    print(f"{chosen} is the default backend")


def _key_set(backend: str) -> int:
    if backend not in KEY_BACKENDS:
        from stt_daemon.server.backend_set import installed_plugins

        if backend not in installed_plugins().classes:
            names = ", ".join(KEY_BACKENDS)
            return _fail(f"no backend named {backend!r} takes a key: {names}, or a plugin's name")
    before = _effective_default()
    if sys.stdin.isatty():
        key = getpass.getpass(f"{backend} API key (not shown): ")
    else:
        key = sys.stdin.readline()
    path = write_key(backend, key)
    print(f"stored the {backend} key in {path}")
    _pin_default(before, backend)
    return _reload_daemon()


def _install(backend: str) -> int:
    before = _effective_default()
    if backend == "parakeet":
        path, installed = install_parakeet()
        print(f"{backend} {'is installed in' if installed else 'was already in'} {path} ({CREDIT})")
    elif (problem := _install_plugin(backend)) is not None:
        return _fail(problem)
    _pin_default(before, backend)
    return _reload_daemon()


def _install_plugin(backend: str) -> str | None:
    """Runs a plugin's `install`; the problem, if any."""
    from stt_daemon.backends.plugins import error_line, has_install
    from stt_daemon.server.backend_set import installed_plugins

    cls = installed_plugins().classes.get(backend)
    if cls is None:
        return f"no backend named {backend!r} to install: parakeet, or a plugin's name"
    if not has_install(cls):
        return f"{backend} has nothing to install"
    instance = cls()
    target = data_dir() / "models" / backend
    key = load_credentials().get(backend)
    try:
        instance.setup(key, load_config().backends.plugins.get(backend, {}))
        target.mkdir(parents=True, exist_ok=True)
        asyncio.run(instance.install(target))
        ready = instance.available()
    except Exception as exc:
        return f"{backend} failed to install: {error_line(exc, key)}"
    if not ready:
        return f"{backend} still reports itself not installed"
    print(f"{backend} is installed in {target}")
    return None


def _remove(backend: str) -> int:
    model = model_dir()
    if not model.exists():
        return _fail(f"{backend} is not installed: there is no model in {model}")
    config = load_config()
    if config.backends.default == backend:
        from stt_daemon.server.backend_set import kinds_of

        setup = _setup()
        others = (n for n, b in setup.backends.items() if n != backend and "mic" in kinds_of(b))
        if (other := next(others, None)) is not None:
            set_config_string("backends", "default", other)
        else:
            remove_config_key("backends", "default")
    shutil.rmtree(model)
    print(f"deleted {model}")
    after = _effective_default()
    print(f"{after} is the default backend" if after else "no default backend")
    return _reload_daemon()


def _set_default(backend: str) -> int:
    setup = _setup()
    if backend not in setup.backends:
        problem = setup.unavailable.get(backend)
        if problem is None:
            names = ", ".join([*setup.backends, *setup.unavailable])
            return _fail(f"there is no backend named {backend!r}; the backends: {names}")
        return _fail(f"{backend} is not set up: {problem.message}")
    if isinstance(chosen := setup.backends[backend], FileBackend):
        return _fail(
            f"{backend} takes transcribe sessions only, so it can't be the default; "
            f"with {chosen.default_for} as the default, transcribe uses {backend}"
        )
    set_config_string("backends", "default", backend)
    print(f"{backend} is the default backend")
    return _reload_daemon()


def _list() -> int:
    data: control.Msg
    try:
        data = control.request("backends_status")
        running = True
    except control.NotRunning:
        data, running = _setup().status_wire(), False
    except (control.DaemonError, OSError) as exc:
        return _fail(f"could not ask the daemon at {socket_path()}: {exc}")
    rows = [
        (b["name"], b["status"] + (", default" if b["default"] else "")) for b in data["backends"]
    ] + [(f["file"], f["status"]) for f in data["failed_files"]]
    width = max((len(name) for name, _ in rows), default=0)
    for name, status in rows:
        print(f"{name:<{width}}  {status}")
    if not running:
        print("(the daemon is not running; this is what it would set up)")
    return 0


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


def _transcribe(args: argparse.Namespace) -> int:
    from stt_daemon.transcribe_client import TranscribeOptions, transcribe

    options = TranscribeOptions(
        backend=args.backend,
        language=args.language,
        json=args.json,
        diarize=args.diarize or args.speakers is not None,
        num_speakers=args.speakers,
    )
    return asyncio.run(transcribe(socket_path(), args.files, options, sys.stdout, sys.stderr))
