import argparse
import sys

from stt_daemon import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stt-daemon",
        description="Speech-to-text daemon: one mic, many clients, over a Unix socket.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("run", help="run the daemon in the foreground (the service runs this)")
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run()
    parser.print_help()
    return 0


def _run() -> int:
    try:
        from stt_daemon import daemon
    except OSError as exc:  # sounddevice raises it when PortAudio can't be loaded
        print(f"stt-daemon: cannot load PortAudio ({exc}); install it, then retry", file=sys.stderr)
        return 1
    return daemon.run()
