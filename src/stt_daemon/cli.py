import argparse

from stt_daemon import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stt-daemon",
        description="Speech-to-text daemon: one mic, many clients, over a Unix socket.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0
