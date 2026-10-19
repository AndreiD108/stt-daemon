"""A stand-in for `pw-cli -m load-module libpipewire-module-echo-cancel '{ … }'`.

With `--dir`, it adds the module's `Audio/Source` node (the `node.name` from the module args) to
that `FakePipeWire` registry, and removes it when it ends. It stays until SIGTERM, like the real
one. Each start and end appends a line to `--log`.
"""

import argparse
import contextlib
import re
import signal
import sys
import time
from pathlib import Path
from types import FrameType

from fake_pw_dump import FakePipeWire

FAKE_PW_CLI = str(Path(__file__))


def command(*options: str) -> list[str]:
    """The `aec_command` prefix: append the real command line's arguments after `pw-cli`."""
    return [sys.executable, FAKE_PW_CLI, *options, "--"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, help="announce the node in this FakePipeWire")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--fail", action="store_true", help="print the load error, stay alive")
    parser.add_argument("--exit-after", type=float, help="end with code 1 after this long")
    parser.add_argument("--ignore-term", action="store_true")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    found = re.search(r'node\.name = "([^"]+)"', " ".join(args.rest))
    name = found[1] if found else "?"

    def log(line: str) -> None:
        if args.log is not None:
            with args.log.open("a") as f:
                f.write(f"{line} {name}\n")

    node: tuple[FakePipeWire, int] | None = None

    def end(code: int) -> None:
        if node is not None:
            node[0].remove(node[1])
        log("end")
        sys.exit(code)

    def on_term(_signum: int, _frame: FrameType | None) -> None:
        end(0)

    # Before the start line: a test that waits for it can send SIGTERM.
    signal.signal(signal.SIGTERM, signal.SIG_IGN if args.ignore_term else on_term)
    log("start")
    if args.fail:
        print('Error: "Could not load module"', file=sys.stderr, flush=True)
    elif args.dir is not None:
        pipewire = FakePipeWire.reopen(args.dir)
        node = (pipewire, pipewire.add_source(name)[0])

    began = time.monotonic()
    while args.exit_after is None or time.monotonic() - began < args.exit_after:
        time.sleep(0.02)
    end(1)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
