"""A stand-in for `pw-record`: s16 mono 16 kHz on stdout in real time, from a WAV, then silence.

Each start appends a line to `--log`, so a test can count restarts and script the nth start.
"""

import argparse
import contextlib
import sys
import time
import wave
from pathlib import Path

CHUNK = 320
"""20 ms at 16 kHz."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--exit-after", type=float, help="exit after this many seconds")
    parser.add_argument("--stall-after", type=float, help="stop writing, stay alive")
    parser.add_argument("--only-starts", type=int, default=0, help="the limits hit these starts")
    args = parser.parse_args()
    start = 0
    if args.log is not None:
        start = len(args.log.read_text().splitlines()) if args.log.exists() else 0
        with args.log.open("a") as log:
            log.write("start\n")
    limited = args.only_starts == 0 or start < args.only_starts
    pcm = b""
    if args.wav is not None:
        with wave.open(str(args.wav), "rb") as w:
            pcm = w.readframes(w.getnframes())
    began = time.monotonic()
    sent = 0
    out = sys.stdout.buffer
    while True:
        elapsed = time.monotonic() - began
        if limited and args.exit_after is not None and elapsed >= args.exit_after:
            return
        if limited and args.stall_after is not None and elapsed >= args.stall_after:
            time.sleep(3600)
        at = sent * CHUNK * 2
        chunk = pcm[at : at + CHUNK * 2]
        out.write(chunk + bytes(CHUNK * 2 - len(chunk)))
        out.flush()
        sent += 1
        time.sleep(max(0.0, began + sent * CHUNK / 16000 - time.monotonic()))


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        main()
