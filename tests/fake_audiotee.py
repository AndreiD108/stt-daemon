"""A stand-in for `audiotee`: its flags, s16 mono at `--sample-rate` on stdout in real time in
`--chunk-duration` chunks, from a WAV, then silence; JSON log lines on stderr.

Test flags: `--log` gets a line per start (with the included pids); `--live` lists the pids that
exist, and an included pid missing from it fails as audiotee does; `--exit-after` and
`--stall-after` act from the start; `--silent` writes nothing, as without the permission.
"""

import argparse
import contextlib
import json
import sys
import time
import wave
from pathlib import Path


def log_line(kind: str, message: str, **context: str) -> None:
    data: dict[str, object] = {"message": message}
    if context:
        data["context"] = context
    line = {"timestamp": "2026-01-01T00:00:00.000Z", "message_type": kind, "data": data}
    print(json.dumps(line), file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--chunk-duration", type=float, default=0.2)
    parser.add_argument("--include-processes", nargs="+", default=[])
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--live", type=Path)
    parser.add_argument("--exit-after", type=float)
    parser.add_argument("--stall-after", type=float)
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()
    if args.log is not None:
        with args.log.open("a") as log:
            log.write(" ".join(["start", *args.include_processes]) + "\n")
    log_line("info", "Starting AudioTee...")
    if args.live is not None:
        live = args.live.read_text().split()
        if missing := [pid for pid in args.include_processes if pid not in live]:
            log_line(
                "error",
                "Failed to translate process IDs to audio objects",
                failed_pids=", ".join(missing),
            )
            sys.exit(1)
    pcm = b""
    if args.wav is not None:
        with wave.open(str(args.wav), "rb") as w:
            pcm = w.readframes(w.getnframes())
    chunk = int(args.sample_rate * args.chunk_duration)
    began = time.monotonic()
    sent = 0
    out = sys.stdout.buffer
    while True:
        elapsed = time.monotonic() - began
        if args.exit_after is not None and elapsed >= args.exit_after:
            log_line("error", "Audio device stopped", device="tap")
            sys.exit(2)
        if args.silent or (args.stall_after is not None and elapsed >= args.stall_after):
            time.sleep(3600)
        at = sent * chunk * 2
        data = pcm[at : at + chunk * 2]
        out.write(data + bytes(chunk * 2 - len(data)))
        out.flush()
        sent += 1
        time.sleep(max(0.0, began + sent * chunk / args.sample_rate - time.monotonic()))


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        main()
