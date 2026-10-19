"""A stand-in for `pw-dump -m -N`, and `FakePipeWire`, the test side that changes its graph.

The script prints the current registry as one pretty-printed array, then each change written to
the feed after it. `FakePipeWire` keeps the registry in files under one directory: `feed` (one
compact array per change), `snapshot.json`, and `live` (serials of the playing streams, which
`fake_pw_record.py --live` reads).
"""

import argparse
import contextlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

FAKE_PW_DUMP = str(Path(__file__))


class FakePipeWire:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.feed = root / "feed"
        self.live = root / "live"
        self.feed.write_text("")
        self._objects: dict[int, dict[str, Any]] = {}
        self._ids = itertools.count(200)
        self._serials = itertools.count(20000)
        self._write_state()

    def command(self, *extra: str) -> list[str]:
        return [sys.executable, FAKE_PW_DUMP, "--dir", str(self.root), *extra]

    def play(
        self, name: str, *, binary: str | None = None, media_name: str | None = None
    ) -> tuple[int, str]:
        """A native client's playback stream (its binary on the client); (registry id, serial)."""
        client_id, node_id, serial = next(self._ids), next(self._ids), next(self._serials)
        client_props: dict[str, Any] = {"application.name": name}
        if binary is not None:
            client_props["application.process.binary"] = binary
        node_props: dict[str, Any] = {
            "application.name": name,
            "client.id": client_id,
            "media.class": "Stream/Output/Audio",
            "node.name": name,
            "object.id": node_id,
            "object.serial": serial,
        }
        if media_name is not None:
            node_props["media.name"] = media_name
        self._change(
            [
                _object(client_id, "Client", client_props),
                _object(node_id, "Node", node_props),
            ]
        )
        return node_id, str(serial)

    def stop(self, node_id: int) -> None:
        client = self._objects[node_id]["info"]["props"]["client.id"]
        self._change([{"id": node_id, "info": None}, {"id": client, "info": None}])

    def _change(self, objects: list[dict[str, Any]]) -> None:
        with self.feed.open("a") as feed:
            feed.write(json.dumps(objects) + "\n")
        for obj in objects:
            if obj["info"] is None:
                self._objects.pop(obj["id"], None)
            else:
                self._objects[obj["id"]] = obj
        self._write_state()

    def _write_state(self) -> None:
        _replace(self.root / "snapshot.json", json.dumps(list(self._objects.values())))
        serials = [
            str(o["info"]["props"]["object.serial"])
            for o in self._objects.values()
            if o["info"]["props"].get("media.class") == "Stream/Output/Audio"
        ]
        _replace(self.live, "".join(f"{s}\n" for s in serials))


def _object(key: int, kind: str, props: dict[str, Any]) -> dict[str, Any]:
    return {"id": key, "type": f"PipeWire:Interface:{kind}", "info": {"props": props}}


def _replace(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _complete_lines(path: Path) -> list[str]:
    text = path.read_text()
    return text[: text.rfind("\n") + 1].splitlines()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, help="each start appends a line")
    parser.add_argument("--exit-after", type=float, help="exit after this many seconds")
    args = parser.parse_args()
    if args.log is not None:
        with args.log.open("a") as log:
            log.write("start\n")
    feed = args.dir / "feed"
    seen = len(_complete_lines(feed))
    out = sys.stdout
    out.write(json.dumps(json.loads((args.dir / "snapshot.json").read_text()), indent=2) + "\n")
    out.flush()
    began = time.monotonic()
    while args.exit_after is None or time.monotonic() - began < args.exit_after:
        lines = _complete_lines(feed)
        for line in lines[seen:]:
            out.write(json.dumps(json.loads(line), indent=2) + "\n")
        out.flush()
        seen = len(lines)
        time.sleep(0.02)


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError, KeyboardInterrupt):
        main()
