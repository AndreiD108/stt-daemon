"""`stt-daemon transcribe` against a scripted daemon: output shapes, file bytes, exit codes."""

import asyncio
import base64
import errno
import io
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import IO, Any

import pytest

from stt_daemon.transcribe_client import CHUNK_BYTES, TranscribeOptions, transcribe

type Msg = dict[str, Any]


class FakeDaemon:
    """Answers each `transcribe` with a session whose text counts the bytes it got."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.received: list[bytes] = []
        self.starts: list[Msg] = []
        self.cancels = 0
        self.refuse_after = 99
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_unix_server(self._handle, path=self.path, limit=4 << 20)

    async def close(self) -> None:
        assert self._server is not None
        self._server.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        def send(kind: str, data: Msg, reply_to: object = None) -> None:
            msg: Msg = {"type": kind, "data": data}
            if reply_to is not None:
                msg["reply_to"] = reply_to
            writer.write(json.dumps(msg).encode() + b"\n")

        audio = bytearray()
        while line := await reader.readline():
            msg = json.loads(line)
            kind, data, id = msg["type"], msg.get("data", {}), msg.get("id")
            if kind == "hello":
                backends = [{"name": "fake", "word_timestamps": True}]
                send("welcome", {"backends": backends, "default_backend": "fake"}, id)
            elif kind == "transcribe":
                self.starts.append(data)
                if len(self.starts) > self.refuse_after:
                    send("error", {"code": "busy", "message": "fake is full"}, id)
                    continue
                audio = bytearray()
                send("session_started", {"session_id": "x", "kind": "transcribe"}, id)
                send("progress", {"session_id": "x", "audio_seconds": 0.5})
            elif kind == "audio":
                audio += base64.b64decode(data["data"])
            elif kind == "cancel_session":
                self.cancels += 1
                send("session_ended", {"session_id": "x", "reason": "cancelled"}, id)
            elif kind == "stop_session":
                self.received.append(bytes(audio))
                text = f"got {len(audio)} bytes"
                word = {"text": text, "start_ms": 100, "end_ms": 900}
                segment = {"session_id": "x", "index": 0, "text": text, "start_ms": 0}
                if self.starts[-1].get("diarize"):
                    segment["speaker"] = "speaker_0"
                send("segment", {**segment, "end_ms": 1000, "words": [word]})
                send(
                    "session_ended",
                    {"session_id": "x", "reason": "stopped", "final_text": text},
                    id,
                )
            await writer.drain()


@pytest.fixture
async def fake_daemon(tmp_path: Path) -> AsyncIterator[FakeDaemon]:
    daemon = FakeDaemon(tmp_path / "stt.sock")
    await daemon.start()
    yield daemon
    await daemon.close()


def files(tmp_path: Path, *sizes: int) -> list[Path]:
    paths: list[Path] = []
    for i, size in enumerate(sizes):
        path = tmp_path / f"clip{i}.m4a"
        path.write_bytes(bytes(range(256)) * (size // 256) + bytes(size % 256))
        paths.append(path)
    return paths


async def run(daemon: FakeDaemon, paths: list[Path], **options: Any) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = await transcribe(daemon.path, paths, TranscribeOptions(**options), out, err)
    return code, out.getvalue(), err.getvalue()


async def test_each_file_streams_whole_and_prints_its_text(
    fake_daemon: FakeDaemon, tmp_path: Path
) -> None:
    paths = files(tmp_path, CHUNK_BYTES * 2 + 5, 10)
    code, out, err = await run(fake_daemon, paths, backend="fake", language="ro")
    assert (code, err) == (0, "")
    assert out == f"got {CHUNK_BYTES * 2 + 5} bytes\n\ngot 10 bytes\n"
    assert fake_daemon.received == [p.read_bytes() for p in paths]
    assert fake_daemon.starts[0] == {"word_timestamps": True, "backend": "fake", "language": "ro"}


async def test_diarize_asks_for_speakers_and_names_them_per_segment(
    fake_daemon: FakeDaemon, tmp_path: Path
) -> None:
    code, out, _ = await run(fake_daemon, files(tmp_path, 3), diarize=True, num_speakers=2)
    assert code == 0
    assert out == "speaker_0: got 3 bytes\n"
    assert fake_daemon.starts[0]["diarize"] is True
    assert fake_daemon.starts[0]["num_speakers"] == 2


async def test_json_prints_one_object_per_file(fake_daemon: FakeDaemon, tmp_path: Path) -> None:
    paths = files(tmp_path, 3, 4)
    code, out, _ = await run(fake_daemon, paths, json=True)
    assert code == 0
    first, second = (json.loads(line) for line in out.splitlines())
    assert first["file"] == str(paths[0])
    assert first["text"] == "got 3 bytes"
    assert first["words"] == [{"text": "got 3 bytes", "start_ms": 100, "end_ms": 900}]
    [segment] = first["segments"]
    assert "session_id" not in segment
    assert (segment["index"], segment["start_ms"], segment["end_ms"]) == (0, 0, 1000)
    assert second["text"] == "got 4 bytes"


async def test_the_first_failed_file_ends_the_run(fake_daemon: FakeDaemon, tmp_path: Path) -> None:
    fake_daemon.refuse_after = 1
    paths = files(tmp_path, 3, 4, 5)
    code, out, err = await run(fake_daemon, paths)
    assert code == 1
    assert out == "got 3 bytes\n"
    assert str(paths[1]) in err
    assert "busy" in err
    assert len(fake_daemon.starts) == 2


async def test_a_missing_file_fails(fake_daemon: FakeDaemon, tmp_path: Path) -> None:
    code, out, err = await run(fake_daemon, [tmp_path / "nope.wav"])
    assert (code, out) == (1, "")
    assert "nope.wav" in err


async def test_a_read_error_mid_file_cancels_and_skips_the_rest(
    fake_daemon: FakeDaemon, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = files(tmp_path, CHUNK_BYTES * 3, 4)

    class FailingRead(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if self.tell():
                raise OSError(errno.EIO, "Input/output error")
            return super().read(size)

    first = paths[0].read_bytes()
    real_open = Path.open

    def open_failing(self: Path, mode: str = "r") -> IO[Any]:
        return FailingRead(first) if self == paths[0] else real_open(self, mode)

    monkeypatch.setattr(Path, "open", open_failing)
    code, out, err = await asyncio.wait_for(run(fake_daemon, paths), 5)
    assert (code, out) == (1, "")
    assert str(paths[0]) in err
    assert "Input/output error" in err
    assert fake_daemon.cancels == 1
    assert len(fake_daemon.starts) == 1


async def test_no_daemon_fails(tmp_path: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    code = await transcribe(tmp_path / "none.sock", [], TranscribeOptions(), out, err)
    assert code == 1
    assert "not running" in err.getvalue()
