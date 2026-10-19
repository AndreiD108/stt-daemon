"""`transcribe` on elevenlabs-batch over the socket, against a local stand-in for the upload API."""

import asyncio
import base64
import threading
from collections.abc import Iterator

import pytest
from fake_elevenlabs_batch import FakeBatch, answer, serve_batch, spacing, word
from harness import FIXTURE, Harness, Msg, TestClient, fake

from stt_daemon.backends.elevenlabs_batch import ElevenLabsBatchBackend
from stt_daemon.config import Config
from stt_daemon.credentials import Credentials

DIARIZED = answer(
    "First phrase. Second phrase.",
    [
        word("First", 1.0, 1.4, "speaker_0"),
        spacing(1.4, 1.5, "speaker_0"),
        word("phrase.", 1.5, 3.7, "speaker_0"),
        spacing(3.7, 5.7, "speaker_0"),
        word("Second", 5.74, 6.3, "speaker_1"),
        spacing(6.3, 6.4, "speaker_1"),
        word("phrase.", 6.4, 7.8, "speaker_1"),
        spacing(7.8, 8.8, "speaker_1"),
    ],
)


@pytest.fixture
def batch() -> Iterator[FakeBatch]:
    with serve_batch() as server:
        yield server


async def start(daemon: Harness, batch: FakeBatch) -> None:
    config = Config()
    batch.configure(config)
    uploads = ElevenLabsBatchBackend.create(config, Credentials(file_keys={"elevenlabs": "k"}))
    await daemon.start(config=config, backends={"elevenlabs": fake(), "elevenlabs-batch": uploads})


def send_file(client: TestClient, data: bytes, chunk: int = 48 * 1024) -> None:
    for at in range(0, len(data), chunk):
        client.send("audio", {"data": base64.b64encode(data[at : at + chunk]).decode()})


async def test_transcribe_uploads_the_file_and_names_speakers(
    daemon: Harness, batch: FakeBatch
) -> None:
    batch.reply(200, DIARIZED)
    await start(daemon, batch)
    client, _ = await daemon.connect("Scribbly")
    started = await client.request("transcribe", {"diarize": True, "word_timestamps": True})
    assert started["type"] == "session_started", started
    assert (started["data"]["backend"], started["data"]["kind"]) == (
        "elevenlabs-batch",
        "transcribe",
    )
    send_file(client, FIXTURE.read_bytes())
    ended = await client.request("stop_session", timeout=15)
    assert ended["type"] == "session_ended", ended
    assert ended["data"]["final_text"] == "First phrase. Second phrase."
    assert ended["data"]["duration_ms"] == 8800
    [upload] = batch.uploads
    assert upload.file == FIXTURE.read_bytes()
    assert upload.value("diarize") == "true"
    first, second = (m["data"] for m in client.of_type("segment"))
    assert (first["speaker"], second["speaker"]) == ("speaker_0", "speaker_1")
    assert [w["speaker"] for w in first["words"]] == ["speaker_0", "speaker_0"]
    assert (first["words"][0]["start_ms"], second["words"][0]["start_ms"]) == (1000, 5740)
    assert client.of_type("progress") == []
    assert client.of_type("partial") == []


async def test_start_session_on_the_batch_backend_is_refused(
    daemon: Harness, batch: FakeBatch
) -> None:
    await start(daemon, batch)
    client, welcome = await daemon.connect("Talkey")
    listed = {b["name"]: b for b in welcome["data"]["backends"]}
    assert listed["elevenlabs-batch"]["kinds"] == ["transcribe"]
    assert listed["elevenlabs-batch"]["diarize"] is True
    assert welcome["data"]["default_backend"] == "elevenlabs"
    reply = await client.request("start_session", {"backend": "elevenlabs-batch"})
    assert (reply["type"], reply["data"]["code"]) == ("error", "unsupported")


async def test_diarize_on_a_backend_without_it_is_refused(
    daemon: Harness, batch: FakeBatch
) -> None:
    await start(daemon, batch)
    client, _ = await daemon.connect()
    reply = await client.request("transcribe", {"backend": "elevenlabs", "diarize": True})
    assert reply["data"] == {
        "code": "unsupported_option",
        "message": reply["data"]["message"],
        "field": "diarize",
    }


async def test_a_marker_needs_at_and_lands_in_its_speakers_segment(
    daemon: Harness, batch: FakeBatch
) -> None:
    batch.reply(200, DIARIZED)
    await start(daemon, batch)
    client, _ = await daemon.connect()
    started = await client.request("transcribe", {"word_timestamps": True})
    refused = await client.request("add_marker", {"marker": "M"})
    assert (refused["type"], refused["data"]["code"]) == ("error", "invalid")
    at = started["data"]["started_at"] + 6350
    accepted = await client.request("add_marker", {"marker": "M", "at": at})
    assert accepted["data"]["at_ms"] == 6350
    send_file(client, FIXTURE.read_bytes())
    await client.request("stop_session", timeout=15)
    first, second = (m["data"] for m in client.of_type("segment"))
    assert "markers" not in first
    assert second["markers"] == [{"marker": "M", "at_ms": 6350, "after_word": 0}]


async def test_disconnect_during_the_upload_ends_the_session(
    daemon: Harness, batch: FakeBatch
) -> None:
    batch.hold = threading.Event()
    await start(daemon, batch)
    client, _ = await daemon.connect("Scribbly")
    watcher, _ = await daemon.connect("observer", observe_all=True)
    await client.request("transcribe", {})
    send_file(client, b"bytes of some file")
    client.send("stop_session")
    async with asyncio.timeout(5):
        while not batch.uploads:
            await asyncio.sleep(0.01)
    client.writer.transport.abort()
    ended: Msg = await watcher.event("session_ended")
    assert ended["data"]["reason"] == "client_gone"
    assert watcher.of_type("segment") == []
