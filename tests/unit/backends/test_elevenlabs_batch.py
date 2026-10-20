import asyncio
import io
import json
import logging
import wave
from collections.abc import Iterator

import pytest
from fake_elevenlabs_batch import FakeBatch, answer, serve_batch, spacing, word

from stt_daemon.backends import elevenlabs_batch
from stt_daemon.backends.base import Committed, FileAudio, OpenFailed, OpenOptions, Word
from stt_daemon.backends.elevenlabs_batch import ElevenLabsBatchBackend, committed_runs
from stt_daemon.config import Config
from stt_daemon.credentials import Credentials
from unit.backends.fake_elevenlabs import TEST_KEY

MP3ISH = b"ID3\x04fake container bytes"


@pytest.fixture
def fake() -> Iterator[FakeBatch]:
    with serve_batch() as server:
        yield server


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(elevenlabs_batch, "RETRY_AFTER_S", 0.0)


def backend_for(fake: FakeBatch, config: Config | None = None) -> ElevenLabsBatchBackend:
    config = config or Config()
    fake.configure(config)
    return ElevenLabsBatchBackend.create(config, Credentials(file_keys={"elevenlabs": TEST_KEY}))


async def test_request_carries_the_key_the_options_and_the_bytes_as_sent(fake: FakeBatch) -> None:
    config = Config()
    config.backends.elevenlabs.keyterms = ["Talkey"]
    backend = backend_for(fake, config)
    options = OpenOptions(language="de", keyterms=("Scribbly",), diarize=True, num_speakers=3)
    await backend.transcribe_file(FileAudio(MP3ISH, None), options)
    [upload] = fake.uploads
    assert upload.headers["xi-api-key"] == TEST_KEY
    assert upload.value("model_id") == "scribe_v2"
    assert upload.value("timestamps_granularity") == "word"
    assert upload.value("tag_audio_events") == "false"
    assert upload.value("diarize") == "true"
    assert upload.value("num_speakers") == "3"
    assert upload.value("language_code") == "de"
    assert upload.values("keyterms") == ["Talkey", "Scribbly"]
    assert upload.value("file_format") is None
    assert upload.file == MP3ISH
    assert upload.file_type == "application/octet-stream"


async def test_auto_language_and_no_diarize_leave_those_fields_out(fake: FakeBatch) -> None:
    await backend_for(fake).transcribe_file(FileAudio(MP3ISH, None), OpenOptions(language="auto"))
    [upload] = fake.uploads
    assert upload.value("language_code") is None
    assert upload.value("diarize") == "false"
    assert upload.value("num_speakers") is None
    assert upload.values("keyterms") == []


async def test_keyterms_elevenlabs_refuses_are_dropped(fake: FakeBatch) -> None:
    terms = ("ok term", "x" * 50, "one two three four five six", "a<b")
    await backend_for(fake).transcribe_file(FileAudio(MP3ISH, None), OpenOptions(keyterms=terms))
    assert fake.uploads[0].values("keyterms") == ["ok term"]


async def test_16k_mono_pcm_goes_as_is_with_its_format(fake: FakeBatch) -> None:
    pcm = bytes(range(256)) * 4
    await backend_for(fake).transcribe_file(FileAudio(pcm, (16000, 1)), OpenOptions())
    [upload] = fake.uploads
    assert upload.value("file_format") == "pcm_s16le_16"
    assert upload.file == pcm


async def test_other_pcm_goes_inside_a_wav_header(fake: FakeBatch) -> None:
    pcm = bytes(range(256)) * 4
    await backend_for(fake).transcribe_file(FileAudio(pcm, (44100, 2)), OpenOptions())
    [upload] = fake.uploads
    assert upload.value("file_format") is None
    with wave.open(io.BytesIO(upload.file)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (44100, 2, 2)
        assert wav.readframes(wav.getnframes()) == pcm


def test_words_keep_word_entries_only_and_text_comes_from_the_answer() -> None:
    data = answer(
        "Hello, world.",
        [
            word("Hello,", 0.5, 0.9),
            spacing(0.9, 1.0),
            {"text": "(laughs)", "type": "audio_event", "start": 1.0, "end": 1.4},
            word("world.", 1.4, 1.8),
            spacing(1.8, 2.5),
        ],
        language="eng",
    )
    [one] = committed_runs(data)
    assert one.text == "Hello, world."
    assert one.words == (Word("Hello,", 0.5, 0.9), Word("world.", 1.4, 1.8))
    assert one.language == "en"
    assert one.audio_end_s == 2.5


def test_an_answer_with_no_words_is_one_empty_commit() -> None:
    assert committed_runs(answer("", [])) == [Committed(0, 1, "", "en", (), 0.0)]


def test_diarized_answer_splits_at_each_speaker_change() -> None:
    data = answer(
        "Hi there. Hello. Bye.",
        [
            word("Hi", 0.1, 0.3, "speaker_0"),
            spacing(0.3, 0.4, "speaker_0"),
            word("there.", 0.4, 0.8, "speaker_0"),
            spacing(0.8, 1.0, "speaker_0"),
            word("Hello.", 1.0, 1.5, "speaker_1"),
            spacing(1.5, 2.0, "speaker_1"),
            word("Bye.", 2.0, 2.4, "speaker_0"),
            spacing(2.4, 3.0, "speaker_0"),
        ],
    )
    runs = committed_runs(data)
    assert [r.text for r in runs] == ["Hi there.", "Hello.", "Bye."]
    assert [{w.speaker for w in r.words or ()} for r in runs] == [
        {"speaker_0"},
        {"speaker_1"},
        {"speaker_0"},
    ]
    assert [r.audio_end_s for r in runs] == [0.8, 1.5, 3.0]
    assert [r.commit_id for r in runs] == [1, 2, 3]


def talk(texts: list[str], step_s: float = 1.0) -> dict[str, object]:
    """One speaker, a word every `step_s`."""
    entries: list[dict[str, object]] = []
    for i, text in enumerate(texts):
        entries += [
            word(text, i * step_s, i * step_s + 0.5),
            spacing(i * step_s + 0.5, (i + 1) * step_s),
        ]
    return answer(" ".join(texts), entries)


def test_one_speaker_is_cut_at_a_sentence_end_once_past_the_minimum() -> None:
    texts = [f"w{i}." if i in (5, 24, 30) else f"w{i}" for i in range(40)]
    runs = committed_runs(talk(texts))
    # w5. ends a sentence before 20 s, so no cut; w24. (24.5 s span) is the first past it
    assert [r.words[-1].text for r in runs if r.words] == ["w24.", "w39"]
    assert runs[0].text == " ".join(texts[:25])
    assert runs[1].words
    assert runs[1].words[0].start_s == 25.0


def test_one_speaker_with_no_sentence_end_is_cut_at_the_maximum() -> None:
    runs = committed_runs(talk([f"w{i}" for i in range(70)]))
    spans = [r.words[-1].end_s - r.words[0].start_s for r in runs if r.words]
    assert len(runs) == 3
    assert max(spans) <= elevenlabs_batch.SEGMENT_MAX_S
    assert sum(len(r.words or ()) for r in runs) == 70


@pytest.mark.parametrize(
    ("status", "detail", "code", "text"),
    [
        (401, {"status": "invalid_api_key", "message": "Invalid key"}, "auth", "Invalid key"),
        (403, "Forbidden here", "auth", "Forbidden here"),
        (402, {"status": "quota_exceeded", "message": "No credits"}, "quota", "No credits"),
        (429, {"status": "rate_limit_exceeded", "message": "Slow"}, "rate_limited", "Slow"),
        (
            429,
            {"code": "concurrent_limit_exceeded", "message": "Too many"},
            "concurrency_limit",
            "Too many",
        ),
        (400, {"message": "Bad file"}, "invalid_request", "Bad file"),
        (413, {"message": "File too big"}, "invalid_request", "File too big"),
        (415, {"message": "Unknown format"}, "invalid_request", "Unknown format"),
        (
            422,
            [{"loc": ["body", "file"], "msg": "Field required"}],
            "invalid_request",
            "Field required",
        ),
        (503, {"message": "Try later"}, "backend_error", "Try later"),
    ],
)
async def test_http_errors_map_to_codes_with_the_providers_text(
    fake: FakeBatch,
    caplog: pytest.LogCaptureFixture,
    status: int,
    detail: object,
    code: str,
    text: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    fake.default = (status, json.dumps({"detail": detail}).encode())
    with pytest.raises(OpenFailed) as caught:
        await backend_for(fake).transcribe_file(FileAudio(MP3ISH, None), OpenOptions())
    assert caught.value.code == code
    assert caught.value.message == f"ElevenLabs HTTP {status}: {text}"
    assert TEST_KEY not in caught.value.message
    assert TEST_KEY not in caplog.text
    assert len(fake.uploads) == (2 if status >= 500 else 1)


async def test_the_key_is_cut_out_of_an_echoing_error(fake: FakeBatch) -> None:
    fake.reply(401, {"detail": {"message": f"key {TEST_KEY} is not valid"}})
    with pytest.raises(OpenFailed) as caught:
        await backend_for(fake).transcribe_file(FileAudio(MP3ISH, None), OpenOptions())
    assert TEST_KEY not in caught.value.message
    assert "[key]" in caught.value.message


async def test_a_non_json_error_body_is_forwarded_as_text(fake: FakeBatch) -> None:
    fake.reply(400, "<html>\n  Bad   gateway  </html>")
    with pytest.raises(OpenFailed) as caught:
        await backend_for(fake).transcribe_file(FileAudio(MP3ISH, None), OpenOptions())
    assert caught.value.message == "ElevenLabs HTTP 400: <html> Bad gateway </html>"


async def test_one_retry_after_a_server_error(fake: FakeBatch) -> None:
    fake.reply(500, {"detail": "boom"})
    fake.reply(200, answer("second try", []))
    [one] = await backend_for(fake).transcribe_file(FileAudio(MP3ISH, None), OpenOptions())
    assert one.text == "second try"
    assert len(fake.uploads) == 2


async def test_client_errors_are_not_retried(fake: FakeBatch) -> None:
    fake.reply(400, {"detail": "bad"})
    fake.reply(200, answer("never", []))
    with pytest.raises(OpenFailed):
        await backend_for(fake).transcribe_file(FileAudio(MP3ISH, None), OpenOptions())
    assert len(fake.uploads) == 1


async def test_network_errors_fail_as_network(fake: FakeBatch) -> None:
    config = Config()
    backend = backend_for(fake, config)
    config.backends.elevenlabs_batch.endpoint = "http://127.0.0.1:9/v1/speech-to-text"
    with pytest.raises(OpenFailed) as caught:
        await backend.transcribe_file(FileAudio(MP3ISH, None), OpenOptions())
    assert caught.value.code == "network"
    assert TEST_KEY not in caught.value.message


async def test_cancel_aborts_the_request(fake: FakeBatch) -> None:
    import threading

    fake.hold = threading.Event()
    task = asyncio.create_task(
        backend_for(fake).transcribe_file(FileAudio(MP3ISH, None), OpenOptions())
    )
    async with asyncio.timeout(5):
        while not fake.uploads:
            await asyncio.sleep(0.01)
    task.cancel()
    async with asyncio.timeout(1):
        with pytest.raises(asyncio.CancelledError):
            await task


def test_needs_the_elevenlabs_key() -> None:
    from stt_daemon.backends.base import BackendUnavailable

    with pytest.raises(BackendUnavailable):
        ElevenLabsBatchBackend.create(Config(), Credentials())
    assert TEST_KEY not in repr(
        ElevenLabsBatchBackend.create(Config(), Credentials(file_keys={"elevenlabs": TEST_KEY}))
    )
