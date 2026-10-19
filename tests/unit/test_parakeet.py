import asyncio
from pathlib import Path

import numpy as np
import pytest

from stt_daemon.audio.frames import Samples
from stt_daemon.backends.base import (
    BackendEvent,
    BackendUnavailable,
    Closed,
    Committed,
    Failed,
    Word,
)
from stt_daemon.backends.parakeet import (
    MODEL_FILES,
    Decoded,
    ParakeetBackend,
    ParakeetConnection,
    group_words,
    hotwords,
    model_dir,
)
from stt_daemon.config import Config
from stt_daemon.credentials import Credentials


def _audio(seconds: float) -> Samples:
    return np.zeros(round(seconds * 16000), np.float32)


def _decoded(*tokens: tuple[str, float, float]) -> Decoded:
    return Decoded(
        "".join(t for t, _, _ in tokens),
        [t for t, _, _ in tokens],
        [at for _, at, _ in tokens],
        [d for _, _, d in tokens],
    )


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def test_tokens_with_a_leading_space_start_words() -> None:
    decoded = _decoded(
        (" R", 0.32, 0.24), ("est", 0.56, 0.16), ("art", 0.88, 0.16), (",", 1.04, 0.08),
        (" the", 1.2, 0.16), (" k", 1.5, 0.08), ("itty", 1.6, 0.0),
    )  # fmt: skip
    assert group_words(decoded, 10.0) == (
        Word("Restart,", 10.32, pytest.approx(11.12)),  # pyright: ignore[reportArgumentType]
        Word("the", 11.2, 11.36),
        Word("kitty", 11.5, 11.6),
    )


def test_a_first_token_without_a_space_starts_a_word() -> None:
    decoded = _decoded(("Hi", 0.0, 0.1), (" there", 0.2, 0.2))
    assert group_words(decoded, 0.0) == (Word("Hi", 0.0, 0.1), Word("there", 0.2, 0.4))


def test_hotwords_join_phrases_and_drop_slashes_and_blanks() -> None:
    assert hotwords(["stt-daemon", "sherpa  onnx", "a/b", " "]) == "stt-daemon/sherpa onnx/a b"
    assert hotwords([]) is None
    assert hotwords(["/"]) is None


class _Decoder:
    """Stands in for the engine: one reply per decode, recording the audio it got."""

    def __init__(self, *replies: Decoded | Exception) -> None:
        self.replies = list(replies)
        self.audio: list[Samples] = []
        self.gate = asyncio.Event()
        self.gate.set()

    async def __call__(self, audio: Samples) -> Decoded:
        self.audio.append(audio)
        await self.gate.wait()
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


async def test_commits_decode_their_own_audio_with_connection_times() -> None:
    decoder = _Decoder(_decoded((" one", 0.1, 0.2)), _decoded((" two", 0.3, 0.1)))
    events: list[BackendEvent] = []
    conn = ParakeetConnection(3, events.append, decoder)
    await conn.send_audio(_audio(1.0))
    await conn.send_audio(_audio(0.5))
    assert await conn.commit() == 1
    await conn.send_audio(_audio(2.0))
    assert await conn.commit() == 2
    await _settle()
    assert [len(a) for a in decoder.audio] == [24000, 32000]
    assert events == [
        Committed(3, 1, "one", None, (Word("one", 0.1, pytest.approx(0.3)),), 1.5),  # pyright: ignore[reportArgumentType]
        Committed(3, 2, "two", None, (Word("two", 1.8, pytest.approx(1.9)),), 3.5),  # pyright: ignore[reportArgumentType]
    ]
    await conn.close()


async def test_empty_and_too_short_commits_answer_without_decoding() -> None:
    decoder = _Decoder()
    events: list[BackendEvent] = []
    conn = ParakeetConnection(1, events.append, decoder)
    assert await conn.commit() == 1
    await conn.send_audio(_audio(0.05))
    assert await conn.commit() == 2
    await _settle()
    assert decoder.audio == []
    assert events == [Committed(1, 1, "", None, (), 0.0), Committed(1, 2, "", None, (), 0.05)]
    await conn.close()


async def test_blank_text_gives_no_words() -> None:
    decoder = _Decoder(Decoded(" ", [" "], [0.0], [0.1]))
    events: list[BackendEvent] = []
    conn = ParakeetConnection(1, events.append, decoder)
    await conn.send_audio(_audio(1.0))
    await conn.commit()
    await _settle()
    assert events == [Committed(1, 1, "", None, (), 1.0)]
    await conn.close()


async def test_a_decode_error_fails_the_connection_once() -> None:
    decoder = _Decoder(RuntimeError("bad shape"))
    events: list[BackendEvent] = []
    conn = ParakeetConnection(2, events.append, decoder)
    await conn.send_audio(_audio(1.0))
    await conn.commit()
    await conn.send_audio(_audio(1.0))
    await conn.commit()
    await _settle()
    assert events == [
        Failed(2, "backend_error", "decoding failed: bad shape", retryable=False),
        Closed(2),
    ]
    await conn.send_audio(_audio(1.0))
    assert await conn.commit() == 3
    await conn.close()
    assert len(events) == 2


async def test_close_during_a_decode_drops_its_result() -> None:
    decoder = _Decoder(_decoded((" late", 0.0, 0.1)))
    decoder.gate.clear()
    events: list[BackendEvent] = []
    conn = ParakeetConnection(4, events.append, decoder)
    await conn.send_audio(_audio(1.0))
    await conn.commit()
    await _settle()
    await conn.close()
    decoder.gate.set()
    await _settle()
    assert events == [Closed(4)]


def _model(tmp: Path) -> Path:
    model = model_dir({"XDG_DATA_HOME": str(tmp)})
    model.mkdir(parents=True)
    for name in MODEL_FILES:
        (model / name).write_text("x")
    return model


def test_create_needs_every_model_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    with pytest.raises(BackendUnavailable) as caught:
        ParakeetBackend.create(Config(), Credentials())
    assert caught.value.code == "no_model"
    model = _model(tmp_path)
    assert isinstance(ParakeetBackend.create(Config(), Credentials()), ParakeetBackend)
    (model / "bpe.vocab").unlink()
    with pytest.raises(BackendUnavailable):
        ParakeetBackend.create(Config(), Credentials())
