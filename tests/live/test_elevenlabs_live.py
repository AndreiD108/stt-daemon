"""Real ElevenLabs API; needs ELEVENLABS_API_KEY. Run with `pytest -m live`."""

import asyncio
import os
from pathlib import Path

import pytest

from stt_daemon.audio.wav_capture import read_wav
from stt_daemon.backends.base import BackendEvent, Committed, Failed, OpenFailed, OpenOptions
from stt_daemon.backends.elevenlabs import ElevenLabsBackend
from stt_daemon.config import ElevenLabsConfig

FIXTURE = Path(__file__).parents[1] / "fixtures" / "two_phrases.wav"
# Speech edges in the fixture (-40 dBFS), seconds.
PHRASE_STARTS = (1.00, 5.74)
RATE = 16000


def _key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        pytest.skip("ELEVENLABS_API_KEY not set")
    return key


async def _commits(events: list[BackendEvent], n: int) -> list[Committed]:
    async with asyncio.timeout(15):
        while len(found := [e for e in events if isinstance(e, Committed)]) < n:
            assert not [e for e in events if isinstance(e, Failed)], events
            await asyncio.sleep(0.05)
    return found


async def test_words_come_back_at_the_speech_edges() -> None:
    backend = ElevenLabsBackend(ElevenLabsConfig(), _key())
    audio = read_wav(FIXTURE)
    events: list[BackendEvent] = []
    options = OpenOptions(keyterms=("Scribbly",), previous_text="a note about the report")
    conn = await backend.open(options, 1, events.append)
    try:
        cut = round(4.5 * RATE)
        for part in (audio[:cut], audio[cut:]):
            for i in range(0, len(part), RATE // 5):
                await conn.send_audio(part[i : i + RATE // 5])
                await asyncio.sleep(0.05)
            await conn.commit()
        # A tail too short for EL on its own: padded, not throttled.
        await conn.send_audio(audio[: RATE // 10])
        await conn.commit()
        first, second, tail = await _commits(events, 3)
    finally:
        await conn.close()
    for segment, edge in zip((first, second), PHRASE_STARTS, strict=True):
        assert segment.language == "en"
        assert segment.words
        assert abs(segment.words[0].start_s - edge) < 0.1, segment.words[0]
    assert [c.commit_id for c in (first, second, tail)] == [1, 2, 3]
    assert tail.audio_end_s == pytest.approx(len(audio) / RATE + 0.1)


async def test_a_bad_key_fails_the_open_without_retrying() -> None:
    _key()
    backend = ElevenLabsBackend(ElevenLabsConfig(), "sk_not_a_real_key")
    with pytest.raises(OpenFailed) as failed:
        await backend.open(OpenOptions(), 1, lambda _: None)
    assert (failed.value.code, failed.value.retryable) == ("auth", False)
