import asyncio
from collections.abc import AsyncIterator

import pytest
from fake_elevenlabs import TEST_KEY, FakeElevenLabs
from slot_harness import SpyCapture
from websockets.asyncio.server import serve

from stt_daemon.backends.elevenlabs import ElevenLabsBackend
from stt_daemon.config import ElevenLabsConfig


@pytest.fixture
async def fake_el() -> AsyncIterator[FakeElevenLabs]:
    el = FakeElevenLabs()
    async with serve(el.handle, "127.0.0.1", 0, process_request=el.process_request) as server:
        el.server = server
        yield el


@pytest.fixture
def el_backend(fake_el: FakeElevenLabs) -> ElevenLabsBackend:
    return ElevenLabsBackend(
        ElevenLabsConfig(endpoint=fake_el.url, keyterms=["Scribbly"]), TEST_KEY
    )


@pytest.fixture
async def silent() -> AsyncIterator[SpyCapture]:
    capture = SpyCapture()
    await capture.start()
    await asyncio.sleep(0.1)
    yield capture
    await capture.stop()
