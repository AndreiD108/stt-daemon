from collections.abc import AsyncIterator

import pytest
from harness import Harness


@pytest.fixture
async def daemon() -> AsyncIterator[Harness]:
    harness = Harness()
    try:
        yield harness
    finally:
        await harness.stop()
