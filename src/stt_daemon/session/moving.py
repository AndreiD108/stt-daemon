"""Waits bounded by progress instead of by a fixed time."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any


async def while_moving(
    work: Coroutine[Any, Any, None], quiet_s: float, mark: Callable[[], object | None]
) -> None:
    """Run `work`; TimeoutError once `mark()` stays the same for `quiet_s`. A None mark means
    paused: that time doesn't count. For waits that can take minutes (the rest of a file, the
    audio kept while suspended), where no fixed bound fits."""
    task = asyncio.ensure_future(work)
    seen = mark()
    try:
        while not (await asyncio.wait([task], timeout=quiet_s))[0]:
            now = mark()
            if now is not None and now == seen:
                raise TimeoutError
            seen = now
    finally:
        task.cancel()
        await asyncio.wait([task])
    task.result()
