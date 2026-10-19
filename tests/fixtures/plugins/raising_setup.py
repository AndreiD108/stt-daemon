from collections.abc import Mapping
from typing import Any

from stt_daemon.plugin import Backend, Options, Result


class RaisingSetup(Backend):
    name = "raising-setup"

    def setup(self, key: str | None, config: Mapping[str, Any]) -> None:
        raise RuntimeError("no device found\nsecond line")

    async def transcribe(self, pcm: bytes, options: Options) -> Result:
        return Result("never")
