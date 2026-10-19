from collections.abc import Mapping
from pathlib import Path
from typing import Any

from stt_daemon.config import data_dir
from stt_daemon.plugin import Backend, Capabilities, Options, Result, Word


class LocalModel(Backend):
    name = "local-model"
    capabilities = Capabilities(word_times=True)

    def setup(self, key: str | None, config: Mapping[str, Any]) -> None:
        self.model = data_dir() / "models" / self.name / "model.bin"

    async def install(self, data_dir: Path) -> None:
        (data_dir / "model.bin").write_bytes(b"weights")

    def available(self) -> bool:
        return self.model.is_file()

    async def transcribe(self, pcm: bytes, options: Options) -> Result:
        seconds = len(pcm) / 32000
        return Result(words=[Word("local", 0.0, seconds / 2), Word("words", seconds / 2, seconds)])
