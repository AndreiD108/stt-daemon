"""An stt-daemon backend for any OpenAI-compatible `/audio/transcriptions` endpoint."""

import io
import wave
from collections.abc import Mapping
from typing import Any

import httpx

from stt_daemon.plugin import Backend, Capabilities, Options, Result

DEFAULT_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"


class OpenAITranscriptions(Backend):
    name = "openai"
    needs_key = True
    # Text only: the daemon estimates word times from the audio.
    capabilities = Capabilities(languages=("auto", "en", "de", "fr", "es"), previous_text=True)

    def setup(self, key: str | None, config: Mapping[str, Any]) -> None:
        self.key = key
        self.endpoint = str(config.get("endpoint", DEFAULT_ENDPOINT))
        self.model = str(config.get("model", "whisper-1"))

    async def transcribe(self, pcm: bytes, options: Options) -> Result:
        fields = {"model": self.model, "response_format": "json"}
        if options.language:
            fields["language"] = options.language
        if options.previous_text:
            fields["prompt"] = options.previous_text
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.key}"},
                data=fields,
                files={"file": ("audio.wav", to_wav(pcm), "audio/wav")},
            )
        if response.is_error:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        return Result(text=response.json()["text"])


def to_wav(pcm: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    return out.getvalue()
