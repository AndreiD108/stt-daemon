"""Transcription backends behind one interface (`base`); `BACKENDS` maps names to classes."""

from .base import Backend
from .elevenlabs import ElevenLabsBackend

BACKENDS: dict[str, type[Backend]] = {"elevenlabs": ElevenLabsBackend}
