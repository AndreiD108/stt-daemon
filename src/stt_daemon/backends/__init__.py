"""Transcription backends behind one interface (`base`); `BACKENDS` maps names to classes."""

from .base import Backend
from .elevenlabs import ElevenLabsBackend
from .elevenlabs_batch import ElevenLabsBatchBackend
from .parakeet import ParakeetBackend

BACKENDS: dict[str, type[Backend]] = {
    "elevenlabs": ElevenLabsBackend,
    "elevenlabs-batch": ElevenLabsBatchBackend,
    "parakeet": ParakeetBackend,
}
