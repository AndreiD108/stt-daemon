"""Session and VAD events as protocol events: (type, data)."""

from ..audio.frames import epoch_ms
from ..audio.vad import SpeechStarted, VadEvent
from ..session.events import (
    PartialText,
    SessionEnded,
    SessionError,
    SessionEvent,
    SessionStarted,
)
from ..session.transcript import Segment
from .slot import Client

type Wire = tuple[str, dict[str, object]]


def session_event(
    event: SessionEvent, session_id: str, client: Client, session_label: str | None
) -> Wire:
    match event:
        case SessionStarted():
            return "session_started", {
                "session_id": session_id,
                "client_id": client.client_id,
                "client_label": client.label,
                "session_label": session_label,
                "kind": event.kind,
                "backend": event.backend,
                "language": event.language,
                "started_at": event.started_at,
                "previous_text": event.previous_text,
            }
        case PartialText():
            return "partial", {"session_id": session_id, "text": event.text}
        case Segment():
            return "segment", segment(event, session_id)
        case SessionError():
            return "session_error", {
                "session_id": session_id,
                "code": event.code,
                "message": event.message,
                "fatal": event.fatal,
            }
        case SessionEnded():
            data: dict[str, object] = {
                "session_id": session_id,
                "reason": event.reason,
                "final_text": event.final_text,
                "duration_ms": event.duration_ms,
                "segment_count": event.segment_count,
            }
            if event.evicted_by is not None:
                by = event.evicted_by
                data["evicted_by"] = {
                    "client_id": by.client_id,
                    "pid": by.pid,
                    "client_label": by.client_label,
                }
            return "session_ended", data


def vad_event(event: VadEvent, buffer_ms: int) -> Wire:
    at = epoch_ms(event.at)
    if isinstance(event, SpeechStarted):
        return "speech_started", {
            "speech_id": event.speech_id,
            "at": at,
            "expires_at": at + buffer_ms,
        }
    return "speech_ended", {"speech_id": event.speech_id, "at": at}


def segment(seg: Segment, session_id: str) -> dict[str, object]:
    data: dict[str, object] = {"session_id": session_id, "index": seg.index, "text": seg.text}
    if seg.language is not None:
        data["language"] = seg.language
    data["start_ms"] = seg.start_ms
    data["end_ms"] = seg.end_ms
    if seg.words is not None:
        data["words"] = [
            {"text": w.text, "start_ms": w.start_ms, "end_ms": w.end_ms} for w in seg.words
        ]
    if seg.markers:
        markers: list[dict[str, object]] = []
        for m in seg.markers:
            marker: dict[str, object] = {"marker": m.marker, "at_ms": m.at_ms}
            if seg.words is not None:
                marker["after_word"] = m.after_word
            markers.append(marker)
        data["markers"] = markers
    return data
