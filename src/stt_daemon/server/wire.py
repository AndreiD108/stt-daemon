"""Session and VAD events as protocol events: (type, data)."""

from ..audio.frames import epoch_ms
from ..audio.vad import SpeechStarted, VadEvent
from ..session.events import (
    PartialText,
    Progress,
    SessionEnded,
    SessionError,
    SessionEvent,
    SessionResumed,
    SessionStarted,
    SessionSuspended,
)
from ..session.markers import TimedWord
from ..session.transcript import Segment
from .slot_state import Client

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
        case Progress():
            return "progress", {"session_id": session_id, "audio_seconds": event.audio_seconds}
        case Segment():
            return "segment", segment(event, session_id)
        case SessionError():
            return "session_error", {
                "session_id": session_id,
                "code": event.code,
                "message": event.message,
                "fatal": event.fatal,
            }
        case SessionSuspended():
            by = event.by
            return "session_suspended", {
                "session_id": session_id,
                "reason": "preempted",
                "by": {"kind": by.kind, "pid": by.pid, "client_label": by.client_label},
            }
        case SessionResumed():
            return "session_resumed", {
                "session_id": session_id,
                "suspended_seconds": event.suspended_seconds,
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
    if seg.speaker is not None:
        data["speaker"] = seg.speaker
    data["start_ms"] = seg.start_ms
    data["end_ms"] = seg.end_ms
    if seg.words is not None:
        data["words"] = [_word(w) for w in seg.words]
    if seg.markers:
        markers: list[dict[str, object]] = []
        for m in seg.markers:
            marker: dict[str, object] = {"marker": m.marker, "at_ms": m.at_ms}
            if seg.words is not None:
                marker["after_word"] = m.after_word
            markers.append(marker)
        data["markers"] = markers
    return data


def _word(word: TimedWord) -> dict[str, object]:
    data: dict[str, object] = {"text": word.text, "start_ms": word.start_ms, "end_ms": word.end_ms}
    if word.speaker is not None:
        data["speaker"] = word.speaker
    return data
