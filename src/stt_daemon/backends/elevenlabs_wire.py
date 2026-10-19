"""ElevenLabs tables and HTTP refusals: language codes, error messages, status → code."""

import json
from typing import Any, cast

from .base import OpenFailed

# ISO 639-1 where one exists (detection reports these), else ISO 639-3; EL takes either.
LANGUAGES = (
    "af", "am", "ar", "hy", "as", "ast", "az", "be", "bn", "bs", "bg", "my", "yue", "ca", "ceb",
    "ny", "hr", "cs", "da", "nl", "en", "et", "fil", "fi", "fr", "ff", "gl", "lg", "ka", "de",
    "el", "gu", "ha", "he", "hi", "hu", "is", "ig", "id", "ga", "it", "ja", "jv", "kea", "kn",
    "kk", "km", "ko", "ku", "ky", "lo", "lv", "ln", "lt", "luo", "lb", "mk", "ms", "ml", "mt",
    "zh", "mi", "mr", "mn", "ne", "nso", "no", "oc", "or", "ps", "fa", "pl", "pt", "pa", "ro",
    "ru", "sr", "sn", "sd", "si", "sk", "sl", "so", "es", "sw", "sv", "ta", "tg", "te", "th",
    "tr", "uk", "umb", "ur", "uz", "vi", "cy", "wo", "xh", "yo", "zu",
)  # fmt: skip

# The file API reports ISO 639-3; the codes above use 639-1 where one exists.
ISO3_TO_LANGUAGE = {
    "afr": "af", "amh": "am", "ara": "ar", "hye": "hy", "asm": "as", "aze": "az", "bel": "be",
    "ben": "bn", "bos": "bs", "bul": "bg", "mya": "my", "cat": "ca", "nya": "ny", "hrv": "hr",
    "ces": "cs", "dan": "da", "nld": "nl", "eng": "en", "est": "et", "fin": "fi", "fra": "fr",
    "ful": "ff", "glg": "gl", "lug": "lg", "kat": "ka", "deu": "de", "ell": "el", "guj": "gu",
    "hau": "ha", "heb": "he", "hin": "hi", "hun": "hu", "isl": "is", "ibo": "ig", "ind": "id",
    "gle": "ga", "ita": "it", "jpn": "ja", "jav": "jv", "kan": "kn", "kaz": "kk", "khm": "km",
    "kor": "ko", "kur": "ku", "kir": "ky", "lao": "lo", "lav": "lv", "lin": "ln", "lit": "lt",
    "ltz": "lb", "mkd": "mk", "msa": "ms", "mal": "ml", "mlt": "mt", "zho": "zh", "mri": "mi",
    "mar": "mr", "mon": "mn", "nep": "ne", "nor": "no", "oci": "oc", "ori": "or", "pus": "ps",
    "fas": "fa", "pol": "pl", "por": "pt", "pan": "pa", "ron": "ro", "rus": "ru", "srp": "sr",
    "sna": "sn", "snd": "sd", "sin": "si", "slk": "sk", "slv": "sl", "som": "so", "spa": "es",
    "swa": "sw", "swe": "sv", "tam": "ta", "tgk": "tg", "tel": "te", "tha": "th", "tur": "tr",
    "ukr": "uk", "urd": "ur", "uzb": "uz", "vie": "vi", "cym": "cy", "wol": "wo", "xho": "xh",
    "yor": "yo", "zul": "zu",
    "cmn": "zh",
}  # fmt: skip

# Error message type -> (code, retryable). The server closes the connection after each.
ERRORS: dict[str, tuple[str, bool]] = {
    "auth_error": ("auth", False),
    "unaccepted_terms": ("auth", False),
    "quota_exceeded": ("quota", False),
    "rate_limited": ("rate_limited", True),
    "commit_throttled": ("rate_limited", True),
    "queue_overflow": ("rate_limited", True),
    "resource_exhausted": ("rate_limited", True),
    "session_time_limit_exceeded": ("session_limit", True),
    "insufficient_audio_activity": ("idle_drop", True),
    "input_error": ("invalid_request", False),
    "invalid_request": ("invalid_request", False),
    "chunk_size_exceeded": ("invalid_request", False),
    "transcriber_error": ("backend_error", True),
    "error": ("backend_error", True),
}

# In an HTTP 429 body: the plan's concurrent connections are all in use. `detail.code` in the
# current error format; the older `detail.status` value.
CONCURRENCY_REFUSALS = (b"concurrent_limit_exceeded", b"too_many_concurrent_requests")

_MESSAGE_LIMIT = 300


def http_failure(status: int, body: bytes | bytearray | None, secret: str) -> OpenFailed:
    """An HTTP error answer as a failure whose message carries ElevenLabs' own text; `secret`
    (the API key) is cut out of that text."""
    body = bytes(body or b"")
    reason = body_text(body).replace(secret, "[key]") if secret else body_text(body)
    message = f"ElevenLabs HTTP {status}" + (f": {reason}" if reason else "")
    if status in (401, 403):
        return OpenFailed("auth", message, retryable=False)
    if status == 402:
        return OpenFailed("quota", message, retryable=False)
    if status == 429 and any(code in body for code in CONCURRENCY_REFUSALS):
        return OpenFailed("concurrency_limit", message, retryable=False)
    if status == 429:
        return OpenFailed("rate_limited", message, retryable=False)
    if status >= 500:
        return OpenFailed("backend_error", message, retryable=True)
    return OpenFailed("invalid_request", message, retryable=False)


def body_text(body: bytes) -> str:
    """`detail.message`, a `detail` string, or the first validation error's `msg`; else the
    body's text. One line, capped."""
    text = body.decode("utf-8", "replace")
    try:
        parsed: Any = json.loads(text)
        detail = parsed.get("detail")
    except (json.JSONDecodeError, AttributeError):
        detail = None
    if isinstance(detail, dict):
        detail = cast(dict[str, Any], detail).get("message")
    elif isinstance(detail, list) and detail and isinstance(detail[0], dict):
        detail = cast(dict[str, Any], detail[0]).get("msg")
    if isinstance(detail, str) and detail.strip():
        text = detail
    text = " ".join(text.split())
    return text if len(text) <= _MESSAGE_LIMIT else text[: _MESSAGE_LIMIT - 1] + "…"
