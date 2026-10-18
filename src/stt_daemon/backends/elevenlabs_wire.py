"""ElevenLabs realtime tables: language codes and error messages."""

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
