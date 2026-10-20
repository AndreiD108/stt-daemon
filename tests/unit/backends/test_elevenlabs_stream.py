import asyncio
import base64
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from stt_daemon.audio.frames import Samples
from stt_daemon.backends import elevenlabs
from stt_daemon.backends.base import (
    BackendEvent,
    Closed,
    Committed,
    Failed,
    OpenFailed,
    OpenOptions,
    Partial,
    Word,
)
from stt_daemon.backends.elevenlabs import ElevenLabsBackend, ElevenLabsConnection
from unit.backends.fake_elevenlabs import TEST_KEY, FakeElevenLabs, commits, committed, el_word
from unit.waiting import wait_until


def _audio(seconds: float, value: float = 0.0) -> Samples:
    return np.full(round(seconds * 16000), value, np.float32)


def _samples(msg: dict[str, Any]) -> int:
    return len(base64.b64decode(msg["audio_base_64"])) // 2


async def _open(
    backend: ElevenLabsBackend, options: OpenOptions | None = None
) -> tuple[ElevenLabsConnection, list[BackendEvent]]:
    events: list[BackendEvent] = []
    conn = await backend.open(options or OpenOptions(), 4, events.append)
    return conn, events


def _query(fake_el: FakeElevenLabs) -> dict[str, list[str]]:
    return parse_qs(urlsplit(fake_el.requests[-1].path).query)


async def test_auto_language_asks_for_detection_and_timestamps(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    await _open(el_backend, OpenOptions(keyterms=("Talkey", "scribbly")))
    query = _query(fake_el)
    assert query["include_timestamps"] == ["true"]
    assert query["include_language_detection"] == ["true"]
    assert query["commit_strategy"] == ["manual"]
    assert query["audio_format"] == ["pcm_16000"]
    assert "language_code" not in query
    assert query["keyterms"] == ["Scribbly", "Talkey"]
    assert fake_el.requests[-1].headers["xi-api-key"] == TEST_KEY


async def test_fixed_language_is_sent_and_unknown_one_fails_before_connecting(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    await _open(el_backend, OpenOptions(language="ro"))
    query = _query(fake_el)
    assert query["language_code"] == ["ro"]
    assert query["include_language_detection"] == ["false"]
    with pytest.raises(OpenFailed) as failed:
        await _open(el_backend, OpenOptions(language="xx"))
    assert (failed.value.code, failed.value.retryable) == ("unsupported_language", False)
    assert len(fake_el.requests) == 1


async def test_keyterms_over_the_limits_are_dropped(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    terms = ("a" * 21, *(f"term{i}" for i in range(60)))
    await _open(el_backend, OpenOptions(keyterms=terms))
    sent = _query(fake_el)["keyterms"]
    assert sent == ["Scribbly", *(f"term{i}" for i in range(49))]


async def test_previous_text_rides_on_the_first_audio_chunk_only(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    long_text = "we were talking about the " + "x" * 10 + " and then the Scribbly canvas"
    conn, _ = await _open(el_backend, OpenOptions(previous_text=long_text))
    await conn.send_audio(_audio(0.2))
    await conn.send_audio(_audio(0.2))
    await wait_until(lambda: len(fake_el.messages) == 2)
    first, second = fake_el.messages
    assert first["previous_text"] == "about the xxxxxxxxxx and then the Scribbly canvas"
    assert "previous_text" not in second


async def test_audio_is_pcm16_in_chunks_of_at_most_a_second(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    conn, _ = await _open(el_backend)
    await conn.send_audio(_audio(2.5, 0.5))
    await wait_until(lambda: len(fake_el.messages) == 3)
    assert [_samples(m) for m in fake_el.messages] == [16000, 16000, 8000]
    pcm = np.frombuffer(base64.b64decode(fake_el.messages[0]["audio_base_64"]), "<i2")
    assert set(pcm.tolist()) == {16383}
    assert all(m["sample_rate"] == 16000 and m["commit"] is False for m in fake_el.messages)


async def test_commit_is_its_own_empty_message_and_the_timestamped_reply_acknowledges_it(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    fake_el.reply = lambda n: committed(
        "hello there",
        [el_word("hello", 0.1, 0.4), {"text": " ", "start": 0.4, "end": 0.5, "type": "spacing"},
         el_word(" there", 0.5, 0.9)],
    )  # fmt: skip
    conn, events = await _open(el_backend)
    await conn.send_audio(_audio(1.0))
    assert await conn.commit() == 1
    await wait_until(lambda: bool(commits(events)))
    assert fake_el.messages[-1] == {
        "message_type": "input_audio_chunk",
        "audio_base_64": "",
        "commit": True,
        "sample_rate": 16000,
    }
    assert events == [
        Committed(
            4, 1, "hello there", "en", (Word("hello", 0.1, 0.4), Word("there", 0.5, 0.9)), 1.0
        )
    ]


async def test_plain_committed_message_alone_acknowledges_nothing(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    fake_el.reply = lambda n: [{"message_type": "committed_transcript", "text": "hi"}]
    conn, events = await _open(el_backend)
    await conn.send_audio(_audio(0.5))
    await conn.commit()
    await wait_until(lambda: fake_el.commits == 1)
    await asyncio.sleep(0.05)
    assert events == []


async def test_short_tail_is_padded_and_word_times_skip_the_pad(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    replies = {
        1: committed("one"),
        2: committed("two"),
        # Wire audio: 0-1.0 first, 1.0-1.1 tail, 1.1-1.35 pad, 1.35-2.35 third (1.1-2.1 sent).
        3: committed("in pad, after", [el_word("in", 1.2, 1.3), el_word("after", 2.0, 2.2)]),
    }
    fake_el.reply = lambda n: replies[n]
    conn, events = await _open(el_backend)
    await conn.send_audio(_audio(1.0))
    await conn.commit()
    await conn.send_audio(_audio(0.1))
    await conn.commit()
    await conn.send_audio(_audio(1.0))
    await conn.commit()
    await wait_until(lambda: len(commits(events)) == 3)
    audio = [_samples(m) for m in fake_el.messages if not m["commit"]]
    assert audio == [16000, 1600, 4000, 16000]
    assert [c.audio_end_s for c in commits(events)] == [1.0, 1.1, 2.1]
    words = commits(events)[2].words
    assert words is not None
    assert [(w.start_s, w.end_s) for w in words] == pytest.approx([(1.1, 1.1), (1.75, 1.95)])


async def test_commit_without_audio_is_answered_here_in_order(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    conn, events = await _open(el_backend)
    fake_el.reply = lambda n: []
    await conn.send_audio(_audio(0.5))
    assert await conn.commit() == 1
    assert await conn.commit() == 2
    await wait_until(lambda: fake_el.commits == 1)
    await asyncio.sleep(0.02)
    assert events == []
    await fake_el.push(*committed("spoken"))
    await wait_until(lambda: len(commits(events)) == 2)
    assert [(c.commit_id, c.text, c.audio_end_s) for c in commits(events)] == [
        (1, "spoken", 0.5),
        (2, "", 0.5),
    ]
    assert fake_el.commits == 1


async def test_a_commit_before_any_audio_is_answered_at_once(
    el_backend: ElevenLabsBackend,
) -> None:
    conn, events = await _open(el_backend)
    await conn.commit()
    assert events == [Committed(4, 1, "", None, (), 0.0)]


async def test_partials_are_forwarded_once_and_never_repeat_committed_text(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    fake_el.reply = lambda n: [
        *committed("move this up"),
        {"message_type": "partial_transcript", "text": "move this up"},
    ]
    conn, events = await _open(el_backend, OpenOptions(partials=True))
    for text in ("move", "move", "", "move this"):
        await fake_el.push({"message_type": "partial_transcript", "text": text})
    await conn.send_audio(_audio(0.5))
    await conn.commit()
    await wait_until(lambda: bool(commits(events)))
    await fake_el.push({"message_type": "partial_transcript", "text": "next"})
    await wait_until(lambda: len(events) == 4)
    assert [e.text for e in events if isinstance(e, Partial)] == ["move", "move this", "next"]


async def test_partials_are_dropped_unless_asked(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    conn, events = await _open(el_backend)
    await fake_el.push({"message_type": "partial_transcript", "text": "move"})
    await conn.send_audio(_audio(0.5))
    await conn.commit()
    await wait_until(lambda: bool(commits(events)))
    assert not any(isinstance(e, Partial) for e in events)


@pytest.mark.parametrize(
    ("kind", "code", "retryable"),
    [
        ("insufficient_audio_activity", "idle_drop", True),
        ("session_time_limit_exceeded", "session_limit", True),
        ("quota_exceeded", "quota", False),
        ("transcriber_error", "backend_error", True),
    ],
)
async def test_error_message_fails_then_closes_the_connection(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend, kind: str, code: str, retryable: bool
) -> None:
    conn, events = await _open(el_backend)
    await fake_el.push({"message_type": kind, "error": "details"})
    await wait_until(lambda: len(events) == 2)
    failed, closed = events
    assert isinstance(failed, Failed)
    assert (failed.code, failed.retryable, failed.fatal) == (code, retryable, True)
    assert closed == Closed(4)
    await conn.send_audio(_audio(0.5))
    await conn.commit()
    await conn.close()
    assert len(events) == 2


async def test_server_dropping_the_connection_is_a_retryable_network_failure(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    _, events = await _open(el_backend)
    await fake_el.connections[-1].close(1011, "gone")
    await wait_until(lambda: len(events) == 2)
    assert [type(e) for e in events] == [Failed, Closed]
    failed = events[0]
    assert isinstance(failed, Failed)
    assert (failed.code, failed.retryable) == ("network", True)


async def test_close_ends_with_closed_only(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    conn, events = await _open(el_backend)
    await conn.close()
    await conn.close()
    assert events == [Closed(4)]
    await wait_until(lambda: fake_el.connections[-1].close_code is not None)


async def test_backend_commits_itself_before_the_server_would(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(elevenlabs, "_SELF_COMMIT_AFTER_S", 1.0)
    conn, events = await _open(el_backend)
    await conn.send_audio(_audio(1.2))
    await wait_until(lambda: len(commits(events)) == 1)
    await conn.send_audio(_audio(0.5))
    assert await conn.commit() == 1
    await wait_until(lambda: len(commits(events)) == 2)
    assert [(c.commit_id, c.audio_end_s) for c in commits(events)] == [(None, 1.0), (1, 1.7)]


async def test_self_commit_cuts_inside_one_long_send(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(elevenlabs, "_SELF_COMMIT_AFTER_S", 2.0)
    conn, events = await _open(el_backend)
    await conn.send_audio(_audio(5.0))
    await wait_until(lambda: len(commits(events)) == 2)
    kinds = ["commit" if m["commit"] else _samples(m) for m in fake_el.messages]
    assert kinds == [16000, 16000, "commit", 16000, 16000, "commit", 16000]
    assert [c.audio_end_s for c in commits(events)] == [2.0, 4.0]


async def test_server_commit_claims_no_audio_and_the_next_commit_covers_the_rest(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    conn, events = await _open(el_backend)
    await conn.send_audio(_audio(0.5))
    await conn.commit()
    await wait_until(lambda: len(commits(events)) == 1)
    await conn.send_audio(_audio(1.0))
    # The server cut somewhere in that second; its reply comes after more audio was sent.
    await conn.send_audio(_audio(0.1))
    await fake_el.push(*committed("server cut"))
    await wait_until(lambda: len(commits(events)) == 2)
    assert await conn.commit() == 2
    await wait_until(lambda: len(commits(events)) == 3)
    assert [(c.commit_id, c.text, c.audio_end_s) for c in commits(events)] == [
        (1, "text 1", 0.5),
        (None, "server cut", 0.5),
        (2, "text 2", 1.6),
    ]
    # The commit after the unknown cut goes to the server, padded in full.
    audio = [_samples(m) for m in fake_el.messages if not m["commit"]]
    assert audio == [8000, 16000, 1600, 5600]
    assert fake_el.commits == 2


async def test_a_bad_message_fails_the_connection(
    fake_el: FakeElevenLabs, el_backend: ElevenLabsBackend
) -> None:
    conn, events = await _open(el_backend)
    await conn.send_audio(_audio(0.5))
    await conn.commit()
    await fake_el.push(
        {"message_type": "committed_transcript_with_timestamps", "text": "x",
         "words": [{"text": "x", "type": "word"}]}
    )  # fmt: skip
    await wait_until(lambda: len(events) >= 2)
    assert [type(e) for e in events] == [Failed, Closed]
    failed = events[0]
    assert isinstance(failed, Failed)
    assert (failed.code, failed.retryable) == ("backend_error", True)
    await conn.close()
    assert len(events) == 2
