import json

import pytest

from stt_daemon.server.messages import (
    RequestError,
    StartRequest,
    encode,
    parse_hello,
    parse_line,
    parse_marker,
    parse_start,
)


def code_of(call: object, *args: object) -> tuple[str, object]:
    with pytest.raises(RequestError) as caught:
        call(*args)  # type: ignore[operator]
    return caught.value.code, caught.value.extra.get("field")


@pytest.mark.parametrize(
    "line",
    [b"{", b"[1]", b'"text"', b"{}", b'{"type": ""}', b'{"type": 3}', b"\xff\xfe"],
)
def test_lines_that_are_not_messages_are_bad_requests(line: bytes) -> None:
    assert code_of(parse_line, line) == ("bad_request", None)


@pytest.mark.parametrize("bad", [{"id": True}, {"id": 1.5}, {"id": []}, {"data": []}])
def test_envelope_fields_are_checked(bad: dict[str, object]) -> None:
    line = json.dumps({"type": "ping", **bad}).encode()
    assert code_of(parse_line, line)[0] == "bad_request"


def test_a_message_keeps_its_id_and_data() -> None:
    message = parse_line(b'{"type": "add_marker", "id": 7, "data": {"marker": "x"}}\n')
    assert (message.type, message.id, dict(message.data)) == ("add_marker", 7, {"marker": "x"})
    assert parse_line(b'{"type": "ping"}').data == {}


def test_encode_writes_one_line_with_reply_to_only_on_replies() -> None:
    line = encode("pong", {}, reply_to="r1")
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    assert json.loads(line)["reply_to"] == "r1"
    event = json.loads(encode("slot", {"state": "idle"}))
    assert "reply_to" not in event
    assert isinstance(event["ts"], int)
    assert json.loads(encode("segment", {"text": "⟦A⟧"}))["data"]["text"] == "⟦A⟧"


@pytest.mark.parametrize("protocol", [None, 2, "1", True])
def test_hello_needs_protocol_1(protocol: object) -> None:
    with pytest.raises(RequestError) as caught:
        parse_hello({"protocol": protocol})
    assert caught.value.code == "protocol_unsupported"
    assert caught.value.to_wire()["supported"] == [1]


def test_hello_options() -> None:
    hello = parse_hello({"protocol": 1, "label": "Talkey", "observe_all": True})
    assert (hello.label, hello.observe_all, hello.vad_events) == ("Talkey", True, False)
    assert parse_hello({"protocol": 1, "label": ""}).label is None
    assert code_of(parse_hello, {"protocol": 1, "vad_events": 1}) == ("bad_request", "vad_events")


def test_start_fields_and_defaults() -> None:
    assert parse_start({}) == StartRequest()
    request = parse_start({"keyterms": ["a", "b"], "partials": True, "context_key": "k"})
    assert request.keyterms == ("a", "b")
    assert request.partials
    assert request.context_key == "k"
    assert parse_start({"previous_text": ""}).previous_text == ""


@pytest.mark.parametrize(
    ("data", "name"),
    [
        ({"label": 3}, "label"),
        ({"backend": None, "language": ["en"]}, "language"),
        ({"keyterms": "a"}, "keyterms"),
        ({"keyterms": ["a", 1]}, "keyterms"),
        ({"word_timestamps": "true"}, "word_timestamps"),
        ({"speech_id": 41}, "speech_id"),
    ],
)
def test_start_type_errors_name_the_field(data: dict[str, object], name: str) -> None:
    assert code_of(parse_start, data) == ("bad_request", name)


def test_marker_errors_are_invalid() -> None:
    assert parse_marker({"marker": "⟦A⟧", "at": 5}).at == 5
    assert parse_marker({"marker": "⟦A⟧"}).at is None
    assert code_of(parse_marker, {})[0] == "invalid"
    assert code_of(parse_marker, {"marker": "  "})[0] == "invalid"
    assert code_of(parse_marker, {"marker": "a", "at": 1.5}) == ("invalid", "at")
    assert code_of(parse_marker, {"marker": "a", "at": True}) == ("invalid", "at")
