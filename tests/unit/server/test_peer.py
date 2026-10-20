"""Peer lookup: Linux against real Unix sockets; the macOS branch on fixture bytes only."""

import asyncio
import os
import socket
import struct
import sys
from pathlib import Path

import pytest

from stt_daemon.server import peer
from stt_daemon.server.peer import Peer, parse_procargs2, peer_of, split_nul

linux_only = pytest.mark.skipif(sys.platform != "linux", reason="SO_PEERCRED and /proc")


@linux_only
def test_socketpair_peer_is_this_process() -> None:
    a, b = socket.socketpair()
    with a, b:
        assert peer_of(a) == Peer(os.getpid(), tuple(sys.orig_argv))


@linux_only
async def test_server_side_sees_the_connecting_process(tmp_path: Path) -> None:
    path = tmp_path / "s.sock"
    seen: asyncio.Future[Peer] = asyncio.get_running_loop().create_future()

    async def on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        seen.set_result(peer_of(writer.get_extra_info("socket")))
        writer.close()

    code = (
        "import socket, time; s = socket.socket(socket.AF_UNIX); "
        f"s.connect({str(path)!r}); time.sleep(5)"
    )
    server = await asyncio.start_unix_server(on_connect, path)
    proc = await asyncio.create_subprocess_exec(sys.executable, "-c", code)
    try:
        found = await asyncio.wait_for(seen, 5)
    finally:
        proc.kill()
        await proc.wait()
        server.close()
    assert found == Peer(proc.pid, (sys.executable, "-c", code))


def test_unknown_platform_gives_unknown_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "sunos5")
    a, b = socket.socketpair()
    with a, b:
        assert peer_of(a) == Peer()


class _Failing:
    def getsockopt(self, level: int, optname: int, buflen: int, /) -> bytes:
        raise OSError("not a unix socket")


def test_failed_lookup_gives_unknown_fields() -> None:
    assert peer_of(_Failing()) == Peer()


@linux_only
def test_gone_process_keeps_its_pid_without_cmdline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(peer._LOOKUPS, "linux", (lambda _sock: 2**22 + 1, peer._linux_cmdline))  # pyright: ignore[reportPrivateUsage]
    assert peer_of(_Failing()) == Peer(2**22 + 1, None)


def test_cmdline_splits_on_nul() -> None:
    assert split_nul(b"scribbly\0serve\0\0--x\0") == ("scribbly", "serve", "", "--x")
    assert split_nul(b"") is None


class _MacPeer:
    def getsockopt(self, level: int, optname: int, buflen: int, /) -> bytes:
        assert (level, optname, buflen) == (0, 0x002, 4)
        return struct.pack("i", 4242)


def test_macos_pid_from_local_peerpid(monkeypatch: pytest.MonkeyPatch) -> None:
    pid_of = peer._macos_pid  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(peer._LOOKUPS, "darwin", (pid_of, lambda pid: (str(pid),)))  # pyright: ignore[reportPrivateUsage]
    assert peer_of(_MacPeer()) == Peer(4242, ("4242",))


def procargs(argc: int, path: bytes, args: list[bytes], env: list[bytes]) -> bytes:
    body = path + b"\0\0\0\0" + b"".join(a + b"\0" for a in args + env)
    return struct.pack("=i", argc) + body


def test_procargs2_takes_argc_args_after_the_exec_path() -> None:
    data = procargs(
        3, b"/usr/local/bin/scribbly", [b"scribbly", b"serve", b"--port=1"], [b"HOME=/x"]
    )
    assert parse_procargs2(data) == ("scribbly", "serve", "--port=1")


def test_procargs2_refuses_short_or_broken_data() -> None:
    assert parse_procargs2(b"\1\0") is None
    assert parse_procargs2(struct.pack("=i", 2) + b"/bin/talkey") is None
    assert parse_procargs2(procargs(0, b"/bin/talkey", [], [])) is None
    truncated = procargs(3, b"/bin/talkey", [b"talkey"], [])
    assert parse_procargs2(truncated) is None
