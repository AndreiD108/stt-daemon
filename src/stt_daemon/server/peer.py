"""Who is on the other end of a Unix socket: the peer's pid and command line, from the kernel.

Linux: `SO_PEERCRED`, then `/proc/<pid>/cmdline`. macOS: `LOCAL_PEERPID`, then the process
table (`KERN_PROCARGS2`); written to the documented layout, untested. Anything else, or any
failure: unknown fields, never an error.
"""

import ctypes
import ctypes.util
import logging
import socket
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

_SOL_LOCAL = 0
_LOCAL_PEERPID = 0x002
_CTL_KERN = 1
_KERN_PROCARGS2 = 49


@dataclass(frozen=True)
class Peer:
    pid: int | None = None
    cmdline: tuple[str, ...] | None = None


class _Sockopts(Protocol):
    def getsockopt(self, level: int, optname: int, buflen: int, /) -> bytes: ...


def peer_of(sock: _Sockopts) -> Peer:
    """Read at connect time: a pid can be reused once the peer exits."""
    lookup = _LOOKUPS.get(sys.platform)
    if lookup is None:
        return Peer()
    pid_of, cmdline_of = lookup
    try:
        pid = pid_of(sock)
        return Peer(pid, cmdline_of(pid) if pid is not None else None)
    except Exception as exc:  # never an error: unknown fields instead
        log.debug("peer lookup failed: %s", exc)
        return Peer()


def _linux_pid(sock: _Sockopts) -> int | None:
    so_peercred: int = getattr(socket, "SO_PEERCRED")  # noqa: B009  (Linux-only constant)
    pid, _uid, _gid = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, so_peercred, 12))
    return pid if pid > 0 else None


def _linux_cmdline(pid: int) -> tuple[str, ...] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return split_nul(raw)


def split_nul(raw: bytes) -> tuple[str, ...] | None:
    """NUL-terminated argv strings; None when there are none (kernel threads, zombies)."""
    args = raw.split(b"\0")
    if args and args[-1] == b"":
        args.pop()
    return tuple(a.decode(errors="replace") for a in args) or None


def _macos_pid(sock: _Sockopts) -> int | None:
    (pid,) = struct.unpack("i", sock.getsockopt(_SOL_LOCAL, _LOCAL_PEERPID, 4))
    return pid if pid > 0 else None


def _macos_cmdline(pid: int) -> tuple[str, ...] | None:
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    mib = (ctypes.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
        return None
    return parse_procargs2(buf.raw[: size.value])


def parse_procargs2(data: bytes) -> tuple[str, ...] | None:
    """`KERN_PROCARGS2`: native int argc, the exec path, NUL padding, then argc NUL-terminated
    args (environment after them)."""
    if len(data) < 4:
        return None
    (argc,) = struct.unpack("=i", data[:4])
    rest = data[4:]
    path_end = rest.find(b"\0")
    if argc <= 0 or path_end < 0:
        return None
    rest = rest[path_end:].lstrip(b"\0")
    args = rest.split(b"\0")[:argc]
    if len(args) < argc:
        return None
    return tuple(a.decode(errors="replace") for a in args)


_LOOKUPS: dict[
    str, tuple[Callable[[_Sockopts], int | None], Callable[[int], tuple[str, ...] | None]]
] = {
    "linux": (_linux_pid, _linux_cmdline),
    "darwin": (_macos_pid, _macos_cmdline),
}
