"""Echo cancellation for a system session's mic track on Linux: PipeWire's echo-cancel module.

A `pw-cli -m` process loads the module for one session; its exit unloads it. In monitor mode the
module takes the default output's monitor as the reference and offers the default mic, with that
playback taken out, as a source node. The mic track records that node with `pw-record`. When the
module can't load, or its `pw-cli` ends, the track takes the plain mic instead.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from ..service.preflight import check_pipewire_aec, check_tool
from .capture import CaptureSource, CaptureUnavailable
from .frames import Attachment, Frame
from .stream_capture import STDERR_TAIL, ToolMissing
from .system_capture import RESTART_BACKOFF_S, AppTarget, SystemCapture

log = logging.getLogger(__name__)

AEC_MODULE = "libpipewire-module-echo-cancel"
NODE_WAIT_S = 2.0
TERM_WAIT_S = 2.0
MIC_RETRY_S = RESTART_BACKOFF_S

type Command = Callable[[str], Sequence[str]]
"""The `pw-cli` command line for a source node name."""


def pw_cli_command(node_name: str) -> list[str]:
    args = f'{{ monitor.mode = true source.props = {{ node.name = "{node_name}" }} }}'
    return ["pw-cli", "-m", "load-module", AEC_MODULE, args]


class AecUnavailable(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NodeWatch(Protocol):
    """Where PipeWire nodes are seen by `node.name` (the source scan)."""

    def node_serial(self, name: str) -> str | None: ...

    async def wait_node(self, name: str) -> str: ...


class NodeTarget:
    """The module's source node as a `pw-record` target; it doesn't come back once gone."""

    def __init__(self, nodes: NodeWatch, name: str, serial: str) -> None:
        self._nodes = nodes
        self._name = name
        self._serial = serial

    @property
    def serial(self) -> str:
        return self._serial

    def current(self) -> str | None:
        return self._serial if self._nodes.node_serial(self._name) == self._serial else None

    async def appeared(self) -> str:
        self._serial = await self._nodes.wait_node(self._name)
        return self._serial


class PipewireAec:
    """One echo-cancel module instance, loaded while its `pw-cli` process runs."""

    def __init__(
        self,
        node_name: str,
        *,
        nodes: NodeWatch,
        command: Command = pw_cli_command,
        wait_s: float = NODE_WAIT_S,
        term_wait_s: float = TERM_WAIT_S,
    ) -> None:
        self.node_name = node_name
        self.on_exit: Callable[[str], None] | None = None
        """Called when `pw-cli` ends by itself after `load`, with a message for the client."""
        self.exit_message: str | None = None
        """That message, once `pw-cli` ended by itself."""
        self._nodes = nodes
        self._command = command
        self._wait_s = wait_s
        self._term_wait_s = term_wait_s
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr = bytearray()
        self._error_seen = asyncio.Event()
        self._watch: asyncio.Task[None] | None = None
        self._diagnosis: asyncio.Task[str | None] | None = None
        self._unloading = False

    def target(self, serial: str) -> AppTarget:
        return NodeTarget(self._nodes, self.node_name, serial)

    async def load(self) -> str:
        """Load the module and wait for its source node; returns the node's serial. Raises
        `AecUnavailable`, with nothing left running, within `wait_s`."""
        deadline = asyncio.get_running_loop().time() + self._wait_s
        # Started now: a failed load names what is missing without waiting past the deadline.
        self._diagnosis = asyncio.create_task(asyncio.to_thread(check_pipewire_aec))
        command = list(self._command(self.node_name))
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            problem = check_tool(command[0]) or f"{command[0]} is not installed"
            raise AecUnavailable(f"echo cancellation is off for this track: {problem}") from None
        except OSError as exc:
            raise AecUnavailable(
                f"echo cancellation is off for this track: {command[0]} could not be started: {exc}"
            ) from None
        self._proc = proc
        stderr = asyncio.create_task(self._read_stderr(proc))
        node = asyncio.create_task(self._nodes.wait_node(self.node_name))
        exited = asyncio.create_task(proc.wait())
        error = asyncio.create_task(self._error_seen.wait())
        try:
            done, _ = await asyncio.wait(
                [node, exited, error], timeout=self._wait_s, return_when=asyncio.FIRST_COMPLETED
            )
        except BaseException:
            await self._stop(node, exited, error, stderr)
            raise
        if node in done:
            for task in (exited, error):
                task.cancel()
            self._watch = asyncio.create_task(self._watch_exit(stderr))
            log.info("echo cancellation: %s loaded, source %s", AEC_MODULE, self.node_name)
            return node.result()
        await self._stop(node, exited, error, stderr)
        if exited in done:
            why = f"{command[0]} exited with code {proc.returncode}"
        elif error in done:
            why = f"{command[0]} could not load {AEC_MODULE}"
        else:
            why = f"its source did not appear within {self._wait_s:g} s"
        raise AecUnavailable(await self._message(why, deadline))

    async def unload(self) -> None:
        """End `pw-cli` (TERM, then KILL after `term_wait_s`), which unloads the module."""
        self._unloading = True
        await self._terminate()
        if self._watch is not None:
            await asyncio.wait([self._watch])

    async def _stop(self, *tasks: asyncio.Task[Any]) -> None:
        """Cancel the waits, kill `pw-cli`; the last task (its stderr) ends with it."""
        for task in tasks[:-1]:
            task.cancel()
        assert self._proc is not None
        if self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.kill()
        await asyncio.wait(tasks)
        await self._proc.wait()

    async def _terminate(self) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            async with asyncio.timeout(self._term_wait_s):
                await proc.wait()
        except TimeoutError:
            log.warning("echo cancellation: pw-cli ignored SIGTERM; killing it")
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()

    async def _watch_exit(self, stderr: asyncio.Task[None]) -> None:
        assert self._proc is not None
        code = await self._proc.wait()
        await asyncio.wait([stderr])
        if self._unloading:
            return
        self._diagnosis = None
        self.exit_message = await self._message(f"pw-cli exited with code {code}")
        if not self._unloading and self.on_exit is not None:
            self.on_exit(self.exit_message)

    async def _read_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        while line := await proc.stderr.readline():
            self._stderr += line
            del self._stderr[:-STDERR_TAIL]
            if line.startswith(b"Error"):
                self._error_seen.set()

    async def _message(self, why: str, deadline: float | None = None) -> str:
        """With `deadline` (loop time), the install check is named only if it is done by then."""
        tail = self._stderr.decode(errors="replace").strip()
        message = f"echo cancellation is off for this track: {why}"
        if tail:
            message += f" ({tail})"
        diagnosis = self._diagnosis or asyncio.create_task(asyncio.to_thread(check_pipewire_aec))
        if deadline is not None:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            await asyncio.wait([diagnosis], timeout=remaining)
        else:
            await asyncio.wait([diagnosis])
        if diagnosis.done() and (problem := diagnosis.result()) is not None:
            message += f". {problem}"
        return message


class EchoCancelledMic(CaptureSource):
    """A system session's mic track: the echo-cancel module's source, recorded by a capture from
    `cleaned`; the plain `mic` when the module can't load, and from the moment its `pw-cli` ends.
    Frames of either go through this source's own hub."""

    def __init__(
        self,
        mic: CaptureSource,
        aec: PipewireAec,
        *,
        cleaned: Callable[[AppTarget], SystemCapture],
    ) -> None:
        super().__init__(sample_rate=mic.sample_rate, buffer_ms=1000, keep_open=False)
        self.on_lost: Callable[[str], None] | None = None
        self.on_restored: Callable[[], None] | None = None
        self.lost_reason: str | None = None
        self.on_unavailable: Callable[[str], None] | None = None
        """Called once when the track takes the plain mic, with why."""
        self.unavailable_reason: str | None = None
        self._mic = mic
        self._aec = aec
        self._make_cleaned = cleaned
        self._cleaned: SystemCapture | None = None
        self._mic_held = False
        self._feed: Attachment | None = None
        self._forward: asyncio.Task[None] | None = None
        self._switch: asyncio.Task[None] | None = None
        self._last_end: float | None = None
        self._closed = False

    @property
    def is_open(self) -> bool:
        return not self._closed

    @property
    def cleaned(self) -> bool:
        """Frames come from the echo-cancel source."""
        return self._cleaned is not None

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    async def hold(self) -> None:
        """Load the module and record its source, else hold the plain mic. Raises what the mic's
        `hold` raises."""
        try:
            serial = await self._aec.load()
        except AecUnavailable as exc:
            await self._hold_plain(exc.message)
            return
        try:
            cleaned = self._make_cleaned(self._aec.target(serial))
            cleaned.on_lost, cleaned.on_restored = self._lost, self._restored
            await cleaned.hold()
        except (ToolMissing, OSError) as exc:
            await self._aec.unload()
            problem = check_tool(str(exc)) if isinstance(exc, ToolMissing) else None
            problem = problem or f"its source could not be recorded: {exc}"
            await self._hold_plain(f"echo cancellation is off for this track: {problem}")
            return
        except BaseException:
            await self._aec.unload()
            raise
        if (message := self._aec.exit_message) is not None:
            await cleaned.release()
            await self._aec.unload()
            await self._hold_plain(message)
            return
        # No await from the check to here: an exit either was seen above or reaches `on_exit`.
        self._cleaned = cleaned
        self._aec.on_exit = self._aec_exited
        self._feed_from(cleaned.attach(None))

    async def _hold_plain(self, message: str) -> None:
        await self._mic.hold()
        self._mic_held = True
        self._report(message)

    async def release(self) -> None:
        self._closed = True
        if self._switch is not None:
            self._switch.cancel()
            await asyncio.wait([self._switch])
        await self._stop_feed()
        await self._release_inner()
        await self._aec.unload()

    def attach(self, since: float | None = None) -> Attachment:
        attachment = self._hub.attach(since)
        if self._mic_held and self._feed is None:
            self._feed_from(self._mic_feed(since))
        return attachment

    def listen(self) -> Attachment:
        return self.attach(time.monotonic())

    def _mic_feed(self, since: float | None) -> Attachment:
        try:
            return self._mic.attach(since)
        except CaptureUnavailable:
            return self._mic.listen()

    def _feed_from(self, feed: Attachment) -> None:
        self._feed = feed
        self._forward = asyncio.create_task(self._pass_on(feed.queue))

    async def _pass_on(self, queue: asyncio.Queue[Frame]) -> None:
        while True:
            frame = await queue.get()
            self._last_end = frame.end
            self._hub.push(frame.samples, frame.end)

    async def _stop_feed(self) -> None:
        if self._feed is not None:
            self._feed.detach()
            self._feed = None
        if self._forward is not None:
            self._forward.cancel()
            await asyncio.wait([self._forward])
            self._forward = None

    async def _release_inner(self) -> None:
        cleaned, self._cleaned = self._cleaned, None
        if cleaned is not None:
            await cleaned.release()
        if self._mic_held:
            self._mic_held = False
            await self._mic.release()

    def _aec_exited(self, message: str) -> None:
        if not self._closed:
            self._switch = asyncio.create_task(self._fall_back(message))

    async def _fall_back(self, message: str) -> None:
        await self._stop_feed()
        await self._release_inner()
        failures = 0
        while True:
            try:
                await self._mic.hold()
                break
            except Exception as exc:
                self._report(message)
                if self.lost_reason is None:
                    self._lost(f"the mic did not open: {exc}")
                await asyncio.sleep(MIC_RETRY_S[min(failures, len(MIC_RETRY_S) - 1)])
                failures += 1
        self._mic_held = True
        # From where the cleaned audio stopped, out of the mic's ring.
        self._feed_from(self._mic_feed(self._last_end))
        self._report(message)
        if self.lost_reason is not None:
            self._restored()

    def _report(self, message: str) -> None:
        if self.unavailable_reason is not None:
            return
        self.unavailable_reason = message
        log.warning("%s", message)
        if self.on_unavailable is not None:
            self.on_unavailable(message)

    def _lost(self, reason: str) -> None:
        self.lost_reason = reason
        if self.on_lost is not None:
            self.on_lost(reason)

    def _restored(self) -> None:
        self.lost_reason = None
        if self.on_restored is not None:
            self.on_restored()
