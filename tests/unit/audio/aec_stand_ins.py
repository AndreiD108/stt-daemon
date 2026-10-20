"""Stand-ins for the echo-cancel tests: a node table, a pushed mic, the `pw-cli` and `pw-record`
stand-ins' command lines."""

import asyncio
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from stt_daemon.audio.aec_pipewire import pw_cli_command
from stt_daemon.audio.capture import CaptureSource, CaptureUnavailable
from stt_daemon.audio.frames import Attachment
from stt_daemon.audio.pw_capture import PwRecordCapture
from stt_daemon.audio.system_capture import AppTarget
from unit.paths import FAKE_PW_CLI, FAKE_PW_RECORD

NAME = "stt-daemon-aec-s1"
MIC_LEVEL = 0.25


class FakeNodes:
    def __init__(self) -> None:
        self.serials: dict[str, str] = {}

    def node_serial(self, name: str) -> str | None:
        return self.serials.get(name)

    async def wait_node(self, name: str) -> str:
        while name not in self.serials:
            await asyncio.sleep(0.01)
        return self.serials[name]

    def appear_after(self, delay_s: float, serial: str = "777") -> None:
        async def later() -> None:
            await asyncio.sleep(delay_s)
            self.serials[NAME] = serial

        asyncio.get_running_loop().create_task(later())


class PushedMic(CaptureSource):
    """Frames at `MIC_LEVEL` every 20 ms while running; counts holds."""

    def __init__(self) -> None:
        super().__init__(sample_rate=16000, buffer_ms=5000, keep_open=True)
        self.holds = 0
        self._task: asyncio.Task[None] | None = None

    @property
    def is_open(self) -> bool:
        return True

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    async def hold(self) -> None:
        self.holds += 1

    async def release(self) -> None:
        self.holds -= 1

    def run(self) -> None:
        async def loop() -> None:
            while True:
                self._hub.push(np.full(320, MIC_LEVEL, dtype=np.float32), time.monotonic())
                await asyncio.sleep(0.02)

        self._task = asyncio.get_running_loop().create_task(loop())

    def stop_running(self) -> None:
        assert self._task is not None
        self._task.cancel()


def pw_cli(tmp_path: Path, *options: str) -> Callable[[str], list[str]]:
    log = tmp_path / "pw-cli.log"

    def command(name: str) -> list[str]:
        real = pw_cli_command(name)
        return [sys.executable, str(FAKE_PW_CLI), "--log", str(log), *options, "--", *real[1:]]

    return command


def pw_cli_log(tmp_path: Path) -> list[str]:
    path = tmp_path / "pw-cli.log"
    return path.read_text().splitlines() if path.exists() else []


async def until(check: Callable[[], bool], timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.01)


def cleaned_capture(tmp_path: Path, targets: list[str]) -> Callable[[AppTarget], PwRecordCapture]:
    """A `pw-record` stand-in writing silence; records each target serial."""

    def make(target: AppTarget) -> PwRecordCapture:
        targets.append(target.serial)
        return PwRecordCapture(
            lambda serial: [sys.executable, str(FAKE_PW_RECORD), "--target", str(serial)],
            target=target,
        )

    return make


async def levels(attachment: Attachment, count: int) -> list[float]:
    out: list[float] = []
    async with asyncio.timeout(5):
        while len(out) < count:
            frame = await attachment.queue.get()
            out.append(float(np.max(np.abs(frame.samples))))
    return out


class FlakyMic(PushedMic):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures
        self.tries = 0

    async def hold(self) -> None:
        self.tries += 1
        if self.tries <= self.failures:
            raise CaptureUnavailable("the mic is busy")
        await super().hold()
