"""A session's start: hold every track's capture, attach on one origin, open the links; `undo`
gives back what was taken when the start fails."""

from collections.abc import Callable, Coroutine
from dataclasses import replace
from typing import Any

from ..audio.capture import CaptureSource, CaptureUnavailable
from ..audio.frames import Attachment
from ..backends.base import OpenOptions
from .tracks import Track


class TrackStart:
    def __init__(self, tracks: list[Track]) -> None:
        """`tracks[0]` gives the origin; the rest are mic tracks."""
        self._tracks = tracks
        self._held: list[Track] = []
        self._first_attachment: Attachment | None = None
        self._begun: list[Track] = []

    async def open(
        self,
        options: OpenOptions,
        make_room: Callable[[], Coroutine[Any, Any, bool]] | None,
        *,
        since: float | None,
        attachment: Attachment | None,
    ) -> float:
        """Returns the origin (monotonic). `attachment`: made by a caller that took a hold on
        the first capture, which is now the session's. Raises `CaptureUnavailable`,
        `ToolMissing` or `OpenFailed`."""
        first, rest = self._tracks[0], self._tracks[1:]
        # The mic track is held before the system capture starts: the mic is open first.
        for track in rest:
            await _hold(track.capture)
            self._held.append(track)
        if attachment is None:
            await first.capture.hold()
            try:
                attachment = first.capture.attach(since)
            except BaseException:
                await first.capture.release()
                raise
        self._first_attachment = attachment
        self._held.insert(0, first)
        origin = attachment.started_at
        for track in self._tracks:
            await track.take_playback(origin)
        first.begin(attachment, origin, options)
        self._begun.append(first)
        for track in rest:
            track.begin(_attach_from(track.capture, origin), origin, replace(options, mic=True))
            self._begun.append(track)
        for track in self._tracks:
            await track.open(make_room)
        return origin

    async def undo(self) -> None:
        if self._first_attachment is not None and self._tracks[0] not in self._begun:
            self._first_attachment.detach()
        for track in self._begun:
            assert track.link is not None
            await track.link.close()
            track.detach()
        for track in self._held:
            await track.capture.release()
        for track in self._tracks:
            await track.release_playback()


async def _hold(capture: CaptureSource) -> None:
    """Hold a shared capture. Raises `CaptureUnavailable` when it can't open."""
    try:
        await capture.hold()
    except CaptureUnavailable:
        raise
    except Exception as exc:
        raise CaptureUnavailable(f"the mic did not open: {exc}") from exc


def _attach_from(capture: CaptureSource, origin: float) -> Attachment:
    """Attach a held capture from `origin`; while it recovers, live frames once it is back."""
    try:
        return capture.attach(origin)
    except CaptureUnavailable:
        return capture.listen()
