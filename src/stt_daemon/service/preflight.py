"""PortAudio check: the Linux `sounddevice` wheels don't bundle it, so the system must have it."""

import sys
from collections.abc import Callable
from pathlib import Path

_INSTALL = {
    "debian": "sudo apt install libportaudio2",
    "ubuntu": "sudo apt install libportaudio2",
    "fedora": "sudo dnf install portaudio",
    "rhel": "sudo dnf install portaudio",
    "centos": "sudo dnf install portaudio",
    "arch": "sudo pacman -S portaudio",
    "suse": "sudo zypper install libportaudio2",
    "opensuse": "sudo zypper install libportaudio2",
}


def _load_sounddevice() -> None:
    import sounddevice  # noqa: F401  # pyright: ignore[reportUnusedImport]


def check_portaudio(
    load: Callable[[], None] = _load_sounddevice,
    platform: str | None = None,
    os_release: Path = Path("/etc/os-release"),
) -> str | None:
    """None when PortAudio loads; else a message that names the install command."""
    try:
        load()
    except OSError as exc:  # sounddevice raises it when it can't load PortAudio
        return f"PortAudio did not load ({exc}). Install it: {install_hint(platform, os_release)}"
    return None


def install_hint(platform: str | None = None, os_release: Path = Path("/etc/os-release")) -> str:
    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        return "brew install portaudio"
    for distro in _distro_ids(os_release):
        if command := _INSTALL.get(distro):
            return command
    return "PortAudio (libportaudio.so.2) from your distribution's packages"


def _distro_ids(os_release: Path) -> list[str]:
    """`ID`, then each `ID_LIKE` entry; opensuse-tumbleweed etc. also give their first part."""
    try:
        text = os_release.read_text()
    except OSError:
        return []
    fields: dict[str, str] = {}
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep:
            fields[name.strip()] = value.strip().strip("\"'")
    ids = [fields.get("ID", ""), *fields.get("ID_LIKE", "").split()]
    return [part for distro in ids if distro for part in (distro, distro.split("-")[0])]
