"""System dependency checks: PortAudio (the Linux `sounddevice` wheels don't bundle it) and the
external tools some sessions run, each with its install command for this system."""

import shutil
import sys
from collections.abc import Callable
from pathlib import Path

_MANAGERS = {
    "debian": "apt",
    "ubuntu": "apt",
    "fedora": "dnf",
    "rhel": "dnf",
    "centos": "dnf",
    "arch": "pacman",
    "suse": "zypper",
    "opensuse": "zypper",
}
_COMMANDS = {
    "apt": "sudo apt install",
    "dnf": "sudo dnf install",
    "pacman": "sudo pacman -S",
    "zypper": "sudo zypper install",
    "brew": "brew install",
}
_PACKAGES = {
    "portaudio": {
        "apt": "libportaudio2",
        "dnf": "portaudio",
        "pacman": "portaudio",
        "zypper": "libportaudio2",
        "brew": "portaudio",
    },
    "ffmpeg": {
        "apt": "ffmpeg",
        "dnf": "ffmpeg-free",
        "pacman": "ffmpeg",
        "zypper": "ffmpeg",
        "brew": "ffmpeg",
    },
    "pw-record": {
        "apt": "pipewire-bin",
        "dnf": "pipewire-utils",
        "pacman": "pipewire",
        "zypper": "pipewire-tools",
    },
}
_FALLBACK = {
    "portaudio": "PortAudio (libportaudio.so.2) from your distribution's packages",
    "ffmpeg": "ffmpeg from your distribution's packages",
    "pw-record": "the PipeWire command-line tools (pw-record) from your distribution's packages",
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


def check_tool(
    tool: str,
    which: Callable[[str], str | None] = shutil.which,
    platform: str | None = None,
    os_release: Path = Path("/etc/os-release"),
) -> str | None:
    """None when `tool` is on PATH; else a message that names the install command."""
    if which(tool) is not None:
        return None
    hint = install_hint(platform, os_release, package=tool)
    return f"{tool} is not installed or not on PATH. Install it: {hint}"


def install_hint(
    platform: str | None = None,
    os_release: Path = Path("/etc/os-release"),
    *,
    package: str = "portaudio",
) -> str:
    platform = sys.platform if platform is None else platform
    packages = _PACKAGES.get(package)
    if packages is None:
        return f"{package} from your distribution's packages"
    if platform == "darwin":
        if "brew" not in packages:
            return _FALLBACK[package]
        return f"{_COMMANDS['brew']} {packages['brew']}"
    for distro in _distro_ids(os_release):
        if manager := _MANAGERS.get(distro):
            return f"{_COMMANDS[manager]} {packages[manager]}"
    return _FALLBACK[package]


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
