"""System dependency checks: PortAudio (the Linux `sounddevice` wheels don't bundle it), the
external tools some sessions run, and PipeWire's echo-cancel module, each with its install command
for this system."""

import shutil
import subprocess
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
_PACKAGES["pw-dump"] = _PACKAGES["pw-cli"] = _PACKAGES["pw-record"]
_PACKAGES["audiotee"] = {}
_PACKAGES["pipewire-aec"] = {
    "apt": "libpipewire-0.3-modules libspa-0.2-modules",
    "dnf": "pipewire-libs",
    "pacman": "pipewire-audio",
    "zypper": "pipewire-modules-0_3 pipewire-spa-plugins-0_2",
}
_FALLBACK = {
    "portaudio": "PortAudio (libportaudio.so.2) from your distribution's packages",
    "ffmpeg": "ffmpeg from your distribution's packages",
    "pw-record": "the PipeWire command-line tools (pw-record) from your distribution's packages",
    "pw-dump": "the PipeWire command-line tools (pw-dump) from your distribution's packages",
    "pw-cli": "the PipeWire command-line tools (pw-cli) from your distribution's packages",
    "pipewire-aec": "PipeWire's echo-cancel module and its WebRTC plugin from your distribution's "
    "packages",
    "audiotee": "build it from https://github.com/makeusabrew/audiotee (macOS 14.2 or later; "
    "`swift build -c release`) and put .build/release/audiotee on PATH",
}
AEC_MODULE_FILE = "libpipewire-module-echo-cancel.so"
AEC_PLUGIN_FILE = "libspa-aec-webrtc.so"
_MODULE_DIR_GLOBS = ("/usr/lib*/pipewire-0.3", "/usr/lib/*/pipewire-0.3")
_PLUGIN_DIR_GLOBS = ("/usr/lib*/spa-0.2", "/usr/lib/*/spa-0.2")


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


def _pkg_config_variable(variable: str, package: str) -> str | None:
    try:
        done = subprocess.run(
            ["pkg-config", f"--variable={variable}", package],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def check_pipewire_aec(
    pkg_config: Callable[[str, str], str | None] = _pkg_config_variable,
    root: Path = Path("/"),
    platform: str | None = None,
    os_release: Path = Path("/etc/os-release"),
) -> str | None:
    """None when PipeWire's echo-cancel module and its WebRTC plugin are installed; else a
    message that names the install command. `root` prefixes the fallback directories."""
    module_dirs = _dirs(pkg_config("moduledir", "libpipewire-0.3"), _MODULE_DIR_GLOBS, root)
    plugin_dirs = _dirs(pkg_config("plugindir", "libspa-0.2"), _PLUGIN_DIR_GLOBS, root)
    module = any((d / AEC_MODULE_FILE).is_file() for d in module_dirs)
    plugin = any((d / sub / AEC_PLUGIN_FILE).is_file() for d in plugin_dirs for sub in ("aec", ""))
    missing = [
        name for name, found in ((AEC_MODULE_FILE, module), (AEC_PLUGIN_FILE, plugin)) if not found
    ]
    if not missing:
        return None
    hint = install_hint(platform, os_release, package="pipewire-aec")
    return f"{' and '.join(missing)} not found. Install them: {hint}"


def _dirs(configured: str | None, globs: tuple[str, ...], root: Path) -> list[Path]:
    if configured is not None:
        return [Path(configured)]
    return [path for pattern in globs for path in sorted(root.glob(pattern.lstrip("/")))]


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
        if (manager := _MANAGERS.get(distro)) and manager in packages:
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
