"""`stt-daemon backend install parakeet`: download, check, unpack, and add `bpe.vocab`."""

import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, TextIO

from .parakeet import BPE_VOCAB, DECODER, ENCODER, JOINER, MODEL_FILES, TOKENS, model_dir

URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
)
SIZE = 487_170_055
SHA256 = "5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf"
ARCHIVE_DIR = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
CREDIT = "Parakeet TDT 0.6B v3 by NVIDIA, licensed CC-BY-4.0"
_CHUNK = 1 << 20

type Opener = Callable[[str], BinaryIO]


class InstallError(Exception):
    pass


def _open_url(url: str) -> BinaryIO:
    return urllib.request.urlopen(url, timeout=30)


def is_installed(target: Path) -> bool:
    return all((target / name).is_file() for name in MODEL_FILES)


def install_parakeet(
    target: Path | None = None, opener: Opener = _open_url, progress: TextIO = sys.stderr
) -> tuple[Path, bool]:
    """Returns the model's directory and whether this call installed it."""
    target = model_dir() if target is None else target
    if is_installed(target):
        return target, False
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=target.parent) as work:
        archive = Path(work) / "model.tar.bz2"
        _download(opener, archive, progress)
        print("unpacking", file=progress)
        staged = Path(work) / "model"
        try:
            _unpack(archive, staged)
            archive.unlink()
            write_bpe_vocab(staged / TOKENS, staged / BPE_VOCAB)
        except (tarfile.TarError, EOFError, OSError) as exc:
            raise InstallError(f"could not unpack the model: {exc}") from None
        if target.exists():
            shutil.rmtree(target)
        staged.rename(target)
    return target, True


def write_bpe_vocab(tokens: Path, out: Path) -> None:
    """sherpa needs a sentencepiece vocab to turn hotwords into tokens; the export ships none.

    The score is minus the token id, so lower ids win where pieces compete."""
    lines: list[str] = []
    for n, line in enumerate(tokens.read_text(encoding="utf-8").splitlines(), start=1):
        piece, _, token_id = line.rpartition(" ")
        if not piece or not token_id.isdigit():
            raise InstallError(f"{tokens}:{n}: expected '<piece> <id>'")
        lines.append(f"{piece}\t{-int(token_id)}\n")
    out.write_text("".join(lines), encoding="utf-8")


def _download(opener: Opener, dest: Path, progress: TextIO) -> None:
    print(f"downloading {SIZE / 1e6:.0f} MB from {URL}", file=progress)
    digest = hashlib.sha256()
    done, shown = 0, -1
    try:
        with opener(URL) as response, dest.open("wb") as out:
            while chunk := response.read(_CHUNK):
                done += len(chunk)
                if done > SIZE:
                    raise InstallError(f"the download is larger than the expected {SIZE} bytes")
                digest.update(chunk)
                out.write(chunk)
                if (percent := done * 100 // SIZE) // 10 > shown:
                    shown = percent // 10
                    print(f"  {percent}%", file=progress)
    except OSError as exc:
        raise InstallError(f"the download failed: {exc}") from None
    if done != SIZE:
        raise InstallError(f"the download ended at {done} of {SIZE} bytes; try again")
    if digest.hexdigest() != SHA256:
        raise InstallError("the download's SHA-256 doesn't match; try again")


def _unpack(archive: Path, dest: Path) -> None:
    dest.mkdir()
    wanted = {f"{ARCHIVE_DIR}/{name}": name for name in (ENCODER, DECODER, JOINER, TOKENS)}
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar:
            name = wanted.pop(member.name, None)
            if name is None:
                continue
            if not member.isfile():
                raise InstallError(f"{member.name} in the archive is not a file")
            source = tar.extractfile(member)
            assert source is not None
            with source, (dest / name).open("wb") as out:
                shutil.copyfileobj(source, out, _CHUNK)
    if wanted:
        raise InstallError(f"the archive has no {', '.join(sorted(wanted))}")
