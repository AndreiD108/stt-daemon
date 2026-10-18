import hashlib
import io
import tarfile
from pathlib import Path
from typing import BinaryIO

import pytest

from stt_daemon.backends import install
from stt_daemon.backends.install import InstallError, install_parakeet, write_bpe_vocab
from stt_daemon.backends.parakeet import MODEL_FILES

TOKENS = "<unk> 0\n▁the 1\nest 2\n, 3\n"


def _archive(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _model_archive(**extra: bytes) -> bytes:
    top = install.ARCHIVE_DIR
    files = {
        f"{top}/encoder.int8.onnx": b"enc",
        f"{top}/decoder.int8.onnx": b"dec",
        f"{top}/joiner.int8.onnx": b"join",
        f"{top}/tokens.txt": TOKENS.encode(),
        f"{top}/test_wavs/en.wav": b"wav",
        **extra,
    }
    return _archive(files)


class _Opener:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.urls: list[str] = []

    def __call__(self, url: str) -> BinaryIO:
        self.urls.append(url)
        return io.BytesIO(self.data)


def _serve(monkeypatch: pytest.MonkeyPatch, data: bytes, *, digest: bytes | None = None) -> _Opener:
    monkeypatch.setattr(install, "SIZE", len(data))
    monkeypatch.setattr(install, "SHA256", hashlib.sha256(digest or data).hexdigest())
    return _Opener(data)


def test_bpe_vocab_scores_are_minus_the_token_id(tmp_path: Path) -> None:
    tokens = tmp_path / "tokens.txt"
    tokens.write_text(TOKENS + "  1000\n")
    write_bpe_vocab(tokens, tmp_path / "bpe.vocab")
    lines = (tmp_path / "bpe.vocab").read_text().splitlines()
    assert lines == ["<unk>\t0", "▁the\t-1", "est\t-2", ",\t-3", " \t-1000"]


def test_bpe_vocab_refuses_a_line_without_an_id(tmp_path: Path) -> None:
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<unk> 0\nbroken\n")
    with pytest.raises(InstallError, match=r"tokens.txt:2"):
        write_bpe_vocab(tokens, tmp_path / "bpe.vocab")


def test_install_unpacks_the_model_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opener = _serve(monkeypatch, _model_archive())
    target = tmp_path / "models" / "parakeet"
    progress = io.StringIO()
    assert install_parakeet(target, opener, progress) == (target, True)
    assert sorted(p.name for p in target.iterdir()) == sorted(MODEL_FILES)
    assert (target / "joiner.int8.onnx").read_bytes() == b"join"
    assert (target / "bpe.vocab").read_text().splitlines()[1] == "▁the\t-1"
    assert "100%" in progress.getvalue()
    assert [p.name for p in target.parent.iterdir()] == ["parakeet"]

    assert install_parakeet(target, opener, io.StringIO()) == (target, False)
    assert len(opener.urls) == 1


def test_install_replaces_a_partial_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opener = _serve(monkeypatch, _model_archive())
    target = tmp_path / "parakeet"
    target.mkdir()
    (target / "encoder.int8.onnx").write_bytes(b"truncated")
    (target / "stray").write_bytes(b"x")
    install_parakeet(target, opener, io.StringIO())
    assert (target / "encoder.int8.onnx").read_bytes() == b"enc"
    assert not (target / "stray").exists()


@pytest.mark.parametrize(
    ("cut", "digest_of", "message"),
    [(0, b"other", "SHA-256"), (10, None, "ended at")],
)
def test_a_bad_download_installs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cut: int, digest_of: bytes | None, message: str
) -> None:
    data = _model_archive()
    _serve(monkeypatch, data, digest=digest_of)
    target = tmp_path / "models" / "parakeet"
    with pytest.raises(InstallError, match=message):
        install_parakeet(target, _Opener(data[: len(data) - cut]), io.StringIO())
    assert list(target.parent.iterdir()) == []


def test_an_archive_without_a_model_file_installs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    top = install.ARCHIVE_DIR
    opener = _serve(monkeypatch, _archive({f"{top}/tokens.txt": TOKENS.encode()}))
    target = tmp_path / "models" / "parakeet"
    with pytest.raises(InstallError, match=r"encoder\.int8\.onnx"):
        install_parakeet(target, opener, io.StringIO())
    assert list(target.parent.iterdir()) == []


def test_a_network_error_names_the_cause(tmp_path: Path) -> None:
    def refuse(_url: str) -> BinaryIO:
        raise OSError("connection refused")

    with pytest.raises(InstallError, match="connection refused"):
        install_parakeet(tmp_path / "parakeet", refuse, io.StringIO())
