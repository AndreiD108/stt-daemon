from pathlib import Path

import pytest

_LIVE_DIR = Path(__file__).parent / "live"


@pytest.fixture(autouse=True)
def _own_data_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keeps a model installed on this machine out of tests that don't ask for it."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path_factory.mktemp("data")))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.is_relative_to(_LIVE_DIR):
            item.add_marker(pytest.mark.live)
