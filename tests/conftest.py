from pathlib import Path

import pytest

_LIVE_DIR = Path(__file__).parent / "live"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.is_relative_to(_LIVE_DIR):
            item.add_marker(pytest.mark.live)
