"""Test configuration. Developed with AI assistance (Claude, Anthropic)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def pytest_collection_modifyitems(config, items):
    """Tests marked 'live' talk to NASA's archive; run them only when asked (SPX_LIVE=1)."""
    import pytest

    if os.environ.get("SPX_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live archive test; set SPX_LIVE=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
