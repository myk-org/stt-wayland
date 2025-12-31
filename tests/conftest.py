"""Pytest configuration and fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: MonkeyPatch) -> None:
    """Reset environment variables before each test.

    This prevents environment variable pollution between tests.
    """
    # Clear any existing STT-related environment variables
    for var in ["GEMINI_API_KEY", "STT_MODEL", "XDG_RUNTIME_DIR"]:
        monkeypatch.delenv(var, raising=False)
    return None
