"""Clipboard operations using wl-copy."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Final

ERR_NO_WL_COPY: Final[str] = "wl-copy not found. Install wl-clipboard package."
ERR_WL_COPY_TIMEOUT: Final[str] = "wl-copy timed out"


def copy_to_clipboard(text: str) -> None:
    """Copy text to clipboard using wl-copy.

    Args:
        text: Text to copy.

    Raises:
        RuntimeError: If wl-copy is not available or fails.

    """
    logger = logging.getLogger(__name__)

    if not shutil.which("wl-copy"):
        msg = ERR_NO_WL_COPY
        raise RuntimeError(msg)

    logger.info("Copying text to clipboard: %s...", text[:50])

    try:
        subprocess.run(
            ["wl-copy"],  # noqa: S607
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
            timeout=5,
        )

        logger.info("Text copied to clipboard successfully")

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode()
        logger.exception("wl-copy failed: %s", stderr)
        msg = f"wl-copy failed: {stderr}"
        raise RuntimeError(msg) from e
    except subprocess.TimeoutExpired as e:
        logger.exception("wl-copy timed out")
        msg = ERR_WL_COPY_TIMEOUT
        raise RuntimeError(msg) from e
