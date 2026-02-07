"""Text typing using wtype."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Final

ERR_NO_WTYPE: Final[str] = "wtype not found. Install wtype package."
ERR_NO_WL_COPY: Final[str] = "wl-copy not found. Install wl-clipboard package."
ERR_WTYPE_TIMEOUT: Final[str] = "wtype timed out"
ERR_WL_COPY_TIMEOUT: Final[str] = "wl-copy timed out"
ERR_TEXT_TOO_LONG: Final[str] = "Text exceeds maximum length of 100KB"

# Maximum text length (100KB)
MAX_TEXT_LENGTH: Final[int] = 100 * 1024


def type_text(text: str) -> None:
    """Type text using wtype.

    Args:
        text: Text to type.

    Raises:
        RuntimeError: If wtype is not available or fails.
        ValueError: If text is invalid or too long.

    """
    logger = logging.getLogger(__name__)

    if not shutil.which("wtype"):
        msg = ERR_NO_WTYPE
        raise RuntimeError(msg)

    # Input validation
    # Strip null bytes (security)
    text = text.replace("\x00", "")

    # Check text length (prevent DoS)
    if len(text.encode("utf-8")) > MAX_TEXT_LENGTH:
        msg = ERR_TEXT_TOO_LONG
        raise ValueError(msg)

    logger.info("Typing text: %s...", text[:50])

    try:
        # Use stdin mode for proper Unicode handling
        subprocess.run(
            ["wtype", "-"],  # noqa: S607
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
            timeout=10,
        )

        logger.info("Text typed successfully")

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode()
        logger.exception("wtype failed: %s", stderr)
        msg = f"wtype failed: {stderr}"
        raise RuntimeError(msg) from e
    except subprocess.TimeoutExpired as e:
        logger.exception("wtype timed out")
        msg = ERR_WTYPE_TIMEOUT
        raise RuntimeError(msg) from e


def paste_text(text: str) -> None:
    """Paste text using wl-copy and wtype Ctrl+V.

    Copies text to clipboard and simulates paste keystroke.
    Use this instead of type_text when text contains newlines,
    as wtype interprets newlines as Enter key presses.

    Args:
        text: Text to paste.

    Raises:
        RuntimeError: If wl-copy or wtype is not available or fails.
        ValueError: If text is invalid or too long.

    """
    logger = logging.getLogger(__name__)

    if not shutil.which("wl-copy"):
        msg = ERR_NO_WL_COPY
        raise RuntimeError(msg)

    if not shutil.which("wtype"):
        msg = ERR_NO_WTYPE
        raise RuntimeError(msg)

    # Input validation
    # Strip null bytes (security)
    text = text.replace("\x00", "")

    # Check text length (prevent DoS)
    if len(text.encode("utf-8")) > MAX_TEXT_LENGTH:
        msg = ERR_TEXT_TOO_LONG
        raise ValueError(msg)

    logger.info("Pasting text via clipboard: %s...", text[:50])

    try:
        # Use DEVNULL instead of capture_output=True because wl-copy forks
        # a background child that keeps pipes open, causing subprocess.run to hang.
        # Copy to clipboard
        subprocess.run(
            ["wl-copy", "--"],  # noqa: S607
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=5,
        )
    except subprocess.CalledProcessError as e:
        logger.exception("wl-copy failed with exit code %d", e.returncode)
        msg = f"wl-copy failed with exit code {e.returncode}"
        raise RuntimeError(msg) from e
    except subprocess.TimeoutExpired as e:
        logger.exception("wl-copy timed out")
        msg = ERR_WL_COPY_TIMEOUT
        raise RuntimeError(msg) from e

    # Brief delay to let the compositor register the clipboard content
    time.sleep(0.05)

    try:
        # Simulate Ctrl+V to paste
        subprocess.run(
            ["wtype", "-M", "ctrl", "v", "-m", "ctrl"],  # noqa: S607
            capture_output=True,
            check=True,
            timeout=10,
        )

        logger.info("Text pasted successfully")

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode()
        logger.exception("wtype paste failed: %s", stderr)
        msg = f"wtype paste failed: {stderr}"
        raise RuntimeError(msg) from e
    except subprocess.TimeoutExpired as e:
        logger.exception("wtype timed out")
        msg = ERR_WTYPE_TIMEOUT
        raise RuntimeError(msg) from e
