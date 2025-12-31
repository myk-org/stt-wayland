"""Text typing using wtype."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Final

ERR_NO_WTYPE: Final[str] = "wtype not found. Install wtype package."
ERR_WTYPE_TIMEOUT: Final[str] = "wtype timed out"
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
