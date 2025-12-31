"""Desktop notification module using notify-send."""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Final

_logger = logging.getLogger(__name__)

# Notification configuration
NOTIFY_SEND: Final[str] = "notify-send"
APP_NAME: Final[str] = "STT Wayland"
ICON_RECORDING: Final[str] = "audio-input-microphone"
ICON_PROCESSING: Final[str] = "system-run"
ICON_COMPLETE: Final[str] = "emblem-default"
ICON_ERROR: Final[str] = "dialog-error"
URGENCY_NORMAL: Final[str] = "normal"
URGENCY_CRITICAL: Final[str] = "critical"


def _send_notification(
    summary: str,
    body: str = "",
    icon: str = "",
    urgency: str = URGENCY_NORMAL,
) -> None:
    """Send a desktop notification using notify-send.

    Args:
        summary: Notification title.
        body: Notification body text.
        icon: Icon name from freedesktop icon theme.
        urgency: Notification urgency level (low, normal, critical).

    Note:
        Non-blocking operation. Errors are logged but don't raise exceptions.

    """
    cmd = [NOTIFY_SEND, "--app-name", APP_NAME]

    if icon:
        cmd.extend(["--icon", icon])

    cmd.extend(["--urgency", urgency])
    cmd.extend([summary, body])

    try:
        subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        _logger.warning("notify-send timeout: %s", summary)
    except FileNotFoundError:
        _logger.warning("notify-send not found, notifications disabled")
    except Exception:
        _logger.exception("Failed to send notification: %s", summary)


def notify_recording_started() -> None:
    """Show 'Recording...' notification."""
    _send_notification(
        summary="Recording",
        body="Speech recording in progress",
        icon=ICON_RECORDING,
        urgency=URGENCY_NORMAL,
    )


def notify_recording_stopped() -> None:
    """Show 'Processing...' notification."""
    _send_notification(
        summary="Processing",
        body="Transcribing audio...",
        icon=ICON_PROCESSING,
        urgency=URGENCY_NORMAL,
    )


def notify_transcription_complete() -> None:
    """Show 'Done' notification."""
    _send_notification(
        summary="Transcription Complete",
        body="Text typed successfully",
        icon=ICON_COMPLETE,
        urgency=URGENCY_NORMAL,
    )


def notify_error(message: str) -> None:
    """Show error notification.

    Args:
        message: Error message to display.

    """
    _send_notification(
        summary="STT Error",
        body=message,
        icon=ICON_ERROR,
        urgency=URGENCY_CRITICAL,
    )
