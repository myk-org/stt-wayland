"""Text output module."""

from .clipboard import copy_to_clipboard
from .notify import (
    notify_error,
    notify_recording_started,
    notify_recording_stopped,
    notify_transcription_complete,
)
from .wtype import type_text

__all__ = [
    "copy_to_clipboard",
    "type_text",
    "notify_error",
    "notify_recording_started",
    "notify_recording_stopped",
    "notify_transcription_complete",
]
