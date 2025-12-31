"""Google Gemini transcription service."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NoReturn

from google import genai
from google.genai import types

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Final

ERR_EMPTY_RESPONSE: Final[str] = "Empty transcription response"
ERR_NO_SPEECH: Final[str] = "No speech detected in audio"
TRANSCRIPTION_PROMPT: Final[str] = (
    "Transcribe the spoken words in this audio file. "
    "Output ONLY the exact words spoken, nothing else. "
    "If the audio is silent or contains no speech, respond with exactly: [NO_SPEECH]"
)


def _raise_empty_response_error() -> NoReturn:
    """Raise error for empty transcription response."""
    msg = ERR_EMPTY_RESPONSE
    raise RuntimeError(msg)


def _raise_no_speech_error() -> NoReturn:
    """Raise error when no speech is detected in audio."""
    msg = ERR_NO_SPEECH
    raise RuntimeError(msg)


class GeminiTranscriber:
    """Transcribes audio using Google Gemini API."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        """Initialize Gemini transcriber.

        Args:
            api_key: Google API key.
            model: Model name to use.

        """
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._logger = logging.getLogger(__name__)

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            Transcribed text.

        Raises:
            RuntimeError: If transcription fails.

        """
        self._logger.info("Transcribing %s with %s", audio_path, self._model)

        try:
            # Upload audio file
            with audio_path.open("rb") as f:
                audio_data = f.read()

            # Create audio part
            audio_part = types.Part.from_bytes(data=audio_data, mime_type="audio/wav")

            # Generate content with audio
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_text(text=TRANSCRIPTION_PROMPT),
                    audio_part,
                ],
            )

            if response.text:
                text: str = str(response.text).strip()
                # Check for known failure patterns
                if (
                    text == "[NO_SPEECH]"
                    or "cannot transcribe" in text.lower()
                    or "appears to be silent" in text.lower()
                ):
                    _raise_no_speech_error()
                self._logger.info("Transcribed: %s...", text[:100])
                return text

            # Empty response - raise error
            _raise_empty_response_error()

        except Exception as e:
            self._logger.exception("Transcription failed")
            msg = f"Transcription failed: {e}"
            raise RuntimeError(msg) from e
