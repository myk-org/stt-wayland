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
    "Transcribe the spoken words in this audio file.\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Return ONLY the transcribed words exactly as spoken\n"
    "- NO explanations whatsoever\n"
    "- NO prefixes like 'The user said' or 'Here is the transcription' or 'The speaker said'\n"
    "- NO meta-commentary or descriptions\n"
    "- NO markdown formatting or code blocks\n"
    "- NO quotation marks around the output\n"
    "- Output the raw text directly with no wrapping\n"
    "- If audio is empty, silent, or unclear, return exactly: [NO_SPEECH]\n\n"
    "CRITICAL: Your entire response must be ONLY the spoken words. Nothing else."
)
TRANSCRIPTION_PROMPT_WITH_REFINEMENT: Final[str] = (
    "Transcribe the spoken words in this audio file. "
    "After transcription, refine the text by correcting any typos, grammatical errors, "
    "and improving clarity while preserving the original meaning.\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Return ONLY the refined transcribed text\n"
    "- NO explanations whatsoever\n"
    "- NO prefixes like 'The corrected version is' or 'Here is the refined text'\n"
    "- NO meta-commentary about what you corrected or changed\n"
    "- NO markdown formatting or code blocks\n"
    "- NO quotation marks around the output\n"
    "- Output the clean refined text directly with no wrapping\n"
    "- If audio is empty, silent, or unclear, return exactly: [NO_SPEECH]\n\n"
    "CRITICAL: Your entire response must be ONLY the refined transcribed words. Nothing else."
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

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", *, refine: bool = False) -> None:
        """Initialize Gemini transcriber.

        Args:
            api_key: Google API key.
            model: Model name to use.
            refine: Enable AI-based typo and grammar correction.

        """
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._refine = refine
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
        refine_mode = "with refinement" if self._refine else "raw transcription"
        self._logger.info("Transcribing %s with %s (%s)", audio_path, self._model, refine_mode)

        try:
            # Upload audio file
            with audio_path.open("rb") as f:
                audio_data = f.read()

            # Create audio part
            audio_part = types.Part.from_bytes(data=audio_data, mime_type="audio/wav")

            # Select prompt based on refine setting
            prompt = TRANSCRIPTION_PROMPT_WITH_REFINEMENT if self._refine else TRANSCRIPTION_PROMPT

            # Generate content with audio
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_text(text=prompt),
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
