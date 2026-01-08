"""Google Gemini transcription service."""

from __future__ import annotations

import logging
import re
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
CUSTOM_INSTRUCTION_PROMPT: Final[str] = (
    "Apply the following instruction to this text:\n\n"
    "INSTRUCTION: {instruction}\n\n"
    "TEXT: {content}\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Return ONLY the processed text\n"
    "- NO explanations or meta-commentary\n"
    "- NO markdown formatting or code blocks\n"
    "- Apply the instruction exactly as specified\n"
    "CRITICAL: Your entire response must be ONLY the processed text. Nothing else."
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

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        *,
        refine: bool = False,
        instruction_keyword: str | None = None,
    ) -> None:
        """Initialize Gemini transcriber.

        Args:
            api_key: Google API key.
            model: Model name to use.
            refine: Enable AI-based typo and grammar correction.
            instruction_keyword: Keyword to separate content from AI instructions.

        """
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._refine = refine
        self._instruction_keyword = instruction_keyword
        self._logger = logging.getLogger(__name__)

    def _parse_instruction(self, text: str) -> tuple[str, str] | None:
        """Parse text for instruction keyword and split into content and instruction.

        Args:
            text: The transcribed text to parse.

        Returns:
            Tuple of (content, instruction) if keyword found, None otherwise.

        """
        if self._instruction_keyword is None:
            return None

        # Case-insensitive search for the keyword as a standalone word
        lower_text = text.lower()
        pattern = r"\b" + re.escape(self._instruction_keyword.lower()) + r"\b"
        match = re.search(pattern, lower_text)

        if match is None:
            return None

        keyword_pos = match.start()

        # Split at the keyword position
        content = text[:keyword_pos].strip()
        instruction = text[keyword_pos + len(self._instruction_keyword) :].strip()

        return (content, instruction)

    def _apply_instruction(self, content: str, instruction: str) -> str:
        """Apply an instruction to content using the Gemini API.

        Args:
            content: The text content to process.
            instruction: The instruction to apply.

        Returns:
            The processed text.

        Raises:
            RuntimeError: If the API call fails.

        """
        prompt = CUSTOM_INSTRUCTION_PROMPT.format(content=content, instruction=instruction)

        response = self._client.models.generate_content(
            model=self._model,
            contents=[types.Part.from_text(text=prompt)],
        )

        if response.text:
            return str(response.text).strip()

        _raise_empty_response_error()

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

                # Check for inline instruction (only if keyword is configured)
                parsed = self._parse_instruction(text)
                if parsed:
                    content, instruction = parsed
                    # Handle edge cases
                    if not content:
                        # "boom" at start - return instruction as content
                        return instruction if instruction else text
                    if not instruction:
                        # "boom" at end - return content as-is
                        return content
                    self._logger.info("Detected instruction: %s", instruction[:50])
                    return self._apply_instruction(content, instruction)

                return text

            # Empty response - raise error
            _raise_empty_response_error()

        except Exception as e:
            self._logger.exception("Transcription failed")
            msg = f"Transcription failed: {e}"
            raise RuntimeError(msg) from e
