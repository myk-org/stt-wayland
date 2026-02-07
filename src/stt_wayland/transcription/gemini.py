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
TRANSCRIPTION_PROMPT_WITH_FORMAT: Final[str] = (
    "Transcribe the spoken words in this audio file. "
    "After transcription, refine the text by correcting any typos, grammatical errors, "
    "and improving clarity while preserving the original meaning.\n\n"
    "FORMATTING RULES:\n"
    "- If the speech contains multiple distinct points, items, or topics, "
    "format them as a numbered list (1. 2. 3.) or a bulleted list using dashes (-)\n"
    "- Use line breaks to separate logically distinct points or paragraphs\n"
    "- Use numbered lists (1. 2. 3.) for sequential or ordered items\n"
    "- Use dashes (-) for unordered items\n"
    "- Keep simple sentences or short phrases as plain flowing text, do NOT over-format\n"
    "- Only apply structure when the spoken content naturally contains it\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Return ONLY the refined transcribed text\n"
    "- NO explanations whatsoever\n"
    "- NO prefixes like 'The corrected version is' or 'Here is the refined text'\n"
    "- NO meta-commentary about what you corrected or changed\n"
    "- NO markdown formatting (no bold, italic, headers, code blocks, or horizontal rules)\n"
    "- Plain-text lists using dashes (-) or numbers (1. 2. 3.) as described above are allowed\n"
    "- NO quotation marks around the output\n"
    "- Output the clean refined text directly\n"
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
ASK_QUERY_PROMPT: Final[str] = (
    "Answer this question or respond to this request concisely: {query}\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Return ONLY the answer\n"
    "- Be concise and direct\n"
    "- NO explanations about what you're doing\n"
    "- NO prefixes like 'The answer is' or 'Here is the response'\n"
    "- NO markdown formatting or code blocks\n"
    "CRITICAL: Your entire response must be ONLY the answer. Nothing else."
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
        format_output: bool = False,
        instruction_keyword: str | None = None,
        ask_keyword: str | None = None,
    ) -> None:
        """Initialize Gemini transcriber.

        Args:
            api_key: Google API key.
            model: Model name to use.
            refine: Enable AI-based typo and grammar correction.
            format_output: Enable plain-text formatting of refined output.
            instruction_keyword: Keyword to separate content from AI instructions.
            ask_keyword: Keyword at start of speech to trigger AI query mode.

        """
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._refine = refine
        self._format_output = format_output

        if self._format_output and not self._refine:
            msg = "format_output requires refine to be enabled"
            raise ValueError(msg)

        self._instruction_keyword = instruction_keyword
        self._ask_keyword = ask_keyword
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

    def _parse_ask_query(self, text: str) -> str | None:
        """Parse text for ask keyword at the start and extract the query.

        Args:
            text: The transcribed text to parse.

        Returns:
            The query portion if ask keyword is found at start, None otherwise.

        """
        if self._ask_keyword is None:
            return None

        # Case-insensitive search for the keyword at the START of text
        stripped_text = text.strip()
        pattern = r"^" + re.escape(self._ask_keyword.lower()) + r"\b"
        match = re.match(pattern, stripped_text.lower())

        if match is None:
            return None

        # Extract query after the keyword
        query = stripped_text[len(self._ask_keyword) :].strip()
        return query

    def _answer_query(self, query: str) -> str:
        """Send a query to the AI and return the answer.

        Args:
            query: The query to send to the AI.

        Returns:
            The AI's answer.

        Raises:
            RuntimeError: If the API call fails.

        """
        prompt = ASK_QUERY_PROMPT.format(query=query)

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
        if self._format_output:
            refine_mode = "with refinement and formatting"
        elif self._refine:
            refine_mode = "with refinement"
        else:
            refine_mode = "raw transcription"
        self._logger.info("Transcribing %s with %s (%s)", audio_path, self._model, refine_mode)

        try:
            # Upload audio file
            with audio_path.open("rb") as f:
                audio_data = f.read()

            # Create audio part
            audio_part = types.Part.from_bytes(data=audio_data, mime_type="audio/wav")

            # Select prompt based on refine setting
            if self._format_output:
                prompt = TRANSCRIPTION_PROMPT_WITH_FORMAT
            elif self._refine:
                prompt = TRANSCRIPTION_PROMPT_WITH_REFINEMENT
            else:
                prompt = TRANSCRIPTION_PROMPT

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

                # Check for ask keyword at START (takes precedence over instruction keyword)
                ask_query = self._parse_ask_query(text)
                if ask_query is not None:
                    if not ask_query:
                        # Keyword only, no query - return empty string
                        return ""
                    self._logger.info("Detected ask query: %s", ask_query[:50])
                    return self._answer_query(ask_query)

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
