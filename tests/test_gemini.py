"""Tests for Gemini transcription module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stt_wayland.transcription.gemini import (
    CUSTOM_INSTRUCTION_PROMPT,
    ERR_EMPTY_RESPONSE,
    GeminiTranscriber,
)


class TestParseInstruction:
    """Test _parse_instruction method."""

    @pytest.fixture
    def transcriber(self) -> GeminiTranscriber:
        """Create a GeminiTranscriber instance with instruction keyword enabled."""
        with patch("stt_wayland.transcription.gemini.genai.Client"):
            return GeminiTranscriber(api_key="test-key", instruction_keyword="boom")

    def test_basic_split(self, transcriber: GeminiTranscriber) -> None:
        """Test basic split with 'boom' keyword."""
        result = transcriber._parse_instruction("Hello world boom refine as poem")

        assert result is not None
        assert result == ("Hello world", "refine as poem")

    def test_case_insensitive_uppercase(self, transcriber: GeminiTranscriber) -> None:
        """Test case insensitive matching with uppercase BOOM."""
        result = transcriber._parse_instruction("Hello BOOM do something")

        assert result is not None
        assert result == ("Hello", "do something")

    def test_case_insensitive_mixed_case(self, transcriber: GeminiTranscriber) -> None:
        """Test case insensitive matching with mixed case Boom."""
        result = transcriber._parse_instruction("Hello Boom do something")

        assert result is not None
        assert result == ("Hello", "do something")

    def test_case_insensitive_alternating_case(self, transcriber: GeminiTranscriber) -> None:
        """Test case insensitive matching with alternating case bOoM."""
        result = transcriber._parse_instruction("Hello bOoM do something")

        assert result is not None
        assert result == ("Hello", "do something")

    def test_no_keyword_returns_none(self, transcriber: GeminiTranscriber) -> None:
        """Test that text without keyword returns None."""
        result = transcriber._parse_instruction("Hello world")

        assert result is None

    def test_boom_at_start_empty_content(self, transcriber: GeminiTranscriber) -> None:
        """Test boom at start results in empty content."""
        result = transcriber._parse_instruction("boom do something")

        assert result is not None
        assert result == ("", "do something")

    def test_boom_at_end_empty_instruction(self, transcriber: GeminiTranscriber) -> None:
        """Test boom at end results in empty instruction."""
        result = transcriber._parse_instruction("Hello world boom")

        assert result is not None
        assert result == ("Hello world", "")

    def test_boom_at_end_with_trailing_spaces(self, transcriber: GeminiTranscriber) -> None:
        """Test boom at end with trailing spaces still results in empty instruction."""
        result = transcriber._parse_instruction("Hello world boom   ")

        assert result is not None
        assert result == ("Hello world", "")

    def test_multiple_booms_uses_first(self, transcriber: GeminiTranscriber) -> None:
        """Test that multiple boom keywords use the first occurrence only."""
        result = transcriber._parse_instruction("Hello boom first boom second")

        assert result is not None
        assert result == ("Hello", "first boom second")

    def test_boom_in_middle_of_word_no_match(self, transcriber: GeminiTranscriber) -> None:
        """Test that boom inside another word does NOT match."""
        # Word boundary regex ensures "kaboom" does not match keyword "boom"
        result = transcriber._parse_instruction("kaboom something")

        assert result is None

    def test_empty_text(self, transcriber: GeminiTranscriber) -> None:
        """Test that empty text returns None."""
        result = transcriber._parse_instruction("")

        assert result is None

    def test_only_boom(self, transcriber: GeminiTranscriber) -> None:
        """Test text containing only 'boom'."""
        result = transcriber._parse_instruction("boom")

        assert result is not None
        assert result == ("", "")

    def test_boom_with_extra_whitespace(self, transcriber: GeminiTranscriber) -> None:
        """Test boom with extra whitespace is handled correctly."""
        result = transcriber._parse_instruction("  Hello world   boom   refine this  ")

        assert result is not None
        assert result == ("Hello world", "refine this")

    def test_boom_with_newlines(self, transcriber: GeminiTranscriber) -> None:
        """Test boom with newlines in text."""
        result = transcriber._parse_instruction("Hello\nworld boom refine\nthis")

        assert result is not None
        assert result == ("Hello\nworld", "refine\nthis")

    def test_boom_with_special_characters(self, transcriber: GeminiTranscriber) -> None:
        """Test boom with special characters around it."""
        result = transcriber._parse_instruction("Hello! boom refine?")

        assert result is not None
        assert result == ("Hello!", "refine?")

    def test_disabled_when_no_keyword(self) -> None:
        """Test that _parse_instruction returns None when no keyword is configured."""
        with patch("stt_wayland.transcription.gemini.genai.Client"):
            transcriber = GeminiTranscriber(api_key="test-key")

        result = transcriber._parse_instruction("Hello world boom refine as poem")

        assert result is None

    def test_custom_keyword(self) -> None:
        """Test that _parse_instruction works with a custom keyword."""
        with patch("stt_wayland.transcription.gemini.genai.Client"):
            transcriber = GeminiTranscriber(api_key="test-key", instruction_keyword="magic")

        result = transcriber._parse_instruction("Hello world magic refine as poem")

        assert result is not None
        assert result == ("Hello world", "refine as poem")

    def test_custom_keyword_case_insensitive(self) -> None:
        """Test that custom keyword matching is case insensitive."""
        with patch("stt_wayland.transcription.gemini.genai.Client"):
            transcriber = GeminiTranscriber(api_key="test-key", instruction_keyword="magic")

        result = transcriber._parse_instruction("Hello world MAGIC refine as poem")

        assert result is not None
        assert result == ("Hello world", "refine as poem")


class TestApplyInstruction:
    """Test _apply_instruction method."""

    @pytest.fixture
    def transcriber(self) -> GeminiTranscriber:
        """Create a GeminiTranscriber instance with mocked client."""
        with patch("stt_wayland.transcription.gemini.genai.Client") as _mock_client_class:
            mock_client = MagicMock()
            _mock_client_class.return_value = mock_client
            transcriber = GeminiTranscriber(api_key="test-key", instruction_keyword="boom")
            return transcriber

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_apply_instruction_success(self, mock_client_class: MagicMock) -> None:
        """Test successful instruction application."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Processed content"
        mock_client.models.generate_content.return_value = mock_response

        transcriber = GeminiTranscriber(api_key="test-key")
        result = transcriber._apply_instruction("Hello world", "refine as poem")

        assert result == "Processed content"
        mock_client.models.generate_content.assert_called_once()

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_apply_instruction_uses_correct_prompt_format(self, mock_client_class: MagicMock) -> None:
        """Test that apply_instruction uses the correct prompt format."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Result"
        mock_client.models.generate_content.return_value = mock_response

        transcriber = GeminiTranscriber(api_key="test-key")
        transcriber._apply_instruction("My content", "my instruction")

        # Verify the call was made with correct model and contents
        call_args = mock_client.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-2.5-flash"
        assert "contents" in call_args.kwargs

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_apply_instruction_strips_whitespace(self, mock_client_class: MagicMock) -> None:
        """Test that result is stripped of leading/trailing whitespace."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "  Processed content  \n"
        mock_client.models.generate_content.return_value = mock_response

        transcriber = GeminiTranscriber(api_key="test-key")
        result = transcriber._apply_instruction("content", "instruction")

        assert result == "Processed content"

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_apply_instruction_empty_response_raises_error(self, mock_client_class: MagicMock) -> None:
        """Test that empty API response raises RuntimeError."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response

        transcriber = GeminiTranscriber(api_key="test-key")

        with pytest.raises(RuntimeError, match=ERR_EMPTY_RESPONSE):
            transcriber._apply_instruction("content", "instruction")

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_apply_instruction_empty_string_response_raises_error(self, mock_client_class: MagicMock) -> None:
        """Test that empty string API response raises RuntimeError."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.models.generate_content.return_value = mock_response

        transcriber = GeminiTranscriber(api_key="test-key")

        with pytest.raises(RuntimeError, match=ERR_EMPTY_RESPONSE):
            transcriber._apply_instruction("content", "instruction")

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_apply_instruction_uses_configured_model(self, mock_client_class: MagicMock) -> None:
        """Test that apply_instruction uses the configured model."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Result"
        mock_client.models.generate_content.return_value = mock_response

        transcriber = GeminiTranscriber(api_key="test-key", model="custom-model")
        transcriber._apply_instruction("content", "instruction")

        call_args = mock_client.models.generate_content.call_args
        assert call_args.kwargs["model"] == "custom-model"


class TestGeminiTranscriberInit:
    """Test GeminiTranscriber initialization."""

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_init_default_values(self, mock_client_class: MagicMock) -> None:
        """Test initialization with default values."""
        transcriber = GeminiTranscriber(api_key="test-key")

        assert transcriber._model == "gemini-2.5-flash"
        assert transcriber._refine is False
        assert transcriber._instruction_keyword is None
        mock_client_class.assert_called_once_with(api_key="test-key")

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_init_custom_model(self, _mock_client_class: MagicMock) -> None:
        """Test initialization with custom model."""
        transcriber = GeminiTranscriber(api_key="test-key", model="gemini-pro")

        assert transcriber._model == "gemini-pro"

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_init_refine_enabled(self, _mock_client_class: MagicMock) -> None:
        """Test initialization with refine enabled."""
        transcriber = GeminiTranscriber(api_key="test-key", refine=True)

        assert transcriber._refine is True

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_init_instruction_keyword(self, _mock_client_class: MagicMock) -> None:
        """Test initialization with instruction keyword."""
        transcriber = GeminiTranscriber(api_key="test-key", instruction_keyword="magic")

        assert transcriber._instruction_keyword == "magic"


class TestCustomInstructionPrompt:
    """Test CUSTOM_INSTRUCTION_PROMPT constant."""

    def test_prompt_contains_placeholders(self) -> None:
        """Test that prompt contains required placeholders."""
        assert "{instruction}" in CUSTOM_INSTRUCTION_PROMPT
        assert "{content}" in CUSTOM_INSTRUCTION_PROMPT

    def test_prompt_format_works(self) -> None:
        """Test that prompt can be formatted correctly."""
        formatted = CUSTOM_INSTRUCTION_PROMPT.format(
            instruction="test instruction",
            content="test content",
        )

        assert "test instruction" in formatted
        assert "test content" in formatted
        assert "{instruction}" not in formatted
        assert "{content}" not in formatted
