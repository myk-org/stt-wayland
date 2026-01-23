"""Tests for Gemini transcription module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stt_wayland.transcription.gemini import (
    ASK_QUERY_PROMPT,
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


class TestParseAskQuery:
    """Test _parse_ask_query method."""

    @pytest.fixture
    def transcriber(self) -> GeminiTranscriber:
        """Create a GeminiTranscriber instance with ask keyword enabled."""
        with patch("stt_wayland.transcription.gemini.genai.Client"):
            return GeminiTranscriber(api_key="test-key", ask_keyword="hey")

    def test_basic_ask_query(self, transcriber: GeminiTranscriber) -> None:
        """Test basic ask query detection."""
        result = transcriber._parse_ask_query("hey what is the capital of France")

        assert result == "what is the capital of France"

    def test_case_insensitive_uppercase(self, transcriber: GeminiTranscriber) -> None:
        """Test case insensitive matching with uppercase HEY."""
        result = transcriber._parse_ask_query("HEY what is the answer")

        assert result == "what is the answer"

    def test_case_insensitive_mixed_case(self, transcriber: GeminiTranscriber) -> None:
        """Test case insensitive matching with mixed case Hey."""
        result = transcriber._parse_ask_query("Hey what is the answer")

        assert result == "what is the answer"

    def test_case_insensitive_alternating_case(self, transcriber: GeminiTranscriber) -> None:
        """Test case insensitive matching with alternating case hEy."""
        result = transcriber._parse_ask_query("hEy what is the answer")

        assert result == "what is the answer"

    def test_keyword_not_at_start_returns_none(self, transcriber: GeminiTranscriber) -> None:
        """Test that keyword not at start returns None."""
        result = transcriber._parse_ask_query("hello hey what is the answer")

        assert result is None

    def test_word_boundary_heyo_not_match(self, transcriber: GeminiTranscriber) -> None:
        """Test that 'heyo' should NOT trigger for keyword 'hey' (word boundary)."""
        result = transcriber._parse_ask_query("heyo what is the answer")

        assert result is None

    def test_word_boundary_they_not_match(self, transcriber: GeminiTranscriber) -> None:
        """Test that 'they' should NOT trigger for keyword 'hey' (word boundary)."""
        result = transcriber._parse_ask_query("they are here")

        assert result is None

    def test_keyword_only_returns_empty_string(self, transcriber: GeminiTranscriber) -> None:
        """Test keyword only returns empty string."""
        result = transcriber._parse_ask_query("hey")

        assert result == ""

    def test_keyword_with_trailing_spaces_only(self, transcriber: GeminiTranscriber) -> None:
        """Test keyword with trailing spaces only returns empty string."""
        result = transcriber._parse_ask_query("hey   ")

        assert result == ""

    def test_empty_text(self, transcriber: GeminiTranscriber) -> None:
        """Test that empty text returns None."""
        result = transcriber._parse_ask_query("")

        assert result is None

    def test_text_with_leading_whitespace(self, transcriber: GeminiTranscriber) -> None:
        """Test text with leading whitespace."""
        result = transcriber._parse_ask_query("  hey what is the answer")

        assert result == "what is the answer"

    def test_disabled_when_no_keyword(self) -> None:
        """Test that _parse_ask_query returns None when no keyword is configured."""
        with patch("stt_wayland.transcription.gemini.genai.Client"):
            transcriber = GeminiTranscriber(api_key="test-key")

        result = transcriber._parse_ask_query("hey what is the answer")

        assert result is None

    def test_custom_keyword(self) -> None:
        """Test that _parse_ask_query works with a custom keyword."""
        with patch("stt_wayland.transcription.gemini.genai.Client"):
            transcriber = GeminiTranscriber(api_key="test-key", ask_keyword="assistant")

        result = transcriber._parse_ask_query("assistant what is the weather")

        assert result == "what is the weather"

    def test_custom_keyword_case_insensitive(self) -> None:
        """Test that custom keyword matching is case insensitive."""
        with patch("stt_wayland.transcription.gemini.genai.Client"):
            transcriber = GeminiTranscriber(api_key="test-key", ask_keyword="assistant")

        result = transcriber._parse_ask_query("ASSISTANT what is the weather")

        assert result == "what is the weather"

    def test_keyword_with_special_chars_in_query(self, transcriber: GeminiTranscriber) -> None:
        """Test keyword with special characters in query."""
        result = transcriber._parse_ask_query("hey what is 2 + 2?")

        assert result == "what is 2 + 2?"


class TestAnswerQuery:
    """Test _answer_query method."""

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_answer_query_success(self, mock_client_class: MagicMock) -> None:
        """Test successful query answering."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "The capital of France is Paris"
        mock_client.models.generate_content.return_value = mock_response

        transcriber = GeminiTranscriber(api_key="test-key", ask_keyword="hey")
        result = transcriber._answer_query("what is the capital of France")

        assert result == "The capital of France is Paris"
        mock_client.models.generate_content.assert_called_once()

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_answer_query_uses_correct_prompt_format(self, mock_client_class: MagicMock) -> None:
        """Test that answer_query uses the correct prompt format."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Result"
        mock_client.models.generate_content.return_value = mock_response

        transcriber = GeminiTranscriber(api_key="test-key", ask_keyword="hey")
        transcriber._answer_query("my query")

        call_args = mock_client.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-2.5-flash"
        assert "contents" in call_args.kwargs

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_answer_query_strips_whitespace(self, mock_client_class: MagicMock) -> None:
        """Test that result is stripped of leading/trailing whitespace."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "  The answer  \n"
        mock_client.models.generate_content.return_value = mock_response

        transcriber = GeminiTranscriber(api_key="test-key", ask_keyword="hey")
        result = transcriber._answer_query("query")

        assert result == "The answer"

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_answer_query_empty_response_raises_error(self, mock_client_class: MagicMock) -> None:
        """Test that empty API response raises RuntimeError."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response

        transcriber = GeminiTranscriber(api_key="test-key", ask_keyword="hey")

        with pytest.raises(RuntimeError, match=ERR_EMPTY_RESPONSE):
            transcriber._answer_query("query")

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_answer_query_empty_string_response_raises_error(self, mock_client_class: MagicMock) -> None:
        """Test that empty string API response raises RuntimeError."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.models.generate_content.return_value = mock_response

        transcriber = GeminiTranscriber(api_key="test-key", ask_keyword="hey")

        with pytest.raises(RuntimeError, match=ERR_EMPTY_RESPONSE):
            transcriber._answer_query("query")

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_answer_query_uses_configured_model(self, mock_client_class: MagicMock) -> None:
        """Test that answer_query uses the configured model."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Result"
        mock_client.models.generate_content.return_value = mock_response

        transcriber = GeminiTranscriber(api_key="test-key", model="custom-model", ask_keyword="hey")
        transcriber._answer_query("query")

        call_args = mock_client.models.generate_content.call_args
        assert call_args.kwargs["model"] == "custom-model"


class TestAskQueryPrompt:
    """Test ASK_QUERY_PROMPT constant."""

    def test_prompt_contains_placeholder(self) -> None:
        """Test that prompt contains required placeholder."""
        assert "{query}" in ASK_QUERY_PROMPT

    def test_prompt_format_works(self) -> None:
        """Test that prompt can be formatted correctly."""
        formatted = ASK_QUERY_PROMPT.format(query="test query")

        assert "test query" in formatted
        assert "{query}" not in formatted


class TestAskKeywordPrecedence:
    """Test that ask-keyword takes precedence over instruction-keyword."""

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_ask_keyword_takes_precedence(self, mock_client_class: MagicMock) -> None:
        """Test that ask-keyword is checked before instruction-keyword."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Both keywords are configured
        transcriber = GeminiTranscriber(
            api_key="test-key",
            instruction_keyword="boom",
            ask_keyword="hey",
        )

        # Text starts with ask keyword - should trigger ask mode, not instruction mode
        text = "hey what is boom doing here"

        # Test _parse_ask_query first
        ask_result = transcriber._parse_ask_query(text)
        assert ask_result == "what is boom doing here"

        # Test _parse_instruction - it also finds "boom" in the text
        # but in transcribe(), ask-keyword is checked first and takes precedence
        instruction_result = transcriber._parse_instruction(text)
        assert instruction_result is not None  # "boom" is present, so this parses too


class TestGeminiTranscriberAskKeywordInit:
    """Test GeminiTranscriber initialization with ask_keyword."""

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_init_ask_keyword(self, _mock_client_class: MagicMock) -> None:
        """Test initialization with ask keyword."""
        transcriber = GeminiTranscriber(api_key="test-key", ask_keyword="hey")

        assert transcriber._ask_keyword == "hey"

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_init_default_ask_keyword_none(self, _mock_client_class: MagicMock) -> None:
        """Test that ask_keyword defaults to None."""
        transcriber = GeminiTranscriber(api_key="test-key")

        assert transcriber._ask_keyword is None

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_init_both_keywords(self, _mock_client_class: MagicMock) -> None:
        """Test initialization with both instruction and ask keywords."""
        transcriber = GeminiTranscriber(
            api_key="test-key",
            instruction_keyword="boom",
            ask_keyword="hey",
        )

        assert transcriber._instruction_keyword == "boom"
        assert transcriber._ask_keyword == "hey"


class TestTranscribeWithAskKeyword:
    """Integration tests for transcribe() method with ask keyword."""

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_transcribe_ask_keyword_only_returns_empty(self, mock_client_class: MagicMock, tmp_path: Path) -> None:
        """Test that saying only the keyword (no query) returns empty string."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock transcription to return just the keyword
        mock_response = MagicMock()
        mock_response.text = "hey"
        mock_client.models.generate_content.return_value = mock_response

        transcriber = GeminiTranscriber(api_key="test-key", ask_keyword="hey")

        # Create a dummy audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        result = transcriber.transcribe(audio_file)

        assert result == ""

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_transcribe_ask_keyword_with_query_returns_ai_answer(
        self, mock_client_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test full flow with a query returns AI answer."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # First call: transcription returns "hey what is Python"
        # Second call: AI answer returns the response
        transcription_response = MagicMock()
        transcription_response.text = "hey what is Python"

        ai_answer_response = MagicMock()
        ai_answer_response.text = "Python is a programming language"

        mock_client.models.generate_content.side_effect = [
            transcription_response,
            ai_answer_response,
        ]

        transcriber = GeminiTranscriber(api_key="test-key", ask_keyword="hey")

        # Create a dummy audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        result = transcriber.transcribe(audio_file)

        assert result == "Python is a programming language"
        # Verify two API calls were made: transcription + AI query
        assert mock_client.models.generate_content.call_count == 2

    @patch("stt_wayland.transcription.gemini.genai.Client")
    def test_transcribe_ask_keyword_precedence_in_full_flow(self, mock_client_class: MagicMock, tmp_path: Path) -> None:
        """Integration test verifying ask-keyword takes precedence over instruction-keyword."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock transcription containing both keywords: "hey boom do something"
        # The ask keyword "hey" should take precedence over instruction keyword "boom"
        transcription_response = MagicMock()
        transcription_response.text = "hey boom do something"

        ai_answer_response = MagicMock()
        ai_answer_response.text = "AI response to boom do something"

        mock_client.models.generate_content.side_effect = [
            transcription_response,
            ai_answer_response,
        ]

        # Create transcriber with BOTH keywords
        transcriber = GeminiTranscriber(
            api_key="test-key",
            instruction_keyword="boom",
            ask_keyword="hey",
        )

        # Create a dummy audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        result = transcriber.transcribe(audio_file)

        # Verify ask mode is triggered (AI answer returned), not instruction mode
        assert result == "AI response to boom do something"
        # Verify two API calls: transcription + AI query (not instruction apply)
        assert mock_client.models.generate_content.call_count == 2
