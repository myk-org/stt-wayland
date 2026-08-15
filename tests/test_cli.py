"""Tests for CLI and daemon wiring."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

import stt_wayland.transcription as transcription_mod
from stt_wayland.__main__ import main
from stt_wayland.daemon import STTDaemon, run
from stt_wayland.transcription import GeminiTranscriber


class TestTranscriptionModuleExports:
    """Test that transcription package exports GeminiTranscriber only."""

    def test_gemini_transcriber_importable_from_package(self) -> None:
        """Test that GeminiTranscriber can be imported from stt_wayland.transcription."""
        assert GeminiTranscriber is not None

    def test_all_includes_gemini_transcriber(self) -> None:
        """Test that __all__ includes GeminiTranscriber."""
        assert "GeminiTranscriber" in transcription_mod.__all__

    def test_all_does_not_include_gemini_live_transcriber(self) -> None:
        """Test that GeminiLiveTranscriber is not exported."""
        assert "GeminiLiveTranscriber" not in transcription_mod.__all__


class TestMainArguments:
    """Test CLI argument parsing."""

    @patch("stt_wayland.__main__.run")
    def test_default_flags(self, mock_run: MagicMock) -> None:
        """Test default flags when none are provided."""
        with patch("sys.argv", ["stt-daemon"]):
            main()

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["refine"] is False
        assert kwargs["format_output"] is False
        assert kwargs["instruction_keyword"] is None
        assert kwargs["ask_keyword"] is None
        assert "live" not in kwargs

    @patch("stt_wayland.__main__.run")
    def test_live_flag_rejected(self, mock_run: MagicMock) -> None:
        """Test that --live is not a valid flag."""
        with patch("sys.argv", ["stt-daemon", "--live"]), pytest.raises(SystemExit):
            main()
        mock_run.assert_not_called()

    @patch("stt_wayland.__main__.run")
    def test_main_no_lang_flag(self, mock_run: MagicMock) -> None:
        """Test that --lang flag is not passed to run()."""
        with patch("sys.argv", ["stt-daemon"]):
            main()

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert "lang" not in kwargs


class TestSTTDaemonTranscriber:
    """Test that STTDaemon uses GeminiTranscriber."""

    @patch("stt_wayland.daemon.AudioRecorder")
    @patch("stt_wayland.daemon.GeminiTranscriber")
    def test_uses_gemini_transcriber(self, mock_gemini: MagicMock, _mock_recorder: MagicMock) -> None:
        """Test that STTDaemon constructs GeminiTranscriber."""
        config = MagicMock()
        config.api_key = "test-key"
        config.model = "gemini-2.5-flash"

        STTDaemon(config)

        mock_gemini.assert_called_once()
        call_kwargs = mock_gemini.call_args.kwargs
        assert call_kwargs.get("api_key") == "test-key"
        assert call_kwargs.get("model") == "gemini-2.5-flash"

    @patch("stt_wayland.daemon.AudioRecorder")
    @patch("stt_wayland.daemon.GeminiTranscriber")
    def test_passes_all_options(self, mock_gemini: MagicMock, _mock_recorder: MagicMock) -> None:
        """Test that STTDaemon passes refine, format_output, instruction_keyword, ask_keyword."""
        config = MagicMock()
        config.api_key = "test-key"
        config.model = "gemini-2.5-flash"

        STTDaemon(
            config,
            refine=True,
            format_output=True,
            instruction_keyword="boom",
            ask_keyword="hey",
        )

        mock_gemini.assert_called_once()
        call_kwargs = mock_gemini.call_args.kwargs
        assert call_kwargs.get("refine") is True
        assert call_kwargs.get("format_output") is True
        assert call_kwargs.get("instruction_keyword") == "boom"
        assert call_kwargs.get("ask_keyword") == "hey"

    @patch("stt_wayland.daemon.AudioRecorder")
    @patch("stt_wayland.daemon.GeminiTranscriber")
    def test_does_not_pass_lang(self, mock_gemini: MagicMock, _mock_recorder: MagicMock) -> None:
        """Test that STTDaemon does not pass lang to GeminiTranscriber."""
        config = MagicMock()
        config.api_key = "test-key"
        config.model = "gemini-2.5-flash"

        STTDaemon(config)

        mock_gemini.assert_called_once()
        call_kwargs = mock_gemini.call_args.kwargs
        assert "lang" not in call_kwargs


class TestDaemonRunParameter:
    """Test that run() and STTDaemon do not accept a live parameter."""

    def test_run_has_no_live_parameter(self) -> None:
        """Test that run() has no live parameter."""
        assert "live" not in inspect.signature(run).parameters

    def test_daemon_init_has_no_live_parameter(self) -> None:
        """Test that STTDaemon.__init__ has no live parameter."""
        assert "live" not in inspect.signature(STTDaemon.__init__).parameters
