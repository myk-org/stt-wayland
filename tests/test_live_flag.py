"""Tests for --live CLI flag wiring across __main__, daemon, and transcription __init__."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import stt_wayland.transcription as transcription_mod
from stt_wayland.__main__ import main
from stt_wayland.daemon import STTDaemon, run
from stt_wayland.transcription import GeminiLiveTranscriber


class TestTranscriptionModuleExports:
    """Test that transcription __init__ exports GeminiLiveTranscriber."""

    def test_gemini_live_transcriber_importable_from_package(self) -> None:
        """Test that GeminiLiveTranscriber can be imported from stt_wayland.transcription."""
        assert GeminiLiveTranscriber is not None

    def test_all_includes_gemini_live_transcriber(self) -> None:
        """Test that __all__ includes GeminiLiveTranscriber."""
        assert "GeminiLiveTranscriber" in transcription_mod.__all__

    def test_all_includes_gemini_transcriber(self) -> None:
        """Test that __all__ still includes GeminiTranscriber."""
        assert "GeminiTranscriber" in transcription_mod.__all__


class TestMainLiveArgument:
    """Test that __main__ parses --live flag and passes it to run()."""

    @patch("stt_wayland.__main__.run")
    def test_live_flag_default_false(self, mock_run: MagicMock) -> None:
        """Test that --live defaults to False when not provided."""
        with patch("sys.argv", ["stt-daemon"]):
            main()

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["live"] is False

    @patch("stt_wayland.__main__.run")
    def test_live_flag_set_true(self, mock_run: MagicMock) -> None:
        """Test that --live flag is passed as True to run()."""
        with patch("sys.argv", ["stt-daemon", "--live"]):
            main()

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["live"] is True


class TestDaemonRunValueErrorHandling:
    """Test that daemon.run() catches ValueError from STTDaemon construction."""

    @patch("stt_wayland.daemon.STTDaemon")
    @patch("stt_wayland.daemon.Config.from_env")
    def test_run_exits_cleanly_when_daemon_init_raises_value_error(
        self, mock_config: MagicMock, mock_daemon_class: MagicMock
    ) -> None:
        """Test that run() catches ValueError from STTDaemon() and exits with code 1."""
        mock_config.return_value = MagicMock()
        mock_daemon_class.side_effect = ValueError("Model 'bad-model' is not a Live API model.")

        with pytest.raises(SystemExit) as exc_info:
            run(live=True)

        assert exc_info.value.code == 1


class TestDaemonRunLiveParameter:
    """Test that daemon.run() accepts and passes live parameter."""

    @patch("stt_wayland.daemon.STTDaemon")
    @patch("stt_wayland.daemon.Config.from_env")
    def test_run_passes_live_false_to_daemon(self, _mock_config: MagicMock, mock_daemon_class: MagicMock) -> None:
        """Test that run() passes live=False to STTDaemon."""
        _mock_config.return_value = MagicMock()
        mock_daemon_class.return_value = MagicMock()

        try:
            run(live=False)
        except SystemExit:
            pass

        mock_daemon_class.assert_called_once()
        _, kwargs = mock_daemon_class.call_args
        assert kwargs["live"] is False

    @patch("stt_wayland.daemon.STTDaemon")
    @patch("stt_wayland.daemon.Config.from_env")
    def test_run_passes_live_true_to_daemon(self, _mock_config: MagicMock, mock_daemon_class: MagicMock) -> None:
        """Test that run() passes live=True to STTDaemon."""
        _mock_config.return_value = MagicMock()
        mock_daemon_class.return_value = MagicMock()

        try:
            run(live=True)
        except SystemExit:
            pass

        mock_daemon_class.assert_called_once()
        _, kwargs = mock_daemon_class.call_args
        assert kwargs["live"] is True


class TestSTTDaemonLiveTranscriber:
    """Test that STTDaemon uses GeminiLiveTranscriber when live=True."""

    @patch("stt_wayland.daemon.GeminiLiveTranscriber")
    @patch("stt_wayland.daemon.GeminiTranscriber")
    def test_live_false_uses_gemini_transcriber(self, mock_gemini: MagicMock, _mock_live: MagicMock) -> None:
        """Test that live=False uses GeminiTranscriber."""
        config = MagicMock()
        config.api_key = "test-key"
        config.model = "gemini-2.5-flash"

        STTDaemon(config, live=False)

        mock_gemini.assert_called_once()
        _mock_live.assert_not_called()

    @patch("stt_wayland.daemon.GeminiLiveTranscriber")
    @patch("stt_wayland.daemon.GeminiTranscriber")
    def test_live_true_uses_gemini_live_transcriber(self, _mock_gemini: MagicMock, mock_live: MagicMock) -> None:
        """Test that live=True uses GeminiLiveTranscriber."""
        config = MagicMock()
        config.api_key = "test-key"
        config.model = "gemini-2.5-flash"

        STTDaemon(config, live=True)

        mock_live.assert_called_once()
        _mock_gemini.assert_not_called()

    @patch("stt_wayland.daemon.GeminiLiveTranscriber")
    @patch("stt_wayland.daemon.GeminiTranscriber")
    def test_live_true_always_passes_config_model(self, _mock_gemini: MagicMock, mock_live: MagicMock) -> None:
        """Test that live=True always passes config.model directly to GeminiLiveTranscriber."""
        config = MagicMock()
        config.api_key = "test-key"
        config.model = "gemini-3.1-flash-live-preview"

        STTDaemon(config, live=True)

        mock_live.assert_called_once()
        call_kwargs = mock_live.call_args[1] if mock_live.call_args[1] else {}
        assert call_kwargs.get("model") == "gemini-3.1-flash-live-preview"

    @patch("stt_wayland.daemon.GeminiLiveTranscriber")
    @patch("stt_wayland.daemon.GeminiTranscriber")
    def test_live_true_passes_all_options(self, _mock_gemini: MagicMock, mock_live: MagicMock) -> None:
        """Test that live=True passes refine, format_output, instruction_keyword, ask_keyword."""
        config = MagicMock()
        config.api_key = "test-key"
        config.model = "gemini-3.1-flash-live-preview"

        STTDaemon(
            config,
            refine=True,
            format_output=True,
            instruction_keyword="boom",
            ask_keyword="hey",
            live=True,
        )

        mock_live.assert_called_once()
        call_kwargs = mock_live.call_args[1] if mock_live.call_args[1] else {}
        assert call_kwargs.get("refine") is True
        assert call_kwargs.get("format_output") is True
        assert call_kwargs.get("instruction_keyword") == "boom"
        assert call_kwargs.get("ask_keyword") == "hey"

    @patch("stt_wayland.daemon.GeminiLiveTranscriber")
    @patch("stt_wayland.daemon.GeminiTranscriber")
    def test_live_true_passes_batch_model_from_config(self, _mock_gemini: MagicMock, mock_live: MagicMock) -> None:
        """Test that live=True passes config.model as batch_model for REST API calls."""
        config = MagicMock()
        config.api_key = "test-key"
        config.model = "gemini-2.5-flash"

        STTDaemon(config, live=True)

        mock_live.assert_called_once()
        call_kwargs = mock_live.call_args[1] if mock_live.call_args[1] else {}
        assert call_kwargs.get("batch_model") == "gemini-2.5-flash"

    @patch("stt_wayland.daemon.GeminiLiveTranscriber")
    @patch("stt_wayland.daemon.GeminiTranscriber")
    def test_live_false_passes_all_options(self, mock_gemini: MagicMock, _mock_live: MagicMock) -> None:
        """Test that live=False passes refine, format_output, instruction_keyword, ask_keyword to GeminiTranscriber."""
        config = MagicMock()
        config.api_key = "test-key"
        config.model = "gemini-2.5-flash"

        STTDaemon(
            config,
            refine=True,
            format_output=True,
            instruction_keyword="boom",
            ask_keyword="hey",
            live=False,
        )

        mock_gemini.assert_called_once()
        call_kwargs = mock_gemini.call_args[1] if mock_gemini.call_args[1] else {}
        assert call_kwargs.get("refine") is True
        assert call_kwargs.get("format_output") is True
        assert call_kwargs.get("instruction_keyword") == "boom"
        assert call_kwargs.get("ask_keyword") == "hey"


class TestNoLangParameterPassthrough:
    """Test that lang parameter has been removed from __main__, daemon, and transcriber."""

    @patch("stt_wayland.__main__.run")
    def test_main_no_lang_flag(self, mock_run: MagicMock) -> None:
        """Test that --lang flag is not passed to run()."""
        with patch("sys.argv", ["stt-daemon"]):
            main()

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert "lang" not in kwargs

    @patch("stt_wayland.daemon.GeminiLiveTranscriber")
    @patch("stt_wayland.daemon.GeminiTranscriber")
    def test_daemon_does_not_pass_lang_to_gemini_transcriber(
        self, mock_gemini: MagicMock, _mock_live: MagicMock
    ) -> None:
        """Test that STTDaemon does not pass lang to GeminiTranscriber."""
        config = MagicMock()
        config.api_key = "test-key"
        config.model = "gemini-2.5-flash"

        STTDaemon(config)

        mock_gemini.assert_called_once()
        call_kwargs = mock_gemini.call_args[1] if mock_gemini.call_args[1] else {}
        assert "lang" not in call_kwargs

    @patch("stt_wayland.daemon.GeminiLiveTranscriber")
    @patch("stt_wayland.daemon.GeminiTranscriber")
    def test_daemon_does_not_pass_lang_to_gemini_live_transcriber(
        self, _mock_gemini: MagicMock, mock_live: MagicMock
    ) -> None:
        """Test that STTDaemon does not pass lang to GeminiLiveTranscriber when live=True."""
        config = MagicMock()
        config.api_key = "test-key"
        config.model = "gemini-3.1-flash-live-preview"

        STTDaemon(config, live=True)

        mock_live.assert_called_once()
        call_kwargs = mock_live.call_args[1] if mock_live.call_args[1] else {}
        assert "lang" not in call_kwargs
