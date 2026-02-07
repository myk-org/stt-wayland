"""Tests for wtype text output module."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from stt_wayland.output.wtype import (
    ERR_NO_WL_COPY,
    ERR_NO_WTYPE,
    ERR_TEXT_TOO_LONG,
    ERR_WL_COPY_TIMEOUT,
    ERR_WTYPE_TIMEOUT,
    MAX_TEXT_LENGTH,
    paste_text,
    type_text,
)


class TestTypeText:
    """Test type_text function."""

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_success(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test successful text typing."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        type_text("Hello, World!")

        mock_which.assert_called_once_with("wtype")
        mock_run.assert_called_once_with(
            ["wtype", "-"],
            input=b"Hello, World!",
            capture_output=True,
            check=True,
            timeout=10,
        )

    @patch("stt_wayland.output.wtype.shutil.which")
    def test_type_text_no_wtype(self, mock_which: MagicMock) -> None:
        """Test that missing wtype raises RuntimeError."""
        mock_which.return_value = None

        with pytest.raises(RuntimeError, match=ERR_NO_WTYPE):
            type_text("test")

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_strips_null_bytes(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test that null bytes are stripped from input."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        type_text("Hello\x00World\x00!")

        # Null bytes should be removed
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["input"] == b"HelloWorld!"

    @patch("stt_wayland.output.wtype.shutil.which")
    def test_type_text_too_long(self, mock_which: MagicMock) -> None:
        """Test that text exceeding MAX_TEXT_LENGTH raises ValueError."""
        mock_which.return_value = "/usr/bin/wtype"

        # Create text that exceeds 100KB
        long_text = "x" * (MAX_TEXT_LENGTH + 1)

        with pytest.raises(ValueError, match=ERR_TEXT_TOO_LONG):
            type_text(long_text)

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_exactly_max_length(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test that text at exactly MAX_TEXT_LENGTH is accepted."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        # Create text at exactly max length
        max_text = "x" * MAX_TEXT_LENGTH

        type_text(max_text)

        mock_run.assert_called_once()

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_unicode(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test that Unicode text is handled correctly."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        unicode_text = "Hello 世界! 🌍 مرحبا"

        type_text(unicode_text)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["input"] == unicode_text.encode("utf-8")

    @patch("stt_wayland.output.wtype.shutil.which")
    def test_type_text_multibyte_length_check(self, mock_which: MagicMock) -> None:
        """Test that length check uses UTF-8 byte count, not character count."""
        mock_which.return_value = "/usr/bin/wtype"

        # Create text with multibyte characters that exceeds byte limit
        # Each emoji is ~4 bytes
        emoji_text = "🌍" * (MAX_TEXT_LENGTH // 4 + 1)

        with pytest.raises(ValueError, match=ERR_TEXT_TOO_LONG):
            type_text(emoji_text)

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_empty_string(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test typing empty string."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        type_text("")

        mock_run.assert_called_once_with(
            ["wtype", "-"],
            input=b"",
            capture_output=True,
            check=True,
            timeout=10,
        )

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_whitespace_only(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test typing whitespace-only text."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        type_text("   \n\t\r  ")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["input"] == b"   \n\t\r  "

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_subprocess_error(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test that subprocess errors are handled."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["wtype", "-"],
            stderr=b"wtype: Wayland connection failed",
        )

        with pytest.raises(RuntimeError, match="wtype failed"):
            type_text("test")

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_timeout(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test that timeout is handled."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["wtype", "-"], timeout=10)

        with pytest.raises(RuntimeError, match=ERR_WTYPE_TIMEOUT):
            type_text("test")

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_special_characters(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test typing text with special characters."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        special_text = "!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/~`"

        type_text(special_text)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["input"] == special_text.encode("utf-8")

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_newlines(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test typing text with newlines."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        multiline_text = "Line 1\nLine 2\nLine 3"

        type_text(multiline_text)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["input"] == b"Line 1\nLine 2\nLine 3"

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_multiple_null_bytes(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test that multiple null bytes are all stripped."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        type_text("\x00\x00Hello\x00\x00\x00World\x00")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["input"] == b"HelloWorld"

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_only_null_bytes(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test typing text that contains only null bytes results in empty string."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        type_text("\x00\x00\x00")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["input"] == b""

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_uses_stdin_mode(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test that wtype is called with stdin mode (-)."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        type_text("test")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["wtype", "-"]

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_timeout_value(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test that subprocess is called with correct timeout."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        type_text("test")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["timeout"] == 10

    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_type_text_capture_output(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Test that subprocess captures output."""
        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        type_text("test")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["check"] is True


class TestMaxTextLength:
    """Test MAX_TEXT_LENGTH constant."""

    def test_max_text_length_value(self) -> None:
        """Test that MAX_TEXT_LENGTH is set to 100KB."""
        assert MAX_TEXT_LENGTH == 100 * 1024
        assert MAX_TEXT_LENGTH == 102400


class TestPasteText:
    """Test paste_text function."""

    @patch("stt_wayland.output.wtype.time.sleep")
    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_paste_text_success(self, mock_run: MagicMock, mock_which: MagicMock, mock_sleep: MagicMock) -> None:
        """Test successful text pasting via clipboard."""
        mock_which.return_value = "/usr/bin/wl-copy"
        mock_run.return_value = MagicMock(returncode=0)

        paste_text("Line 1\nLine 2\nLine 3")

        assert mock_run.call_count == 2
        # First call: wl-copy
        wl_copy_call = mock_run.call_args_list[0]
        assert wl_copy_call[0][0] == ["wl-copy", "--"]
        assert wl_copy_call[1]["input"] == b"Line 1\nLine 2\nLine 3"
        assert wl_copy_call[1]["timeout"] == 5
        # Second call: wtype Ctrl+V
        wtype_call = mock_run.call_args_list[1]
        assert wtype_call[0][0] == ["wtype", "-M", "ctrl", "v", "-m", "ctrl"]
        assert wtype_call[1]["timeout"] == 10
        # Verify sleep was called between the two
        mock_sleep.assert_called_once_with(0.05)

    @patch("stt_wayland.output.wtype.shutil.which")
    def test_paste_text_no_wl_copy(self, mock_which: MagicMock) -> None:
        """Test that missing wl-copy raises RuntimeError."""
        mock_which.return_value = None

        with pytest.raises(RuntimeError, match=ERR_NO_WL_COPY):
            paste_text("test")

    @patch("stt_wayland.output.wtype.shutil.which")
    def test_paste_text_no_wtype(self, mock_which: MagicMock) -> None:
        """Test that missing wtype (with wl-copy present) raises RuntimeError."""
        mock_which.side_effect = lambda cmd: "/usr/bin/wl-copy" if cmd == "wl-copy" else None

        with pytest.raises(RuntimeError, match=ERR_NO_WTYPE):
            paste_text("test")

    @patch("stt_wayland.output.wtype.time.sleep")
    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_paste_text_strips_null_bytes(
        self, mock_run: MagicMock, mock_which: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        """Test that null bytes are stripped before pasting."""
        mock_which.return_value = "/usr/bin/wl-copy"
        mock_run.return_value = MagicMock(returncode=0)

        paste_text("Hello\x00World\x00!")

        wl_copy_call = mock_run.call_args_list[0]
        assert wl_copy_call[1]["input"] == b"HelloWorld!"

    @patch("stt_wayland.output.wtype.shutil.which")
    def test_paste_text_too_long(self, mock_which: MagicMock) -> None:
        """Test that text exceeding MAX_TEXT_LENGTH raises ValueError."""
        mock_which.return_value = "/usr/bin/wl-copy"

        long_text = "x" * (MAX_TEXT_LENGTH + 1)

        with pytest.raises(ValueError, match=ERR_TEXT_TOO_LONG):
            paste_text(long_text)

    @patch("stt_wayland.output.wtype.time.sleep")
    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_paste_text_wl_copy_failure(
        self, mock_run: MagicMock, mock_which: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        """Test that wl-copy subprocess error is handled."""
        mock_which.return_value = "/usr/bin/wl-copy"
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["wl-copy", "--"],
            stderr=b"wl-copy: connection failed",
        )

        with pytest.raises(RuntimeError, match="wl-copy failed"):
            paste_text("test")

    @patch("stt_wayland.output.wtype.time.sleep")
    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_paste_text_wtype_failure(self, mock_run: MagicMock, mock_which: MagicMock, _mock_sleep: MagicMock) -> None:
        """Test that wtype subprocess error is handled."""
        mock_which.return_value = "/usr/bin/wtype"

        def side_effect(*_args: object, **_kwargs: object) -> MagicMock:
            # First call (wl-copy) succeeds, second call (wtype) fails
            if mock_run.call_count <= 1:
                return MagicMock(returncode=0)
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=["wtype", "-M", "ctrl", "v", "-m", "ctrl"],
                stderr=b"wtype: Wayland connection failed",
            )

        mock_run.side_effect = side_effect

        with pytest.raises(RuntimeError, match="wtype paste failed"):
            paste_text("test")

    @patch("stt_wayland.output.wtype.time.sleep")
    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_paste_text_wl_copy_timeout(
        self, mock_run: MagicMock, mock_which: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        """Test that wl-copy timeout uses correct error message."""
        mock_which.return_value = "/usr/bin/wl-copy"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["wl-copy", "--"], timeout=5)

        with pytest.raises(RuntimeError, match=ERR_WL_COPY_TIMEOUT):
            paste_text("test")

    @patch("stt_wayland.output.wtype.time.sleep")
    @patch("stt_wayland.output.wtype.shutil.which")
    @patch("stt_wayland.output.wtype.subprocess.run")
    def test_paste_text_wtype_timeout(self, mock_run: MagicMock, mock_which: MagicMock, _mock_sleep: MagicMock) -> None:
        """Test that wtype timeout uses correct error message."""
        mock_which.return_value = "/usr/bin/wtype"

        def side_effect(*_args: object, **_kwargs: object) -> MagicMock:
            if mock_run.call_count <= 1:
                return MagicMock(returncode=0)
            raise subprocess.TimeoutExpired(cmd=["wtype", "-M", "ctrl", "v", "-m", "ctrl"], timeout=10)

        mock_run.side_effect = side_effect

        with pytest.raises(RuntimeError, match=ERR_WTYPE_TIMEOUT):
            paste_text("test")
