"""Tests for Gemini Live API transcription module."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from stt_wayland.transcription.gemini import ERR_EMPTY_RESPONSE, GeminiTranscriber
from stt_wayland.transcription.gemini_live import (
    AUDIO_CHUNK_SIZE,
    LIVE_TRANSCRIPTION_INSTRUCTION,
    LIVE_TRANSCRIPTION_INSTRUCTION_FORMATTED,
    LIVE_TRANSCRIPTION_INSTRUCTION_REFINED,
    GeminiLiveTranscriber,
)

# Patch target for genai.Client -- both gemini.py and gemini_live.py share the same
# google.genai module object, so we only need to patch it once.
_GENAI_CLIENT_PATCH = "stt_wayland.transcription.gemini.genai.Client"


def _make_wav_file(path: Path, pcm_data: bytes = b"\x00" * 100) -> None:
    """Create a minimal valid WAV file with the given PCM data.

    Constructs a 44-byte RIFF/WAVE header followed by the raw PCM data.

    Args:
        path: Path to write the WAV file to.
        pcm_data: Raw PCM audio bytes to include after the header.

    """
    # Build a minimal 44-byte WAV header
    num_channels = 1
    sample_rate = 16000
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_data)
    # RIFF chunk size = 36 + data_size
    riff_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    path.write_bytes(header + pcm_data)


class _AsyncIterator:
    """Async iterator wrapper for testing."""

    def __init__(self, items: list[MagicMock]) -> None:
        self._items = items
        self._index = 0

    def __aiter__(self) -> _AsyncIterator:
        return self

    async def __anext__(self) -> MagicMock:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


def _async_iter(items: list[MagicMock]) -> _AsyncIterator:
    """Create an async iterator from a list of items.

    Args:
        items: List of mock objects to iterate over.

    Returns:
        An async iterator that yields the given items.

    """
    return _AsyncIterator(items)


def _make_client_factory() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Create a genai.Client mock factory that returns separate parent and live clients.

    The factory uses side_effect to distinguish between:
    - Parent client call: genai.Client(api_key=...) -- no http_options
    - Live client call: genai.Client(api_key=..., http_options=...) -- with http_options

    Returns:
        A tuple of (mock_class, mock_parent_client, mock_live_client).

    """
    mock_parent_client = MagicMock(name="parent_client")
    mock_live_client = MagicMock(name="live_client")

    def _factory(**kwargs: object) -> MagicMock:
        if "http_options" in kwargs:
            return mock_live_client
        return mock_parent_client

    mock_class = MagicMock(side_effect=_factory)
    return mock_class, mock_parent_client, mock_live_client


class TestGeminiLiveTranscriberInit:
    """Test GeminiLiveTranscriber initialization."""

    @patch(_GENAI_CLIENT_PATCH)
    def test_init_default_model(self, _mock_class: MagicMock) -> None:
        """Test that default model is gemini-3.1-flash-live-preview."""
        transcriber = GeminiLiveTranscriber(api_key="test-key")

        assert transcriber._model == "gemini-3.1-flash-live-preview"

    @patch(_GENAI_CLIENT_PATCH)
    def test_init_custom_model(self, _mock_class: MagicMock) -> None:
        """Test initialization with custom model."""
        transcriber = GeminiLiveTranscriber(api_key="test-key", model="custom-live-model")

        assert transcriber._model == "custom-live-model"

    @patch(_GENAI_CLIENT_PATCH)
    def test_init_inherits_refine(self, _mock_class: MagicMock) -> None:
        """Test that refine parameter is passed to parent."""
        transcriber = GeminiLiveTranscriber(api_key="test-key", refine=True)

        assert transcriber._refine is True

    @patch(_GENAI_CLIENT_PATCH)
    def test_init_inherits_format_output(self, _mock_class: MagicMock) -> None:
        """Test that format_output parameter is passed to parent."""
        transcriber = GeminiLiveTranscriber(api_key="test-key", refine=True, format_output=True)

        assert transcriber._format_output is True

    @patch(_GENAI_CLIENT_PATCH)
    def test_init_inherits_instruction_keyword(self, _mock_class: MagicMock) -> None:
        """Test that instruction_keyword is passed to parent."""
        transcriber = GeminiLiveTranscriber(api_key="test-key", instruction_keyword="boom")

        assert transcriber._instruction_keyword == "boom"

    @patch(_GENAI_CLIENT_PATCH)
    def test_init_inherits_ask_keyword(self, _mock_class: MagicMock) -> None:
        """Test that ask_keyword is passed to parent."""
        transcriber = GeminiLiveTranscriber(api_key="test-key", ask_keyword="hey")

        assert transcriber._ask_keyword == "hey"

    @patch(_GENAI_CLIENT_PATCH)
    def test_init_creates_two_clients(self, mock_class: MagicMock) -> None:
        """Test that genai.Client is called twice: once for parent, once for live."""
        GeminiLiveTranscriber(api_key="test-key")

        assert mock_class.call_count == 2
        # First call: parent client (no http_options)
        parent_call = mock_class.call_args_list[0]
        assert parent_call == ((), {"api_key": "test-key"})
        # Second call: live client (with http_options for v1beta)
        live_call = mock_class.call_args_list[1]
        assert live_call.kwargs["api_key"] == "test-key"
        assert live_call.kwargs["http_options"].api_version == "v1beta"

    @patch(_GENAI_CLIENT_PATCH)
    def test_init_live_client_uses_v1beta(self, mock_class: MagicMock) -> None:
        """Test that the live client is created with api_version='v1beta'."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        transcriber = GeminiLiveTranscriber(api_key="test-key")

        assert transcriber._live_client is mock_live

    @patch(_GENAI_CLIENT_PATCH)
    def test_is_subclass_of_gemini_transcriber(self, _mock_class: MagicMock) -> None:
        """Test that GeminiLiveTranscriber is a subclass of GeminiTranscriber."""
        transcriber = GeminiLiveTranscriber(api_key="test-key")

        assert isinstance(transcriber, GeminiTranscriber)

    @patch(_GENAI_CLIENT_PATCH)
    def test_format_output_without_refine_raises(self, _mock_class: MagicMock) -> None:
        """Test that format_output without refine raises ValueError (inherited behavior)."""
        with pytest.raises(ValueError, match="format_output requires refine"):
            GeminiLiveTranscriber(api_key="test-key", format_output=True)

    @patch(_GENAI_CLIENT_PATCH)
    def test_init_non_live_model_raises_value_error(self, _mock_class: MagicMock) -> None:
        """Test that a non-live model raises ValueError."""
        with pytest.raises(ValueError, match="not a Live API model"):
            GeminiLiveTranscriber(api_key="test-key", model="gemini-2.5-flash")

    @patch(_GENAI_CLIENT_PATCH)
    def test_init_accepts_model_with_live_in_name(self, _mock_class: MagicMock) -> None:
        """Test that a model with 'live' in the name is accepted."""
        transcriber = GeminiLiveTranscriber(api_key="test-key", model="gemini-3.1-flash-live-preview")
        assert transcriber._model == "gemini-3.1-flash-live-preview"

    @patch(_GENAI_CLIENT_PATCH)
    def test_batch_model_defaults_to_gemini_2_5_flash(self, _mock_class: MagicMock) -> None:
        """Test that _batch_model defaults to gemini-2.5-flash."""
        transcriber = GeminiLiveTranscriber(api_key="test-key")

        assert transcriber._batch_model == "gemini-2.5-flash"

    @patch(_GENAI_CLIENT_PATCH)
    def test_batch_model_custom_value(self, _mock_class: MagicMock) -> None:
        """Test that _batch_model can be set to a custom value."""
        transcriber = GeminiLiveTranscriber(api_key="test-key", batch_model="gemini-2.0-flash")

        assert transcriber._batch_model == "gemini-2.0-flash"


class TestExtractPcmFromWav:
    """Test _extract_pcm_from_wav static method."""

    def test_extracts_pcm_data(self, tmp_path: Path) -> None:
        """Test that PCM data is extracted correctly from a valid WAV file."""
        # PCM data must be frame-aligned (2 bytes per sample for 16-bit mono)
        pcm_data = b"\x01\x02\x03\x04\x05\x06"
        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file, pcm_data)

        result = GeminiLiveTranscriber._extract_pcm_from_wav(wav_file)

        assert result == pcm_data

    def test_invalid_file_raises_value_error(self, tmp_path: Path) -> None:
        """Test that an invalid (non-WAV) file raises ValueError."""
        wav_file = tmp_path / "bad.wav"
        wav_file.write_bytes(b"\x00" * 20)

        with pytest.raises(ValueError, match="Not a valid WAV file"):
            GeminiLiveTranscriber._extract_pcm_from_wav(wav_file)

    def test_corrupt_wav_raises_value_error(self, tmp_path: Path) -> None:
        """Test that a corrupt file with RIFF but invalid WAV structure raises ValueError."""
        wav_file = tmp_path / "corrupt.wav"
        data = bytearray(b"\x00" * 44)
        data[0:4] = b"RIFF"
        data[8:12] = b"XXXX"  # Not WAVE
        wav_file.write_bytes(bytes(data))

        with pytest.raises(ValueError, match="Not a valid WAV file"):
            GeminiLiveTranscriber._extract_pcm_from_wav(wav_file)

    def test_empty_pcm_data(self, tmp_path: Path) -> None:
        """Test extraction from a WAV file with no PCM data after header."""
        wav_file = tmp_path / "empty.wav"
        _make_wav_file(wav_file, pcm_data=b"")

        result = GeminiLiveTranscriber._extract_pcm_from_wav(wav_file)

        assert result == b""

    def test_large_pcm_data(self, tmp_path: Path) -> None:
        """Test extraction with large PCM data (larger than chunk size)."""
        pcm_data = b"\xab" * (AUDIO_CHUNK_SIZE * 3 + 100)
        wav_file = tmp_path / "large.wav"
        _make_wav_file(wav_file, pcm_data)

        result = GeminiLiveTranscriber._extract_pcm_from_wav(wav_file)

        assert result == pcm_data


class TestRawTranscribeLive:
    """Test _raw_transcribe method of GeminiLiveTranscriber."""

    @patch(_GENAI_CLIENT_PATCH)
    def test_raw_transcribe_returns_transcribed_text(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test successful transcription returns joined text parts."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        # Create mock session with MagicMock for receive (returns async iterable, not coroutine)
        mock_session = MagicMock()

        # Mock the receive() to yield messages with input_transcription
        msg1 = MagicMock()
        msg1.server_content = MagicMock()
        msg1.server_content.input_transcription = MagicMock()
        msg1.server_content.input_transcription.text = "Hello "
        msg1.server_content.turn_complete = False

        msg2 = MagicMock()
        msg2.server_content = MagicMock()
        msg2.server_content.input_transcription = MagicMock()
        msg2.server_content.input_transcription.text = "world"
        msg2.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg1, msg2])
        mock_session.send_realtime_input = AsyncMock()

        # Mock the async context manager
        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key")

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)

        result = transcriber._raw_transcribe(wav_file)

        assert result == "Hello world"

    @patch(_GENAI_CLIENT_PATCH)
    def test_raw_transcribe_empty_response_raises_error(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that empty response raises RuntimeError."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()

        # Mock receive() to yield a message with no input_transcription and turn_complete
        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = None
        msg.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key")

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)

        with pytest.raises(RuntimeError, match=ERR_EMPTY_RESPONSE):
            transcriber._raw_transcribe(wav_file)

    @patch(_GENAI_CLIENT_PATCH)
    def test_raw_transcribe_sends_audio_in_chunks(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that audio is sent in AUDIO_CHUNK_SIZE chunks."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()

        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "transcribed"
        msg.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key")

        # Create WAV with PCM data larger than one chunk
        pcm_data = b"\xab" * (AUDIO_CHUNK_SIZE + 100)
        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file, pcm_data)

        transcriber._raw_transcribe(wav_file)

        # Should have sent 2 audio chunks + 1 end-of-stream signal
        send_calls = mock_session.send_realtime_input.call_args_list
        # 2 audio chunks + 1 audio_stream_end
        assert len(send_calls) == 3

        # Verify audio_stream_end was sent last
        last_call = send_calls[-1]
        assert last_call.kwargs.get("audio_stream_end") is True

    @patch(_GENAI_CLIENT_PATCH)
    def test_raw_transcribe_uses_audio_param_not_media(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that send_realtime_input uses 'audio' param, not deprecated 'media'."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()

        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "text"
        msg.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key")

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file, b"\x00" * 100)

        transcriber._raw_transcribe(wav_file)

        # Check all audio chunk calls (exclude the audio_stream_end call)
        audio_calls = [
            call for call in mock_session.send_realtime_input.call_args_list if not call.kwargs.get("audio_stream_end")
        ]
        assert len(audio_calls) > 0

        for call in audio_calls:
            # Must use 'audio' param, NOT deprecated 'media'
            assert "audio" in call.kwargs, f"Expected 'audio' param but got: {call.kwargs}"
            assert "media" not in call.kwargs, f"Deprecated 'media' param should not be used: {call.kwargs}"
            # Verify it's a Blob with correct MIME type
            blob = call.kwargs["audio"]
            assert isinstance(blob, types.Blob)
            assert blob.mime_type == "audio/pcm;rate=16000"

    @patch(_GENAI_CLIENT_PATCH)
    def test_raw_transcribe_sends_audio_stream_end(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that audio_stream_end=True is sent after all chunks."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()

        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "text"
        msg.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key")

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file, b"\x00" * 100)

        transcriber._raw_transcribe(wav_file)

        # Find the call with audio_stream_end
        end_calls = [
            call for call in mock_session.send_realtime_input.call_args_list if call.kwargs.get("audio_stream_end")
        ]
        assert len(end_calls) == 1
        assert end_calls[0].kwargs["audio_stream_end"] is True

    @patch(_GENAI_CLIENT_PATCH)
    def test_raw_transcribe_stops_at_turn_complete(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that receiving stops when turn_complete is True."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()

        msg1 = MagicMock()
        msg1.server_content = MagicMock()
        msg1.server_content.input_transcription = MagicMock()
        msg1.server_content.input_transcription.text = "Hello"
        msg1.server_content.turn_complete = True

        # This message should NOT be consumed (turn already complete)
        msg2 = MagicMock()
        msg2.server_content = MagicMock()
        msg2.server_content.input_transcription = MagicMock()
        msg2.server_content.input_transcription.text = " should not appear"
        msg2.server_content.turn_complete = False

        mock_session.receive.return_value = _async_iter([msg1, msg2])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key")

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)

        result = transcriber._raw_transcribe(wav_file)

        assert result == "Hello"
        assert "should not appear" not in result

    @patch(_GENAI_CLIENT_PATCH)
    def test_raw_transcribe_strips_whitespace(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that result is stripped of leading/trailing whitespace."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()

        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "  Hello world  \n"
        msg.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key")

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)

        result = transcriber._raw_transcribe(wav_file)

        assert result == "Hello world"

    @patch(_GENAI_CLIENT_PATCH)
    def test_raw_transcribe_uses_audio_response_modality(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that LiveConnectConfig uses AUDIO response modality with input_audio_transcription."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()

        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "text"
        msg.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key")

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)

        transcriber._raw_transcribe(wav_file)

        # Verify connect was called with correct config on the live client
        connect_call = mock_live.aio.live.connect.call_args
        config = connect_call.kwargs["config"]
        assert config.response_modalities == ["AUDIO"]
        assert isinstance(config.input_audio_transcription, types.AudioTranscriptionConfig)

    @patch(_GENAI_CLIENT_PATCH)
    def test_raw_transcribe_uses_configured_model(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that the configured model is passed to live.connect."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()

        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "text"
        msg.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key", model="custom-live")

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)

        transcriber._raw_transcribe(wav_file)

        connect_call = mock_live.aio.live.connect.call_args
        assert connect_call.kwargs["model"] == "custom-live"

    @patch(_GENAI_CLIENT_PATCH)
    def test_raw_transcribe_invalid_wav_raises_value_error(self, _mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that an invalid WAV file raises ValueError."""
        transcriber = GeminiLiveTranscriber(api_key="test-key")

        wav_file = tmp_path / "bad.wav"
        wav_file.write_bytes(b"\x00" * 20)

        with pytest.raises(ValueError, match="Not a valid WAV file"):
            transcriber._raw_transcribe(wav_file)

    @patch(_GENAI_CLIENT_PATCH)
    def test_raw_transcribe_does_not_check_no_speech(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that _raw_transcribe returns [NO_SPEECH] as-is without raising."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()

        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "[NO_SPEECH]"
        msg.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key")

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)

        result = transcriber._raw_transcribe(wav_file)

        assert result == "[NO_SPEECH]"


class TestLiveConstants:
    """Test module-level constants."""

    def test_audio_chunk_size(self) -> None:
        """Test AUDIO_CHUNK_SIZE constant value."""
        assert AUDIO_CHUNK_SIZE == 32768

    def test_live_transcription_instruction(self) -> None:
        """Test LIVE_TRANSCRIPTION_INSTRUCTION is a non-empty string."""
        assert isinstance(LIVE_TRANSCRIPTION_INSTRUCTION, str)
        assert len(LIVE_TRANSCRIPTION_INSTRUCTION) > 0
        assert "[NO_SPEECH]" in LIVE_TRANSCRIPTION_INSTRUCTION

    def test_live_transcription_instruction_refined(self) -> None:
        """Test LIVE_TRANSCRIPTION_INSTRUCTION_REFINED is a non-empty string with expected content."""
        assert isinstance(LIVE_TRANSCRIPTION_INSTRUCTION_REFINED, str)
        assert len(LIVE_TRANSCRIPTION_INSTRUCTION_REFINED) > 0
        assert "[NO_SPEECH]" in LIVE_TRANSCRIPTION_INSTRUCTION_REFINED
        assert "refine" in LIVE_TRANSCRIPTION_INSTRUCTION_REFINED.lower()

    def test_live_transcription_instruction_formatted(self) -> None:
        """Test LIVE_TRANSCRIPTION_INSTRUCTION_FORMATTED is a non-empty string with expected content."""
        assert isinstance(LIVE_TRANSCRIPTION_INSTRUCTION_FORMATTED, str)
        assert len(LIVE_TRANSCRIPTION_INSTRUCTION_FORMATTED) > 0
        assert "[NO_SPEECH]" in LIVE_TRANSCRIPTION_INSTRUCTION_FORMATTED
        assert "numbered list" in LIVE_TRANSCRIPTION_INSTRUCTION_FORMATTED.lower()


class TestInstructionSelection:
    """Test that _transcribe_live selects the correct system instruction."""

    @patch(_GENAI_CLIENT_PATCH)
    def test_default_instruction_when_no_refine_no_format(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that default instruction is used when neither refine nor format is set."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()
        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "text"
        msg.server_content.turn_complete = True
        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key")

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)
        transcriber._raw_transcribe(wav_file)

        config = mock_live.aio.live.connect.call_args.kwargs["config"]
        instruction_text = config.system_instruction.parts[0].text
        assert instruction_text == LIVE_TRANSCRIPTION_INSTRUCTION

    @patch(_GENAI_CLIENT_PATCH)
    def test_refined_instruction_when_refine_only(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that refined instruction is used when refine is True and format is False."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()
        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "text"
        msg.server_content.turn_complete = True
        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key", refine=True)

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)
        transcriber._raw_transcribe(wav_file)

        config = mock_live.aio.live.connect.call_args.kwargs["config"]
        instruction_text = config.system_instruction.parts[0].text
        assert instruction_text == LIVE_TRANSCRIPTION_INSTRUCTION_REFINED

    @patch(_GENAI_CLIENT_PATCH)
    def test_formatted_instruction_when_format_output_set(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that formatted instruction is used when format_output is True."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()
        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "text"
        msg.server_content.turn_complete = True
        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key", refine=True, format_output=True)

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)
        transcriber._raw_transcribe(wav_file)

        config = mock_live.aio.live.connect.call_args.kwargs["config"]
        instruction_text = config.system_instruction.parts[0].text
        assert instruction_text == LIVE_TRANSCRIPTION_INSTRUCTION_FORMATTED


class TestLiveTranscriberIntegration:
    """Integration tests for GeminiLiveTranscriber with parent's transcribe() method."""

    @patch(_GENAI_CLIENT_PATCH)
    def test_transcribe_delegates_to_raw_transcribe(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that parent's transcribe() calls the overridden _raw_transcribe."""
        mock_class_obj, mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()

        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "Hello world"
        msg.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key")

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)

        result = transcriber.transcribe(wav_file)

        assert result == "Hello world"
        # Verify it used live_client.aio.live.connect, NOT parent's models.generate_content
        mock_live.aio.live.connect.assert_called_once()
        mock_parent.models.generate_content.assert_not_called()

    @patch(_GENAI_CLIENT_PATCH)
    def test_transcribe_no_speech_raises(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that [NO_SPEECH] from live API triggers no speech error via parent."""
        mock_class_obj, _mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_session = MagicMock()

        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "[NO_SPEECH]"
        msg.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        transcriber = GeminiLiveTranscriber(api_key="test-key")

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)

        with pytest.raises(RuntimeError, match="No speech detected"):
            transcriber.transcribe(wav_file)

    @patch(_GENAI_CLIENT_PATCH)
    def test_transcribe_with_ask_keyword(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that ask keyword works through parent's transcribe() with live backend."""
        mock_class_obj, mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        # Live API returns text starting with ask keyword
        mock_session = MagicMock()

        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "hey what is Python"
        msg.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        # Mock the generate_content for the ask query answer (uses parent client)
        ai_answer_response = MagicMock()
        ai_answer_response.text = "Python is a programming language"
        mock_parent.models.generate_content.return_value = ai_answer_response

        transcriber = GeminiLiveTranscriber(api_key="test-key", ask_keyword="hey")

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)

        result = transcriber.transcribe(wav_file)

        assert result == "Python is a programming language"

    @patch(_GENAI_CLIENT_PATCH)
    def test_apply_instruction_uses_batch_model(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that _apply_instruction uses _batch_model, not _model."""
        mock_class_obj, mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        # Live API returns text with instruction keyword
        mock_session = MagicMock()

        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "some content boom translate to French"
        msg.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        # Mock the generate_content for the instruction processing (uses parent client)
        instruction_response = MagicMock()
        instruction_response.text = "du contenu"
        mock_parent.models.generate_content.return_value = instruction_response

        transcriber = GeminiLiveTranscriber(
            api_key="test-key",
            instruction_keyword="boom",
            batch_model="gemini-2.5-flash",
        )

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)

        result = transcriber.transcribe(wav_file)

        assert result == "du contenu"
        # Verify generate_content was called with batch_model, NOT the live model
        call_args = mock_parent.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-2.5-flash"

    @patch(_GENAI_CLIENT_PATCH)
    def test_answer_query_uses_batch_model(self, mock_class: MagicMock, tmp_path: Path) -> None:
        """Test that _answer_query uses _batch_model, not _model."""
        mock_class_obj, mock_parent, mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        # Live API returns text starting with ask keyword
        mock_session = MagicMock()

        msg = MagicMock()
        msg.server_content = MagicMock()
        msg.server_content.input_transcription = MagicMock()
        msg.server_content.input_transcription.text = "hey what is the capital of France"
        msg.server_content.turn_complete = True

        mock_session.receive.return_value = _async_iter([msg])
        mock_session.send_realtime_input = AsyncMock()

        mock_connect = AsyncMock()
        mock_connect.__aenter__.return_value = mock_session
        mock_live.aio.live.connect.return_value = mock_connect

        # Mock the generate_content for the ask query answer (uses parent client)
        ai_answer_response = MagicMock()
        ai_answer_response.text = "Paris"
        mock_parent.models.generate_content.return_value = ai_answer_response

        transcriber = GeminiLiveTranscriber(
            api_key="test-key",
            ask_keyword="hey",
            batch_model="gemini-2.5-flash",
        )

        wav_file = tmp_path / "test.wav"
        _make_wav_file(wav_file)

        result = transcriber.transcribe(wav_file)

        assert result == "Paris"
        # Verify generate_content was called with batch_model, NOT the live model
        call_args = mock_parent.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-2.5-flash"


class TestNoLiveLangParameter:
    """Test that lang parameter has been removed from GeminiLiveTranscriber."""

    @patch(_GENAI_CLIENT_PATCH)
    def test_init_does_not_accept_lang(self, _mock_class: MagicMock) -> None:
        """Test that GeminiLiveTranscriber does not accept a lang parameter."""
        with pytest.raises(TypeError):
            GeminiLiveTranscriber(api_key="test-key", lang="English")  # type: ignore[call-arg]

    @patch(_GENAI_CLIENT_PATCH)
    def test_no_lang_attribute(self, _mock_class: MagicMock) -> None:
        """Test that GeminiLiveTranscriber has no _lang attribute."""
        transcriber = GeminiLiveTranscriber(api_key="test-key")
        assert not hasattr(transcriber, "_lang")

    @patch(_GENAI_CLIENT_PATCH)
    def test_live_apply_instruction_no_lang_directive(self, mock_class: MagicMock) -> None:
        """Test that live _apply_instruction does not append language directive."""
        mock_class_obj, mock_parent, _mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_response = MagicMock()
        mock_response.text = "Result"
        mock_parent.models.generate_content.return_value = mock_response

        transcriber = GeminiLiveTranscriber(api_key="test-key")
        transcriber._apply_instruction("content", "instruction")

        call_args = mock_parent.models.generate_content.call_args
        contents = call_args.kwargs["contents"]
        prompt_text = contents[0].text
        assert "CRITICAL LANGUAGE REQUIREMENT" not in prompt_text

    @patch(_GENAI_CLIENT_PATCH)
    def test_live_answer_query_no_lang_directive(self, mock_class: MagicMock) -> None:
        """Test that live _answer_query does not append language directive."""
        mock_class_obj, mock_parent, _mock_live = _make_client_factory()
        mock_class.side_effect = mock_class_obj.side_effect

        mock_response = MagicMock()
        mock_response.text = "Answer"
        mock_parent.models.generate_content.return_value = mock_response

        transcriber = GeminiLiveTranscriber(api_key="test-key", ask_keyword="hey")
        transcriber._answer_query("what is Python")

        call_args = mock_parent.models.generate_content.call_args
        contents = call_args.kwargs["contents"]
        prompt_text = contents[0].text
        assert "CRITICAL LANGUAGE REQUIREMENT" not in prompt_text
