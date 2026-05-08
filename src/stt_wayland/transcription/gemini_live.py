"""Google Gemini Live API transcription service."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import wave
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from .gemini import (
    ASK_QUERY_PROMPT,
    CUSTOM_INSTRUCTION_PROMPT,
    GeminiTranscriber,
    _raise_empty_response_error,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Final

    from stt_wayland.audio.recorder import AudioRecorder

# Audio chunk size for streaming (32KB)
AUDIO_CHUNK_SIZE: Final[int] = 32768

# Timeout for live transcription (seconds)
LIVE_TRANSCRIBE_TIMEOUT: Final[float] = 120

# Live API transcription system instruction
LIVE_TRANSCRIPTION_INSTRUCTION: Final[str] = (
    "You are a speech-to-text transcription service. "
    "Transcribe the spoken words in the audio exactly as spoken. "
    "Return ONLY the transcribed words. "
    "NO explanations, NO prefixes, NO meta-commentary, NO markdown. "
    "If the audio is empty or silent, return exactly: [NO_SPEECH]"
)

LIVE_TRANSCRIPTION_INSTRUCTION_REFINED: Final[str] = (
    "You are a speech-to-text transcription service. "
    "Transcribe the spoken words in the audio. "
    "After transcription, refine the text by correcting any typos, grammatical errors, "
    "and improving clarity while preserving the original meaning. "
    "Return ONLY the refined transcribed text. "
    "NO explanations, NO prefixes, NO meta-commentary, NO markdown. "
    "If the audio is empty or silent, return exactly: [NO_SPEECH]"
)

LIVE_TRANSCRIPTION_INSTRUCTION_FORMATTED: Final[str] = (
    "You are a speech-to-text transcription service. "
    "Transcribe the spoken words in the audio. "
    "After transcription, refine the text by correcting any typos, grammatical errors, "
    "and improving clarity while preserving the original meaning. "
    "If the speech contains multiple distinct points, format them as a numbered list (1. 2. 3.) "
    "or a bulleted list using dashes (-). Use line breaks to separate distinct points. "
    "Keep simple sentences as plain flowing text. "
    "Return ONLY the refined transcribed text. "
    "NO explanations, NO prefixes, NO meta-commentary, NO markdown formatting "
    "(no bold, italic, headers, code blocks). "
    "Plain-text lists using dashes or numbers are allowed. "
    "If the audio is empty or silent, return exactly: [NO_SPEECH]"
)


class GeminiLiveTranscriber(GeminiTranscriber):
    """Transcribes audio using Google Gemini Live API (WebSocket streaming)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-live-preview",
        *,
        batch_model: str = "gemini-2.5-flash",
        refine: bool = False,
        format_output: bool = False,
        instruction_keyword: str | None = None,
        ask_keyword: str | None = None,
    ) -> None:
        """Initialize Gemini Live transcriber.

        Args:
            api_key: Google API key.
            model: Model name to use for Live API (WebSocket streaming).
            batch_model: Model name for REST API calls (generate_content).
                The live model only supports WebSocket, so post-processing
                (instruction/ask keywords) uses this separate model.
            refine: Enable AI-based typo and grammar correction.
            format_output: Enable plain-text formatting of refined output.
            instruction_keyword: Keyword to separate content from AI instructions.
            ask_keyword: Keyword at start of speech to trigger AI query mode.

        """
        super().__init__(
            api_key=api_key,
            model=model,
            refine=refine,
            format_output=format_output,
            instruction_keyword=instruction_keyword,
            ask_keyword=ask_keyword,
        )
        self._logger = logging.getLogger(__name__)

        if "live" not in model.lower():
            msg = (
                f"Model '{model}' is not a Live API model. "
                f"Live mode requires a model with 'live' in its name "
                f"(e.g., 'gemini-3.1-flash-live-preview'). "
                f"Remove --live flag or use a Live API model."
            )
            raise ValueError(msg)

        # Replace parent's client with one that uses v1beta API version.
        # v1beta is required for Live API and is backward compatible
        # with generate_content calls used for batch operations.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        self._batch_model = batch_model

        # Create a persistent event loop in a background daemon thread
        # to avoid creating/destroying loops on every transcription call.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="gemini-live-event-loop",
            daemon=True,
        )
        self._loop_thread.start()
        self._closed = False
        self._client_closed = False
        self._streaming_future: concurrent.futures.Future[str] | None = None

    def _raw_transcribe(self, audio_path: Path) -> str:
        """Transcribe audio using the Gemini Live API.

        Reads the WAV file, strips the header to get raw PCM data,
        and sends it through the Live API WebSocket for transcription.

        Args:
            audio_path: Path to WAV audio file.

        Returns:
            Raw transcribed text from the API.

        Raises:
            RuntimeError: If the API returns an empty response or transcriber is closed.
            ValueError: If the audio file is not a valid WAV file.

        """
        if self._closed or self._client_closed:
            msg = "Transcriber is closed"
            raise RuntimeError(msg)

        # Read WAV file and strip header to get raw PCM
        pcm_data = self._extract_pcm_from_wav(audio_path)
        self._logger.info("Extracted %d bytes of PCM data from %s", len(pcm_data), audio_path)

        # Submit to the persistent event loop thread
        future = asyncio.run_coroutine_threadsafe(self._transcribe_live(pcm_data), self._loop)
        try:
            return future.result(timeout=LIVE_TRANSCRIBE_TIMEOUT)
        except TimeoutError:
            if not future.cancel():
                self._logger.warning("Could not cancel timed-out live transcription task")
            msg = f"Live transcription timed out after {LIVE_TRANSCRIBE_TIMEOUT} seconds"
            raise RuntimeError(msg) from None

    @staticmethod
    def _extract_pcm_from_wav(audio_path: Path) -> bytes:
        """Extract raw PCM data from a WAV file.

        Args:
            audio_path: Path to WAV audio file.

        Returns:
            Raw PCM audio data.

        Raises:
            ValueError: If the file is not a valid WAV file.

        """
        try:
            with wave.open(str(audio_path), "rb") as wf:
                return wf.readframes(wf.getnframes())
        except wave.Error as e:
            msg = f"Not a valid WAV file: {audio_path}"
            raise ValueError(msg) from e

    async def _transcribe_live(self, pcm_data: bytes) -> str:
        """Send audio to Gemini Live API and collect transcription.

        Args:
            pcm_data: Raw PCM audio data (16-bit, 16kHz, mono, little-endian).

        Returns:
            Transcribed text.

        Raises:
            RuntimeError: If the API returns an empty response.

        """
        # Select system instruction based on refine/format settings
        if self._format_output:
            instruction = LIVE_TRANSCRIPTION_INSTRUCTION_FORMATTED
        elif self._refine:
            instruction = LIVE_TRANSCRIPTION_INSTRUCTION_REFINED
        else:
            instruction = LIVE_TRANSCRIPTION_INSTRUCTION

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=types.Content(parts=[types.Part.from_text(text=instruction)]),
        )

        async with self._client.aio.live.connect(
            model=self._model,
            config=config,
        ) as session:
            # Send audio in chunks
            for offset in range(0, len(pcm_data), AUDIO_CHUNK_SIZE):
                chunk = pcm_data[offset : offset + AUDIO_CHUNK_SIZE]
                await session.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"),
                )

            # Signal end of audio stream
            await session.send_realtime_input(audio_stream_end=True)

            # Collect input audio transcription
            text_parts: list[str] = []
            async for msg in session.receive():
                if msg.server_content:
                    # Input transcription contains the STT result
                    if msg.server_content.input_transcription and msg.server_content.input_transcription.text:
                        text_parts.append(msg.server_content.input_transcription.text)
                    # Stop when turn is complete
                    if msg.server_content.turn_complete:
                        break

            result = "".join(text_parts).strip()
            if not result:
                _raise_empty_response_error()

        return result

    def start_streaming(self, recorder: AudioRecorder) -> None:
        """Start real-time streaming transcription.

        Opens a Live API WebSocket session and begins consuming audio chunks
        from the recorder in the background. Call stop_streaming() to finish
        and retrieve the transcription result.

        Args:
            recorder: AudioRecorder instance (must already be in streaming mode).

        Raises:
            RuntimeError: If transcriber is closed or streaming already active.

        """
        if self._closed or self._client_closed:
            msg = "Transcriber is closed"
            raise RuntimeError(msg)

        self._logger.info("Starting real-time streaming transcription with %s", self._model)
        self._streaming_future = asyncio.run_coroutine_threadsafe(self._transcribe_stream_live(recorder), self._loop)

    def stop_streaming(self) -> str:
        """Stop streaming transcription and return the result.

        Waits for the background transcription to complete (the recorder
        must have been stopped first to signal end of stream).

        Returns:
            Transcribed text.

        Raises:
            RuntimeError: If transcription fails or times out.

        """
        if self._streaming_future is None:
            msg = "No active streaming transcription"
            raise RuntimeError(msg)

        try:
            result = self._streaming_future.result(timeout=LIVE_TRANSCRIBE_TIMEOUT)
        except TimeoutError:
            if not self._streaming_future.cancel():
                self._logger.warning("Could not cancel timed-out streaming transcription")
            msg = f"Streaming transcription timed out after {LIVE_TRANSCRIBE_TIMEOUT} seconds"
            raise RuntimeError(msg) from None
        finally:
            self._streaming_future = None

        return result

    async def _transcribe_stream_live(self, recorder: AudioRecorder) -> str:
        """Stream audio from recorder to Live API in real-time.

        Sends audio and receives transcription concurrently. The Live API
        emits transcription results while audio is still streaming, so both
        must happen in parallel.

        Args:
            recorder: AudioRecorder instance to read chunks from.

        Returns:
            Transcribed text.

        """
        if self._format_output:
            instruction = LIVE_TRANSCRIPTION_INSTRUCTION_FORMATTED
        elif self._refine:
            instruction = LIVE_TRANSCRIPTION_INSTRUCTION_REFINED
        else:
            instruction = LIVE_TRANSCRIPTION_INSTRUCTION

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=types.Content(parts=[types.Part.from_text(text=instruction)]),
        )

        self._logger.info("Connecting to Live API for streaming transcription")
        async with self._client.aio.live.connect(
            model=self._model,
            config=config,
        ) as session:
            self._logger.info("Live API session established")

            # Collect transcription results concurrently while sending audio
            text_parts: list[str] = []
            chunks_sent = 0
            bytes_sent = 0

            async def _receive_transcription() -> None:
                """Receive transcription results from the Live API."""
                self._logger.info("Receive task started")
                async for msg in session.receive():
                    if msg.server_content:
                        if msg.server_content.input_transcription and msg.server_content.input_transcription.text:
                            self._logger.info(
                                "Received transcription chunk: %s",
                                msg.server_content.input_transcription.text[:80],
                            )
                            text_parts.append(msg.server_content.input_transcription.text)
                        if msg.server_content.turn_complete:
                            self._logger.info("Received turn_complete")
                            break

            # Start receiving in the background — the API sends results
            # while audio is still streaming, so we must listen concurrently
            receive_task = asyncio.create_task(_receive_transcription())

            # Stream audio chunks as they arrive from the recorder
            loop = asyncio.get_running_loop()
            while True:
                chunk = await loop.run_in_executor(None, recorder.read_chunk, 0.5)
                if chunk is None:
                    self._logger.info("End of audio stream (recorder stopped)")
                    break
                if chunk:  # Skip empty chunks (timeout with no data)
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"),
                    )
                    chunks_sent += 1
                    bytes_sent += len(chunk)

            self._logger.info("Sent %d chunks (%d bytes) to Live API", chunks_sent, bytes_sent)

            # Signal end of audio stream and wait for final transcription
            await session.send_realtime_input(audio_stream_end=True)
            self._logger.info("Sent audio_stream_end, waiting for transcription result")

            # Wait for receive task with a timeout to avoid hanging forever
            try:
                await asyncio.wait_for(asyncio.shield(receive_task), timeout=30)
            except TimeoutError:
                self._logger.warning("Receive task timed out after 30s, cancelling")
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass

            result = "".join(text_parts).strip()
            self._logger.info("Streaming transcription result: %s", result[:100] if result else "(empty)")
            if not result:
                _raise_empty_response_error()

        return result

    def close(self) -> None:
        """Close the transcriber and release resources.

        Closes the genai client, stops the persistent event loop, and joins
        the background thread. Safe to call multiple times.
        """
        if self._closed:
            return

        # Close the genai client (sync + async sides) — only once
        if not self._client_closed:
            try:
                if self._loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(self._client.aio.aclose(), self._loop)
                    future.result(timeout=5)
                self._client.close()
                self._client_closed = True
            except Exception:
                self._logger.exception("Error closing genai client")

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)
        if self._loop_thread.is_alive():
            self._logger.warning("Event loop thread did not stop in time; close() can be retried")
        else:
            if not self._loop.is_closed():
                self._loop.close()
            self._closed = True
            self._logger.info("Gemini Live transcriber closed")

    def _apply_instruction(self, content: str, instruction: str) -> str:
        """Apply an instruction using the batch model (REST API).

        The live model only supports WebSocket, so post-processing
        uses a separate batch model for generate_content calls.

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
            model=self._batch_model,
            contents=[types.Part.from_text(text=prompt)],
        )

        if response.text:
            return str(response.text).strip()

        _raise_empty_response_error()

    def _answer_query(self, query: str) -> str:
        """Answer a query using the batch model (REST API).

        The live model only supports WebSocket, so post-processing
        uses a separate batch model for generate_content calls.

        Args:
            query: The query to send to the AI.

        Returns:
            The AI's answer.

        Raises:
            RuntimeError: If the API call fails.

        """
        prompt = ASK_QUERY_PROMPT.format(query=query)

        response = self._client.models.generate_content(
            model=self._batch_model,
            contents=[types.Part.from_text(text=prompt)],
        )

        if response.text:
            return str(response.text).strip()

        _raise_empty_response_error()
