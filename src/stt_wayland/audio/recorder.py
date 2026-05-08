"""Audio recording using PipeWire/PulseAudio."""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Final

TEMP_RECORDING_PATH: Final[str] = "/tmp/claude/stt_recording.wav"  # noqa: S108

# Error messages
ERR_NO_RECORDER: Final[str] = "No audio recorder found. Install pipewire-utils or pulseaudio-utils."
ERR_ALREADY_RECORDING: Final[str] = "Recording already in progress"
ERR_NO_RECORDING: Final[str] = "No recording in progress"


class AudioRecorder:
    """Records audio using pw-record (PipeWire) or parecord (PulseAudio)."""

    def __init__(self, output_path: Path | None = None) -> None:
        """Initialize audio recorder.

        Args:
            output_path: Path to save audio recording. Defaults to /tmp/claude/stt_recording.wav

        """
        if output_path is None:
            output_path = Path(TEMP_RECORDING_PATH)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self._process: subprocess.Popen[bytes] | None = None
        self._logger = logging.getLogger(__name__)
        self._recorder_cmd = self._detect_recorder()

    def _detect_recorder(self) -> str:
        """Detect available audio recorder.

        Returns:
            Command name ('pw-record' or 'parecord').

        Raises:
            RuntimeError: If no recorder is available.

        """
        if shutil.which("pw-record"):
            self._logger.info("Using pw-record (PipeWire)")
            return "pw-record"
        if shutil.which("parecord"):
            self._logger.info("Using parecord (PulseAudio)")
            return "parecord"
        msg = ERR_NO_RECORDER
        raise RuntimeError(msg)

    def start_recording(self) -> None:
        """Start audio recording.

        Raises:
            RuntimeError: If already recording.

        """
        if self._process is not None:
            msg = ERR_ALREADY_RECORDING
            raise RuntimeError(msg)

        # Remove old recording if exists
        self.output_path.unlink(missing_ok=True)

        # Common args: 16kHz mono WAV
        # pw-record: --rate 16000 --channels 1 output.wav
        # parecord: --rate=16000 --channels=1 output.wav
        if self._recorder_cmd == "pw-record":
            cmd = [
                "pw-record",
                "--rate",
                "16000",
                "--channels",
                "1",
                str(self.output_path),
            ]
        else:  # parecord
            cmd = [
                "parecord",
                "--rate=16000",
                "--channels=1",
                str(self.output_path),
            ]

        self._logger.info("Starting recording: %s", " ".join(cmd))
        self._process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)  # noqa: S603

    def stop_recording(self) -> Path:
        """Stop audio recording.

        Returns:
            Path to recorded audio file.

        Raises:
            RuntimeError: If not recording.

        """
        if self._process is None:
            msg = ERR_NO_RECORDING
            raise RuntimeError(msg)

        self._logger.info("Stopping recording")
        self._process.terminate()

        try:
            _, stderr = self._process.communicate(timeout=5)
            if self._process.returncode not in {0, -15}:  # -15 is SIGTERM
                self._logger.warning("Recorder stderr: %s", stderr.decode())
        except subprocess.TimeoutExpired:
            self._logger.warning("Recorder did not terminate, killing")
            self._process.kill()
            self._process.communicate()

        self._process = None

        if not self.output_path.exists():
            msg = f"Recording file not created: {self.output_path}"
            raise RuntimeError(msg)

        self._logger.info("Recording saved: %s", self.output_path)
        return self.output_path

    def start_streaming(self) -> None:
        """Start audio recording in streaming mode (raw PCM to stdout).

        Audio chunks are buffered internally and can be read via read_chunk().

        Raises:
            RuntimeError: If already recording.

        """
        if self._process is not None:
            msg = ERR_ALREADY_RECORDING
            raise RuntimeError(msg)

        # For streaming, we output raw PCM to stdout instead of WAV to file
        if self._recorder_cmd == "pw-record":
            cmd = [
                "pw-record",
                "--rate",
                "16000",
                "--channels",
                "1",
                "--format",
                "s16",
                "-",  # stdout
            ]
        else:  # parecord
            cmd = [
                "parecord",
                "--rate=16000",
                "--channels=1",
                "--format=s16le",
                "--raw",
            ]

        self._logger.info("Starting streaming recording: %s", " ".join(cmd))
        self._chunk_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=100)
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603

        # Background thread to read chunks from recorder stdout
        self._reader_thread = threading.Thread(
            target=self._read_audio_chunks,
            daemon=True,
            name="audio-chunk-reader",
        )
        self._reader_thread.start()

    def _read_audio_chunks(self) -> None:
        """Read PCM chunks from recorder stdout and put them in the queue."""
        assert self._process is not None  # noqa: S101
        assert self._process.stdout is not None  # noqa: S101
        try:
            while True:
                chunk = self._process.stdout.read(32768)  # 32KB chunks
                if not chunk:
                    break
                try:
                    self._chunk_queue.put(chunk, timeout=1.0)
                except queue.Full:
                    self._logger.warning("Audio chunk queue full, dropping chunk")
        except Exception:
            self._logger.exception("Error reading audio chunks")
        finally:
            self._chunk_queue.put(None)  # Sentinel to signal end of stream

    def read_chunk(self, timeout: float = 1.0) -> bytes | None:
        """Read the next audio chunk from the streaming buffer.

        Args:
            timeout: Maximum time to wait for a chunk.

        Returns:
            Raw PCM audio data, or None if the stream has ended.

        """
        try:
            return self._chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return b""

    def stop_streaming(self) -> None:
        """Stop streaming recording.

        Raises:
            RuntimeError: If not recording.

        """
        if self._process is None:
            msg = ERR_NO_RECORDING
            raise RuntimeError(msg)

        self._logger.info("Stopping streaming recording")
        self._process.terminate()

        try:
            self._process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self._logger.warning("Recorder did not terminate, killing")
            self._process.kill()
            self._process.communicate()

        self._process = None
        self._logger.info("Streaming recording stopped")

    def is_recording(self) -> bool:
        """Check if currently recording.

        Returns:
            True if recording is in progress.

        """
        return self._process is not None
