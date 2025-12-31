"""Audio recording using PipeWire/PulseAudio."""

from __future__ import annotations

import logging
import shutil
import subprocess
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

    def is_recording(self) -> bool:
        """Check if currently recording.

        Returns:
            True if recording is in progress.

        """
        return self._process is not None
