"""STT daemon main logic."""

from __future__ import annotations

import errno
import logging
import os
import signal
import sys
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Final

from .audio import AudioRecorder
from .config import Config
from .output import (
    notify_error,
    notify_recording_started,
    notify_recording_stopped,
    notify_transcription_complete,
    type_text,
)
from .state_machine import Event, State, StateMachine
from .transcription import GeminiTranscriber

# Error messages
ERR_DAEMON_RUNNING: Final[str] = "Daemon already running with PID {pid}. PID file: {pid_file}"
ERR_CONFIG: Final[str] = "GEMINI_API_KEY environment variable is required. Set it in .env or export it."


class STTDaemon:
    """Speech-to-Text daemon for Wayland."""

    def __init__(self, config: Config, *, refine: bool = False, instruction_keyword: str | None = None) -> None:
        """Initialize daemon.

        Args:
            config: Daemon configuration.
            refine: Enable AI-based typo and grammar correction.
            instruction_keyword: Keyword to separate content from AI instructions.

        """
        self.config = config
        self.state_machine = StateMachine()
        self.recorder = AudioRecorder()
        self.transcriber = GeminiTranscriber(
            api_key=config.api_key,
            model=config.model,
            refine=refine,
            instruction_keyword=instruction_keyword,
        )
        self._logger = logging.getLogger(__name__)
        self._audio_path: Path | None = None
        self._transcribed_text: str | None = None

        # Signal flags (async-safe)
        self._toggle_requested = False
        self._shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGUSR1, self._handle_toggle_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)

    def _handle_toggle_signal(self, _signum: int, _frame: object) -> None:
        """Handle SIGUSR1 signal to toggle recording.

        Args:
            _signum: Signal number.
            _frame: Current stack frame.

        Note:
            Signal-safe: Only sets a flag, actual processing happens in main loop.

        """
        # Async-safe: only set a flag
        self._toggle_requested = True

    def _handle_shutdown_signal(self, _signum: int, _frame: object) -> None:
        """Handle SIGTERM/SIGINT signal for graceful shutdown.

        Args:
            _signum: Signal number.
            _frame: Current stack frame.

        Note:
            Signal-safe: Only sets a flag, actual processing happens in main loop.

        """
        # Async-safe: only set a flag
        self._shutdown_requested = True

    def _write_pid_file(self) -> None:
        """Write PID file atomically.

        Raises:
            RuntimeError: If PID file already exists and daemon is running.

        """
        pid_file = self.config.pid_file
        pid_file.parent.mkdir(parents=True, exist_ok=True)

        # Atomic creation using O_CREAT | O_EXCL
        try:
            fd = os.open(
                str(pid_file),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                # File exists, check if process is running
                try:
                    existing_pid = int(pid_file.read_text().strip())
                    # Check if process is running
                    os.kill(existing_pid, 0)
                    msg = ERR_DAEMON_RUNNING.format(pid=existing_pid, pid_file=pid_file)
                    raise RuntimeError(msg) from e
                except (ProcessLookupError, ValueError):
                    # Process not running or invalid PID, remove stale file
                    self._logger.warning("Removing stale PID file: %s", pid_file)
                    pid_file.unlink()
                    # Retry atomically
                    fd = os.open(
                        str(pid_file),
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o644,
                    )
            else:
                raise

        try:
            os.write(fd, f"{os.getpid()}\n".encode())
        finally:
            os.close(fd)

        self._logger.info("PID file created: %s", pid_file)

    def _remove_pid_file(self) -> None:
        """Remove PID file."""
        try:
            self.config.pid_file.unlink(missing_ok=True)
            self._logger.info("PID file removed")
        except Exception:
            self._logger.exception("Failed to remove PID file")

    # State transition handlers
    def _start_recording(self) -> None:
        """Start recording (IDLE → RECORDING)."""
        self._logger.info("Starting recording")
        try:
            self.recorder.start_recording()
            self.state_machine.set_state(State.RECORDING)
            notify_recording_started()
        except Exception:
            self._logger.exception("Failed to start recording")
            notify_error("Failed to start recording")
            self.state_machine.transition(Event.ERROR)

    def _stop_recording(self) -> None:
        """Stop recording (RECORDING → TRANSCRIBING)."""
        self._logger.info("Stopping recording")
        try:
            self._audio_path = self.recorder.stop_recording()
            self.state_machine.set_state(State.TRANSCRIBING)
            notify_recording_stopped()
            # Immediately trigger transcription
            self.state_machine.transition(Event.RECORDING_STOPPED)
        except Exception:
            self._logger.exception("Failed to stop recording")
            notify_error("Failed to stop recording")
            # Ensure recorder is stopped on error
            if self.recorder.is_recording():
                try:
                    self.recorder.stop_recording()
                except Exception:
                    self._logger.exception("Failed to force-stop recorder")
            self.state_machine.transition(Event.ERROR)
            self.state_machine.set_state(State.IDLE)

    def _transcribe_audio(self) -> None:
        """Transcribe audio (TRANSCRIBING → TYPING)."""
        if not self._audio_path:
            self._logger.error("No audio file to transcribe")
            notify_error("No audio file to transcribe")
            self.state_machine.set_state(State.IDLE)
            return

        self._logger.info("Transcribing %s", self._audio_path)
        try:
            self._transcribed_text = self.transcriber.transcribe(self._audio_path)
            self.state_machine.set_state(State.TYPING)
            # Immediately trigger typing
            self.state_machine.transition(Event.TRANSCRIPTION_COMPLETE)
        except Exception:
            self._logger.exception("Transcription failed")
            notify_error("Transcription failed")
            self.state_machine.transition(Event.ERROR)
            self.state_machine.set_state(State.IDLE)
        finally:
            # Always cleanup audio file after transcription attempt
            if self._audio_path and self._audio_path.exists():
                try:
                    self._audio_path.unlink()
                    self._logger.info("Deleted audio file: %s", self._audio_path)
                except Exception:
                    self._logger.exception("Failed to delete audio file: %s", self._audio_path)

    def _type_text(self) -> None:
        """Type transcribed text (TYPING → IDLE)."""
        if not self._transcribed_text:
            self._logger.error("No text to type")
            notify_error("No text to type")
            self.state_machine.set_state(State.IDLE)
            return

        self._logger.info("Typing text")
        try:
            type_text(self._transcribed_text)
            self.state_machine.set_state(State.IDLE)
            notify_transcription_complete()
        except Exception:
            self._logger.exception("Failed to type text")
            notify_error("Failed to type text")
            self.state_machine.transition(Event.ERROR)
            self.state_machine.set_state(State.IDLE)
        finally:
            # Always clean up state
            self._audio_path = None
            self._transcribed_text = None

    def _handle_error(self) -> None:
        """Handle error state."""
        self._logger.error("Error occurred, returning to IDLE")
        self.state_machine.set_state(State.IDLE)

    def run(self) -> NoReturn:
        """Run daemon main loop."""
        self._logger.info("Starting STT daemon")
        self._write_pid_file()

        # Define state machine handlers
        handlers = {
            # Toggle recording in IDLE starts recording
            (State.IDLE, Event.TOGGLE_RECORDING): self._start_recording,
            # Toggle recording in RECORDING stops it
            (State.RECORDING, Event.TOGGLE_RECORDING): self._stop_recording,
            # After recording stopped, transcribe
            (State.TRANSCRIBING, Event.RECORDING_STOPPED): self._transcribe_audio,
            # After transcription, type
            (State.TYPING, Event.TRANSCRIPTION_COMPLETE): self._type_text,
            # Error handling
            (State.IDLE, Event.ERROR): self._handle_error,
            (State.RECORDING, Event.ERROR): self._handle_error,
            (State.TRANSCRIBING, Event.ERROR): self._handle_error,
            (State.TYPING, Event.ERROR): self._handle_error,
        }

        try:
            self._logger.info("Daemon ready, waiting for SIGUSR1 to toggle recording")
            while True:
                # Process signal flags (async-safe pattern)
                if self._toggle_requested:
                    self._toggle_requested = False
                    self._logger.info("Received SIGUSR1 (toggle recording)")
                    self.state_machine.transition(Event.TOGGLE_RECORDING)

                if self._shutdown_requested:
                    self._shutdown_requested = False
                    self._logger.info("Received shutdown signal")
                    self.state_machine.transition(Event.SHUTDOWN)

                # Process state machine events
                if not self.state_machine.process_events(handlers):
                    break
        except KeyboardInterrupt:
            self._logger.info("Interrupted by user")
        finally:
            self._cleanup()

        sys.exit(0)

    def _cleanup(self) -> None:
        """Cleanup resources."""
        self._logger.info("Cleaning up")

        # Stop recording if in progress
        if self.recorder.is_recording():
            try:
                self.recorder.stop_recording()
            except Exception:
                self._logger.exception("Error stopping recorder")

        self._remove_pid_file()
        self._logger.info("Daemon stopped")


def run(*, refine: bool = False, instruction_keyword: str | None = None) -> NoReturn:
    """Run the STT daemon.

    Args:
        refine: Enable AI-based typo and grammar correction.
        instruction_keyword: Keyword to separate content from AI instructions.

    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)

    try:
        config = Config.from_env()
    except ValueError:
        logger.exception("Configuration error")
        sys.exit(1)

    daemon = STTDaemon(config, refine=refine, instruction_keyword=instruction_keyword)
    daemon.run()
