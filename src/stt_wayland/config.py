"""Configuration management for STT daemon."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

from dotenv import load_dotenv

if TYPE_CHECKING:
    from typing import Final

ERR_NO_API_KEY: Final[str] = "GEMINI_API_KEY environment variable is required. Set it in .env or export it."
LIVE_MODEL_DEFAULT: Final[str] = "gemini-3.1-flash-live-preview"
MAX_RECORDING_DURATION_DEFAULT: Final[float] = 300.0  # 5 minutes


@dataclass(frozen=True, slots=True)
class Config:
    """STT daemon configuration."""

    api_key: str
    model: str
    runtime_dir: Path
    max_recording_duration: float

    @classmethod
    def from_env(cls) -> Self:
        """Load configuration from environment variables.

        Raises:
            ValueError: If GEMINI_API_KEY is not set.

        """
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            msg = ERR_NO_API_KEY
            raise ValueError(msg)

        model = os.getenv("STT_MODEL", "gemini-2.5-flash")

        runtime_dir_str = os.getenv("XDG_RUNTIME_DIR")
        runtime_dir = Path("/tmp") if not runtime_dir_str else Path(runtime_dir_str)  # noqa: S108

        max_recording_str = os.getenv("STT_MAX_RECORDING")
        if max_recording_str:
            try:
                max_recording_duration = float(max_recording_str)
            except ValueError:
                max_recording_duration = MAX_RECORDING_DURATION_DEFAULT
            if max_recording_duration <= 0:
                max_recording_duration = MAX_RECORDING_DURATION_DEFAULT
        else:
            max_recording_duration = MAX_RECORDING_DURATION_DEFAULT

        return cls(
            api_key=api_key,
            model=model,
            runtime_dir=runtime_dir,
            max_recording_duration=max_recording_duration,
        )

    @property
    def pid_file(self) -> Path:
        """Path to PID file."""
        return self.runtime_dir / "stt-wayland.pid"
