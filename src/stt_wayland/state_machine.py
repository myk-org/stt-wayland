"""State machine for STT daemon."""

from __future__ import annotations

import logging
from enum import Enum, auto
from queue import Empty, Full, Queue
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class State(Enum):
    """Daemon states."""

    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()
    TYPING = auto()


class Event(Enum):
    """State transition events."""

    TOGGLE_RECORDING = auto()
    RECORDING_STARTED = auto()
    RECORDING_STOPPED = auto()
    TRANSCRIPTION_COMPLETE = auto()
    TYPING_COMPLETE = auto()
    ERROR = auto()
    SHUTDOWN = auto()


class StateMachine:
    """Thread-safe state machine for STT daemon."""

    def __init__(self) -> None:
        """Initialize state machine."""
        self._state = State.IDLE
        # Use RLock to allow re-entrant locking (handlers can call set_state)
        self._lock = RLock()
        # Bounded queue to prevent unbounded memory growth
        self._event_queue: Queue[Event] = Queue(maxsize=10)
        self._logger = logging.getLogger(__name__)

    @property
    def state(self) -> State:
        """Get current state (thread-safe)."""
        with self._lock:
            return self._state

    def transition(self, event: Event) -> None:
        """Request a state transition.

        Args:
            event: Event triggering the transition.

        Raises:
            RuntimeError: If event queue is full.

        """
        try:
            self._event_queue.put(event, timeout=1.0)
        except Full as e:
            self._logger.exception("Event queue full, dropping event %s", event.name)
            msg = f"Event queue full, cannot process {event.name}"
            raise RuntimeError(msg) from e

    def process_events(self, handlers: dict[tuple[State, Event], Callable[[], None]]) -> bool:
        """Process pending events and execute handlers.

        Args:
            handlers: Map of (state, event) -> handler function.

        Returns:
            True if should continue running, False if shutdown requested.

        Note:
            Handlers can safely call set_state() due to RLock usage.

        """
        try:
            event = self._event_queue.get(timeout=0.1)
        except Empty:
            return True

        with self._lock:
            current = self._state
            key = (current, event)

            self._logger.info("Processing event %s in state %s", event.name, current.name)

            if event == Event.SHUTDOWN:
                self._logger.info("Shutdown event received")
                return False

            handler = handlers.get(key)
            if handler:
                try:
                    handler()
                except Exception:
                    self._logger.exception("Handler error")
                    # On error, return to IDLE
                    self._state = State.IDLE
            else:
                self._logger.warning("No handler for event %s in state %s", event.name, current.name)

        return True

    def set_state(self, new_state: State) -> None:
        """Set new state (called by handlers).

        Args:
            new_state: Target state.

        """
        with self._lock:
            old_state = self._state
            self._state = new_state
            self._logger.info("State transition: %s → %s", old_state.name, new_state.name)
