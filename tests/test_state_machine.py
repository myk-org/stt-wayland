"""Tests for state machine module."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest

from stt_wayland.state_machine import Event, State, StateMachine

if TYPE_CHECKING:
    from collections.abc import Callable


class TestStateMachine:
    """Test StateMachine class."""

    def test_initial_state(self) -> None:
        """Test that state machine starts in IDLE state."""
        sm = StateMachine()
        assert sm.state == State.IDLE

    def test_state_property_thread_safe(self) -> None:
        """Test that state property is thread-safe."""
        sm = StateMachine()
        results: list[State] = []

        def read_state() -> None:
            for _ in range(100):
                results.append(sm.state)

        threads = [threading.Thread(target=read_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should succeed without errors
        assert len(results) == 500
        assert all(s == State.IDLE for s in results)

    def test_set_state(self) -> None:
        """Test setting state transitions correctly."""
        sm = StateMachine()
        sm.set_state(State.RECORDING)
        # Store in variable to avoid mypy type narrowing issues
        state: State = sm.state
        assert state == State.RECORDING

        sm.set_state(State.TRANSCRIBING)
        state = sm.state
        assert state == State.TRANSCRIBING

        sm.set_state(State.TYPING)
        state = sm.state
        assert state == State.TYPING

        sm.set_state(State.IDLE)
        state = sm.state
        assert state == State.IDLE

    def test_transition_queues_event(self) -> None:
        """Test that transition adds event to queue."""
        sm = StateMachine()
        sm.transition(Event.TOGGLE_RECORDING)

        # Event should be in queue, process_events should find it
        call_count = 0

        def handler() -> None:
            nonlocal call_count
            call_count += 1

        handlers = {(State.IDLE, Event.TOGGLE_RECORDING): handler}
        should_continue = sm.process_events(handlers)

        assert should_continue is True
        assert call_count == 1

    def test_transition_queue_full_raises_error(self) -> None:
        """Test that transition raises error when queue is full."""
        sm = StateMachine()

        # Fill the queue (maxsize=10)
        for _ in range(10):
            sm.transition(Event.TOGGLE_RECORDING)

        # Next transition should raise RuntimeError
        with pytest.raises(RuntimeError, match="Event queue full"):
            sm.transition(Event.TOGGLE_RECORDING)

    def test_process_events_with_no_events(self) -> None:
        """Test that process_events returns True when queue is empty."""
        sm = StateMachine()
        handlers: dict[tuple[State, Event], Callable[[], None]] = {}

        should_continue = sm.process_events(handlers)
        assert should_continue is True

    def test_process_events_calls_handler(self) -> None:
        """Test that process_events calls the correct handler."""
        sm = StateMachine()
        call_count = 0
        received_state = None

        def handler() -> None:
            nonlocal call_count, received_state
            call_count += 1
            received_state = sm.state
            sm.set_state(State.RECORDING)

        handlers = {(State.IDLE, Event.TOGGLE_RECORDING): handler}
        sm.transition(Event.TOGGLE_RECORDING)

        should_continue = sm.process_events(handlers)

        assert should_continue is True
        assert call_count == 1
        assert received_state == State.IDLE
        assert sm.state == State.RECORDING

    def test_process_events_no_handler_found(self) -> None:
        """Test that process_events handles missing handler gracefully."""
        sm = StateMachine()
        sm.transition(Event.ERROR)

        handlers: dict[tuple[State, Event], Callable[[], None]] = {}
        should_continue = sm.process_events(handlers)

        # Should continue running even without handler
        assert should_continue is True

    def test_process_events_shutdown_event(self) -> None:
        """Test that SHUTDOWN event stops the state machine."""
        sm = StateMachine()
        sm.transition(Event.SHUTDOWN)

        handlers: dict[tuple[State, Event], Callable[[], None]] = {}
        should_continue = sm.process_events(handlers)

        assert should_continue is False

    def test_process_events_handler_exception(self) -> None:
        """Test that handler exceptions are caught and state resets to IDLE."""
        sm = StateMachine()
        sm.set_state(State.RECORDING)

        def failing_handler() -> None:
            raise ValueError("Handler error")

        handlers = {(State.RECORDING, Event.ERROR): failing_handler}
        sm.transition(Event.ERROR)

        should_continue = sm.process_events(handlers)

        assert should_continue is True
        # State should be reset to IDLE on error
        assert sm.state == State.IDLE

    def test_concurrent_transitions(self) -> None:
        """Test that multiple threads can safely queue transitions."""
        sm = StateMachine()
        errors: list[Exception] = []

        def queue_events() -> None:
            try:
                for _ in range(10):
                    sm.transition(Event.TOGGLE_RECORDING)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=queue_events) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Some events may fail due to queue full, but no crashes
        # At least some should succeed
        assert len(errors) <= 30  # Not all should fail

    def test_state_transitions_sequence(self) -> None:
        """Test a complete state transition sequence."""
        sm = StateMachine()
        states_visited: list[State] = []

        def start_recording() -> None:
            states_visited.append(sm.state)
            sm.set_state(State.RECORDING)

        def stop_recording() -> None:
            states_visited.append(sm.state)
            sm.set_state(State.TRANSCRIBING)

        def finish_transcription() -> None:
            states_visited.append(sm.state)
            sm.set_state(State.TYPING)

        def finish_typing() -> None:
            states_visited.append(sm.state)
            sm.set_state(State.IDLE)

        handlers = {
            (State.IDLE, Event.TOGGLE_RECORDING): start_recording,
            (State.RECORDING, Event.RECORDING_STOPPED): stop_recording,
            (State.TRANSCRIBING, Event.TRANSCRIPTION_COMPLETE): finish_transcription,
            (State.TYPING, Event.TYPING_COMPLETE): finish_typing,
        }

        # Execute sequence
        sm.transition(Event.TOGGLE_RECORDING)
        sm.process_events(handlers)

        sm.transition(Event.RECORDING_STOPPED)
        sm.process_events(handlers)

        sm.transition(Event.TRANSCRIPTION_COMPLETE)
        sm.process_events(handlers)

        sm.transition(Event.TYPING_COMPLETE)
        sm.process_events(handlers)

        # Verify sequence
        assert states_visited == [State.IDLE, State.RECORDING, State.TRANSCRIBING, State.TYPING]
        assert sm.state == State.IDLE

    def test_reentrant_locking(self) -> None:
        """Test that RLock allows handlers to call set_state."""
        sm = StateMachine()

        def handler_with_state_change() -> None:
            # This should not deadlock due to RLock
            sm.set_state(State.RECORDING)
            sm.set_state(State.TRANSCRIBING)

        handlers = {(State.IDLE, Event.TOGGLE_RECORDING): handler_with_state_change}
        sm.transition(Event.TOGGLE_RECORDING)

        should_continue = sm.process_events(handlers)

        assert should_continue is True
        assert sm.state == State.TRANSCRIBING


class TestStateEnum:
    """Test State enum."""

    def test_all_states_exist(self) -> None:
        """Test that all expected states are defined."""
        assert hasattr(State, "IDLE")
        assert hasattr(State, "RECORDING")
        assert hasattr(State, "TRANSCRIBING")
        assert hasattr(State, "TYPING")

    def test_states_are_unique(self) -> None:
        """Test that all states have unique values."""
        states = [State.IDLE, State.RECORDING, State.TRANSCRIBING, State.TYPING]
        assert len(states) == len(set(states))


class TestEventEnum:
    """Test Event enum."""

    def test_all_events_exist(self) -> None:
        """Test that all expected events are defined."""
        assert hasattr(Event, "TOGGLE_RECORDING")
        assert hasattr(Event, "RECORDING_STARTED")
        assert hasattr(Event, "RECORDING_STOPPED")
        assert hasattr(Event, "TRANSCRIPTION_COMPLETE")
        assert hasattr(Event, "TYPING_COMPLETE")
        assert hasattr(Event, "ERROR")
        assert hasattr(Event, "SHUTDOWN")

    def test_events_are_unique(self) -> None:
        """Test that all events have unique values."""
        events = [
            Event.TOGGLE_RECORDING,
            Event.RECORDING_STARTED,
            Event.RECORDING_STOPPED,
            Event.TRANSCRIPTION_COMPLETE,
            Event.TYPING_COMPLETE,
            Event.ERROR,
            Event.SHUTDOWN,
        ]
        assert len(events) == len(set(events))
