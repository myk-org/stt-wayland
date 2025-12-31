# Test Suite for stt-wayland

This directory contains the test suite for the stt-wayland project.

## Test Organization

- **test_state_machine.py** - Tests for the state machine module
  - State transitions and sequencing
  - Thread safety with concurrent access
  - Event queue management
  - Error handling and recovery

- **test_config.py** - Tests for configuration management
  - Loading from environment variables
  - Loading from .env files
  - Validation of required fields
  - Default value handling

- **test_wtype.py** - Tests for wtype text output
  - Input validation (text length, null bytes)
  - Unicode handling
  - Error handling
  - Subprocess management

## Running Tests

### Install Test Dependencies

```bash
uv pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_state_machine.py
pytest tests/test_config.py
pytest tests/test_wtype.py
```

### Run Specific Test

```bash
pytest tests/test_state_machine.py::TestStateMachine::test_concurrent_transitions
```

### Run with Coverage

```bash
pytest --cov=stt_wayland --cov-report=html
```

Coverage report will be generated in `htmlcov/index.html`.

### Run with Verbose Output

```bash
pytest -v
```

### Run in Parallel (if pytest-xdist is installed)

```bash
pytest -n auto
```

## Test Design Principles

1. **Test Pyramid** - Many unit tests, fewer integration tests
2. **Arrange-Act-Assert** - Clear test structure
3. **Test Behavior, Not Implementation** - Focus on public APIs
4. **No Flakiness** - All tests are deterministic
5. **Fast Feedback** - Tests run quickly

## Coverage Goals

- **State Machine**: 100% - Critical component with complex threading
- **Config**: 90%+ - Configuration edge cases covered
- **wtype**: 90%+ - Input validation thoroughly tested

## Notes

- Tests use mocking for external dependencies (subprocess, file system)
- Environment variables are reset before each test (see conftest.py)
- No actual system calls are made during tests
- Thread safety tests use controlled concurrency
