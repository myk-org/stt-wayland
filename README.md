# stt-wayland

Wayland-native Speech-to-Text daemon using Google Gemini API.

## Features

- **Wayland-native**: Uses `wtype` for text input and `wl-copy` for clipboard
- **PipeWire/PulseAudio**: Supports both `pw-record` and `parecord` for audio recording
- **Signal-based control**: Toggle recording via `SIGUSR1` signal
- **State machine**: Clean state transitions (IDLE → RECORDING → TRANSCRIBING → TYPING)
- **Desktop notifications**: Visual feedback for each state transition
- **Google Gemini**: Fast transcription using Gemini 2.0 Flash

## Requirements

### Python

- **Python 3.14+** is required

### Package Manager

- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager and installer

### System Tools

- **wtype** - Wayland text input simulation
- **wl-clipboard** - Wayland clipboard utilities (provides `wl-copy`)
- **pipewire-utils** or **pulseaudio-utils** - Audio recording (`pw-record` or `parecord`)
- **libnotify** - Desktop notification support (provides `notify-send`)

See the [Dependencies](#dependencies) section below for installation commands.

## Installation

### Clone the repository

```bash
git clone https://github.com/myk-org/stt-wayland.git
cd stt-wayland
```

### Development (editable)

```bash
uv sync
```

### System-wide

```bash
uv tool install .
```

This installs `stt-daemon` globally so it's available in your PATH.

## Updating

### Development

```bash
git pull
uv sync
```

### System-wide

```bash
cd stt-wayland
git pull
uv tool install . -U
```

## Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
STT_MODEL=gemini-2.5-flash  # Optional, this is the default
```

Or export the environment variable:

```bash
export GEMINI_API_KEY=your_api_key_here
```

## Dependencies

### System packages (Fedora/RHEL)

```bash
sudo dnf install pipewire-utils wtype wl-clipboard libnotify
```

### System packages (Debian/Ubuntu)

```bash
sudo apt install pipewire-audio wtype wl-clipboard libnotify-bin
```

## Usage

### Start the daemon

```bash
stt-daemon
```

The daemon will:

1. Create a PID file at `$XDG_RUNTIME_DIR/stt-wayland.pid`
2. Wait for SIGUSR1 signal to toggle recording

### Toggle recording

Send SIGUSR1 to the daemon process:

```bash
pkill -USR1 stt-daemon
```

Or using the PID file:

```bash
kill -USR1 $(cat $XDG_RUNTIME_DIR/stt-wayland.pid)
```

### Stop the daemon

```bash
pkill -TERM stt-daemon
```

Or:

```bash
kill -TERM $(cat $XDG_RUNTIME_DIR/stt-wayland.pid)
```

## Workflow

1. **First SIGUSR1**: Start recording (IDLE → RECORDING) - notification shown
2. **Second SIGUSR1**: Stop recording and transcribe (RECORDING → TRANSCRIBING) - notification shown
3. **Auto-type**: Transcribed text is automatically typed using `wtype` - success notification shown
4. **Return to IDLE**: Ready for next recording

Desktop notifications appear at each stage to provide visual feedback on the current state.

## State Machine

```
IDLE ──SIGUSR1──→ RECORDING ──SIGUSR1──→ TRANSCRIBING ──auto──→ TYPING ──auto──→ IDLE
  ↑                                                                              │
  └──────────────────────────────────────────────────────────────────────────────┘
```

## Sway/SwayFX Integration

Add to your Sway config:

```sway
# Start the daemon in background (runs in IDLE state, waiting for signals)
exec env GEMINI_API_KEY="<YOUR_GEMINI_API_KEY>" $HOME/.local/bin/stt-daemon


# Press Super+R to toggle: first press starts recording, second press stops and transcribes
bindsym $mod+r exec pkill -USR1 stt-daemon
```

## Architecture

```
src/stt_wayland/
├── daemon.py           # Main daemon logic, signal handling, PID file
├── state_machine.py    # Thread-safe state machine
├── config.py           # Environment configuration
├── audio/
│   └── recorder.py     # pw-record/parecord wrapper
├── transcription/
│   └── gemini.py       # Google Gemini API client
└── output/
    ├── wtype.py        # wtype text output
    └── clipboard.py    # wl-copy fallback
```

## Development

### Install with test dependencies

```bash
uv sync --extra test
```

### Running Tests

```bash
tox
# or
uv run pytest
```

### Format code

```bash
uv run ruff format src/
```

### Lint

```bash
uv run ruff check src/
```

### Type check

```bash
uv run mypy src/
```

## License

MIT
