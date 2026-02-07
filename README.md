# stt-wayland

Wayland-native Speech-to-Text daemon using Google Gemini API.

## Features

- **Wayland-native**: Uses `wtype` for text input and `wl-copy` for clipboard
- **PipeWire/PulseAudio**: Supports both `pw-record` and `parecord` for audio recording
- **Signal-based control**: Toggle recording via `SIGUSR1` signal
- **State machine**: Clean state transitions (IDLE → RECORDING → TRANSCRIBING → TYPING)
- **Desktop notifications**: Visual feedback for each state transition
- **Google Gemini**: Fast transcription using Gemini 2.0 Flash
- **AI refinement**: Optional typo and grammar correction via `--refine`
- **Inline AI instructions**: Speak custom AI instructions using a keyword separator
- **Voice query mode**: Ask AI questions and get answers typed out via `--ask-keyword`

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

## Command-line Options

### `--refine`

Enable AI-based typo and grammar correction on transcribed text.

```bash
stt-daemon --refine
```

### `--format`

Enable plain-text formatting of refined output. Requires `--refine`.

When the spoken content contains natural structure (lists, multiple points, distinct topics), the AI will format the output with line breaks, numbered lists, and bulleted items. Multi-line formatted output is pasted via clipboard (`wl-copy` + Ctrl+V) to avoid triggering Enter key presses in the active window.

```bash
stt-daemon --refine --format
```

**Example:**
- You say: "I have a few things to say. This is okay, this is bad, and this needs approval."
- Output:
  ```
  I have a few things to say:
  1. This is okay
  2. This is bad
  3. This needs approval
  ```

### `--instruction-keyword KEYWORD`

Enable inline AI instructions by specifying a separator keyword. When you speak the keyword, everything after it is treated as an instruction for the AI to apply to the text before the keyword.

```bash
stt-daemon --instruction-keyword boom
```

**Example:**
- You say: "Hello world boom refine as a poem"
- The system parses: content = "Hello world", instruction = "refine as a poem"
- AI applies your instruction and outputs the processed text

**Notes:**
- Keyword matching is case-insensitive ("boom", "BOOM", "Boom" all work)
- If no keyword is detected in your speech, the text is output as-is
- You can combine with `--refine`: `stt-daemon --refine --instruction-keyword boom`

### `--ask-keyword KEYWORD`

Enable voice query mode by specifying a trigger keyword. When your speech STARTS with the keyword, the rest is sent to the AI as a question, and the AI's ANSWER is typed out instead of your original words.

```bash
stt-daemon --ask-keyword hey
```

**Example:**
- You say: "hey what is the capital of France"
- The system types: "The capital of France is Paris"

**Notes:**
- The keyword must be at the START of your speech
- Keyword matching is case-insensitive ("hey", "HEY", "Hey" all work)
- Word boundaries are respected ("heyo" will NOT trigger for keyword "hey")
- If keyword is spoken alone without a query, nothing is typed
- You can combine with other flags: `stt-daemon --ask-keyword hey --instruction-keyword boom`
- When both keywords are configured, `--ask-keyword` takes precedence (checked first)

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
3. **Auto-type**: Transcribed text is typed via `wtype` (or pasted via clipboard for multi-line output) - success notification shown
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
# Basic usage
exec env GEMINI_API_KEY="<YOUR_KEY>" $HOME/.local/bin/stt-daemon

# With refinement
exec env GEMINI_API_KEY="<YOUR_KEY>" $HOME/.local/bin/stt-daemon --refine

# With refinement and formatting
exec env GEMINI_API_KEY="<YOUR_KEY>" $HOME/.local/bin/stt-daemon --refine --format

# With inline AI instructions (say "boom" to give AI instructions)
exec env GEMINI_API_KEY="<YOUR_KEY>" $HOME/.local/bin/stt-daemon --instruction-keyword boom

# With voice query mode (say "hey" at start to ask AI questions)
exec env GEMINI_API_KEY="<YOUR_KEY>" $HOME/.local/bin/stt-daemon --ask-keyword hey

# Combined: voice queries + inline instructions
exec env GEMINI_API_KEY="<YOUR_KEY>" $HOME/.local/bin/stt-daemon --ask-keyword hey --instruction-keyword boom

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
    ├── wtype.py        # wtype text output and clipboard paste
    └── clipboard.py    # wl-copy clipboard operations
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
