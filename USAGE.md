# STT Wayland Daemon - Quick Usage Guide

## Setup

1. **Configure API key**:
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Verify installation**:
   ```bash
   uv run stt-daemon --help
   # You should see an error about GEMINI_API_KEY (expected without .env configured)
   ```

## Running the Daemon

### Start daemon:
```bash
uv run stt-daemon
```

The daemon will:
- Create PID file at `$XDG_RUNTIME_DIR/stt-wayland.pid`
- Wait for SIGUSR1 signal to toggle recording
- Log to stdout

### Toggle recording:
```bash
# Send signal to toggle recording on/off
pkill -USR1 stt-daemon

# Or using PID file:
kill -USR1 $(cat $XDG_RUNTIME_DIR/stt-wayland.pid)
```

### Stop daemon:
```bash
# Graceful shutdown
pkill -TERM stt-daemon

# Or using PID file:
kill -TERM $(cat $XDG_RUNTIME_DIR/stt-wayland.pid)
```

## Sway Integration

Add to `~/.config/sway/config`:

```sway
# Auto-start STT daemon
exec uv run --directory /home/myakove/git/stt-wayland stt-daemon

# Bind Super+R to toggle recording
bindsym $mod+r exec pkill -USR1 stt-daemon
```

Reload Sway: `Super+Shift+C`

## Workflow Example

1. Press `Super+R` → Recording starts (LED/notification)
2. Speak: "Hello world, this is a test"
3. Press `Super+R` → Recording stops, transcription begins
4. Wait ~1-2 seconds → Text is automatically typed at cursor

## Troubleshooting

### Check if daemon is running:
```bash
ps aux | grep stt-daemon
cat $XDG_RUNTIME_DIR/stt-wayland.pid
```

### View logs:
```bash
# If running in terminal, logs go to stdout
# For background daemon, redirect to file:
uv run stt-daemon > /tmp/stt-daemon.log 2>&1 &
tail -f /tmp/stt-daemon.log
```

### Test audio recording:
```bash
# Record 5 seconds with pw-record
pw-record --rate 16000 --channels 1 /tmp/test.wav &
sleep 5
pkill pw-record

# Play back
pw-play /tmp/test.wav
```

### Test wtype:
```bash
# Focus a text editor, then run:
echo "Hello from wtype" | wtype -
```

## Development

### Run linter:
```bash
uvx ruff check src/
uvx ruff format src/
```

### Type check:
```bash
uvx --with python-dotenv --with google-genai mypy src/stt_wayland
```

### Test imports:
```bash
uv run test_import.py
```

## Architecture Overview

```
SIGUSR1 Signal
     ↓
State Machine: IDLE
     ↓ (toggle)
State Machine: RECORDING (pw-record starts)
     ↓ (toggle)
State Machine: TRANSCRIBING (Gemini API call)
     ↓ (auto)
State Machine: TYPING (wtype output)
     ↓ (auto)
State Machine: IDLE (ready for next recording)
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | - | Google Gemini API key |
| `STT_MODEL` | No | `gemini-2.5-flash` | Gemini model to use |
| `XDG_RUNTIME_DIR` | No | `/tmp` | Runtime directory for PID file |

## System Dependencies

### Fedora/RHEL:
```bash
sudo dnf install pipewire-utils wtype wl-clipboard
```

### Debian/Ubuntu:
```bash
sudo apt install pipewire-audio wtype wl-clipboard
```
