#!/usr/bin/env zsh
# run.sh — BCI FYP launcher
# Optionally flashes the Arduino, lets you pick the audio output device,
# then starts main.py.

set -e
cd "$(dirname "$0")"

PIO="$HOME/.platformio/penv/bin/pio"
PYTHON="$(pwd)/venv/bin/python3"
ARDUINO_DIR="$(pwd)/arduino"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo ""
echo "${BOLD}${CYAN}╔══════════════════════════════════════╗${NC}"
echo "${BOLD}${CYAN}║        BCI FYP Experiment Runner     ║${NC}"
echo "${BOLD}${CYAN}╚══════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Flash Arduino? ────────────────────────────────────────────────────
echo "${BOLD}Flash Arduino firmware?${NC} (y/N): \c"
read FLASH
if [[ "$FLASH" =~ ^[Yy]$ ]]; then
    echo ""
    echo "${CYAN}Flashing Arduino...${NC}"
    "$PIO" run --target upload --project-dir "$ARDUINO_DIR"
    echo "${GREEN}Flash complete.${NC}"
else
    echo "Skipping flash."
fi

echo ""

# ── Step 2: Select audio output device ───────────────────────────────────────
echo "${BOLD}Available audio output devices:${NC}"
echo ""
"$PYTHON" - <<'PYEOF'
import sounddevice as sd
devices = sd.query_devices()
for i, d in enumerate(devices):
    if d['max_output_channels'] > 0:
        marker = ' <-- DEFAULT' if i == sd.default.device[1] else ''
        print(f"  [{i}] {d['name']}{marker}")
PYEOF

echo ""
echo "${BOLD}Enter audio device index (leave blank for system default):${NC} \c"
read AUDIO_DEVICE

echo ""

# ── Step 3: Launch experiment ─────────────────────────────────────────────────
echo "${CYAN}Starting experiment...${NC}"
echo ""
if [[ -z "$AUDIO_DEVICE" ]]; then
    echo "Using system default audio device."
    "$PYTHON" main.py
else
    echo "Using device $AUDIO_DEVICE."
    "$PYTHON" main.py --audio-out-device "$AUDIO_DEVICE"
fi
