"""
haptics.py — Real-time haptic feedback controller for BCI FYP.

Manages auditory (stereo-panned sine tone) and vibration motor feedback
during experiment sessions, guided by rotary encoder position vs. target.

Session modes
-------------
  "auditory"      — stereo-panned 440 Hz sine tone via USB-C audio output
                    left channel louder → turn CCW; right → turn CW
  "vibrations"    — motor 1 = CCW cue, motor 2 = CW cue (PWM intensity ∝ error)
  "shape_changing" — encoder tracking + task-error markers only;
                    servo motor integration is a future stub

Configuration constants can be overridden by importing and reassigning before
constructing a HapticsController, or by editing this file directly.
"""

import json
import threading
import time
from typing import Dict, List, Optional

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
HAPTIC_TARGETS_FILE   = "haptic_targets.json"  # Edit this file between experiments
HAPTIC_MAX_ERROR      = 10000   # Ticks at which feedback reaches maximum intensity
HAPTIC_DEAD_ZONE      = 5     # Ticks of silence around the target (±)
HAPTIC_TONE_HZ        = 440   # Sine tone frequency for auditory feedback (Hz)
HAPTIC_MOTOR_INTERVAL = 0   # Minimum seconds between vibration motor command updates


# ── Target loader ─────────────────────────────────────────────────────────────

def load_haptic_targets(tasks_per_device: int = 5) -> Dict[str, List[int]]:
    """
    Load per-session target tick positions from haptic_targets.json.

    Returns a dict mapping session name → list of target ticks (one per task).
    Missing keys or a missing file fall back to all-zeros so the experiment
    can still run without any target file.
    """
    defaults: Dict[str, List[int]] = {
        "Auditory":       [0] * tasks_per_device,
        "Vibrations":     [0] * tasks_per_device,
        "Shape Changing": [0] * tasks_per_device,
    }
    try:
        with open(HAPTIC_TARGETS_FILE) as f:
            loaded = json.load(f)
        for key in defaults:
            if key in loaded:
                defaults[key] = loaded[key]
    except FileNotFoundError:
        print(f"[Haptics] {HAPTIC_TARGETS_FILE} not found — using zero targets.")
    except Exception as e:
        print(f"[Haptics] Could not load {HAPTIC_TARGETS_FILE}: {e} — using zero targets.")
    return defaults


# ── Controller ────────────────────────────────────────────────────────────────

class HapticsController:
    """
    Real-time haptic feedback controller.

    Usage:
        haptics = HapticsController(arduino_bridge, "auditory", audio_out_device=2)
        haptics.start()
        haptics.set_target(10)      # new task: target = +10 ticks from here
        haptics.update_encoder(+3)  # call for each encoder delta received
        haptics.stop()              # silence everything at session end
    """

    def __init__(self, arduino_bridge, session_mode: str,
                 audio_out_device: Optional[int] = None):
        """
        Args:
            arduino_bridge:   ArduinoBridge instance (or None when Arduino disabled).
            session_mode:     "auditory" | "vibrations" | "shape_changing"
            audio_out_device: sounddevice output device index (None = system default).
        """
        self._arduino = arduino_bridge
        self._mode = session_mode
        self._audio_out_device = audio_out_device

        self.target: int = 0              # Target tick offset for the current task
        self.current_position: int = 0    # Accumulated encoder ticks since set_target()

        # Audio state (auditory mode only)
        self._audio_stream = None
        self._left_gain: float = 0.0
        self._right_gain: float = 0.0
        self._audio_phase: float = 0.0   # Continuous sine phase accumulator
        self._audio_lock = threading.Lock()

        # Motor throttle (vibrations mode only)
        self._last_motor_update: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the audio output stream (auditory mode only)."""
        if self._mode == "auditory":
            import sounddevice as sd
            self._audio_stream = sd.OutputStream(
                samplerate=44100,
                channels=2,
                dtype='float32',
                blocksize=256,
                device=self._audio_out_device,
                callback=self._audio_callback,
            )
            self._audio_stream.start()
            print(f"[Haptics] Audio output stream started (device={self._audio_out_device})")

    def stop(self) -> None:
        """Silence audio and zero both vibration motors."""
        with self._audio_lock:
            self._left_gain = 0.0
            self._right_gain = 0.0
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception as e:
                print(f"[Haptics] Error closing audio stream: {e}")
            self._audio_stream = None
        if self._arduino is not None:
            self._arduino.send_command({"cmd": "set_vibration", "motor": 1, "intensity": 0})
            self._arduino.send_command({"cmd": "set_vibration", "motor": 2, "intensity": 0})

    def set_target(self, target_ticks: int) -> None:
        """Set a new target for the current task and reset the encoder position to 0."""
        self.target = target_ticks
        self.current_position = 0
        self._apply_feedback()

    def update_encoder(self, delta: int) -> None:
        """Accumulate an encoder delta and update feedback output accordingly."""
        self.current_position += delta
        self._apply_feedback()

    # ── Internal dispatchers ──────────────────────────────────────────────────

    def _apply_feedback(self) -> None:
        error = self.current_position - self.target
        if self._mode == "auditory":
            self._update_audio(error)
        elif self._mode == "vibrations":
            self._update_motors(error)
        elif self._mode == "shape_changing":
            self._update_shape(error)

    def _update_audio(self, error: int) -> None:
        """Update left/right gains for the stereo sine tone based on encoder error."""
        if abs(error) <= HAPTIC_DEAD_ZONE:
            with self._audio_lock:
                self._left_gain = 0.0
                self._right_gain = 0.0
            return
        volume = min(abs(error) / HAPTIC_MAX_ERROR, 1.0)
        # error < 0 → need to go CW  → left channel only
        # error > 0 → need to go CCW → right channel only
        if error < 0:
            left, right = volume, 0.0
        else:
            left, right = 0.0, volume
        with self._audio_lock:
            self._left_gain  = left
            self._right_gain = right

    def _update_motors(self, error: int) -> None:
        """Send vibration intensity to motors, throttled to HAPTIC_MOTOR_INTERVAL."""
        now = time.time()
        if now - self._last_motor_update < HAPTIC_MOTOR_INTERVAL:
            return
        self._last_motor_update = now

        if self._arduino is None:
            return

        volume = min(abs(error) / HAPTIC_MAX_ERROR, 1.0)
        if error < 0:       # Need to go CCW → motor 1
            m1, m2 = int(volume * 255), 0
        elif error > 0:     # Need to go CW  → motor 2
            m1, m2 = 0, int(volume * 255)
        else:
            m1 = m2 = 0

        self._arduino.send_command({"cmd": "set_vibration", "motor": 1, "intensity": m1})
        self._arduino.send_command({"cmd": "set_vibration", "motor": 2, "intensity": m2})

    def _update_shape(self, error: int) -> None:
        """Stub for shape-changing device (servo motor — future implementation)."""
        pass  # TODO: send servo position command when hardware is integrated

    # ── Audio callback (runs in sounddevice audio thread) ─────────────────────

    def _audio_callback(self, outdata, frames, time_info, status):
        """Generate a stereo-panned sine tone at HAPTIC_TONE_HZ."""
        with self._audio_lock:
            left_gain  = self._left_gain
            right_gain = self._right_gain

        omega = 2.0 * np.pi * HAPTIC_TONE_HZ / 44100.0
        phases = self._audio_phase + omega * np.arange(frames)
        sine = np.sin(phases).astype(np.float32)
        self._audio_phase = (self._audio_phase + omega * frames) % (2.0 * np.pi)

        outdata[:, 0] = sine * left_gain
        outdata[:, 1] = sine * right_gain
