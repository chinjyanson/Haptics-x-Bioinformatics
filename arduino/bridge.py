"""
arduino/bridge.py
=================
Python communication layer for the Arduino Uno R3 peripheral.

Serial protocol (115200 baud, JSON lines) — see src/main.cpp for the C++ side.

  Python → Arduino:
    {"cmd":"start","task":1}                        — begin task N; ack returned
    {"cmd":"stop"}                                  — stop, zero both motors; ack returned
    {"cmd":"set_vibration","motor":1,"intensity":N} — set motor 1 PWM 0-255; no ack
    {"cmd":"set_vibration","motor":2,"intensity":N} — set motor 2 PWM 0-255; no ack
    {"cmd":"set_servo","angle":N}                   — set servo to angle 0-180°; no ack

  Arduino → Python:
    {"type":"ready"}                      — sent once on boot
    {"type":"encoder","delta":N}          — ticks since last report (50 ms intervals)
    {"type":"ack","cmd":"start","task":N} — start acknowledgement
    {"type":"ack","cmd":"stop"}           — stop acknowledgement

  Pin assignments: ENC_A=2(INT0)  ENC_B=3(INT1)  VIB1=5(PWM/T0)  VIB2=11(PWM/T2)  SERVO=6
"""

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import serial

# ── Defaults (override via ARDUINO_DEFAULT_PORT in main.py or --arduino-port CLI) ──
ARDUINO_DEFAULT_PORT: Optional[str] = "/dev/tty.usbmodem1401"
ARDUINO_DEFAULT_BAUD: int = 115200


# ── Dataclass ────────────────────────────────────────────────────────────────
@dataclass
class TimestampedArduinoEvent:
    """Single event received from the Arduino (encoder delta, button press, ack)."""
    timestamp: float       # Unix timestamp assigned on receipt
    event_type: str        # "encoder", "button", "ack", "error"
    data: Dict[str, Any]   # Parsed JSON fields minus 'type'


# ── Bridge class ─────────────────────────────────────────────────────────────
class ArduinoBridge:
    """
    Encapsulates all Python-side Arduino serial communication.

    Composed into SynchronizedCollector as self.arduino.
    Owns: serial.Serial handle, serial lock, ready event, connected/error state.
    Borrows (injected at construction): data_store, shared _lock, _stop_event,
    _start_recording.
    """

    def __init__(
        self,
        port: Optional[str],
        baud: int,
        data_store,                             # SynchronizedDataStore
        lock: threading.Lock,                   # shared collector._lock
        stop_event: threading.Event,            # shared collector._stop_event
        start_recording_event: threading.Event, # shared collector._record_event
        paused_event: threading.Event,          # collector._arduino_paused
    ) -> None:
        self.port = port
        self.baud = baud
        self._data_store = data_store
        self._lock = lock
        self._stop_event = stop_event
        self._start_recording = start_recording_event
        self.paused_event = paused_event

        self._serial: Optional[serial.Serial] = None
        self._serial_lock = threading.Lock()

        # Public state — read by SynchronizedCollector properties and GUI screens
        self.connected: bool = False
        self.error: Optional[str] = None
        self.ready = threading.Event()

    # ── Connection probe ──────────────────────────────────────────────────────
    def connect(self) -> bool:
        """
        Open the serial port and verify the Arduino is responding.
        Sleeps 2 s for the DTR-reset boot, then checks for the ready message.
        If the ready message was missed (Arduino already booted), sends a stop
        command and waits for the ack to confirm the link is live.
        Returns True if successful.
        """
        try:
            if not self.port:
                raise Exception("No Arduino port specified (set ARDUINO_DEFAULT_PORT or use --arduino-port)")
            ser = serial.Serial(port=self.port, baudrate=self.baud, timeout=1.0)
            time.sleep(2.0)  # Uno resets on serial open via DTR toggle
            # Do NOT reset_input_buffer — the {"type":"ready"} message may be waiting

            # Look for any JSON line (ready message or encoder delta) in the buffer
            deadline = time.time() + 1.0
            got_line = False
            while time.time() < deadline:
                if ser.in_waiting:
                    ser.readline()
                    got_line = True
                    break
                time.sleep(0.05)

            if not got_line:
                # Arduino was already running before we opened the port (no DTR reset),
                # so there's no ready message. Ping it with a stop command and wait for ack.
                ser.reset_input_buffer()
                ser.write(b'{"cmd":"stop"}\n')
                ser.flush()
                deadline = time.time() + 3.0
                while time.time() < deadline:
                    if ser.in_waiting:
                        ser.readline()
                        got_line = True
                        break
                    time.sleep(0.05)

            ser.close()
            if not got_line:
                raise Exception("No response from Arduino — check port/baud/sketch")

            self.connected = True
            self.error = None
            print(f"[Arduino] Connection test successful on {self.port}")
            return True
        except Exception as e:
            self.error = str(e)
            self.connected = False
            return False

    # ── Command writer ────────────────────────────────────────────────────────
    def send_command(self, cmd_dict: Dict[str, Any]) -> bool:
        """
        Thread-safe JSON write to the Arduino over serial.
        Returns True on success, False if the port is not open.
        """
        with self._serial_lock:
            if self._serial is None or not self._serial.is_open:
                return False
            try:
                line = json.dumps(cmd_dict) + "\n"
                self._serial.write(line.encode('utf-8'))
                self._serial.flush()
                return True
            except Exception as e:
                print(f"[Arduino] send_command error: {e}")
                return False

    # ── Collection thread ─────────────────────────────────────────────────────
    def collection_thread(self, duration: Optional[float]) -> None:
        """
        Thread target for Arduino bidirectional communication.

        Opens serial port ONCE and keeps it open for the full experiment.
        Pauses between sessions by draining the buffer, resumes on _start_recording.
        Sends {"cmd":"start"} on each session start and {"cmd":"stop"} only at shutdown.
        """
        try:
            if not self.port:
                raise Exception("port not set; cannot open serial port.")

            print(f"[Arduino] Opening {self.port} @ {self.baud} baud...")
            with self._serial_lock:
                self._serial = serial.Serial(
                    port=self.port,
                    baudrate=self.baud,
                    timeout=0.05,  # 50 ms non-blocking readline
                )
            time.sleep(2.0)  # Wait for Uno DTR-reset boot
            with self._serial_lock:
                self._serial.reset_input_buffer()

            self.connected = True
            print("[Arduino] Ready, waiting for synchronized start...")
            self.ready.set()

            total_events = 0

            # Outer loop — runs once per session slot; serial stays open throughout
            while not self._stop_event.is_set():
                # Between sessions: drain incoming bytes to prevent buffer overflow
                while not self._start_recording.is_set() and not self._stop_event.is_set():
                    with self._serial_lock:
                        if self._serial.in_waiting:
                            self._serial.read(self._serial.in_waiting)
                    time.sleep(0.05)

                if self._stop_event.is_set():
                    break

                self.send_command({"cmd": "start", "task": 1})
                print("[Arduino] Starting data collection...")
                event_count = 0

                # Inner recording loop
                while self._start_recording.is_set() and not self._stop_event.is_set():
                    with self._serial_lock:
                        raw = self._serial.readline()

                    if not raw:
                        continue

                    timestamp = time.time()
                    line_str = raw.decode('utf-8', errors='replace').strip()
                    if not line_str:
                        continue

                    try:
                        msg = json.loads(line_str)
                    except json.JSONDecodeError:
                        print(f"[Arduino] Non-JSON line: {line_str!r}")
                        continue

                    event_type = msg.get("type", "unknown")
                    data = {k: v for k, v in msg.items() if k != "type"}

                    with self._lock:
                        self._data_store.add_arduino_event(timestamp, event_type, data)
                    event_count += 1

                total_events += event_count
                print(f"[Arduino] Session paused. {event_count} events this session ({total_events} total).")
                self.paused_event.set()

            print(f"[Arduino] Collection complete. {total_events} total events.")

        except Exception as e:
            print(f"[Arduino] Error: {e}")
            self.error = str(e)
            self.ready.set()        # Unblock GUI on error
            self.paused_event.set() # Unblock end_session() on error

        finally:
            self.send_command({"cmd": "stop"})
            time.sleep(0.1)
            with self._serial_lock:
                if self._serial and self._serial.is_open:
                    try:
                        self._serial.close()
                    except Exception as e:
                        print(f"[Arduino] Error closing serial: {e}")
                self._serial = None
            self.connected = False

    # ── Teardown ──────────────────────────────────────────────────────────────
    def disconnect(self) -> None:
        """Forcefully close the serial port. Called from disconnect_devices()."""
        with self._serial_lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.write(b'{"cmd":"stop"}\n')
                    self._serial.flush()
                    self._serial.close()
                except Exception:
                    pass
            self._serial = None
        self.connected = False
