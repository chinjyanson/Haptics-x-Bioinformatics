"""
Synchronized Multi-Device Data Collection System
Collects data from Muse 2 EEG and Polar H10 HR monitor simultaneously
with timestamp synchronization for cognitive load analysis.
"""

import asyncio
import random
import time
import json
import threading
import signal
import sys
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import numpy as np

# Import device classes
from muse import MuseBrainFlowProcessor
from polar import PolarH10, HEART_RATE_MEASUREMENT_UUID
from esense import ESenseGSR
from arduino.bridge import ArduinoBridge, TimestampedArduinoEvent, ARDUINO_DEFAULT_PORT, ARDUINO_DEFAULT_BAUD

# ── Experiment configuration ──────────────────────────────────────────────────
TASKS_PER_DEVICE = 5   # Number of tasks per device session (change here to adjust)

# ── Device enable flags — set to False to run without a device ────────────────
USE_MUSE    = True
USE_POLAR   = False
USE_GSR     = False
USE_ARDUINO = True

# ── Baseline recording flag — set to False to skip baseline at session start ──
RUN_BASELINE = True

from haptics import HapticsController, load_haptic_targets
import baseline
from gui import (
    show_participant_screen, show_connection_screen, show_consent_screen,
    show_reconnect_screen, show_countdown_screen, show_experiment_screen,
    show_red_circle_count_screen, show_nasa_tlx_screen, show_device_swap_screen, show_completion_screen,
    show_error_popup, show_baseline_screen, show_calibration_screen,
)

@dataclass
class TimestampedEEGSample:
    """Single EEG sample with synchronized timestamp"""
    timestamp: float  # Unix timestamp (seconds since epoch)
    channels: Dict[str, float]  # Channel name -> value mapping

@dataclass
class TimestampedHRSample:
    """Single heart rate sample with synchronized timestamp"""
    timestamp: float  # Unix timestamp
    heart_rate: int  # bpm
    rr_intervals: List[float]  # RR intervals in ms (may be empty)

@dataclass
class TimestampedGSRSample:
    """Single GSR sample with synchronized timestamp"""
    timestamp: float  # Unix timestamp
    raw_audio: float  # Raw audio amplitude
    filtered_signal: float  # Lowpass filtered signal
    gsr_uS: float  # Calibrated conductance in microsiemens

@dataclass
class TaskMarker:
    """Task transition marker with synchronized timestamp"""
    timestamp: float  # Unix timestamp
    task_number: int  # Task index that just ended
    event: str = "task_end"
    extra: Dict[str, Any] = field(default_factory=dict)  # Optional metadata (e.g. encoder_error)

@dataclass
class SynchronizedDataStore:
    """
    Central data store for synchronized multi-device data collection.
    Uses a common time reference for all data points.
    """
    session_start: float = field(default_factory=time.time)
    eeg_data: List[TimestampedEEGSample] = field(default_factory=list)
    hr_data: List[TimestampedHRSample] = field(default_factory=list)
    gsr_data: List[TimestampedGSRSample] = field(default_factory=list)
    task_markers: List[TaskMarker] = field(default_factory=list)
    arduino_data: List[TimestampedArduinoEvent] = field(default_factory=list)

    # Metadata
    muse_sampling_rate: int = 256
    muse_channels: List[str] = field(default_factory=lambda: ['TP9', 'AF7', 'AF8', 'TP10'])
    gsr_sampling_rate: int = 50  # Downsampled rate

    def add_eeg_sample(self, timestamp: float, channel_values: Dict[str, float]):
        """Add a timestamped EEG sample"""
        sample = TimestampedEEGSample(timestamp=timestamp, channels=channel_values)
        self.eeg_data.append(sample)

    def add_hr_sample(self, timestamp: float, heart_rate: int, rr_intervals: List[float]):
        """Add a timestamped heart rate sample"""
        sample = TimestampedHRSample(
            timestamp=timestamp,
            heart_rate=heart_rate,
            rr_intervals=rr_intervals
        )
        self.hr_data.append(sample)

    def add_gsr_sample(self, timestamp: float, raw_audio: float, filtered_signal: float, gsr_uS: float):
        """Add a timestamped GSR sample"""
        sample = TimestampedGSRSample(
            timestamp=timestamp,
            raw_audio=raw_audio,
            filtered_signal=filtered_signal,
            gsr_uS=gsr_uS
        )
        self.gsr_data.append(sample)

    def add_task_marker(self, timestamp: float, task_number: int, event: str = "task_end",
                        extra: Optional[Dict[str, Any]] = None):
        """Add a timestamped task marker with optional extra metadata."""
        marker = TaskMarker(timestamp=timestamp, task_number=task_number, event=event,
                            extra=extra or {})
        self.task_markers.append(marker)

    def add_arduino_event(self, timestamp: float, event_type: str, data: Dict[str, Any]):
        """Add a timestamped Arduino event (encoder delta, button press, ack, etc.)."""
        event = TimestampedArduinoEvent(timestamp=timestamp, event_type=event_type, data=data)
        self.arduino_data.append(event)

    def get_relative_time(self, timestamp: float) -> float:
        """Convert absolute timestamp to relative time from session start"""
        return timestamp - self.session_start

    def to_dict(self) -> Dict[str, Any]:
        """Convert data store to dictionary for JSON serialization.
        Time values start from 0 (beginning of experiment after countdown)."""
        return {
            'metadata': {
                'session_start_unix': self.session_start,
                'session_start_iso': datetime.fromtimestamp(self.session_start).isoformat(),
                'muse_sampling_rate': self.muse_sampling_rate,
                'muse_channels': self.muse_channels,
                'gsr_sampling_rate': self.gsr_sampling_rate,
            },
            'eeg_data': [
                {
                    'time': round(self.get_relative_time(s.timestamp), 6),
                    'channels': s.channels
                }
                for s in self.eeg_data
            ],
            'hr_data': [
                {
                    'time': round(self.get_relative_time(s.timestamp), 6),
                    'heart_rate': s.heart_rate,
                    'rr_intervals': s.rr_intervals
                }
                for s in self.hr_data
            ],
            'gsr_data': [
                {
                    'time': round(self.get_relative_time(s.timestamp), 6),
                    'raw_audio': s.raw_audio,
                    'filtered_signal': s.filtered_signal,
                    'gsr_uS': s.gsr_uS
                }
                for s in self.gsr_data
            ],
            'task_markers': [
                {
                    'time': round(self.get_relative_time(m.timestamp), 6),
                    'task_number': m.task_number,
                    'event': m.event
                }
                for m in self.task_markers
            ],
            'arduino_data': [
                {
                    'time': round(self.get_relative_time(e.timestamp), 6),
                    'event_type': e.event_type,
                    'data': e.data,
                }
                for e in self.arduino_data
            ],
            'summary': {
                'total_eeg_samples': len(self.eeg_data),
                'total_hr_samples': len(self.hr_data),
                'total_gsr_samples': len(self.gsr_data),
                'total_task_markers': len(self.task_markers),
                'total_arduino_events': len(self.arduino_data),
                'duration_seconds': round(self.get_relative_time(time.time()), 2)
            }
        }

    def save_json(self, filepath: str):
        """Save data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Data saved to {filepath}")

    def save_csv(self, base_filepath: str):
        """Save data to separate CSV files for EEG and HR data.
        Time starts from 0 (beginning of experiment after countdown)."""
        # Save EEG data
        eeg_filepath = f"{base_filepath}_eeg.csv"
        with open(eeg_filepath, 'w') as f:
            # Header - time is relative time starting from 0
            f.write("time," + ",".join(self.muse_channels) + "\n")
            for sample in self.eeg_data:
                rel_time = self.get_relative_time(sample.timestamp)
                channel_vals = ",".join(str(sample.channels.get(ch, 0)) for ch in self.muse_channels)
                f.write(f"{rel_time:.6f},{channel_vals}\n")
        print(f"EEG data saved to {eeg_filepath}")

        # Save HR data
        hr_filepath = f"{base_filepath}_hr.csv"
        with open(hr_filepath, 'w') as f:
            # Header - time is relative time starting from 0
            f.write("time,heart_rate,rr_intervals\n")
            for sample in self.hr_data:
                rel_time = self.get_relative_time(sample.timestamp)
                rr_str = ";".join(f"{rr:.2f}" for rr in sample.rr_intervals)
                f.write(f"{rel_time:.6f},{sample.heart_rate},{rr_str}\n")
        print(f"HR data saved to {hr_filepath}")

        # Save GSR data
        gsr_filepath = f"{base_filepath}_gsr.csv"
        with open(gsr_filepath, 'w') as f:
            # Header - time is relative time starting from 0
            f.write("time,raw_audio,filtered_signal,gsr_uS\n")
            for sample in self.gsr_data:
                rel_time = self.get_relative_time(sample.timestamp)
                f.write(f"{rel_time:.6f},{sample.raw_audio},{sample.filtered_signal},{sample.gsr_uS}\n")
        print(f"GSR data saved to {gsr_filepath}")

        # Save Arduino event data
        arduino_filepath = f"{base_filepath}_arduino.csv"
        with open(arduino_filepath, 'w') as f:
            f.write("time,event_type,data_json\n")
            for event in self.arduino_data:
                rel_time = self.get_relative_time(event.timestamp)
                data_str = json.dumps(event.data).replace('"', '""')
                f.write(f'{rel_time:.6f},{event.event_type},"{data_str}"\n')
        print(f"Arduino data saved to {arduino_filepath}")

    def save_task_markers(self, base_filepath: str):
        """Save task markers to JSON and CSV files."""
        markers_json = f"{base_filepath}_markers.json"
        markers_csv = f"{base_filepath}_markers.csv"

        with open(markers_json, 'w') as f:
            json.dump(
                {
                    'session_start_unix': self.session_start,
                    'markers': [
                        {
                            'time': round(self.get_relative_time(m.timestamp), 6),
                            'task_number': m.task_number,
                            'event': m.event,
                            **m.extra,
                        }
                        for m in self.task_markers
                    ]
                },
                f,
                indent=2
            )
        print(f"Task markers saved to {markers_json}")

        with open(markers_csv, 'w') as f:
            # Collect all extra keys across all markers for consistent columns
            extra_keys = []
            for m in self.task_markers:
                for k in m.extra:
                    if k not in extra_keys:
                        extra_keys.append(k)
            f.write("time,task_number,event" + ("," + ",".join(extra_keys) if extra_keys else "") + "\n")
            for m in self.task_markers:
                rel_time = self.get_relative_time(m.timestamp)
                extra_vals = ",".join(str(m.extra.get(k, "")) for k in extra_keys)
                row = f"{rel_time:.6f},{m.task_number},{m.event}"
                if extra_keys:
                    row += "," + extra_vals
                f.write(row + "\n")
        print(f"Task markers saved to {markers_csv}")


class SynchronizedCollector:
    """
    Main controller for synchronized data collection from multiple devices.

    Handles:
    - Simultaneous connection to Muse 2 and Polar H10
    - Common timestamp reference for synchronization
    - Thread-safe data storage
    - Graceful shutdown
    """

    def __init__(self,
                 muse_serial_port: Optional[str] = None,
                 muse_mac_address: Optional[str] = None,
                 polar_device_name: str = "Polar H10",
                 gsr_device: Optional[int] = None,
                 buffer_duration: int = 10,
                 apply_filter: bool = True,
                 use_muse: bool = True,
                 use_polar: bool = True,
                 use_gsr: bool = True,
                 arduino_port: Optional[str] = None,
                 arduino_baud: int = 115200,
                 use_arduino: bool = True,
                 audio_out_device: Optional[int] = None):
        """
        Initialize the synchronized collector.

        Args:
            muse_serial_port: Serial port for BLED112 dongle (optional)
            muse_mac_address: MAC address of Muse 2 (optional)
            polar_device_name: Name of Polar device to scan for
            gsr_device: Audio input device index for eSense GSR (optional)
            buffer_duration: EEG buffer duration in seconds
            apply_filter: Whether to apply bandpass/notch filters to EEG
            arduino_port: Serial port for Arduino Uno R3 (e.g. /dev/ttyACM0)
            arduino_baud: Baud rate for Arduino serial (default 115200)
            use_arduino: Whether to use the Arduino device
            audio_out_device: Output device index for haptic audio (None = system default)
        """
        self.muse_serial_port = muse_serial_port
        self.muse_mac_address = muse_mac_address
        self.polar_device_name = polar_device_name
        self.gsr_device = gsr_device
        self.buffer_duration = buffer_duration
        self.apply_filter = apply_filter
        self.audio_out_device = audio_out_device

        # Device enable flags
        self.use_muse    = use_muse
        self.use_polar   = use_polar
        self.use_gsr     = use_gsr
        self.use_arduino = use_arduino

        # Data store with synchronized timestamps
        self.data_store = SynchronizedDataStore()

        # Thread synchronization
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Device instances (created during collection)
        self.muse: Optional[MuseBrainFlowProcessor] = None
        self.polar: Optional[PolarH10] = None
        self.gsr: Optional[ESenseGSR] = None

        # Status tracking
        self.muse_connected  = False
        self.polar_connected = False
        self.gsr_connected   = False
        self.muse_error:  Optional[str] = None
        self.polar_error: Optional[str] = None
        self.gsr_error:   Optional[str] = None

        # Synchronization flags for simultaneous start
        self._muse_ready    = threading.Event()
        self._muse_failed   = False   # True when connection exhausted all retries
        self._polar_ready   = threading.Event()
        self._gsr_ready     = threading.Event()
        self._start_recording = threading.Event()  # Signal to start actual recording

        # Arduino bridge (None when use_arduino=False)
        self.arduino: Optional[ArduinoBridge] = (
            ArduinoBridge(
                port=arduino_port,
                baud=arduino_baud,
                data_store=self.data_store,
                lock=self._lock,
                stop_event=self._stop_event,
                start_recording_event=self._start_recording,
            )
            if use_arduino else None
        )

    # ── Arduino delegation properties ────────────────────────────────────────
    # These let GUI code reference collector.arduino_connected / .arduino_error /
    # ._arduino_ready without knowing about ArduinoBridge internals.

    @property
    def arduino_connected(self) -> bool:
        return self.arduino.connected if self.arduino else False

    @property
    def arduino_error(self) -> Optional[str]:
        return self.arduino.error if self.arduino else None

    @property
    def _arduino_ready(self) -> threading.Event:
        if self.arduino is None:
            e = threading.Event()
            e.set()
            return e
        return self.arduino.ready

    # ── Device connection helpers ─────────────────────────────────────────────

    def connect_muse(self) -> bool:
        """
        Connect to Muse device. Returns True if successful.
        Call this before starting collection.
        """
        try:
            self.muse = MuseBrainFlowProcessor(
                buffer_duration=self.buffer_duration,
                serial_port=self.muse_serial_port,
                mac_address=self.muse_mac_address
            )
            self.muse_connected = True
            self.data_store.muse_sampling_rate = self.muse.sampling_rate
            self.muse_error = None
            return True
        except Exception as e:
            self.muse_error = str(e)
            self.muse_connected = False
            return False

    def connect_polar(self) -> bool:
        """
        Connect to Polar device. Returns True if successful.
        Call this before starting collection.
        """
        try:
            self.polar = PolarH10(device_name=self.polar_device_name)

            async def test_polar_connection():
                """Find device and test actual BLE connection"""
                await self.polar.find_device()
                if not self.polar.device:
                    raise Exception("Device not found during scan")

                # Actually test the BLE connection
                from bleak import BleakClient
                async with BleakClient(self.polar.device.address) as client:
                    if not client.is_connected:
                        raise Exception("Failed to establish BLE connection")
                    # Connection successful - we'll reconnect during experiment
                    print(f"[Polar] Connection test successful")
                return True

            asyncio.run(test_polar_connection())
            self.polar_connected = True
            self.polar_error = None
            return True
        except Exception as e:
            self.polar_error = str(e)
            self.polar_connected = False
            return False

    def connect_gsr(self) -> bool:
        """
        Connect to eSense GSR device. Returns True if successful.
        Tests audio input availability.
        """
        try:
            import sounddevice as sd

            # Create GSR instance (doesn't start streaming yet)
            self.gsr = ESenseGSR(
                device=self.gsr_device,
                downsample_rate=50,
                buffer_size=50  # Smaller buffer for more frequent updates
            )

            # Test that we can access the audio device
            devices = sd.query_devices()
            if self.gsr_device is not None:
                if self.gsr_device >= len(devices):
                    raise Exception(f"Audio device {self.gsr_device} not found")
                device_info = devices[self.gsr_device]
                if device_info['max_input_channels'] < 1:
                    raise Exception(f"Device {self.gsr_device} has no input channels")
                print(f"[GSR] Using device: {device_info['name']}")
            else:
                # Check default input device
                default_input = sd.default.device[0]
                if default_input is None or default_input < 0:
                    raise Exception("No default audio input device found")
                print(f"[GSR] Using default input device: {devices[default_input]['name']}")

            print(f"[GSR] Connection test successful")
            self.gsr_connected = True
            self.gsr_error = None
            self.data_store.gsr_sampling_rate = self.gsr.downsample_rate
            return True
        except Exception as e:
            self.gsr_error = str(e)
            self.gsr_connected = False
            return False

    def disconnect_devices(self):
        """Disconnect all devices"""
        if self.muse:
            try:
                self.muse.stop()
            except:
                pass
            self.muse = None
            self.muse_connected = False
        self.polar = None
        self.polar_connected = False
        if self.gsr:
            try:
                self.gsr.stop()
            except:
                pass
            self.gsr = None
            self.gsr_connected = False
        if self.arduino:
            self.arduino.disconnect()

    def _muse_collection_thread(self, duration: Optional[float]):
        """
        Thread function for Muse EEG data collection.
        Runs in a separate thread since BrainFlow is synchronous.

        Args:
            duration: Collection duration in seconds, or None for indefinite
        """
        try:
            # Only create new connection if not already connected
            if not self.muse_connected or self.muse is None:
                # Retry loop — BrainFlow's BLE stack can take 10-15s to fully
                # release after a previous session (e.g. post-baseline). We wait
                # BEFORE each attempt so the stack has time to settle.
                max_attempts  = 6
                initial_delay = 15.0  # seconds before first attempt
                retry_delay   = 8.0   # seconds between subsequent attempts
                last_err = None

                print(f"[Muse] Waiting {initial_delay:.0f}s for BLE stack to settle...")
                for i in range(int(initial_delay)):
                    if self._stop_event.is_set():
                        return
                    time.sleep(1)

                for attempt in range(1, max_attempts + 1):
                    if self._stop_event.is_set():
                        return
                    try:
                        print(f"[Muse] Connecting (attempt {attempt}/{max_attempts})...")
                        self.muse = MuseBrainFlowProcessor(
                            buffer_duration=self.buffer_duration,
                            serial_port=self.muse_serial_port,
                            mac_address=self.muse_mac_address
                        )
                        # Verify the stream is alive before declaring success
                        time.sleep(1.0)
                        self.muse.get_data()
                        self.muse_connected = True
                        self.data_store.muse_sampling_rate = self.muse.sampling_rate
                        last_err = None
                        print(f"[Muse] Connection verified on attempt {attempt}.")
                        break
                    except Exception as e:
                        last_err = e
                        print(f"[Muse] Connection attempt {attempt} failed: {e}")
                        # Clean up any partial board state before retrying
                        if self.muse is not None:
                            try:
                                self.muse.stop()
                            except Exception:
                                pass
                            self.muse = None
                        if attempt < max_attempts:
                            print(f"[Muse] Retrying in {retry_delay}s...")
                            for i in range(int(retry_delay)):
                                if self._stop_event.is_set():
                                    return
                                time.sleep(1)
                if last_err is not None:
                    self._muse_failed = True
                    raise last_err
            else:
                print("[Muse] Using existing connection...")

            # Clear any buffered data before signaling ready
            self.muse.get_data()

            # Signal that Muse is ready
            print("[Muse] Ready, waiting for synchronized start...")
            self._muse_ready.set()

            # Wait for start signal (or stop signal)
            while not self._start_recording.is_set() and not self._stop_event.is_set():
                # Drain any data that accumulates while waiting
                self.muse.get_data()
                time.sleep(0.01)

            if self._stop_event.is_set():
                return

            # Clear buffer one more time right before recording starts
            self.muse.get_data()

            print("[Muse] Starting data collection...")
            start_time = time.time()
            sample_count = 0

            while not self._stop_event.is_set():
                # Check duration
                elapsed = time.time() - start_time
                if duration and elapsed >= duration:
                    break

                # Get new data from Muse — hardware timestamps come from the device
                eeg_data, hw_timestamps = self.muse.get_data()

                if eeg_data is not None and eeg_data.shape[1] > 0:
                    n_samples = eeg_data.shape[1]

                    for i in range(n_samples):
                        # Use the hardware timestamp from the Muse 2's internal clock
                        # (aligned to host clock by BrainFlow). This avoids OS scheduling
                        # jitter from calling time.time() after batch arrival.
                        sample_timestamp = float(hw_timestamps[i])

                        # Create channel value dict
                        channel_values = {
                            self.muse.channels[ch]: float(eeg_data[ch, i])
                            for ch in range(self.muse.n_channels)
                        }

                        # Thread-safe append
                        with self._lock:
                            self.data_store.add_eeg_sample(sample_timestamp, channel_values)

                        sample_count += 1

                # Small sleep to prevent busy-waiting
                time.sleep(0.01)

            print(f"[Muse] Collection complete. {sample_count} samples collected.")

        except Exception as e:
            print(f"[Muse] Error: {e}")
            self._muse_ready.set()  # Set ready even on error so we don't hang
        finally:
            if self.muse:
                try:
                    self.muse.stop()
                except Exception as e:
                    print(f"[Muse] Error stopping: {e}")
                self.muse = None
                self.muse_connected = False

    def _polar_hr_callback(self, sender, data):
        """
        Callback for Polar heart rate notifications.
        Parses data and adds to synchronized store with timestamp.
        Only records data after _start_recording is set.
        """
        # Only record if recording has started
        if not self._start_recording.is_set():
            return

        timestamp = time.time()

        # Parse heart rate data (same logic as PolarH10.parse_heart_rate_data)
        flags = data[0]
        hr_format = flags & 0x01
        rr_present = (flags & 0x10) != 0

        # Parse heart rate
        if hr_format == 0:
            heart_rate = data[1]
            offset = 2
        else:
            import struct
            heart_rate = struct.unpack('<H', data[1:3])[0]
            offset = 3

        # Parse RR intervals
        rr_intervals = []
        if rr_present:
            import struct
            rr_data = data[offset:]
            num_rr = len(rr_data) // 2

            for i in range(num_rr):
                rr_raw = struct.unpack('<H', rr_data[i*2:(i+1)*2])[0]
                rr_ms = (rr_raw / 1024.0) * 1000.0
                rr_intervals.append(rr_ms)

        # Thread-safe append
        with self._lock:
            self.data_store.add_hr_sample(timestamp, heart_rate, rr_intervals)

        # Also update Polar's internal buffers for HRV calculation
        if self.polar:
            self.polar.heart_rates.append(heart_rate)
            for rr in rr_intervals:
                self.polar.rr_intervals.append(rr)

        # Print status
        rel_time = self.data_store.get_relative_time(timestamp)
        rr_str = f" | RR: {[f'{rr:.0f}ms' for rr in rr_intervals]}" if rr_intervals else ""
        print(f"[Polar] t={rel_time:.1f}s | HR: {heart_rate} bpm{rr_str}")

    async def _polar_collection_async(self, duration: Optional[float]):
        """
        Async function for Polar data collection.
        Uses bleak's async BLE interface.

        Args:
            duration: Collection duration in seconds, or None for indefinite
        """
        try:
            # Only find device if not already found during connection screen
            if self.polar is None or self.polar.device is None:
                print("[Polar] Initializing connection...")
                self.polar = PolarH10(device_name=self.polar_device_name)
                await self.polar.find_device()

                if not self.polar.device:
                    raise Exception("Polar device not found")
            else:
                print("[Polar] Using existing device...")
                # Clear any old data from connection phase
                self.polar.heart_rates.clear()
                self.polar.rr_intervals.clear()

            print(f"[Polar] Connecting to {self.polar.device.name}...")

            from bleak import BleakClient
            async with BleakClient(self.polar.device.address) as client:
                self.polar_connected = True
                print(f"[Polar] Connected: {client.is_connected}")

                # Subscribe to heart rate notifications with our callback
                await client.start_notify(
                    HEART_RATE_MEASUREMENT_UUID,
                    self._polar_hr_callback
                )

                # Signal that Polar is ready
                print("[Polar] Ready, waiting for synchronized start...")
                self._polar_ready.set()

                # Wait for start signal (or stop signal)
                while not self._start_recording.is_set() and not self._stop_event.is_set():
                    await asyncio.sleep(0.01)

                if self._stop_event.is_set():
                    await client.stop_notify(HEART_RATE_MEASUREMENT_UUID)
                    return

                print(f"[Polar] Starting data collection...")

                # Wait for duration or stop signal
                start_time = time.time()
                while not self._stop_event.is_set():
                    elapsed = time.time() - start_time
                    if duration and elapsed >= duration:
                        break
                    await asyncio.sleep(0.1)

                # Stop notifications
                await client.stop_notify(HEART_RATE_MEASUREMENT_UUID)

            self.polar_connected = False

            # Calculate HRV metrics
            metrics = self.polar.calculate_hrv_metrics()
            if metrics:
                print("\n[Polar] HRV Metrics:")
                print(f"  Mean HR: {metrics['mean_hr']:.1f} bpm")
                print(f"  SDNN: {metrics['sdnn']:.1f} ms")
                print(f"  RMSSD: {metrics['rmssd']:.1f} ms")
                print(f"  pNN50: {metrics['pnn50']:.1f}%")

            print(f"[Polar] Collection complete.")

        except Exception as e:
            print(f"[Polar] Error: {e}")
            self.polar_connected = False

    def _gsr_collection_thread(self, duration: Optional[float]):
        """
        Thread function for GSR data collection.
        Uses the ESenseGSR callback mechanism.

        Args:
            duration: Collection duration in seconds, or None for indefinite
        """
        try:
            # Initialize GSR if needed
            if self.gsr is None:
                print("[GSR] Initializing connection...")
                self.gsr = ESenseGSR(
                    device=self.gsr_device,
                    downsample_rate=50,
                    buffer_size=50
                )
            else:
                print("[GSR] Using existing connection...")

            # Override the GSR's internal callback to feed our data store
            original_callback = self.gsr._audio_callback

            def synchronized_callback(indata, frames, time_info, status):
                """Custom callback that feeds data to our synchronized store."""
                if status:
                    print(f"[GSR] Audio status: {status}")

                # Only record if recording has started
                if not self._start_recording.is_set():
                    # Still need to process to maintain filter state
                    raw = indata[:, 0].copy()
                    self.gsr._apply_filter(raw)
                    return

                # Process audio data
                raw = indata[:, 0].copy()
                filtered = self.gsr._apply_filter(raw)

                # Downsample and store
                self.gsr._downsample_accumulator.extend(zip(raw, filtered))

                while len(self.gsr._downsample_accumulator) >= self.gsr.downsample_factor:
                    chunk = self.gsr._downsample_accumulator[:self.gsr.downsample_factor]
                    self.gsr._downsample_accumulator = self.gsr._downsample_accumulator[self.gsr.downsample_factor:]

                    raw_vals = [c[0] for c in chunk]
                    filt_vals = [c[1] for c in chunk]

                    raw_ds = float(np.mean(raw_vals))
                    filt_ds = float(np.mean(filt_vals))
                    gsr_uS = self.gsr._convert_to_gsr(filt_ds)
                    timestamp = time.time()

                    # Thread-safe append to our data store
                    with self._lock:
                        self.data_store.add_gsr_sample(timestamp, raw_ds, filt_ds, gsr_uS)

            # Start the audio stream with our custom callback
            import sounddevice as sd
            self.gsr._init_filter()
            self.gsr._downsample_accumulator = []

            self.gsr.stream = sd.InputStream(
                callback=synchronized_callback,
                channels=1,
                samplerate=self.gsr.fs,
                device=self.gsr.device,
                dtype='float32'
            )
            self.gsr.stream.start()
            self.gsr.is_streaming = True

            # Signal that GSR is ready
            print("[GSR] Ready, waiting for synchronized start...")
            self._gsr_ready.set()

            # Wait for start signal
            while not self._start_recording.is_set() and not self._stop_event.is_set():
                time.sleep(0.01)

            if self._stop_event.is_set():
                self.gsr.stream.stop()
                self.gsr.stream.close()
                self.gsr.is_streaming = False
                return

            print("[GSR] Starting data collection...")
            start_time = time.time()
            last_print = start_time

            while not self._stop_event.is_set():
                elapsed = time.time() - start_time
                if duration and elapsed >= duration:
                    break

                # Print status every 5 seconds
                if time.time() - last_print >= 5.0:
                    with self._lock:
                        gsr_count = len(self.data_store.gsr_data)
                        if gsr_count > 0:
                            latest_gsr = self.data_store.gsr_data[-1].gsr_uS
                            rel_time = self.data_store.get_relative_time(time.time())
                            print(f"[GSR] t={rel_time:.1f}s | GSR: {latest_gsr:.4f} µS | Samples: {gsr_count}")
                    last_print = time.time()

                time.sleep(0.1)

            # Stop streaming
            self.gsr.stream.stop()
            self.gsr.stream.close()
            self.gsr.is_streaming = False

            with self._lock:
                sample_count = len(self.data_store.gsr_data)
            print(f"[GSR] Collection complete. {sample_count} samples collected.")

        except Exception as e:
            print(f"[GSR] Error: {e}")
            self._gsr_ready.set()  # Set ready even on error to prevent hanging
        finally:
            if self.gsr and self.gsr.is_streaming:
                try:
                    self.gsr.stream.stop()
                    self.gsr.stream.close()
                    self.gsr.is_streaming = False
                except:
                    pass

    def collect(self, duration: float = 60, output_path: Optional[str] = None):
        """
        Start synchronized data collection from both devices.

        Args:
            duration: Collection duration in seconds
            output_path: Base path for output files (without extension)
        """
        print("=" * 70)
        print("SYNCHRONIZED DATA COLLECTION")
        print(f"Duration: {duration} seconds")
        print(f"Session start: {datetime.now().isoformat()}")
        print("=" * 70)

        # Record session start
        self.data_store.session_start = time.time()

        # Set up signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\n\nReceived interrupt signal. Stopping collection...")
            self._stop_event.set()

        original_handler = signal.signal(signal.SIGINT, signal_handler)

        try:
            # Start Muse collection in separate thread
            muse_thread = threading.Thread(
                target=self._muse_collection_thread,
                args=(duration,),
                daemon=True
            )
            muse_thread.start()

            # Run Polar collection in async event loop (main thread)
            asyncio.run(self._polar_collection_async(duration))

            # Wait for Muse thread to finish
            self._stop_event.set()
            muse_thread.join(timeout=5)

        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)

        # Print summary
        print("\n" + "=" * 70)
        print("COLLECTION SUMMARY")
        print("=" * 70)
        total_duration = self.data_store.get_relative_time(time.time())
        print(f"Total duration: {total_duration:.1f} seconds")
        print(f"EEG samples collected: {len(self.data_store.eeg_data)}")
        print(f"HR samples collected: {len(self.data_store.hr_data)}")

        if self.data_store.eeg_data:
            eeg_rate = len(self.data_store.eeg_data) / total_duration
            print(f"Effective EEG sample rate: {eeg_rate:.1f} Hz")

        if self.data_store.hr_data:
            hr_rate = len(self.data_store.hr_data) / total_duration
            print(f"Effective HR sample rate: {hr_rate:.2f} Hz")

        # Save data if output path provided
        if output_path:
            self.save_data(output_path)

        return self.data_store

    def save_data(self, base_path: str):
        """
        Save collected data to files.

        Args:
            base_path: Base path for output files (without extension)
        """
        # Create output directory if needed
        output_dir = Path(base_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON (complete data with metadata)
        json_path = f"{base_path}.json"
        self.data_store.save_json(json_path)

        # Save CSVs (easier for analysis)
        self.data_store.save_csv(base_path)

        # Save task markers (separate JSON and CSV)
        self.data_store.save_task_markers(base_path)

        print(f"\nData saved to:")
        print(f"  - {json_path}")
        print(f"  - {base_path}_eeg.csv")
        print(f"  - {base_path}_hr.csv")
        print(f"  - {base_path}_gsr.csv")
        print(f"  - {base_path}_markers.json")
        print(f"  - {base_path}_markers.csv")


def create_participant_folder(participant_id: str) -> Path:
    """Create folder structure for participant data"""
    base_path = Path("data") / participant_id
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path



def _reset_data_store(collector: SynchronizedCollector) -> None:
    """Reset collector state for a new session."""
    collector.data_store = SynchronizedDataStore()
    collector._stop_event.clear()
    collector._muse_ready.clear()
    collector._muse_failed = False
    collector._polar_ready.clear()
    collector._gsr_ready.clear()
    collector._start_recording.clear()
    if collector.arduino is not None:
        collector.arduino._data_store = collector.data_store


def start_collection_threads(collector: SynchronizedCollector):
    """Start all device collection threads and return them as a tuple."""
    muse_thread = threading.Thread(
        target=collector._muse_collection_thread, args=(None,), daemon=True
    )
    polar_thread = threading.Thread(
        target=lambda: asyncio.run(collector._polar_collection_async(None)), daemon=True
    )
    gsr_thread = threading.Thread(
        target=collector._gsr_collection_thread, args=(None,), daemon=True
    )
    if collector.arduino is not None:
        arduino_thread = threading.Thread(
            target=collector.arduino.collection_thread, args=(None,), daemon=True
        )
    else:
        arduino_thread = threading.Thread(target=lambda: None, daemon=True)

    muse_thread.start()
    polar_thread.start()
    gsr_thread.start()
    arduino_thread.start()
    return muse_thread, polar_thread, gsr_thread, arduino_thread


def main():
    """Main entry point for synchronized data collection with GUI"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Synchronized Muse 2 EEG, Polar H10 HR, and eSense GSR data collection'
    )
    parser.add_argument(
        '--muse-serial',
        type=str,
        default=None,
        help='Serial port for BLED112 dongle (e.g. /dev/ttyACM0 or COM3); omit for native Bluetooth'
    )
    parser.add_argument(
        '--muse-mac',
        type=str,
        default=None,
        help='MAC address of Muse 2 device'
    )
    parser.add_argument(
        '--polar-name',
        type=str,
        default="Polar H10",
        help='Name of Polar device to scan for (default: "Polar H10")'
    )
    parser.add_argument(
        '--gsr-device',
        type=int,
        default=None,
        help='Audio input device index for eSense GSR (use --list-audio-devices to see options)'
    )

    parser.add_argument(
        '--audio-out-device',
        type=int,
        default=None,
        help='Audio output device index for haptic feedback (use --list-audio-devices to see options)'
    )
    parser.add_argument(
        '--list-audio-devices',
        action='store_true',
        help='List available audio input/output devices and exit'
    )

    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Disable bandpass/notch filtering for EEG data'
    )
    parser.add_argument(
        '--arduino-port',
        type=str,
        default=ARDUINO_DEFAULT_PORT,
        help='Serial port for Arduino Uno R3 (e.g. /dev/ttyACM0 or COM3)'
    )
    parser.add_argument(
        '--arduino-baud',
        type=int,
        default=ARDUINO_DEFAULT_BAUD,
        help='Baud rate for Arduino serial communication (default: 115200)'
    )


    args = parser.parse_args()

    # Handle list audio devices
    if args.list_audio_devices:
        import sounddevice as sd
        devices = sd.query_devices()
        print("\nAvailable audio INPUT devices (for --gsr-device):")
        print("-" * 50)
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                marker = " [DEFAULT]" if i == sd.default.device[0] else ""
                print(f"  [{i}] {dev['name']}{marker}  ({dev['default_samplerate']:.0f} Hz)")
        print("\nAvailable audio OUTPUT devices (for --audio-out-device):")
        print("-" * 50)
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                marker = " [DEFAULT]" if i == sd.default.device[1] else ""
                print(f"  [{i}] {dev['name']}{marker}  ({dev['default_samplerate']:.0f} Hz)")
        print("-" * 50)
        return

    # Create collector
    collector = SynchronizedCollector(
        muse_serial_port=args.muse_serial,
        muse_mac_address=args.muse_mac,
        polar_device_name=args.polar_name,
        gsr_device=args.gsr_device,
        apply_filter=not args.no_filter,
        use_muse=USE_MUSE,
        use_polar=USE_POLAR,
        use_gsr=USE_GSR,
        arduino_port=args.arduino_port,
        arduino_baud=args.arduino_baud,
        use_arduino=USE_ARDUINO,
        audio_out_device=args.audio_out_device,
    )

    # Load haptic targets once — used by all three device sessions
    haptic_targets_all = load_haptic_targets()

    DEVICES = ['Auditory', 'Vibrations', 'Shape Changing']
    if True:

        try:
            # Step 1: Participant ID
            participant_id = show_participant_screen()
            if not participant_id:
                print("Experiment cancelled at participant registration.")
                return

            # Create participant folder
            participant_folder = create_participant_folder(participant_id)
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            print(f"Participant: {participant_id}")
            print(f"Data will be saved under: {participant_folder}")

            # Step 2: Device Connection (done once — sensors stay on throughout)
            if not show_connection_screen(collector):
                print("Experiment cancelled at device connection.")
                collector.disconnect_devices()
                return

            # Step 3: Consent Form
            if not show_consent_screen():
                print("Experiment cancelled - consent not given.")
                collector.disconnect_devices()
                return

            # Step 4: Baseline recording (skipped if RUN_BASELINE = False).
            session_id = f"session_{session_timestamp}"
            if RUN_BASELINE:
                # Release the existing Muse session so the calibration/baseline
                # screens can open their own fresh connections.
                if collector.muse is not None:
                    try:
                        collector.muse.stop()
                    except Exception:
                        pass
                    collector.muse = None
                    collector.muse_connected = False

                # Step 4a: EEG calibration — live signal check before baseline.
                calib_muse = show_calibration_screen(gsr=collector.gsr if collector.use_gsr else None)
                if calib_muse is None:
                    print("Experiment cancelled at EEG calibration.")
                    collector.disconnect_devices()
                    return

                # Pass the live muse session into baseline to avoid reconnect.
                if not show_baseline_screen(
                    participant_id=participant_id,
                    session_id=session_id,
                    data_dir='data',
                    muse=calib_muse,
                ):
                    print("Experiment cancelled at baseline recording.")
                    collector.disconnect_devices()
                    return

                baseline.preprocess_baseline(
                    participant_id=participant_id,
                    session_id=session_id,
                    data_dir='data',
                    out_dir='data',
                )
                baseline.extract_baseline_features(
                    participant_id=participant_id,
                    session_id=session_id,
                    out_dir='data',
                )

            # Run three device sessions
            completed_sessions = []

            # The Muse collection thread handles its own BLE settle delay internally.

            # Pre-start threads for the first session.
            _reset_data_store(collector)
            session_threads = start_collection_threads(collector)

            for session_idx, device_name in enumerate(DEVICES):
                print(f"\n{'='*60}")
                print(f"SESSION {session_idx + 1}/3: {device_name}")
                print(f"{'='*60}")

                # Per-session output path
                device_slug = device_name.lower().replace(' ', '_')
                output_path = str(
                    participant_folder / f"session_{session_timestamp}_{device_slug}"
                )

                # Wait for all devices to be fully ready before the countdown.
                # This covers both session 1 (initial connect) and sessions 2/3
                # (reconnect after swap). The screen auto-advances once all ready.
                if not show_reconnect_screen(collector):
                    print(f"Experiment aborted at reconnect screen for {device_name}.")
                    collector._stop_event.set()
                    collector.disconnect_devices()
                    return

                # Countdown
                if not show_countdown_screen():
                    print(f"Experiment cancelled at countdown for {device_name}.")
                    collector._stop_event.set()
                    collector.disconnect_devices()
                    return

                # Build a per-session HapticsController
                session_mode = {
                    "Auditory":       "auditory",
                    "Vibrations":     "vibrations",
                    "Shape Changing": "shape_changing",
                }.get(device_name, "auditory")
                haptics = HapticsController(
                    arduino_bridge=collector.arduino,
                    session_mode=session_mode,
                    audio_out_device=collector.audio_out_device,
                )
                session_targets = haptic_targets_all.get(device_name, [0] * TASKS_PER_DEVICE)

                # Run the experiment — threads already started and ready
                show_experiment_screen(
                    collector, output_path,
                    threads=session_threads,
                    haptics=haptics,
                    haptic_targets=session_targets,
                )

                # Red circle count (attention check for oddball task)
                actual_red_count = sum(
                    1 for m in collector.data_store.task_markers
                    if m.event == "oddball_onset"
                )
                red_count = show_red_circle_count_screen(device_name, actual_red_count)
                if red_count is None:
                    print(f"Red circle count skipped for {device_name}.")

                # NASA TLX for this device
                tlx_scores = show_nasa_tlx_screen(device_name)
                if tlx_scores is None:
                    print(f"NASA TLX skipped for {device_name}.")

                # Save TLX scores (and red circle count) alongside sensor data
                if tlx_scores:
                    tlx_path = f"{output_path}_nasa_tlx.json"
                    with open(tlx_path, 'w') as f:
                        json.dump({
                            'device': device_name,
                            'participant_id': participant_id,
                            'red_circle_count': red_count,
                            'actual_red_circle_count': actual_red_count,
                            'scores': tlx_scores,
                            'average': sum(tlx_scores.values()) / len(tlx_scores)
                        }, f, indent=2)
                    print(f"NASA TLX scores saved to {tlx_path}")
                elif red_count is not None:
                    count_path = f"{output_path}_red_circle_count.json"
                    with open(count_path, 'w') as f:
                        json.dump({
                            'device': device_name,
                            'participant_id': participant_id,
                            'red_circle_count': red_count,
                            'actual_red_circle_count': actual_red_count,
                        }, f, indent=2)
                    print(f"Red circle count saved to {count_path}")

                completed_sessions.append({
                    'device': device_name,
                    'data_store': collector.data_store,  # snapshot before reset
                    'output_path': output_path,
                    'tlx_scores': tlx_scores,
                    'red_circle_count': red_count,
                    'actual_red_circle_count': actual_red_count,
                })

                # Device swap screen (not shown after the last session)
                if session_idx < len(DEVICES) - 1:
                    next_device = DEVICES[session_idx + 1]
                    if not show_device_swap_screen(next_device):
                        print("Experiment aborted at device swap.")
                        collector.disconnect_devices()
                        break

                    # User confirmed swap — reset state and start connecting the
                    # next Muse session immediately. show_reconnect_screen at the
                    # top of the next iteration will block until all are ready.
                    _reset_data_store(collector)
                    print(f"[Session] Starting device threads for {next_device}...")
                    session_threads = start_collection_threads(collector)

            # Step 8: Final completion screen
            show_completion_screen(completed_sessions, str(participant_folder))

            print("\nExperiment completed successfully!")

        except Exception as e:
            show_error_popup(f'An error occurred:\n{str(e)}')
            print(f"\nError during experiment: {e}")
            raise
        finally:
            collector.disconnect_devices()


if __name__ == "__main__":
    main()
