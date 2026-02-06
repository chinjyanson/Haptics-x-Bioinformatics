"""
Synchronized Multi-Device Data Collection System
Collects data from Muse 2 EEG and Polar H10 HR monitor simultaneously
with timestamp synchronization for cognitive load analysis.
"""

import asyncio
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
class SynchronizedDataStore:
    """
    Central data store for synchronized multi-device data collection.
    Uses a common time reference for all data points.
    """
    session_start: float = field(default_factory=time.time)
    eeg_data: List[TimestampedEEGSample] = field(default_factory=list)
    hr_data: List[TimestampedHRSample] = field(default_factory=list)

    # Metadata
    muse_sampling_rate: int = 256
    muse_channels: List[str] = field(default_factory=lambda: ['TP9', 'AF7', 'AF8', 'TP10'])

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
            'summary': {
                'total_eeg_samples': len(self.eeg_data),
                'total_hr_samples': len(self.hr_data),
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
                 buffer_duration: int = 10,
                 apply_ica: bool = True,
                 apply_filter: bool = True):
        """
        Initialize the synchronized collector.

        Args:
            muse_serial_port: Serial port for BLED112 dongle (optional)
            muse_mac_address: MAC address of Muse 2 (optional)
            polar_device_name: Name of Polar device to scan for
            buffer_duration: EEG buffer duration in seconds
            apply_ica: Whether to apply ICA denoising to EEG
            apply_filter: Whether to apply bandpass/notch filters to EEG
        """
        self.muse_serial_port = muse_serial_port
        self.muse_mac_address = muse_mac_address
        self.polar_device_name = polar_device_name
        self.buffer_duration = buffer_duration
        self.apply_ica = apply_ica
        self.apply_filter = apply_filter

        # Data store with synchronized timestamps
        self.data_store = SynchronizedDataStore()

        # Thread synchronization
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Device instances (created during collection)
        self.muse: Optional[MuseBrainFlowProcessor] = None
        self.polar: Optional[PolarH10] = None

        # Status tracking
        self.muse_connected = False
        self.polar_connected = False
        self.muse_error: Optional[str] = None
        self.polar_error: Optional[str] = None

        # Synchronization flags for simultaneous start
        self._muse_ready = threading.Event()
        self._polar_ready = threading.Event()
        self._start_recording = threading.Event()  # Signal to start actual recording

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
                print("[Muse] Initializing connection...")
                self.muse = MuseBrainFlowProcessor(
                    buffer_duration=self.buffer_duration,
                    serial_port=self.muse_serial_port,
                    mac_address=self.muse_mac_address
                )
                self.muse_connected = True
                self.data_store.muse_sampling_rate = self.muse.sampling_rate
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

                # Get new data from Muse
                eeg_data = self.muse.get_data()

                if eeg_data is not None and eeg_data.shape[1] > 0:
                    # Timestamp for this batch
                    current_time = time.time()
                    n_samples = eeg_data.shape[1]

                    # Calculate timestamp for each sample based on sampling rate
                    sample_interval = 1.0 / self.muse.sampling_rate

                    for i in range(n_samples):
                        # Estimate timestamp for each sample
                        sample_timestamp = current_time - (n_samples - 1 - i) * sample_interval

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

        print(f"\nData saved to:")
        print(f"  - {json_path}")
        print(f"  - {base_path}_eeg.csv")
        print(f"  - {base_path}_hr.csv")


def create_participant_folder(participant_id: str) -> Path:
    """Create folder structure for participant data"""
    base_path = Path("data") / participant_id
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def show_participant_screen() -> Optional[str]:
    """Show participant ID input screen. Returns participant ID or None if cancelled."""
    import FreeSimpleGUI as sg

    sg.theme('LightBlue2')

    layout = [
        [sg.Text('BCI Experiment', font=('Helvetica', 24, 'bold'), justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Text('Please enter your Participant ID:', font=('Helvetica', 14))],
        [sg.Input(key='-PARTICIPANT_ID-', font=('Helvetica', 14), size=(30, 1), justification='center')],
        [sg.Text('')],
        [sg.Text('', key='-ERROR-', text_color='red', font=('Helvetica', 10))],
        [sg.Button('Continue', size=(15, 1), font=('Helvetica', 12)),
         sg.Button('Exit', size=(15, 1), font=('Helvetica', 12))]
    ]

    window = sg.Window('BCI Experiment - Participant Registration', layout,
                       element_justification='center', finalize=True, size=(500, 300))

    participant_id = None

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        if event == 'Continue':
            pid = values['-PARTICIPANT_ID-'].strip()
            if not pid:
                window['-ERROR-'].update('Please enter a valid Participant ID')
            elif not pid.replace('_', '').replace('-', '').isalnum():
                window['-ERROR-'].update('Participant ID should only contain letters, numbers, - and _')
            else:
                participant_id = pid
                break

    window.close()
    return participant_id


def show_connection_screen(collector: SynchronizedCollector) -> bool:
    """Show device connection screen with status updates. Returns True if both connected."""
    import FreeSimpleGUI as sg

    sg.theme('LightBlue2')

    layout = [
        [sg.Text('Connecting to Devices', font=('Helvetica', 20, 'bold'), justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Text('Please ensure both devices are powered on and in range.', font=('Helvetica', 11))],
        [sg.Text('')],

        # Muse status
        [sg.Frame('Muse 2 EEG Headband', [
            [sg.Text('Status:', font=('Helvetica', 11)),
             sg.Text('Waiting...', key='-MUSE_STATUS-', font=('Helvetica', 11, 'bold'), text_color='gray')],
            [sg.ProgressBar(100, orientation='h', size=(30, 20), key='-MUSE_PROGRESS-', bar_color=('blue', 'lightgray'))]
        ], font=('Helvetica', 12))],

        [sg.Text('')],

        # Polar status
        [sg.Frame('Polar H10 Heart Rate Monitor', [
            [sg.Text('Status:', font=('Helvetica', 11)),
             sg.Text('Waiting...', key='-POLAR_STATUS-', font=('Helvetica', 11, 'bold'), text_color='gray')],
            [sg.ProgressBar(100, orientation='h', size=(30, 20), key='-POLAR_PROGRESS-', bar_color=('blue', 'lightgray'))]
        ], font=('Helvetica', 12))],

        [sg.Text('')],
        [sg.Button('Connect Devices', size=(15, 1), font=('Helvetica', 12), key='-CONNECT-'),
         sg.Button('Skip Connection', size=(15, 1), font=('Helvetica', 12), key='-SKIP-', visible=False),
         sg.Button('Continue', size=(15, 1), font=('Helvetica', 12), key='-CONTINUE-', disabled=True),
         sg.Button('Cancel', size=(15, 1), font=('Helvetica', 12))]
    ]

    window = sg.Window('BCI Experiment - Device Connection', layout,
                       element_justification='center', finalize=True, size=(550, 400))

    muse_connected = False
    polar_connected = False

    def connect_muse_thread():
        nonlocal muse_connected
        muse_connected = collector.connect_muse()

    def connect_polar_thread():
        nonlocal polar_connected
        polar_connected = collector.connect_polar()

    while True:
        event, values = window.read(timeout=100)

        if event in (sg.WIN_CLOSED, 'Cancel'):
            collector.disconnect_devices()
            window.close()
            return False

        if event == '-CONNECT-':
            window['-CONNECT-'].update(disabled=True)
            window['-MUSE_STATUS-'].update('Connecting...', text_color='orange')
            window['-POLAR_STATUS-'].update('Connecting...', text_color='orange')
            window['-MUSE_PROGRESS-'].update(50)
            window['-POLAR_PROGRESS-'].update(50)
            window.refresh()

            # Connect devices in threads
            muse_thread = threading.Thread(target=connect_muse_thread)
            polar_thread = threading.Thread(target=connect_polar_thread)

            muse_thread.start()
            polar_thread.start()

            # Wait for connections with GUI updates
            while muse_thread.is_alive() or polar_thread.is_alive():
                event2, _ = window.read(timeout=100)
                if event2 in (sg.WIN_CLOSED, 'Cancel'):
                    collector.disconnect_devices()
                    window.close()
                    return False

            # Update Muse status
            if muse_connected:
                window['-MUSE_STATUS-'].update('Connected', text_color='green')
                window['-MUSE_PROGRESS-'].update(100)
            else:
                error_msg = collector.muse_error or 'Connection failed'
                window['-MUSE_STATUS-'].update(f'Failed: {error_msg[:30]}', text_color='red')
                window['-MUSE_PROGRESS-'].update(0)

            # Update Polar status
            if polar_connected:
                window['-POLAR_STATUS-'].update('Connected', text_color='green')
                window['-POLAR_PROGRESS-'].update(100)
            else:
                error_msg = collector.polar_error or 'Connection failed'
                window['-POLAR_STATUS-'].update(f'Failed: {error_msg[:30]}', text_color='red')
                window['-POLAR_PROGRESS-'].update(0)

            # Enable continue if at least one device connected
            if muse_connected and polar_connected:
                window['-CONTINUE-'].update(disabled=False)
            else:
                window['-CONNECT-'].update(disabled=False, text='Retry Connection')
                window['-SKIP-'].update(visible=True)

        if event == '-SKIP-':
            # Allow continuing with partial connection for testing
            window.close()
            return True

        if event == '-CONTINUE-':
            window.close()
            return True

    window.close()
    return False


def show_consent_screen() -> bool:
    """Show consent form screen. Returns True if consent given."""
    import FreeSimpleGUI as sg

    sg.theme('LightBlue2')

    consent_text = """
INFORMED CONSENT FOR BCI EXPERIMENT

Purpose of the Study:
This experiment collects brain activity (EEG) and heart rate data to evaluate
cognitive load in multimodal user interfaces.

Data Collection:
- EEG signals from the Muse 2 headband (4 channels)
- Heart rate and heart rate variability from the Polar H10

Your Rights:
- Participation is voluntary
- You may withdraw at any time without penalty
- Your data will be anonymized and stored securely
- Data will only be used for research purposes

Procedure:
1. Devices will record your physiological data
2. You will perform tasks as instructed
3. You may stop at any time by pressing the Stop button

Duration:
The experiment will continue until you choose to stop it or you have completed the task.

By clicking "I Agree", you confirm that:
- You have read and understood the above information
- You voluntarily agree to participate
- You are at least 18 years of age
"""

    layout = [
        [sg.Text('Consent Form', font=('Helvetica', 20, 'bold'), justification='center', expand_x=True)],
        [sg.Multiline(consent_text, size=(60, 20), font=('Helvetica', 10), disabled=True,
                      background_color='white', key='-CONSENT_TEXT-')],
        [sg.Text('')],
        [sg.Checkbox('I have read and understood the consent form', key='-READ-', font=('Helvetica', 11))],
        [sg.Checkbox('I voluntarily agree to participate in this study', key='-AGREE-', font=('Helvetica', 11))],
        [sg.Text('')],
        [sg.Button('I Agree & Continue', size=(18, 1), font=('Helvetica', 12), key='-CONSENT-', disabled=True),
         sg.Button('I Do Not Consent', size=(18, 1), font=('Helvetica', 12), key='-NO_CONSENT-')]
    ]

    window = sg.Window('BCI Experiment - Consent Form', layout,
                       element_justification='center', finalize=True, size=(600, 550))

    while True:
        event, values = window.read(timeout=100)

        if event in (sg.WIN_CLOSED, '-NO_CONSENT-'):
            window.close()
            return False

        # Enable consent button only when both checkboxes are checked
        both_checked = values['-READ-'] and values['-AGREE-']
        window['-CONSENT-'].update(disabled=not both_checked)

        if event == '-CONSENT-' and both_checked:
            window.close()
            return True

    window.close()
    return False


def show_countdown_screen() -> bool:
    """Show countdown before experiment starts. Returns True when complete."""
    import FreeSimpleGUI as sg

    sg.theme('LightBlue2')

    layout = [
        [sg.Text('Get Ready!', font=('Helvetica', 24, 'bold'), justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Text('Experiment starting in:', font=('Helvetica', 14), justification='center', expand_x=True)],
        [sg.Text('3', font=('Helvetica', 72, 'bold'), key='-COUNTDOWN-', justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Text('Please remain still and relaxed.', font=('Helvetica', 12), justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Button('Cancel', size=(15, 1), font=('Helvetica', 12))]
    ]

    window = sg.Window('BCI Experiment - Starting', layout,
                       element_justification='center', finalize=True, size=(400, 350))

    for i in range(3, 0, -1):
        window['-COUNTDOWN-'].update(str(i))
        window.refresh()

        # Wait 1 second, checking for cancel
        start = time.time()
        while time.time() - start < 1.0:
            event, _ = window.read(timeout=50)
            if event in (sg.WIN_CLOSED, 'Cancel'):
                window.close()
                return False

    window['-COUNTDOWN-'].update('GO!')
    window.refresh()
    time.sleep(0.5)

    window.close()
    return True


def show_experiment_screen(collector: SynchronizedCollector, output_path: str) -> None:
    """Show experiment running screen with stop button. Runs until user stops."""
    import FreeSimpleGUI as sg

    sg.theme('LightBlue2')

    layout = [
        [sg.Text('Experiment in Progress', font=('Helvetica', 20, 'bold'), justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Text('Initializing devices...', key='-STATUS-', font=('Helvetica', 12))],
        [sg.Text('')],

        # Status display
        [sg.Frame('Recording Status', [
            [sg.Text('Duration:', font=('Helvetica', 11)),
             sg.Text('00:00:00', key='-DURATION-', font=('Helvetica', 14, 'bold'))],
            [sg.Text('EEG Samples:', font=('Helvetica', 11)),
             sg.Text('0', key='-EEG_COUNT-', font=('Helvetica', 11, 'bold'))],
            [sg.Text('HR Samples:', font=('Helvetica', 11)),
             sg.Text('0', key='-HR_COUNT-', font=('Helvetica', 11, 'bold'))],
            [sg.Text('Latest HR:', font=('Helvetica', 11)),
             sg.Text('-- bpm', key='-LATEST_HR-', font=('Helvetica', 11, 'bold'))]
        ], font=('Helvetica', 12))],

        [sg.Text('')],
        [sg.Text('Press STOP when you are ready to end the experiment.', font=('Helvetica', 10))],
        [sg.Text('')],
        [sg.Button('STOP EXPERIMENT', size=(20, 2), font=('Helvetica', 14, 'bold'),
                   button_color=('white', 'red'), key='-STOP-', disabled=True)]
    ]

    window = sg.Window('BCI Experiment - Recording', layout,
                       element_justification='center', finalize=True, size=(500, 400),
                       disable_close=True)  # Prevent accidental close

    # Reset synchronization events
    collector._stop_event.clear()
    collector._muse_ready.clear()
    collector._polar_ready.clear()
    collector._start_recording.clear()

    # Start Muse collection thread
    muse_thread = threading.Thread(
        target=collector._muse_collection_thread,
        args=(None,),  # None = run indefinitely
        daemon=True
    )
    muse_thread.start()

    # Start Polar collection in a thread (we'll run the async loop there)
    def run_polar_async():
        asyncio.run(collector._polar_collection_async(None))  # None = run indefinitely

    polar_thread = threading.Thread(target=run_polar_async, daemon=True)
    polar_thread.start()

    # Wait for both devices to be ready
    window['-STATUS-'].update('Waiting for Muse to be ready...')
    window.refresh()

    while not collector._muse_ready.is_set():
        event, _ = window.read(timeout=100)
        if event == '-STOP-' or event == sg.WIN_CLOSED:
            collector._stop_event.set()
            window.close()
            return

    window['-STATUS-'].update('Waiting for Polar to be ready...')
    window.refresh()

    while not collector._polar_ready.is_set():
        event, _ = window.read(timeout=100)
        if event == '-STOP-' or event == sg.WIN_CLOSED:
            collector._stop_event.set()
            window.close()
            return

    # Both devices ready - set session_start and signal to start recording
    window['-STATUS-'].update('Starting synchronized recording...')
    window.refresh()

    # Set the session start time RIGHT NOW - this is time=0
    collector.data_store.session_start = time.time()

    # Signal both threads to start recording simultaneously
    collector._start_recording.set()

    window['-STATUS-'].update('Data is being recorded...')
    window['-STOP-'].update(disabled=False)
    window.refresh()

    print(f"[Sync] Both devices ready. Recording started at t=0")

    # Update loop
    while True:
        event, _ = window.read(timeout=500)  # Update every 500ms

        if event == '-STOP-':
            # Confirm stop
            if sg.popup_yes_no('Are you sure you want to stop the experiment?',
                              title='Confirm Stop', font=('Helvetica', 11)) == 'Yes':
                break

        # Update display
        elapsed = time.time() - collector.data_store.session_start
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        window['-DURATION-'].update(f'{hours:02d}:{minutes:02d}:{seconds:02d}')

        with collector._lock:
            eeg_count = len(collector.data_store.eeg_data)
            hr_count = len(collector.data_store.hr_data)
            if collector.data_store.hr_data:
                latest_hr = collector.data_store.hr_data[-1].heart_rate
            else:
                latest_hr = None

        window['-EEG_COUNT-'].update(str(eeg_count))
        window['-HR_COUNT-'].update(str(hr_count))
        if latest_hr:
            window['-LATEST_HR-'].update(f'{latest_hr} bpm')

    # Stop collection
    collector._stop_event.set()
    window['-STOP-'].update('Saving data...', disabled=True)
    window.refresh()

    # Wait for threads to finish
    muse_thread.join(timeout=5)
    polar_thread.join(timeout=5)

    # Save data
    collector.save_data(output_path)

    window.close()


def show_completion_screen(collector: SynchronizedCollector, output_path: str) -> None:
    """Show experiment completion summary."""
    import FreeSimpleGUI as sg

    sg.theme('LightBlue2')

    # Calculate summary stats
    total_duration = collector.data_store.get_relative_time(time.time())
    eeg_count = len(collector.data_store.eeg_data)
    hr_count = len(collector.data_store.hr_data)

    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    duration_str = f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    # HRV metrics if available
    hrv_text = "Not available"
    if collector.polar:
        metrics = collector.polar.calculate_hrv_metrics()
        if metrics:
            hrv_text = f"Mean HR: {metrics['mean_hr']:.1f} bpm\n"
            hrv_text += f"SDNN: {metrics['sdnn']:.1f} ms\n"
            hrv_text += f"RMSSD: {metrics['rmssd']:.1f} ms"

    layout = [
        [sg.Text('Experiment Complete!', font=('Helvetica', 24, 'bold'),
                 text_color='green', justification='center', expand_x=True)],
        [sg.Text('')],
        [sg.Text('Thank you for participating.', font=('Helvetica', 14))],
        [sg.Text('')],

        [sg.Frame('Session Summary', [
            [sg.Text(f'Duration: {duration_str}', font=('Helvetica', 12))],
            [sg.Text(f'EEG Samples: {eeg_count:,}', font=('Helvetica', 12))],
            [sg.Text(f'HR Samples: {hr_count:,}', font=('Helvetica', 12))],
        ], font=('Helvetica', 12))],

        [sg.Text('')],

        [sg.Frame('HRV Metrics', [
            [sg.Text(hrv_text, font=('Helvetica', 11))]
        ], font=('Helvetica', 12))],

        [sg.Text('')],
        [sg.Text(f'Data saved to: {output_path}', font=('Helvetica', 10))],
        [sg.Text('')],
        [sg.Button('Close', size=(15, 1), font=('Helvetica', 12))]
    ]

    window = sg.Window('BCI Experiment - Complete', layout,
                       element_justification='center', finalize=True, size=(500, 450))

    while True:
        event, _ = window.read()
        if event in (sg.WIN_CLOSED, 'Close'):
            break

    window.close()


def main():
    """Main entry point for synchronized data collection with GUI"""
    import argparse
    import FreeSimpleGUI as sg

    parser = argparse.ArgumentParser(
        description='Synchronized Muse 2 EEG and Polar H10 HR data collection'
    )
    parser.add_argument(
        '--muse-serial',
        type=str,
        default=None,
        help='Serial port for Muse BLED112 dongle (e.g., COM3 or /dev/ttyUSB0)'
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
        '--no-ica',
        action='store_true',
        help='Disable ICA denoising for EEG data'
    )
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Disable bandpass/notch filtering for EEG data'
    )
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run without GUI (command-line mode)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=None,
        help='Collection duration in seconds (only for --no-gui mode)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path (only for --no-gui mode)'
    )

    args = parser.parse_args()

    # Create collector
    collector = SynchronizedCollector(
        muse_serial_port=args.muse_serial,
        muse_mac_address=args.muse_mac,
        polar_device_name=args.polar_name,
        apply_ica=not args.no_ica,
        apply_filter=not args.no_filter
    )

    if args.no_gui:
        # Command-line mode (original behavior)
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"data/raw/session_{timestamp}"

        duration = args.duration if args.duration else 60

        try:
            data_store = collector.collect(
                duration=duration,
                output_path=args.output
            )
            print("\nCollection completed successfully!")
        except KeyboardInterrupt:
            print("\nCollection interrupted by user.")
        except Exception as e:
            print(f"\nError during collection: {e}")
            raise
    else:
        # GUI mode
        try:
            # Step 1: Participant ID
            participant_id = show_participant_screen()
            if not participant_id:
                print("Experiment cancelled at participant registration.")
                return

            # Create participant folder
            participant_folder = create_participant_folder(participant_id)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(participant_folder / f"session_{timestamp}")

            print(f"Participant: {participant_id}")
            print(f"Data will be saved to: {output_path}")

            # Step 2: Device Connection
            if not show_connection_screen(collector):
                print("Experiment cancelled at device connection.")
                collector.disconnect_devices()
                return

            # Step 3: Consent Form
            if not show_consent_screen():
                print("Experiment cancelled - consent not given.")
                collector.disconnect_devices()
                return

            # Step 4: Countdown
            if not show_countdown_screen():
                print("Experiment cancelled at countdown.")
                collector.disconnect_devices()
                return

            # Step 5: Run Experiment
            show_experiment_screen(collector, output_path)

            # Step 6: Completion Screen
            show_completion_screen(collector, output_path)

            print("\nExperiment completed successfully!")

        except Exception as e:
            sg.popup_error(f'An error occurred:\n{str(e)}', title='Error')
            print(f"\nError during experiment: {e}")
            raise
        finally:
            collector.disconnect_devices()


if __name__ == "__main__":
    main()
