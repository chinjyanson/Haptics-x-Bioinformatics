"""
Real-Time GSR Acquisition & CSV Logging for eSense Skin Response

Device: eSense Skin Response (3.5 mm TRRS headphone jack version)
Signal path: GSR sensor → audio modulation → USB audio interface → digital samples

Key specifications:
- Audio sampling rate: 44,100 Hz
- GSR bandwidth: 0-5 Hz (lowpass filtered)
- Downsample target: 10-50 Hz
- Output: CSV with timestamp, raw_audio, filtered_signal, gsr_uS
"""

import sounddevice as sd
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi
import time
import os
from datetime import datetime
import threading
from threading import Lock
from typing import Optional


class ESenseGSR:
    """Real-time GSR acquisition from eSense via audio input."""

    def __init__(
        self,
        fs: int = 44100,
        cutoff: float = 5.0,
        filter_order: int = 4,
        downsample_rate: int = 50,
        calibration_a: float = 10.0,
        calibration_b: float = 0.0,
        output_dir: str = "output",
        buffer_size: int = 100,
        device: Optional[int] = None
    ):
        """
        Initialize GSR acquisition system.

        Args:
            fs: Audio sampling rate (Hz)
            cutoff: Lowpass filter cutoff frequency (Hz)
            filter_order: Butterworth filter order
            downsample_rate: Target sample rate after downsampling (Hz)
            calibration_a: Linear calibration slope (µS = a * amplitude + b)
            calibration_b: Linear calibration offset
            output_dir: Directory for CSV output files
            buffer_size: Number of samples to buffer before CSV write
            device: Audio input device index (None for default)
        """
        self.fs = fs
        self.cutoff = cutoff
        self.filter_order = filter_order
        self.downsample_rate = downsample_rate
        self.downsample_factor = fs // downsample_rate

        # Calibration parameters — gain trim applied before physics model
        self.calibration_a = calibration_a
        self.calibration_b = calibration_b

        # eSense circuit constants (from Mindfield documentation)
        # 0.61 V DC source with 61.5 kΩ source resistance
        self.V_SOURCE = 0.61      # Volts
        self.R_SOURCE = 61500.0   # Ohms

        # Open-circuit carrier baseline (set by measure_baseline())
        self.amplitude_baseline = 0.0

        self.output_dir = output_dir
        self.buffer_size = buffer_size
        self.device = device

        # Initialize filter (use second-order sections for numerical stability)
        self._init_filter()

        # Data buffer with thread safety
        self.buffer = []
        self.buffer_lock = Lock()

        # State
        self.is_streaming = False
        self.csv_path: Optional[str] = None
        self.sample_count = 0
        self.start_time: Optional[float] = None

        # Downsampling accumulator
        self._downsample_accumulator = []

    def _init_filter(self):
        """Initialize Butterworth lowpass filter with initial conditions."""
        nyq = 0.5 * self.fs
        normalized_cutoff = self.cutoff / nyq
        self.sos = butter(self.filter_order, normalized_cutoff, btype='low', output='sos')
        self.zi = sosfilt_zi(self.sos)

    def _apply_filter(self, data: np.ndarray) -> np.ndarray:
        """Rectify then lowpass filter to extract AM envelope."""
        rectified = np.abs(data)
        filtered, self.zi = sosfilt(self.sos, rectified, zi=self.zi)
        return filtered

    def _convert_to_gsr(self, amplitude: float) -> float:
        """
        Convert filtered audio amplitude to skin conductance (µS).

        The eSense carrier is AM-modulated by skin conductance: higher
        conductance damps the carrier less → higher amplitude. So the
        relationship is monotonically increasing: higher amplitude = higher
        conductance. After subtracting the open-circuit baseline (clips off),
        a linear scale maps the amplitude delta to µS.

        calibration_a: user gain trim (default 1.0; increase if µS too low)
        calibration_b: offset trim (default 0.0)
        """
        amp = amplitude - self.amplitude_baseline
        if amp <= 0:
            return 0.0
        return max(0.0, amp * self.calibration_a * (1e6 / self.R_SOURCE) + self.calibration_b)

    def _audio_callback(self, indata, frames, time_info, status):
        """Process incoming audio data in real-time."""
        if status:
            print(f"Audio status: {status}")

        # Extract mono channel
        raw = indata[:, 0].copy()

        # Apply lowpass filter
        filtered = self._apply_filter(raw)

        # Downsample and process
        self._downsample_accumulator.extend(zip(raw, filtered))

        while len(self._downsample_accumulator) >= self.downsample_factor:
            # Take chunk for downsampling
            chunk = self._downsample_accumulator[:self.downsample_factor]
            self._downsample_accumulator = self._downsample_accumulator[self.downsample_factor:]

            # Use mean of chunk for downsampled value
            raw_vals = [c[0] for c in chunk]
            filt_vals = [c[1] for c in chunk]

            timestamp = time.time()

            # Skip samples during filter warm-up to avoid initial transient
            if timestamp < self._warmup_until:
                continue

            raw_ds = np.mean(raw_vals)
            filt_ds = float(np.mean(filt_vals))
            gsr_uS = self._convert_to_gsr(filt_ds)

            with self.buffer_lock:
                self.buffer.append([timestamp, raw_ds, filt_ds, gsr_uS])
                self.sample_count += 1

                # Flush buffer to CSV when full
                if len(self.buffer) >= self.buffer_size:
                    self._flush_buffer()

    def _flush_buffer(self):
        """Write buffered data to CSV file."""
        if not self.buffer or not self.csv_path:
            return

        df = pd.DataFrame(
            self.buffer,
            columns=["timestamp", "raw_audio", "filtered_signal", "gsr_uS"]
        )

        # Append to CSV (write header only if file doesn't exist)
        write_header = not os.path.exists(self.csv_path)
        df.to_csv(self.csv_path, mode='a', index=False, header=write_header)

        self.buffer = []

    def measure_baseline(self, duration: float = 4.0, save_path: str = "gsr_baseline.json"):
        """
        Measure open-circuit amplitude with clips disconnected.

        Uses the same stateful filter path as normal recording so the
        baseline matches the values seen in the main stream exactly.
        Sets self.amplitude_baseline and saves it to save_path so it
        can be reloaded next time without re-measuring.

        Args:
            duration: Seconds to record (first 1s discarded as filter warm-up)
            save_path: JSON file to persist the baseline value
        """
        print("Measuring baseline (keep clips disconnected)...", end="", flush=True)

        # Use a temporary CSV path that we discard afterwards
        tmp_path = os.path.join(self.output_dir, "_baseline_tmp.csv")

        # Start the normal stream so _apply_filter state builds up identically
        self.start(filename=os.path.basename(tmp_path))
        time.sleep(duration)
        self.stop()

        # Read back the filtered_signal values (warmup already trimmed by _warmup_until)
        try:
            df = pd.read_csv(tmp_path)
            if len(df) > 0:
                self.amplitude_baseline = float(df['filtered_signal'].mean())
            else:
                self.amplitude_baseline = 0.0
        except Exception:
            self.amplitude_baseline = 0.0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        print(f" done. Baseline amplitude: {self.amplitude_baseline:.5f}")
        self.save_baseline(save_path)

    def save_baseline(self, path: str = "gsr_baseline.json"):
        """Save the current amplitude baseline to a JSON file for reuse."""
        import json
        data = {"amplitude_baseline": self.amplitude_baseline}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Baseline saved to {path}")

    def load_baseline(self, path: str = "gsr_baseline.json"):
        """Load a previously saved amplitude baseline from a JSON file."""
        import json
        if not os.path.exists(path):
            print(f"No baseline file found at {path}. Using 0.0.")
            return
        with open(path) as f:
            data = json.load(f)
        self.amplitude_baseline = float(data["amplitude_baseline"])
        print(f"Baseline loaded from {path}: {self.amplitude_baseline:.5f}")

    def list_devices(self):
        """List available audio input devices."""
        print("\nAvailable audio input devices:")
        print("-" * 50)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                marker = " [DEFAULT]" if i == sd.default.device[0] else ""
                print(f"  [{i}] {dev['name']}{marker}")
                print(f"      Sample rates: {dev['default_samplerate']} Hz")
        print("-" * 50)

    def start(self, filename: Optional[str] = None):
        """
        Start real-time GSR streaming.

        Args:
            filename: Output CSV filename (auto-generated if None)
        """
        if self.is_streaming:
            print("Already streaming!")
            return

        # Setup output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Generate filename with timestamp
        if filename is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gsr_{timestamp_str}.csv"

        self.csv_path = os.path.join(self.output_dir, filename)

        # Reset state
        self._init_filter()
        self.buffer = []
        self._downsample_accumulator = []
        self.sample_count = 0
        self.start_time = time.time()
        # Discard samples during filter warm-up (4th-order Butterworth at 5 Hz
        # needs ~0.8s to settle; use 1s to be safe)
        self._warmup_until = self.start_time + 1.0

        # Start audio stream
        self.stream = sd.InputStream(
            callback=self._audio_callback,
            channels=1,
            samplerate=self.fs,
            device=self.device,
            dtype='float32'
        )
        self.stream.start()
        self.is_streaming = True

        print(f"GSR streaming started")
        print(f"  Output: {self.csv_path}")
        print(f"  Sample rate: {self.downsample_rate} Hz (downsampled from {self.fs} Hz)")
        print(f"  Filter: {self.filter_order}th order Butterworth, {self.cutoff} Hz cutoff")
        print(f"  Press Ctrl+C to stop")

    def stop(self):
        """Stop streaming and flush remaining data."""
        if not self.is_streaming:
            return

        self.stream.stop()
        self.stream.close()
        self.is_streaming = False

        # Flush remaining buffer
        with self.buffer_lock:
            self._flush_buffer()

        duration = time.time() - self.start_time if self.start_time else 0

        print(f"\nStreaming stopped")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Samples recorded: {self.sample_count}")
        print(f"  Saved to: {self.csv_path}")

    def get_latest_value(self) -> Optional[float]:
        """Get most recent GSR value (µS) for real-time display."""
        with self.buffer_lock:
            if self.buffer:
                return self.buffer[-1][3]  # gsr_uS column
        return None


class SimulatedGSR:
    """
    Drop-in replacement for ESenseGSR that generates synthetic EDA data.
    Produces realistic tonic + phasic skin conductance signals for pipeline testing.
    """

    def __init__(self, downsample_rate: int = 50, output_dir: str = "output",
                 buffer_size: int = 100, baseline_uS: float = 2.0, **kwargs):
        self.downsample_rate = downsample_rate
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        self.baseline_uS = baseline_uS

        self.is_streaming = False
        self.csv_path: Optional[str] = None
        self.sample_count = 0
        self.start_time: Optional[float] = None
        self.buffer = []
        self.buffer_lock = Lock()
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    def _simulation_thread(self):
        rng = np.random.default_rng()
        interval = 1.0 / self.downsample_rate
        scr_decay = 0.0

        while not self._stop_event.is_set():
            t = time.time() - self.start_time
            tonic = self.baseline_uS + 0.5 * np.sin(2 * np.pi * t / 60.0)
            if rng.random() < interval / 20.0:
                scr_decay += rng.uniform(0.5, 2.5)
            scr_decay *= np.exp(-interval / 3.0)
            gsr_uS = max(0.0, tonic + scr_decay + rng.normal(0, 0.02))

            with self.buffer_lock:
                self.buffer.append([time.time(), 0.0, 0.0, gsr_uS])
                self.sample_count += 1
                if len(self.buffer) >= self.buffer_size:
                    self._flush_buffer()

            time.sleep(interval)

    def _flush_buffer(self):
        if not self.buffer or not self.csv_path:
            return
        df = pd.DataFrame(self.buffer,
                          columns=["timestamp", "raw_audio", "filtered_signal", "gsr_uS"])
        write_header = not os.path.exists(self.csv_path)
        df.to_csv(self.csv_path, mode='a', index=False, header=write_header)
        self.buffer = []

    def list_devices(self):
        print("[SIMULATED] No real audio devices used.")

    def measure_baseline(self, duration: float = 3.0, save_path: str = "gsr_baseline.json"):  # noqa: ARG002
        print("[SIMULATED] Baseline measurement skipped.")

    def save_baseline(self, path: str = "gsr_baseline.json"):  # noqa: ARG002
        pass

    def load_baseline(self, path: str = "gsr_baseline.json"):  # noqa: ARG002
        pass

    def start(self, filename=None):
        if self.is_streaming:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        if filename is None:
            filename = f"gsr_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_path = os.path.join(self.output_dir, filename)
        self.sample_count = 0
        self.start_time = time.time()
        self.buffer = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._simulation_thread, daemon=True)
        self._thread.start()
        self.is_streaming = True
        print(f"[SIMULATED] GSR simulation started → {self.csv_path}")

    def stop(self):
        if not self.is_streaming:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self.is_streaming = False
        with self.buffer_lock:
            self._flush_buffer()
        print(f"[SIMULATED] GSR simulation stopped. {self.sample_count} samples.")

    def get_latest_value(self):
        with self.buffer_lock:
            if self.buffer:
                return self.buffer[-1][3]
        return None


def plot_gsr_data(filepath: str, save_path: Optional[str] = None):
    """
    Plot GSR data from a CSV file.

    Creates a 3-panel figure showing:
    1. Raw audio signal
    2. Filtered signal
    3. GSR conductance (µS)

    Args:
        filepath: Path to GSR CSV file
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt

    # Load data
    df = pd.read_csv(filepath)

    # Convert timestamp to relative time (starting from 0)
    if 'timestamp' in df.columns:
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    elif 'time' not in df.columns:
        raise ValueError("CSV must have 'timestamp' or 'time' column")

    # Detect calibration: real µS values are >= 0.001; raw audio amplitude is << 1e-3
    is_calibrated = df['gsr_uS'].abs().max() > 1e-3
    unit = "µS" if is_calibrated else "a.u."
    gsr_label = f"GSR ({unit})"
    calibration_note = "" if is_calibrated else "\n⚠ Uncalibrated — values are raw audio amplitude, not µS"

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('eSense GSR Recording Analysis', fontsize=14, fontweight='bold')

    # Panel 1: Raw audio signal
    ax1 = axes[0]
    ax1.plot(df['time'], df['raw_audio'], 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('Raw Audio\nAmplitude')
    ax1.set_title('Raw Audio Signal (Microphone Input)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Panel 2: Filtered signal
    ax2 = axes[1]
    ax2.plot(df['time'], df['filtered_signal'], 'g-', linewidth=0.8)
    ax2.set_ylabel('Filtered Signal\nAmplitude')
    ax2.set_title('Lowpass Filtered Signal (5 Hz cutoff)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Panel 3: GSR
    ax3 = axes[2]
    ax3.plot(df['time'], df['gsr_uS'], 'r-', linewidth=1)
    ax3.set_ylabel(gsr_label)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('Galvanic Skin Response (Conductance)')
    ax3.grid(True, alpha=0.3)

    # Add statistics annotation
    duration = df['time'].iloc[-1]
    n_samples = len(df)
    sample_rate = n_samples / duration if duration > 0 else 0

    # GSR statistics
    gsr_mean = df['gsr_uS'].mean()
    gsr_std = df['gsr_uS'].std()
    gsr_min = df['gsr_uS'].min()
    gsr_max = df['gsr_uS'].max()

    stats_text = (
        f"Duration: {duration:.1f}s | Samples: {n_samples:,} | Rate: {sample_rate:.1f} Hz\n"
        f"GSR - Mean: {gsr_mean:.2e} | Std: {gsr_std:.2e} | Range: [{gsr_min:.2e}, {gsr_max:.2e}]"
        f"{calibration_note}"
    )
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat' if is_calibrated else 'lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    return fig


def plot_gsr_analysis(filepath: str, window_size: float = 5.0, save_path: Optional[str] = None):
    """
    Plot detailed GSR analysis with rolling statistics.

    Creates a 4-panel figure showing:
    1. GSR signal with rolling mean
    2. Rolling standard deviation (variability)
    3. Rate of change (derivative)
    4. Distribution histogram

    Args:
        filepath: Path to GSR CSV file
        window_size: Rolling window size in seconds
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    from scipy import signal as scipy_signal

    # Load data
    df = pd.read_csv(filepath)

    # Convert timestamp to relative time
    if 'timestamp' in df.columns:
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]

    # Estimate sample rate
    duration = df['time'].iloc[-1]
    n_samples = len(df)
    sample_rate = n_samples / duration if duration > 0 else 50
    window_samples = int(window_size * sample_rate)

    # Calculate rolling statistics
    df['rolling_mean'] = df['gsr_uS'].rolling(window=window_samples, center=True).mean()
    df['rolling_std'] = df['gsr_uS'].rolling(window=window_samples, center=True).std()

    # Calculate derivative (rate of change)
    df['derivative'] = np.gradient(df['gsr_uS'], df['time'])

    # Detect calibration: real µS values are >= 0.001; raw audio amplitude is << 1e-3
    is_calibrated = df['gsr_uS'].abs().max() > 1e-3
    unit = "µS" if is_calibrated else "a.u."
    gsr_label = f"GSR ({unit})"
    calibration_note = "" if is_calibrated else "⚠ Uncalibrated — values are raw audio amplitude, not µS"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('eSense GSR Detailed Analysis', fontsize=14, fontweight='bold')

    # Panel 1: GSR with rolling mean
    ax1 = axes[0, 0]
    ax1.plot(df['time'], df['gsr_uS'], 'lightblue', linewidth=0.5, alpha=0.7, label='Raw GSR')
    ax1.plot(df['time'], df['rolling_mean'], 'b-', linewidth=1.5, label=f'Rolling Mean ({window_size}s)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel(gsr_label)
    ax1.set_title('GSR Signal with Trend')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Rolling standard deviation
    ax2 = axes[0, 1]
    ax2.plot(df['time'], df['rolling_std'], 'orange', linewidth=1)
    ax2.fill_between(df['time'], 0, df['rolling_std'], alpha=0.3, color='orange')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel(f'Std Dev ({unit})')
    ax2.set_title(f'GSR Variability (Rolling Std, {window_size}s window)')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Rate of change
    ax3 = axes[1, 0]
    ax3.plot(df['time'], df['derivative'], 'green', linewidth=0.5)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel(f'dGSR/dt ({unit}/s)')
    ax3.set_title('Rate of Change (Phasic Activity Indicator)')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Distribution
    ax4 = axes[1, 1]
    ax4.hist(df['gsr_uS'], bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax4.axvline(x=df['gsr_uS'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['gsr_uS'].mean():.2e}")
    ax4.axvline(x=df['gsr_uS'].median(), color='orange', linestyle='--', linewidth=2, label=f"Median: {df['gsr_uS'].median():.2e}")
    ax4.set_xlabel(gsr_label)
    ax4.set_ylabel('Count')
    ax4.set_title('GSR Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if calibration_note:
        fig.text(0.5, 0.01, calibration_note, ha='center', fontsize=10,
                 color='darkorange', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.subplots_adjust(bottom=0.07)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Analysis plot saved to {save_path}")

    plt.show()
    return fig


def run_two_point_calibration(
    gsr,
    settle_time: float = 3.0,
    measure_time: float = 5.0,
    config_path: str = "gsr_calibration.json"
):
    """
    Interactive two-point linear calibration for eSense GSR.

    Connect eSense clips to two known resistances in sequence. The function
    measures the audio amplitude at each reference and computes a, b such that:
        µS = a * amplitude + b

    Typical resistor values:
        200 kΩ  →  5.0 µS
        500 kΩ  →  2.0 µS
        1 MΩ    →  1.0 µS
        2 MΩ    →  0.5 µS

    Args:
        gsr: ESenseGSR instance (must not be streaming)
        settle_time: Seconds to wait for signal to stabilise after connecting reference
        measure_time: Seconds to average for each reference point
        config_path: JSON file to save calibration coefficients
    """
    import json

    print("\n" + "=" * 60)
    print("TWO-POINT GSR CALIBRATION")
    print("=" * 60)
    print("You will need two precision resistors to simulate known skin")
    print("conductance. Suggested values:")
    print("  Ref 1: 200 kΩ  →  5.0 µS")
    print("  Ref 2: 2 MΩ    →  0.5 µS")
    print("Connect the eSense finger clips across the resistor leads.")
    print("=" * 60)

    points = []  # list of (amplitude, conductance_uS) tuples

    for ref_num in range(1, 3):
        resistance_str = input(f"\nRef {ref_num}: Enter resistance in kΩ (e.g. 200): ").strip()
        try:
            resistance_kohm = float(resistance_str)
        except ValueError:
            print("Invalid input. Exiting calibration.")
            return None

        conductance_uS = 1e6 / (resistance_kohm * 1e3)  # G = 1/R in siemens → µS
        print(f"  → Conductance: {conductance_uS:.4f} µS")

        input(f"  Connect clips to {resistance_kohm:.0f} kΩ resistor, then press Enter...")

        # Start streaming and collect samples over settle + measure window
        gsr.start(filename=f"cal_ref{ref_num}.csv")

        print(f"  Settling for {settle_time:.0f}s...", end="", flush=True)
        time.sleep(settle_time)
        print(" done.")

        print(f"  Measuring for {measure_time:.0f}s...", end="", flush=True)
        samples = []
        measure_start = time.time()
        while time.time() - measure_start < measure_time:
            val = gsr.get_latest_value()
            if val is not None:
                # With default a=1, b=0 this is raw amplitude — exactly what we need
                samples.append(val)
            time.sleep(0.05)

        gsr.stop()

        if not samples:
            print("\nNo samples collected. Exiting calibration.")
            return None

        # Reset filter state for next reference
        gsr._init_filter()
        gsr.buffer = []
        gsr._downsample_accumulator = []

        mean_amplitude = float(np.mean(samples))
        std_amplitude = float(np.std(samples))
        print(f" done.")
        print(f"  Mean amplitude: {mean_amplitude:.4e} ± {std_amplitude:.4e}")
        points.append((mean_amplitude, conductance_uS))

    # Solve for a, b: µS = a * amplitude + b using two-point formula
    amp1, cond1 = points[0]
    amp2, cond2 = points[1]

    if abs(amp1 - amp2) < 1e-12:
        print("\nERROR: Both reference amplitudes are identical — cannot calibrate.")
        print("Ensure the eSense clips are making good contact with each resistor.")
        return None

    a = (cond1 - cond2) / (amp1 - amp2)
    b = cond1 - a * amp1

    print("\n" + "=" * 60)
    print("CALIBRATION RESULT")
    print("=" * 60)
    print(f"  Slope  a = {a:.6e}  µS / amplitude")
    print(f"  Offset b = {b:.6e}  µS")
    print(f"  Formula: µS = {a:.4e} × amplitude + {b:.4e}")
    print()
    print("Verification:")
    for amp, cond in points:
        predicted = a * amp + b
        print(f"  Amplitude {amp:.4e} → predicted {predicted:.4f} µS  (reference: {cond:.4f} µS)")

    # Save coefficients to JSON
    cal = {"calibration_a": a, "calibration_b": b, "reference_points": points}
    with open(config_path, "w") as f:
        json.dump(cal, f, indent=2)

    print(f"\nCalibration saved to: {config_path}")
    print(f"To use: python esense.py --load-cal")
    print(f"Or:     python esense.py --cal-a {a:.6e} --cal-b {b:.6e}")
    print("=" * 60)

    return a, b


def load_calibration(config_path: str = "gsr_calibration.json"):
    """
    Load saved calibration coefficients from JSON file.

    Returns:
        (a, b) tuple, or (1.0, 0.0) if no calibration file found.
    """
    import json
    if not os.path.exists(config_path):
        print(f"No calibration file found at {config_path}. Using default (a=1.0, b=0.0).")
        return 1.0, 0.0
    with open(config_path) as f:
        cal = json.load(f)
    a = cal["calibration_a"]
    b = cal["calibration_b"]
    print(f"Loaded calibration from {config_path}: a={a:.4e}, b={b:.4e}")
    return a, b


def main():
    """Main entry point for GSR acquisition."""
    import argparse

    parser = argparse.ArgumentParser(description="eSense GSR Real-Time Acquisition")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--duration", type=float, default=None, help="Recording duration (seconds)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV filename")
    parser.add_argument("--cal-a", type=float, default=10.0,
                        help="Amplitude gain trim before physics model (default 10; increase if µS values too low)")
    parser.add_argument("--cal-b", type=float, default=0.0,
                        help="Amplitude offset trim before physics model")
    parser.add_argument("--downsample", type=int, default=50, help="Downsample rate (Hz)")
    parser.add_argument("--calibrate", "--calibrate-two-point", action="store_true",
                        help="Run interactive two-point calibration using known resistors")
    parser.add_argument("--load-cal", action="store_true",
                        help="Load saved calibration from gsr_calibration.json")
    parser.add_argument("--plot", type=str, default=None, help="Plot GSR data from CSV file")
    parser.add_argument("--plot-analysis", type=str, default=None, help="Plot detailed GSR analysis from CSV file")
    parser.add_argument("--save-plot", type=str, default=None, help="Save plot to file (use with --plot or --plot-analysis)")
    parser.add_argument("--no-plot", action="store_true", help="Disable automatic plotting after recording")
    parser.add_argument("--baseline", action="store_true",
                        help="Measure open-circuit baseline before recording (keep clips disconnected when prompted)")
    parser.add_argument("--load-baseline", action="store_true",
                        help="Load saved baseline from gsr_baseline.json instead of re-measuring")
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulated GSR data instead of real sensor (for pipeline testing)")

    args = parser.parse_args()

    # Handle plot mode
    if args.plot:
        plot_gsr_data(args.plot, save_path=args.save_plot)
        return

    if args.plot_analysis:
        plot_gsr_analysis(args.plot_analysis, save_path=args.save_plot)
        return

    # Load calibration from file if requested (overrides --cal-a / --cal-b)
    if args.load_cal:
        args.cal_a, args.cal_b = load_calibration()

    if args.simulate:
        gsr = SimulatedGSR(downsample_rate=args.downsample, output_dir="output")
    else:
        gsr = ESenseGSR(
            device=args.device,
            calibration_a=args.cal_a,
            calibration_b=args.cal_b,
            downsample_rate=args.downsample
        )

    if args.list_devices:
        gsr.list_devices()
        return

    if args.calibrate:
        run_two_point_calibration(gsr)
        return

    if args.load_baseline:
        gsr.load_baseline()
    elif args.baseline:
        input("  Disconnect finger clips from skin, then press Enter to measure baseline...")
        gsr.measure_baseline()  # auto-saves to gsr_baseline.json

    # Start streaming
    gsr.start(filename=args.output)

    try:
        if args.duration:
            time.sleep(args.duration)
        else:
            # Stream indefinitely
            while True:
                time.sleep(1)
                val = gsr.get_latest_value()
                if val is not None:
                    elapsed = time.time() - gsr.start_time
                    print(f"[{elapsed:6.1f}s] GSR: {val:.4f} µS | Samples: {gsr.sample_count}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        gsr.stop()

        # Automatically plot results after recording (unless --no-plot)
        if not args.no_plot and gsr.csv_path and gsr.sample_count > 0:
            print("\nGenerating plots...")
            try:
                # Save plot alongside CSV
                plot_save_path = gsr.csv_path.replace('.csv', '_plot.png')
                plot_gsr_data(gsr.csv_path, save_path=plot_save_path)

                # Also show detailed analysis
                analysis_save_path = gsr.csv_path.replace('.csv', '_analysis.png')
                plot_gsr_analysis(gsr.csv_path, save_path=analysis_save_path)
            except Exception as e:
                print(f"Warning: Could not generate plots: {e}")


if __name__ == "__main__":
    main()
