
import asyncio
import struct
from bleak import BleakClient, BleakScanner
import numpy as np
from datetime import datetime
from collections import deque
from scipy import signal as scipy_signal
from scipy import integrate
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import threading

# Polar H10 BLE Service UUIDs
HF_LOW  = 0.15   # Hz
HF_HIGH = 0.40   # Hz
RR_INTERP_FS = 4  # Hz — uniform resampling rate for spectral HRV


def compute_hf_power(rr_array: np.ndarray) -> float:
    """
    Compute HF power (0.15–0.40 Hz) from an array of RR intervals (ms).

    Steps:
      1. Build cumulative time axis from RR intervals.
      2. Interpolate to a uniform 4 Hz time series.
      3. Apply Welch's method to estimate PSD.
      4. Integrate power in the HF band (0.15–0.40 Hz).

    Returns HF power in ms² (log-transform with np.log() for normality if needed).
    Returns np.nan if insufficient data.
    """
    if len(rr_array) < 10:
        return np.nan

    # Cumulative time in seconds (RR intervals are in ms)
    t_rr = np.cumsum(rr_array) / 1000.0
    t_rr -= t_rr[0]  # start at 0

    # Uniform time grid at RR_INTERP_FS Hz
    t_uniform = np.arange(0, t_rr[-1], 1.0 / RR_INTERP_FS)
    if len(t_uniform) < 8:
        return np.nan

    rr_uniform = np.interp(t_uniform, t_rr, rr_array)

    # Welch PSD
    nperseg = min(len(rr_uniform), int(RR_INTERP_FS * 60))  # up to 60-s segments
    freqs, psd = scipy_signal.welch(rr_uniform, fs=RR_INTERP_FS,
                                    window='hann', nperseg=nperseg,
                                    noverlap=nperseg // 2)

    # Integrate HF band
    hf_mask = (freqs >= HF_LOW) & (freqs <= HF_HIGH)
    if hf_mask.sum() < 2:
        return np.nan
    return float(integrate.trapezoid(psd[hf_mask], freqs[hf_mask]))


# Polar H10 BLE Service UUIDs
HEART_RATE_SERVICE_UUID     = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# Rolling window size for HRV metrics (number of RR intervals)
HRV_WINDOW = 30


class PolarH10:
    def __init__(self, device_name="Polar H10"):
        self.device_name = device_name
        self.device = None
        self.client = None
        self.rr_intervals = deque(maxlen=1000)
        self.heart_rates = deque(maxlen=100)

        # Time-series history for live plot (elapsed seconds)
        self.timestamps = deque(maxlen=300)
        self.rmssd_history = deque(maxlen=300)
        self.sdnn_history = deque(maxlen=300)
        self.hr_history = deque(maxlen=300)
        self.rr_history = deque(maxlen=300)

        self._start_time = None

    async def find_device(self):
        print(f"Scanning for {self.device_name}...")
        devices = await BleakScanner.discover(timeout=10.0)
        for device in devices:
            if device.name and self.device_name in device.name:
                print(f"Found device: {device.name} ({device.address})")
                self.device = device
                return device
        raise Exception(f"Could not find {self.device_name}")

    def parse_heart_rate_data(self, sender, data):
        """Parse BLE Heart Rate Measurement characteristic."""
        flags = data[0]
        hr_format = flags & 0x01
        rr_present = (flags & 0x10) != 0

        if hr_format == 0:
            heart_rate = data[1]
            offset = 2
        else:
            heart_rate = struct.unpack('<H', data[1:3])[0]
            offset = 3

        self.heart_rates.append(heart_rate)

        if rr_present:
            rr_data = data[offset:]
            num_rr = len(rr_data) // 2
            new_rr = []
            for i in range(num_rr):
                rr_raw = struct.unpack('<H', rr_data[i*2:(i+1)*2])[0]
                rr_ms = (rr_raw / 1024.0) * 1000.0
                self.rr_intervals.append(rr_ms)
                new_rr.append(rr_ms)

            # Record rolling metrics for each new RR interval received
            for rr_ms in new_rr:
                self.rr_history.append(rr_ms)
                self.hr_history.append(heart_rate)

                elapsed = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
                self.timestamps.append(elapsed)

                metrics = self._rolling_hrv_metrics()
                self.rmssd_history.append(metrics['rmssd'])
                self.sdnn_history.append(metrics['sdnn'])

            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] HR: {heart_rate} bpm | RR: {[f'{rr:.1f}ms' for rr in new_rr]}")

    def _rolling_hrv_metrics(self):
        """Compute HRV metrics over the last HRV_WINDOW RR intervals."""
        window = list(self.rr_intervals)[-HRV_WINDOW:]
        if len(window) < 2:
            return {'rmssd': 0.0, 'sdnn': 0.0}
        arr = np.array(window)
        diff = np.diff(arr)
        rmssd = float(np.sqrt(np.mean(diff**2)))
        sdnn = float(np.std(arr, ddof=1))
        return {'rmssd': rmssd, 'sdnn': sdnn}

    def calculate_hrv_metrics(self):
        """Final summary HRV metrics over all collected RR intervals."""
        if len(self.rr_intervals) < 2:
            return None
        rr_array = np.array(self.rr_intervals)
        mean_rr = np.mean(rr_array)
        sdnn = np.std(rr_array, ddof=1)
        diff_rr = np.diff(rr_array)
        rmssd = np.sqrt(np.mean(diff_rr**2))
        hf_power = compute_hf_power(rr_array)
        return {
            'mean_hr': 60000 / mean_rr if mean_rr > 0 else 0,
            'sdnn': sdnn,
            'rmssd': rmssd,
            'hf_power': hf_power,
            'num_intervals': len(rr_array),
        }

    def start_live_plot(self):
        """Launch the live HRV plot (must be called from the main thread)."""
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a2e')
        gs = gridspec.GridSpec(3, 1, hspace=0.45)

        ax_rr   = fig.add_subplot(gs[0])
        ax_hrv  = fig.add_subplot(gs[1])
        ax_hr   = fig.add_subplot(gs[2])

        for ax in (ax_rr, ax_hrv, ax_hr):
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='#e0e0e0')
            ax.xaxis.label.set_color('#e0e0e0')
            ax.yaxis.label.set_color('#e0e0e0')
            ax.title.set_color('#e0e0e0')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444466')

        line_rr,    = ax_rr.plot([], [], color='#00b4d8', lw=1.2)
        line_rmssd, = ax_hrv.plot([], [], color='#f72585', lw=1.5, label='RMSSD')
        line_sdnn,  = ax_hrv.plot([], [], color='#7209b7', lw=1.5, label='SDNN')
        line_hr,    = ax_hr.plot([], [], color='#4cc9f0', lw=1.5)

        ax_rr.set_title('RR Interval (ms)')
        ax_rr.set_ylabel('ms')
        ax_hrv.set_title(f'Rolling HRV (window = {HRV_WINDOW} beats)')
        ax_hrv.set_ylabel('ms')
        ax_hrv.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='#e0e0e0', fontsize=8)
        ax_hr.set_title('Heart Rate (bpm)')
        ax_hr.set_ylabel('bpm')
        ax_hr.set_xlabel('Time (s)')

        txt_rmssd = ax_hrv.text(0.99, 0.88, '', transform=ax_hrv.transAxes,
                                ha='right', va='top', color='#f72585', fontsize=9)
        txt_sdnn  = ax_hrv.text(0.99, 0.68, '', transform=ax_hrv.transAxes,
                                ha='right', va='top', color='#7209b7', fontsize=9)

        def update(_):
            if not self.timestamps:
                return line_rr, line_rmssd, line_sdnn, line_hr, txt_rmssd, txt_sdnn

            t = list(self.timestamps)
            rr    = list(self.rr_history)
            rmssd = list(self.rmssd_history)
            sdnn  = list(self.sdnn_history)
            hr    = list(self.hr_history)

            n = min(len(t), len(rr), len(rmssd), len(sdnn), len(hr))
            t, rr, rmssd, sdnn, hr = t[-n:], rr[-n:], rmssd[-n:], sdnn[-n:], hr[-n:]

            for line, y in [(line_rr, rr), (line_rmssd, rmssd), (line_sdnn, sdnn), (line_hr, hr)]:
                line.set_data(t, y)

            for ax, data in [(ax_rr, rr), (ax_hrv, rmssd + sdnn), (ax_hr, hr)]:
                ax.set_xlim(max(0, t[0]), t[-1] + 1)
                if data:
                    margin = (max(data) - min(data)) * 0.15 or 5
                    ax.set_ylim(min(data) - margin, max(data) + margin)

            if rmssd:
                txt_rmssd.set_text(f'RMSSD: {rmssd[-1]:.1f} ms')
            if sdnn:
                txt_sdnn.set_text(f'SDNN:  {sdnn[-1]:.1f} ms')

            return line_rr, line_rmssd, line_sdnn, line_hr, txt_rmssd, txt_sdnn

        ani = FuncAnimation(fig, update, interval=500, blit=False, cache_frame_data=False)
        plt.suptitle('Polar H10 — Live HRV Metrics', color='#e0e0e0', fontsize=13, y=0.98)

        # Keep a reference so GC doesn't kill the animation
        self._ani = ani
        try:
            plt.show()
        except KeyboardInterrupt:
            pass

    async def connect_and_stream(self):
        if not self.device:
            await self.find_device()
        if not self.device:
            raise Exception("Device not found, cannot connect")

        print(f"\nConnecting to {self.device.name}...")

        async with BleakClient(self.device.address) as client:
            self.client = client
            print(f"Connected: {client.is_connected}")
            self._start_time = datetime.now()

            await client.start_notify(HEART_RATE_MEASUREMENT_UUID, self.parse_heart_rate_data)
            print("Streaming HRV data — press Ctrl+C to stop.\n")

            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass

            await client.stop_notify(HEART_RATE_MEASUREMENT_UUID)


def run_ble(polar):
    """Run the asyncio BLE loop in a background thread."""
    async def _run():
        task = asyncio.create_task(polar.connect_and_stream())
        polar._ble_task = task
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())


if __name__ == "__main__":
    polar = PolarH10(device_name="Polar H10")

    # BLE runs in a background thread; plot stays on the main thread (macOS requirement)
    ble_thread = threading.Thread(target=run_ble, args=(polar,), daemon=True)
    ble_thread.start()

    try:
        # start_live_plot blocks on plt.show() — must be called from the main thread
        polar.start_live_plot()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopped — printing final summary...")
        # Cancel the BLE task so connect_and_stream exits cleanly
        task = getattr(polar, '_ble_task', None)
        if task and not task.done():
            task.cancel()
        ble_thread.join(timeout=3)

        metrics = polar.calculate_hrv_metrics()
        if metrics:
            print("\n--- Final HRV Summary ---")
            print(f"Number of RR intervals : {metrics['num_intervals']}")
            print(f"Mean Heart Rate        : {metrics['mean_hr']:.1f} bpm")
            print(f"SDNN                   : {metrics['sdnn']:.1f} ms")
            print(f"RMSSD                  : {metrics['rmssd']:.1f} ms")
            hf = metrics['hf_power']
            print(f"HF Power               : {hf:.2f} ms²" if not np.isnan(hf) else "HF Power               : N/A (insufficient data)")
            print("="*70)
