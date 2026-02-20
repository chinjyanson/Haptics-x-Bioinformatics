"""
Muse 2 EEG Stream Reader with Real-time Processing
Connects directly to Muse 2 using BrainFlow and applies denoising/filtering

Data Processing Pipeline:
    RAW -> Save -> LIGHT (bandpass+notch) -> Save -> HEAVY (full pipeline) -> Save -> Analyze

Key Notes:
    - Muse 2 has 4 EEG channels via Bluetooth: TP9, AF7, AF8, TP10
    - Fpz is hardware reference (not accessible as separate channel)
    - All EEG measurements are differential: EEG_measured = EEG_site - EEG_Fpz
    - ERP polarity is inverted (negative-going P300) due to Fpz reference
"""
import time
import argparse
import logging
import threading
from collections import deque
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from scipy.signal import welch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import mne
from mne.preprocessing import ICA
import pywt
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA

# =============================================================================
# CONFIGURATION SECTION - All tunable parameters
# =============================================================================

# Device Configuration
MUSE_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
MUSE_SAMPLING_RATE = 256  # Hz (will be overwritten by actual device rate)

# Light Filtering Parameters (for ERP analysis)
LIGHT_BANDPASS_LOW = 0.1   # Hz
LIGHT_BANDPASS_HIGH = 30.0  # Hz
LIGHT_NOTCH_FREQ = 50.0     # Hz (UK power line, use 60 for US)

# Heavy Processing Parameters
HEAVY_BANDPASS_LOW = 1.0    # Hz (FIR bandpass)
HEAVY_BANDPASS_HIGH = 40.0  # Hz (FIR bandpass)

# EOG Regression Parameters (using AF7/AF8 frontal channels as EOG proxy)
EOG_PROXY_CHANNELS = ['AF7', 'AF8']  # Frontal channels contain most eye artifacts
EOG_REGRESSION_LAGS = list(range(11))  # Lags 0-10 samples

# ICA Parameters
ICA_N_COMPONENTS = None  # None = use n_channels
ICA_RANDOM_STATE = 42    # Fixed seed for reproducibility
ICA_MAX_ITER = 500
ICA_VARIANCE_THRESHOLD = 1.5  # Exclude components with variance > mean + threshold*std

# Wavelet Denoising Parameters
WAVELET_TYPE = 'db4'     # Daubechies-4 wavelet
WAVELET_LEVEL = 4        # Decomposition level
WAVELET_THRESHOLD_MODE = 'soft'

# CCA Parameters (EMG suppression)
CCA_N_COMPONENTS = 2     # Number of canonical components

# ERP Detection Parameters
ERP_BASELINE_START = -0.2  # seconds before stimulus
ERP_BASELINE_END = 0.0     # stimulus onset
ERP_P300_WINDOW_START = 0.25  # seconds after stimulus
ERP_P300_WINDOW_END = 0.5     # seconds after stimulus
ERP_DETECT_NEGATIVE = True    # True because Fpz reference inverts polarity

# Band Power Parameters
BAND_DEFINITIONS = {
    'alpha': (8, 12),    # Hz
    'beta': (13, 30),    # Hz
    'gamma': (30, 40),   # Hz
}
WELCH_WINDOW_SIZE = 2.0   # seconds
WELCH_OVERLAP = 0.5       # 50% overlap

# Processing Pipeline Toggles
ENABLE_EOG_REGRESSION = True
ENABLE_ICA = True
ENABLE_WAVELET = True
ENABLE_CCA = True

# Output Configuration
OUTPUT_FORMAT = '.fif'    # MNE native format
VERBOSE_LOGGING = True

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO if VERBOSE_LOGGING else logging.WARNING,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Read EEG from Muse 2 headband')
    parser.add_argument('--serial-port', type=str, help='Serial port for BLED112 dongle (e.g., COM3)')
    parser.add_argument('--mac-address', type=str, help='MAC address of your Muse 2')
    return parser.parse_args()


class MuseBrainFlowProcessor:
    """
    Muse 2 EEG processor with comprehensive artifact removal and analysis.

    Maintains backward compatibility with main.py through:
        - __init__(buffer_duration, serial_port, mac_address)
        - get_data() -> ndarray(4, n_samples) or None
        - stop()
        - attributes: sampling_rate, channels, n_channels
    """

    def __init__(self, buffer_duration=10, serial_port=None, mac_address=None):
        """
        Initialize the Muse BrainFlow stream reader.

        Args:
            buffer_duration: How many seconds of data to keep in buffer
            serial_port: Serial port for BLED112 dongle (optional)
            mac_address: MAC address of Muse 2 (optional)
        """
        self.buffer_duration = buffer_duration

        # Channel names for Muse 2 (4 channels via Bluetooth)
        self.channels = MUSE_CHANNELS.copy()
        self.n_channels = len(self.channels)

        # Set up BrainFlow connection
        params = BrainFlowInputParams()

        if serial_port:
            logger.info(f"Using BLED dongle on port: {serial_port}")
            params.serial_port = serial_port
            self.board_id = BoardIds.MUSE_2_BLED_BOARD
        else:
            logger.info("Using native Bluetooth connection")
            self.board_id = BoardIds.MUSE_2_BOARD
            if mac_address:
                params.mac_address = mac_address
                logger.info(f"Connecting to specific device: {mac_address}")

        # Create board and connect
        logger.info("Connecting to Muse 2...")
        self.board = BoardShim(self.board_id, params)
        self.board.prepare_session()

        # Get sampling rate and EEG channels from BrainFlow
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.buffer_size = int(buffer_duration * self.sampling_rate)

        # Initialize buffers for each channel
        self.buffers = [deque(maxlen=self.buffer_size) for _ in range(self.n_channels)]

        # Storage for processed data
        self.raw_data = None
        self.light_data = None
        self.heavy_data = None

        # ERP and band power results
        self.erp_results = None
        self.band_power_results = None

        # Start streaming
        self.board.start_stream()
        logger.info(f"Connected to Muse 2 (sampling rate: {self.sampling_rate} Hz)")

    # =========================================================================
    # CORE COMPATIBILITY METHODS (Required by main.py)
    # =========================================================================

    def get_data(self):
        """
        Pull available data from the board.

        Returns:
            ndarray(4, n_samples) or None if no data available
        """
        data = self.board.get_board_data()
        if data.size == 0:
            return None
        # Extract EEG channels
        eeg_data = data[self.eeg_channels, :]
        return eeg_data

    def stop(self):
        """Stop streaming and release the session."""
        self.board.stop_stream()
        self.board.release_session()
        logger.info("Session ended.")

    # =========================================================================
    # DATA SAVING METHODS
    # =========================================================================

    def _create_mne_raw(self, data, description=""):
        """
        Create MNE RawArray from numpy data.

        Args:
            data: numpy array (n_channels x n_samples)
            description: Description string for the info object

        Returns:
            mne.io.RawArray object
        """
        info = mne.create_info(
            ch_names=self.channels,
            sfreq=self.sampling_rate,
            ch_types='eeg'
        )
        info['description'] = description

        # Scale to volts if data is in microvolts
        # MNE expects data in volts
        data_volts = data * 1e-6 if np.max(np.abs(data)) > 1 else data

        raw = mne.io.RawArray(data_volts, info, verbose=False)
        return raw

    def save_raw(self, data, filepath, description="RAW"):
        """
        Save data to MNE .fif format.

        Args:
            data: numpy array (n_channels x n_samples)
            filepath: Output path (with or without .fif extension)
            description: Description of the data stage
        """
        filepath = Path(filepath)
        if filepath.suffix != '.fif':
            filepath = filepath.with_suffix('.fif')

        raw = self._create_mne_raw(data, description)
        raw.save(str(filepath), overwrite=True, verbose=False)
        logger.info(f"Saved {description} data to {filepath}")

    # =========================================================================
    # LIGHT FILTERING (For ERP Analysis)
    # =========================================================================

    def apply_light_filter(self, data):
        """
        Apply light filtering for ERP analysis.

        Pipeline:
            1. Bandpass filter (0.1-30 Hz)
            2. Notch filter (50 Hz)

        Args:
            data: Raw EEG data (n_channels x n_samples)

        Returns:
            Lightly filtered data
        """
        logger.info("Applying light filtering (bandpass + notch)...")

        filtered = data.copy()

        # Apply bandpass filter
        filtered = self._apply_bandpass(
            filtered,
            lowcut=LIGHT_BANDPASS_LOW,
            highcut=LIGHT_BANDPASS_HIGH
        )

        # Apply notch filter
        filtered = self._apply_notch(filtered, freq=LIGHT_NOTCH_FREQ)

        logger.info("Light filtering complete.")
        return filtered

    def _apply_bandpass(self, data, lowcut, highcut, order=4):
        """Apply butterworth bandpass filter."""
        nyquist = self.sampling_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist

        # Ensure frequency bounds are valid
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))

        try:
            b, a = signal.butter(order, [low, high], btype='band')
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch] = signal.filtfilt(b, a, data[ch])
            return filtered
        except Exception as e:
            logger.warning(f"Bandpass filter failed: {e}. Returning original data.")
            return data

    def _apply_notch(self, data, freq, Q=30):
        """Apply notch filter to remove power line interference."""
        nyquist = self.sampling_rate / 2
        w0 = freq / nyquist

        if w0 >= 1.0:
            logger.warning(f"Notch frequency {freq} Hz too high for sampling rate. Skipping.")
            return data

        try:
            b, a = signal.iirnotch(w0, Q)
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch] = signal.filtfilt(b, a, data[ch])
            return filtered
        except Exception as e:
            logger.warning(f"Notch filter failed: {e}. Returning original data.")
            return data

    # =========================================================================
    # HEAVY PROCESSING PIPELINE
    # =========================================================================

    def apply_heavy_processing(self, data):
        """
        Apply full heavy processing pipeline for oscillatory analysis.

        Pipeline (all toggleable via configuration):
            1. FIR Bandpass (1-40 Hz)
            2. EOG Regression (using AF7/AF8 as proxy)
            3. ICA artifact removal
            4. Wavelet denoising (db4)
            5. CCA for EMG suppression

        Args:
            data: Input EEG data (n_channels x n_samples)

        Returns:
            Heavily processed data
        """
        logger.info("Starting heavy processing pipeline...")

        processed = data.copy()

        # Step 1: FIR Bandpass
        logger.info("  Step 1/5: Applying FIR bandpass filter...")
        processed = self._apply_bandpass(
            processed,
            lowcut=HEAVY_BANDPASS_LOW,
            highcut=HEAVY_BANDPASS_HIGH
        )

        # Step 2: EOG Regression
        if ENABLE_EOG_REGRESSION:
            logger.info("  Step 2/5: Applying EOG regression...")
            processed = self.perform_regression(processed)
        else:
            logger.info("  Step 2/5: EOG regression disabled, skipping.")

        # Step 3: ICA
        if ENABLE_ICA:
            logger.info("  Step 3/5: Applying ICA artifact removal...")
            processed = self.perform_ICA(processed)
        else:
            logger.info("  Step 3/5: ICA disabled, skipping.")

        # Step 4: Wavelet Denoising
        if ENABLE_WAVELET:
            logger.info("  Step 4/5: Applying wavelet denoising...")
            processed = self.wavelet_denoise(processed)
        else:
            logger.info("  Step 4/5: Wavelet denoising disabled, skipping.")

        # Step 5: CCA for EMG suppression
        if ENABLE_CCA:
            logger.info("  Step 5/5: Applying CCA for EMG suppression...")
            processed = self.apply_CCA(processed)
        else:
            logger.info("  Step 5/5: CCA disabled, skipping.")

        logger.info("Heavy processing pipeline complete.")
        return processed

    def perform_regression(self, data):
        """
        Apply time-lagged regression for EOG artifact removal.

        Uses AF7/AF8 frontal channels as EOG proxy since Muse 2
        doesn't have dedicated EOG channels.

        Formula: eeg_clean(t) = EEG(t) - sum_{g=0}^{L} beta_g * EOG(t-g)

        Args:
            data: EEG data (n_channels x n_samples)

        Returns:
            Data with EOG artifacts regressed out
        """
        if data.shape[1] < max(EOG_REGRESSION_LAGS) + 10:
            logger.warning("Not enough samples for EOG regression. Skipping.")
            return data

        # Get EOG proxy channel indices
        eog_indices = [self.channels.index(ch) for ch in EOG_PROXY_CHANNELS
                       if ch in self.channels]

        if not eog_indices:
            logger.warning("EOG proxy channels not found. Skipping regression.")
            return data

        # Create EOG reference as average of frontal channels
        eog_ref = np.mean(data[eog_indices, :], axis=0)

        # Build lagged design matrix
        n_samples = data.shape[1]
        max_lag = max(EOG_REGRESSION_LAGS)

        # Create lagged features
        X = np.zeros((n_samples - max_lag, len(EOG_REGRESSION_LAGS)))
        for i, lag in enumerate(EOG_REGRESSION_LAGS):
            X[:, i] = eog_ref[max_lag - lag:n_samples - lag]

        # Apply regression to each channel
        cleaned = data.copy()
        for ch in range(self.n_channels):
            # Skip EOG proxy channels themselves
            if ch in eog_indices:
                continue

            y = data[ch, max_lag:]

            reg = LinearRegression()
            reg.fit(X, y)

            # Subtract predicted EOG artifacts
            predicted_eog = reg.predict(X)
            cleaned[ch, max_lag:] = y - predicted_eog

        return cleaned

    def perform_ICA(self, data):
        """
        Apply ICA-based artifact removal using MNE.

        Excludes components with variance > mean + threshold * std

        Args:
            data: EEG data (n_channels x n_samples)

        Returns:
            Data with artifact ICA components removed
        """
        if data.shape[1] < 100:
            logger.warning("Not enough samples for ICA. Skipping.")
            return data

        n_components = ICA_N_COMPONENTS or self.n_channels

        try:
            raw = self._create_mne_raw(data, "ICA input")

            ica = ICA(
                n_components=n_components,
                random_state=ICA_RANDOM_STATE,
                max_iter=ICA_MAX_ITER,
                verbose=False
            )
            ica.fit(raw)

            # Identify artifact components by variance
            sources = ica.get_sources(raw).get_data()
            component_var = np.var(sources, axis=1)
            threshold = np.mean(component_var) + ICA_VARIANCE_THRESHOLD * np.std(component_var)
            artifact_indices = np.where(component_var > threshold)[0].tolist()

            if artifact_indices:
                logger.info(f"    Excluding {len(artifact_indices)} ICA components: {artifact_indices}")

            ica.exclude = artifact_indices
            raw_clean = ica.apply(raw.copy())

            # Convert back to original scale
            denoised = raw_clean.get_data() * 1e6  # Convert back to microvolts

            return denoised

        except Exception as e:
            logger.warning(f"ICA failed: {e}. Returning original data.")
            return data

    def wavelet_denoise(self, data):
        """
        Apply wavelet denoising using PyWavelets.

        Uses db4 wavelet with soft thresholding.

        Args:
            data: EEG data (n_channels x n_samples)

        Returns:
            Wavelet-denoised data
        """
        denoised = np.zeros_like(data)

        for ch in range(data.shape[0]):
            try:
                # Decompose
                coeffs = pywt.wavedec(data[ch], WAVELET_TYPE, level=WAVELET_LEVEL)

                # Calculate threshold using universal threshold
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(data[ch])))

                # Apply soft thresholding to detail coefficients
                denoised_coeffs = [coeffs[0]]  # Keep approximation
                for c in coeffs[1:]:
                    denoised_coeffs.append(pywt.threshold(c, threshold, mode=WAVELET_THRESHOLD_MODE))

                # Reconstruct
                denoised[ch] = pywt.waverec(denoised_coeffs, WAVELET_TYPE)[:data.shape[1]]

            except Exception as e:
                logger.warning(f"Wavelet denoising failed for channel {ch}: {e}")
                denoised[ch] = data[ch]

        return denoised

    def apply_CCA(self, data):
        """
        Apply Canonical Correlation Analysis for EMG suppression.

        Uses delayed copies of the signal to identify and suppress
        high-frequency EMG artifacts.

        Args:
            data: EEG data (n_channels x n_samples)

        Returns:
            Data with EMG suppressed
        """
        if data.shape[1] < 100:
            logger.warning("Not enough samples for CCA. Skipping.")
            return data

        try:
            # Create delayed version for CCA
            delay = 1  # 1 sample delay
            X = data[:, :-delay].T
            Y = data[:, delay:].T

            cca = CCA(n_components=min(CCA_N_COMPONENTS, self.n_channels))
            cca.fit(X, Y)

            # Transform and inverse transform to suppress noise
            X_c, Y_c = cca.transform(X, Y)

            # Reconstruct using only the canonical components
            # This suppresses components that don't correlate well across time
            X_reconstructed = cca.inverse_transform(X_c)

            # Create output array matching input dimensions
            result = np.zeros_like(data)
            result[:, :-delay] = X_reconstructed.T
            result[:, -delay:] = data[:, -delay:]  # Keep last samples unchanged

            return result

        except Exception as e:
            logger.warning(f"CCA failed: {e}. Returning original data.")
            return data

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def detect_ERP(self, data, events, event_id=1, tmin=-0.2, tmax=0.8):
        """
        Detect ERP components from epoched data.

        IMPORTANT: Due to Fpz hardware reference, P300 appears as NEGATIVE peak
        (measured = true - Fpz_reference -> polarity inversion)

        Args:
            data: EEG data (n_channels x n_samples) - should be RAW or LIGHT
            events: Event array (n_events x 3) in MNE format [sample, 0, event_id]
            event_id: Event code to epoch
            tmin: Epoch start time relative to event (default -0.2s)
            tmax: Epoch end time relative to event (default 0.8s)

        Returns:
            dict with ERP waveforms and detected peaks
        """
        logger.info("Detecting ERP components...")

        try:
            raw = self._create_mne_raw(data, "ERP analysis")

            # Create epochs
            epochs = mne.Epochs(
                raw, events, event_id,
                tmin=tmin, tmax=tmax,
                baseline=(ERP_BASELINE_START, ERP_BASELINE_END),
                preload=True, verbose=False
            )

            # Compute evoked response
            evoked = epochs.average()

            # Get data and times
            erp_data = evoked.data * 1e6  # Convert to microvolts
            times = evoked.times

            # Detect P300 (negative peak due to Fpz reference)
            p300_mask = (times >= ERP_P300_WINDOW_START) & (times <= ERP_P300_WINDOW_END)
            p300_times = times[p300_mask]

            peaks = {}
            for ch_idx, ch_name in enumerate(self.channels):
                ch_data = erp_data[ch_idx, p300_mask]

                if ERP_DETECT_NEGATIVE:
                    # Find negative peak (P300 with Fpz reference)
                    peak_idx = np.argmin(ch_data)
                    peak_amplitude = ch_data[peak_idx]
                else:
                    # Find positive peak (standard P300)
                    peak_idx = np.argmax(ch_data)
                    peak_amplitude = ch_data[peak_idx]

                peak_latency = p300_times[peak_idx]

                peaks[ch_name] = {
                    'latency': peak_latency,
                    'amplitude': peak_amplitude
                }

            self.erp_results = {
                'evoked_data': erp_data,
                'times': times,
                'peaks': peaks,
                'n_epochs': len(epochs),
                'channels': self.channels
            }

            logger.info(f"ERP detection complete. {len(epochs)} epochs averaged.")
            for ch, p in peaks.items():
                logger.info(f"  {ch}: P300 at {p['latency']*1000:.0f}ms, {p['amplitude']:.2f}uV")

            return self.erp_results

        except Exception as e:
            logger.error(f"ERP detection failed: {e}")
            return None

    def compute_band_power(self, data, window_duration=None):
        """
        Compute band power using Welch's method.

        Bands: Alpha (8-12 Hz), Beta (13-30 Hz), Gamma (30-40 Hz)

        Args:
            data: EEG data (n_channels x n_samples) - should be HEAVY processed
            window_duration: Window size in seconds (default from config)

        Returns:
            dict with band powers for each channel
        """
        logger.info("Computing band power using Welch PSD...")

        if window_duration is None:
            window_duration = WELCH_WINDOW_SIZE

        nperseg = int(window_duration * self.sampling_rate)
        noverlap = int(nperseg * WELCH_OVERLAP)

        band_powers = {band: {} for band in BAND_DEFINITIONS}
        psd_data = {}

        for ch_idx, ch_name in enumerate(self.channels):
            # Compute PSD using Welch's method
            freqs, psd = welch(
                data[ch_idx],
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap
            )

            psd_data[ch_name] = {'freqs': freqs, 'psd': psd}

            # Extract band powers
            for band_name, (low_freq, high_freq) in BAND_DEFINITIONS.items():
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                band_powers[band_name][ch_name] = band_power

        self.band_power_results = {
            'band_powers': band_powers,
            'psd_data': psd_data,
            'channels': self.channels,
            'band_definitions': BAND_DEFINITIONS
        }

        logger.info("Band power computation complete.")
        for band, ch_powers in band_powers.items():
            avg_power = np.mean(list(ch_powers.values()))
            logger.info(f"  {band.capitalize()}: avg={avg_power:.4f}")

        return self.band_power_results

    def compute_band_power_over_time(self, data, window_duration=None, step_duration=None):
        """
        Compute band power over time using sliding windows.

        Args:
            data: EEG data (n_channels x n_samples)
            window_duration: Window size in seconds
            step_duration: Step size in seconds (default: window_duration/2)

        Returns:
            dict with time series of band powers
        """
        if window_duration is None:
            window_duration = WELCH_WINDOW_SIZE
        if step_duration is None:
            step_duration = window_duration / 2

        window_samples = int(window_duration * self.sampling_rate)
        step_samples = int(step_duration * self.sampling_rate)
        n_samples = data.shape[1]

        time_points = []
        band_power_series = {band: {ch: [] for ch in self.channels}
                            for band in BAND_DEFINITIONS}

        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            window_data = data[:, start:end]

            time_points.append((start + window_samples/2) / self.sampling_rate)

            # Compute band power for this window
            for ch_idx, ch_name in enumerate(self.channels):
                freqs, psd = welch(
                    window_data[ch_idx],
                    fs=self.sampling_rate,
                    nperseg=min(window_samples, 256),
                    noverlap=min(window_samples // 2, 128)
                )

                for band_name, (low_freq, high_freq) in BAND_DEFINITIONS.items():
                    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    if np.any(freq_mask):
                        band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                    else:
                        band_power = 0
                    band_power_series[band_name][ch_name].append(band_power)

        return {
            'time_points': np.array(time_points),
            'band_power_series': band_power_series
        }

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def generate_final_report(self, raw_data, light_data, heavy_data,
                              erp_results=None, band_power_results=None,
                              save_path=None):
        """
        Generate 5-panel visualization report.

        Panels:
            1. ERP waveforms (from RAW/LIGHT)
            2. Time-domain heavy-processed signals
            3. Alpha power over time
            4. Beta power over time
            5. Gamma power over time

        Args:
            raw_data: RAW EEG data
            light_data: LIGHT filtered data
            heavy_data: HEAVY processed data
            erp_results: Results from detect_ERP()
            band_power_results: Results from compute_band_power_over_time()
            save_path: Path to save figure (optional)
        """
        logger.info("Generating final report visualization...")

        fig = plt.figure(figsize=(16, 12))

        # Panel 1: ERP Waveforms
        ax1 = fig.add_subplot(3, 2, 1)
        if erp_results is not None:
            times = erp_results['times'] * 1000  # Convert to ms
            for ch_idx, ch_name in enumerate(self.channels):
                ax1.plot(times, erp_results['evoked_data'][ch_idx],
                        label=ch_name, linewidth=1.5)
            ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Stimulus')
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Amplitude (uV)')
            ax1.set_title('ERP Waveforms (Note: P300 is negative due to Fpz reference)')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No ERP data available',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('ERP Waveforms')

        # Panel 2: Time-domain heavy signals
        ax2 = fig.add_subplot(3, 2, 2)
        time_axis = np.arange(heavy_data.shape[1]) / self.sampling_rate
        for ch_idx, ch_name in enumerate(self.channels):
            offset = ch_idx * 100  # Offset for visibility
            ax2.plot(time_axis, heavy_data[ch_idx] + offset,
                    label=ch_name, linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude (uV) + offset')
        ax2.set_title('Heavy-Processed EEG Signals')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panels 3-5: Band power over time
        if band_power_results is not None:
            time_points = band_power_results['time_points']
            band_series = band_power_results['band_power_series']

            for idx, (band_name, (low, high)) in enumerate(BAND_DEFINITIONS.items()):
                ax = fig.add_subplot(3, 2, 3 + idx)
                for ch_name in self.channels:
                    ax.plot(time_points, band_series[band_name][ch_name],
                           label=ch_name, linewidth=1)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Power')
                ax.set_title(f'{band_name.capitalize()} Power ({low}-{high} Hz)')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
        else:
            for idx in range(3):
                ax = fig.add_subplot(3, 2, 3 + idx)
                band_name = list(BAND_DEFINITIONS.keys())[idx]
                ax.text(0.5, 0.5, f'No {band_name} power data available',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{band_name.capitalize()} Power')

        # Panel 6: Power spectrum comparison
        ax6 = fig.add_subplot(3, 2, 6)
        for ch_idx, ch_name in enumerate(self.channels):
            freqs, psd = welch(heavy_data[ch_idx], fs=self.sampling_rate, nperseg=512)
            ax6.semilogy(freqs, psd, label=ch_name, linewidth=1)
        ax6.set_xlabel('Frequency (Hz)')
        ax6.set_ylabel('PSD (log scale)')
        ax6.set_title('Power Spectral Density')
        ax6.set_xlim(0, 50)
        ax6.legend(loc='upper right', fontsize=8)
        ax6.grid(True, alpha=0.3)

        # Add vertical lines for band boundaries
        for band_name, (low, high) in BAND_DEFINITIONS.items():
            ax6.axvspan(low, high, alpha=0.1, label=f'{band_name}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Report saved to {save_path}")

        plt.show()

        return fig

    # =========================================================================
    # FULL PROCESSING WORKFLOW
    # =========================================================================

    def run_full_processing(self, data, output_dir, events=None):
        """
        Run the complete processing pipeline with strict data saving order.

        Order: Acquire -> Save RAW -> Light filter -> Save LIGHT ->
               Heavy process -> Save HEAVY -> Analyze -> Generate Report

        Args:
            data: Raw EEG data (n_channels x n_samples)
            output_dir: Directory to save output files
            events: Optional event array for ERP analysis

        Returns:
            dict with all results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("STARTING FULL PROCESSING PIPELINE")
        logger.info("=" * 60)

        # Step 1: Save RAW data
        logger.info("\n[1/6] Saving RAW data...")
        self.raw_data = data.copy()
        self.save_raw(self.raw_data, output_dir / 'data_RAW.fif', 'RAW')

        # Step 2: Apply light filtering
        logger.info("\n[2/6] Applying light filtering...")
        self.light_data = self.apply_light_filter(self.raw_data)

        # Step 3: Save LIGHT data
        logger.info("\n[3/6] Saving LIGHT data...")
        self.save_raw(self.light_data, output_dir / 'data_LIGHT.fif', 'LIGHT')

        # Step 4: Apply heavy processing
        logger.info("\n[4/6] Applying heavy processing...")
        self.heavy_data = self.apply_heavy_processing(self.light_data)

        # Step 5: Save HEAVY data
        logger.info("\n[5/6] Saving HEAVY data...")
        self.save_raw(self.heavy_data, output_dir / 'data_HEAVY.fif', 'HEAVY')

        # Step 6: Analysis
        logger.info("\n[6/6] Running analysis...")

        # ERP detection (uses RAW or LIGHT data)
        erp_results = None
        if events is not None and len(events) > 0:
            erp_results = self.detect_ERP(self.light_data, events)

        # Band power analysis (uses HEAVY data)
        band_power_results = self.compute_band_power(self.heavy_data)
        band_power_time_results = self.compute_band_power_over_time(self.heavy_data)

        # Generate report
        logger.info("\nGenerating final report...")
        self.generate_final_report(
            self.raw_data,
            self.light_data,
            self.heavy_data,
            erp_results=erp_results,
            band_power_results=band_power_time_results,
            save_path=output_dir / 'analysis_report.png'
        )

        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output files saved to: {output_dir}")

        return {
            'raw_data': self.raw_data,
            'light_data': self.light_data,
            'heavy_data': self.heavy_data,
            'erp_results': erp_results,
            'band_power_results': band_power_results,
            'band_power_time_results': band_power_time_results
        }

    # =========================================================================
    # LEGACY METHODS (for backward compatibility)
    # =========================================================================

    def apply_bandpass_filter(self, data, lowcut=1, highcut=40):
        """
        Apply bandpass filter to remove noise.

        Args:
            data: Raw EEG data (1D array for single channel)
            lowcut: Low frequency cutoff (Hz)
            highcut: High frequency cutoff (Hz)

        Returns:
            Filtered data
        """
        nyquist = self.sampling_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist

        try:
            result = signal.butter(4, [low, high], btype='band')
            if result is None or len(result) != 2:
                filtered = data
            else:
                b, a = result
                filtered = signal.filtfilt(b, a, data)
        except Exception:
            filtered = data
        return filtered

    def apply_notch_filter(self, data, freq=60):
        """
        Apply notch filter to remove power line interference.

        Args:
            data: Raw EEG data (1D array for single channel)
            freq: Frequency to remove (Hz, default 60 for US)

        Returns:
            Filtered data
        """
        nyquist = self.sampling_rate / 2
        w0 = freq / nyquist
        Q = 30

        if w0 >= 1.0:
            return data

        b, a = signal.iirnotch(w0, Q)
        filtered = signal.filtfilt(b, a, data)
        return filtered

    def apply_ica_denoising(self, data, n_components=None):
        """
        Apply ICA-based artifact removal using MNE.
        Legacy method for backward compatibility.

        Args:
            data: Raw EEG data (n_channels x n_samples)
            n_components: Number of ICA components (default: n_channels)

        Returns:
            Denoised data with artifact components removed
        """
        return self.perform_ICA(data)

    def process_stream(self, duration=None, apply_filter=True, apply_ica=True):
        """
        Stream and process data in real-time.

        Args:
            duration: How long to stream (seconds). None = infinite
            apply_filter: Whether to apply bandpass and notch filters
            apply_ica: Whether to apply ICA-based artifact removal

        Yields:
            (raw_data, processed_data) tuples
        """
        start_time = time.time()

        try:
            while True:
                # Check duration
                if duration is not None:
                    if time.time() - start_time > duration:
                        break

                # Get new data batch
                eeg_data = self.get_data()

                if eeg_data is not None and eeg_data.shape[1] > 0:
                    # Add samples to buffers
                    for sample_idx in range(eeg_data.shape[1]):
                        for ch in range(self.n_channels):
                            self.buffers[ch].append(eeg_data[ch, sample_idx])

                # Get current buffer data
                raw_data = np.array([np.array(self.buffers[ch]) for ch in range(self.n_channels)])

                # Apply processing
                if apply_filter and len(self.buffers[0]) > 100:
                    processed_data = raw_data.copy()

                    # Apply ICA denoising first (artifact removal)
                    if apply_ica and processed_data.shape[1] > 100:
                        processed_data = self.apply_ica_denoising(processed_data)

                    # Then apply notch and bandpass filters
                    filtered_data = np.zeros_like(processed_data)
                    for ch in range(self.n_channels):
                        # Apply notch filter (remove 60 Hz)
                        temp = self.apply_notch_filter(processed_data[ch])
                        # Apply bandpass filter (1-40 Hz)
                        filtered_data[ch] = self.apply_bandpass_filter(temp)

                    processed_data = filtered_data
                else:
                    processed_data = raw_data

                yield raw_data, processed_data

                # Small delay to prevent busy-waiting
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nStreaming stopped by user")

    def collect_and_analyze(self, duration=60, save_dir=None):
        """
        Collect data for specified duration, then analyze and display results.

        This is a POST-HOC analysis - collects data first, then shows:
            1. Detected P300 candidates (negative peaks in light-filtered data)
            2. Alpha power over time
            3. Beta power over time
            4. Gamma power over time
            5. Power Spectral Density
            6. Summary statistics

        Args:
            duration: Collection duration in seconds (default 60)
            save_dir: Optional directory to save output files

        Returns:
            dict with collected data and analysis results
        """
        logger.info(f"Collecting data for {duration} seconds...")
        print(f"Collecting EEG data for {duration} seconds...")
        print("Please remain still and relaxed.")

        # Collect raw data
        all_data = []
        start_time = time.time()

        while time.time() - start_time < duration:
            eeg_data = self.get_data()
            if eeg_data is not None and eeg_data.shape[1] > 0:
                all_data.append(eeg_data)

            # Progress indicator
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                remaining = duration - elapsed
                if remaining > 0:
                    print(f"  {int(remaining)}s remaining...")

            time.sleep(0.01)

        if not all_data:
            logger.error("No data collected!")
            return None

        # Concatenate all collected data
        raw_data = np.hstack(all_data)
        logger.info(f"Collected {raw_data.shape[1]} samples ({raw_data.shape[1]/self.sampling_rate:.1f}s)")
        print(f"\nCollection complete! {raw_data.shape[1]} samples collected.")
        print("Processing and analyzing data...")

        # Apply processing pipelines
        light_data = self.apply_light_filter(raw_data)
        heavy_data = self.apply_heavy_processing(raw_data)

        # Detect P300 candidates
        p300_candidates = self._detect_p300_candidates(light_data)

        # Compute band power over time
        band_power_time = self.compute_band_power_over_time(heavy_data)

        # Compute overall band power
        band_power = self.compute_band_power(heavy_data)

        # Save data if directory provided (create directory first!)
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.save_raw(raw_data, save_dir / 'data_RAW.fif', 'RAW')
            self.save_raw(light_data, save_dir / 'data_LIGHT.fif', 'LIGHT')
            self.save_raw(heavy_data, save_dir / 'data_HEAVY.fif', 'HEAVY')

        # Generate visualization (after directory is created)
        self._plot_analysis_results(
            raw_data, light_data, heavy_data,
            p300_candidates, band_power_time, band_power,
            save_dir
        )

        return {
            'raw_data': raw_data,
            'light_data': light_data,
            'heavy_data': heavy_data,
            'p300_candidates': p300_candidates,
            'band_power_time': band_power_time,
            'band_power': band_power
        }

    def _detect_p300_candidates(self, light_data, min_amplitude=10, max_amplitude=100):
        """
        Detect P300 candidate events in light-filtered data.

        P300 appears as NEGATIVE peaks due to Fpz reference.
        Looks for significant negative deflections that could be P300.

        Args:
            light_data: Light-filtered EEG data (n_channels x n_samples)
            min_amplitude: Minimum peak amplitude (uV) to consider
            max_amplitude: Maximum peak amplitude (uV) to consider (reject artifacts)

        Returns:
            dict with detected candidates per channel
        """
        from scipy.signal import find_peaks

        candidates = {}

        for ch_idx, ch_name in enumerate(self.channels):
            # Invert signal to find negative peaks as positive peaks
            inverted = -light_data[ch_idx]

            # Find peaks (which are negative peaks in original)
            # Require minimum distance of 200ms between peaks
            min_distance = int(0.2 * self.sampling_rate)

            peaks, properties = find_peaks(
                inverted,
                height=(min_amplitude, max_amplitude),
                distance=min_distance,
                prominence=5  # Minimum prominence to be considered a real peak
            )

            # Convert to times and amplitudes
            peak_times = peaks / self.sampling_rate
            peak_amplitudes = -inverted[peaks]  # Back to original (negative) values

            # Filter to keep only the most prominent candidates (top 10)
            if len(peaks) > 10:
                prominences = properties.get('prominences', np.abs(peak_amplitudes))
                top_indices = np.argsort(prominences)[-10:]
                peaks = peaks[top_indices]
                peak_times = peak_times[top_indices]
                peak_amplitudes = peak_amplitudes[top_indices]

            candidates[ch_name] = {
                'sample_indices': peaks,
                'times': peak_times,
                'amplitudes': peak_amplitudes,
                'count': len(peaks)
            }

            logger.info(f"  {ch_name}: {len(peaks)} P300 candidates detected")

        return candidates

    def _plot_analysis_results(self, raw_data, light_data, heavy_data,
                               p300_candidates, band_power_time, band_power,
                               save_dir=None):
        """
        Generate 6-panel visualization of analysis results.

        Panels:
            1. P300 candidates marked on filtered EEG
            2. Alpha power over time
            3. Beta power over time
            4. Gamma power over time
            5. Power Spectral Density
            6. Summary statistics
        """
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Muse 2 EEG Analysis Results', fontsize=14)

        time_axis = np.arange(light_data.shape[1]) / self.sampling_rate

        # Panel 1: P300 candidates on filtered EEG
        ax1 = fig.add_subplot(3, 2, 1)
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_channels))

        for ch_idx, ch_name in enumerate(self.channels):
            # Plot filtered signal with offset
            offset = ch_idx * 80
            ax1.plot(time_axis, light_data[ch_idx] + offset,
                    color=colors[ch_idx], linewidth=0.5, alpha=0.7, label=ch_name)

            # Mark P300 candidates
            candidates = p300_candidates[ch_name]
            if candidates['count'] > 0:
                ax1.scatter(candidates['times'],
                           candidates['amplitudes'] + offset,
                           color=colors[ch_idx], s=50, marker='v',
                           edgecolors='black', linewidths=0.5, zorder=5)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude (uV) + offset')
        ax1.set_title('P300 Candidates (negative peaks marked with triangles)')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Add annotation
        total_candidates = sum(c['count'] for c in p300_candidates.values())
        ax1.text(0.02, 0.98, f'Total P300 candidates: {total_candidates}',
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Panels 2-4: Band power over time
        time_points = band_power_time['time_points']
        band_series = band_power_time['band_power_series']

        for idx, (band_name, (low, high)) in enumerate(BAND_DEFINITIONS.items()):
            ax = fig.add_subplot(3, 2, 2 + idx)

            for ch_idx, ch_name in enumerate(self.channels):
                ax.plot(time_points, band_series[band_name][ch_name],
                       color=colors[ch_idx], label=ch_name, linewidth=1)

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Power')
            ax.set_title(f'{band_name.capitalize()} Power ({low}-{high} Hz)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Panel 5: Power Spectral Density
        ax5 = fig.add_subplot(3, 2, 5)

        for ch_idx, ch_name in enumerate(self.channels):
            freqs, psd = welch(heavy_data[ch_idx], fs=self.sampling_rate, nperseg=512)
            ax5.semilogy(freqs, psd, color=colors[ch_idx], label=ch_name, linewidth=1)

        # Add band shading
        for band_name, (low, high) in BAND_DEFINITIONS.items():
            ax5.axvspan(low, high, alpha=0.15)
            ax5.text((low + high) / 2, ax5.get_ylim()[1] * 0.5, band_name,
                    ha='center', fontsize=8, alpha=0.7)

        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Power Spectral Density')
        ax5.set_title('Power Spectral Density (Heavy-Processed)')
        ax5.set_xlim(0, 50)
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, alpha=0.3)

        # Panel 6: Summary statistics
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.axis('off')

        # Build summary text
        duration = light_data.shape[1] / self.sampling_rate
        summary_lines = [
            f"Recording Duration: {duration:.1f} seconds",
            f"Samples Collected: {light_data.shape[1]:,}",
            f"Sampling Rate: {self.sampling_rate} Hz",
            "",
            "P300 Candidates Detected:",
        ]

        for ch_name, candidates in p300_candidates.items():
            if candidates['count'] > 0:
                avg_amp = np.mean(candidates['amplitudes'])
                summary_lines.append(f"  {ch_name}: {candidates['count']} peaks (avg: {avg_amp:.1f} uV)")
            else:
                summary_lines.append(f"  {ch_name}: No candidates")

        summary_lines.extend(["", "Average Band Power:"])
        band_powers = band_power['band_powers']
        for band_name in BAND_DEFINITIONS:
            avg = np.mean(list(band_powers[band_name].values()))
            summary_lines.append(f"  {band_name.capitalize()}: {avg:.4f}")

        summary_text = "\n".join(summary_lines)
        ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        ax6.set_title('Analysis Summary')

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / 'analysis_results.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Analysis plot saved to {save_path}")

        plt.show()

        return fig

    def save_data(self, filename, duration=10, apply_ica=True):
        """
        Save streamed data to CSV file.

        Args:
            filename: Output CSV filename
            duration: How long to stream (seconds)
            apply_ica: Whether to apply ICA-based artifact removal
        """
        print(f"Streaming for {duration} seconds...")
        data_list = []

        for raw, processed in self.process_stream(duration=duration, apply_ica=apply_ica):
            # Store processed data
            for ch in range(self.n_channels):
                if len(processed[ch]) == self.buffer_size:
                    data_list.append(processed[ch, -1])

        # Reshape and save
        data_array = np.array(data_list).reshape(-1, self.n_channels)
        np.savetxt(filename, data_array, delimiter=',',
                   header=','.join(self.channels), comments='')
        print(f"Data saved to {filename}")


if __name__ == "__main__":
    args = parse_args()

    # Initialize processor with BrainFlow connection
    processor = MuseBrainFlowProcessor(
        buffer_duration=10,
        serial_port=args.serial_port,
        mac_address=args.mac_address
    )

    try:
        # Collect data for 60 seconds, then analyze and show results
        # This will display:
        #   - P300 candidate peaks (negative deflections)
        #   - Alpha, Beta, Gamma power over time
        #   - Power Spectral Density
        #   - Summary statistics
        results = processor.collect_and_analyze(
            duration=60,
            save_dir='./output'
        )

    finally:
        processor.stop()
