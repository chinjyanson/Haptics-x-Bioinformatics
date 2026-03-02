"""
Muse 2 EEG Stream Reader
Connects to Muse 2 using BrainFlow.

Advanced denoising (ICA, wavelet, CCA, EOG regression) is done post-hoc
in analysis.py on the saved CSV.

Key Notes:
    - Muse 2 has 4 EEG channels via Bluetooth: TP9, AF7, AF8, TP10
    - Fpz is the hardware reference (not accessible as a separate channel)
    - All EEG measurements are differential: EEG_measured = EEG_site - EEG_Fpz
"""
import logging

import numpy as np
from scipy import signal
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# =============================================================================
# CONFIGURATION
# =============================================================================

MUSE_CHANNELS      = ['TP9', 'AF7', 'AF8', 'TP10']
MUSE_SAMPLING_RATE = 256   # Hz — overwritten by actual device rate at runtime

NOTCH_FREQ         = 50.0  # Hz (UK power line; use 60 for US)
BASIC_BP_LOW       = 0.1   # Hz — bandpass low cut
BASIC_BP_HIGH      = 40.0  # Hz — bandpass high cut

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class MuseBrainFlowProcessor:
    """
    Muse 2 EEG reader via BrainFlow.

    Interface required by main.py:
        __init__(buffer_duration, serial_port, mac_address)
        get_data()   -> ndarray(4, n_samples) or None
        stop()
        attributes:  sampling_rate, channels, n_channels
    """

    def __init__(self, buffer_duration=10, serial_port=None, mac_address=None):
        self.buffer_duration = buffer_duration
        self.channels   = MUSE_CHANNELS.copy()
        self.n_channels = len(self.channels)

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

        logger.info("Connecting to Muse 2...")
        self.board = BoardShim(self.board_id, params)
        self.board.prepare_session()

        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels  = BoardShim.get_eeg_channels(self.board_id)

        self.board.start_stream()
        logger.info(f"Connected to Muse 2 (sampling rate: {self.sampling_rate} Hz)")

    # -------------------------------------------------------------------------
    # Core interface (required by main.py)
    # -------------------------------------------------------------------------

    def get_data(self):
        """Pull available EEG data from the board.

        Returns:
            ndarray(4, n_samples) or None if no data available.
        """
        data = self.board.get_board_data()
        if data.size == 0:
            return None
        return data[self.eeg_channels, :]

    def stop(self):
        """Stop streaming and release the BrainFlow session."""
        self.board.stop_stream()
        self.board.release_session()
        logger.info("Session ended.")

    # -------------------------------------------------------------------------
    # Basic filtering helpers (used by main.py's data pipeline)
    # -------------------------------------------------------------------------

    def _apply_notch(self, data, freq=None, Q=30):
        """Apply IIR notch filter to remove power-line interference."""
        if freq is None:
            freq = NOTCH_FREQ
        nyquist = self.sampling_rate / 2
        w0 = freq / nyquist
        if w0 >= 1.0:
            logger.warning(f"Notch frequency {freq} Hz >= Nyquist. Skipping.")
            return data
        try:
            b, a = signal.iirnotch(w0, Q)
            out = np.zeros_like(data)
            for ch in range(data.shape[0]):
                out[ch] = signal.filtfilt(b, a, data[ch])
            return out
        except Exception as e:
            logger.warning(f"Notch filter failed: {e}. Returning original data.")
            return data

    def _apply_bandpass(self, data, lowcut=None, highcut=None, order=4):
        """Apply zero-phase Butterworth bandpass filter."""
        if lowcut is None:
            lowcut = BASIC_BP_LOW
        if highcut is None:
            highcut = BASIC_BP_HIGH
        nyquist = self.sampling_rate / 2
        low  = max(lowcut  / nyquist, 1e-4)
        high = min(highcut / nyquist, 0.999)
        if low >= high:
            return data
        try:
            b, a = signal.butter(order, [low, high], btype='band')
            out = np.zeros_like(data)
            for ch in range(data.shape[0]):
                out[ch] = signal.filtfilt(b, a, data[ch])
            return out
        except Exception as e:
            logger.warning(f"Bandpass filter failed: {e}. Returning original data.")
            return data
