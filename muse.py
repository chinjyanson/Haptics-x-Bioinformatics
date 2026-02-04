"""
Muse 2 EEG Stream Reader with Real-time Processing
Connects directly to Muse 2 using BrainFlow and applies denoising/filtering
"""
import time
import argparse
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from mne.preprocessing import ICA
import mne

def parse_args():
    parser = argparse.ArgumentParser(description='Read EEG from Muse 2 headband')
    parser.add_argument('--serial-port', type=str, help='Serial port for BLED112 dongle (e.g., COM3)')
    parser.add_argument('--mac-address', type=str, help='MAC address of your Muse 2')
    return parser.parse_args()

class MuseBrainFlowProcessor:
    def __init__(self, buffer_duration=10, serial_port=None, mac_address=None):
        """
        Initialize the Muse BrainFlow stream reader

        Args:
            buffer_duration: How many seconds of data to keep in buffer
            serial_port: Serial port for BLED112 dongle (optional)
            mac_address: MAC address of Muse 2 (optional)
        """
        self.buffer_duration = buffer_duration

        # Channel names for Muse 2
        self.channels = ['TP9', 'AF7', 'AF8', 'TP10']
        self.n_channels = len(self.channels)

        # Set up BrainFlow connection
        params = BrainFlowInputParams()

        if serial_port:
            print(f"Using BLED dongle on port: {serial_port}")
            params.serial_port = serial_port
            self.board_id = BoardIds.MUSE_2_BLED_BOARD
        else:
            print("Using native Bluetooth connection")
            self.board_id = BoardIds.MUSE_2_BOARD
            if mac_address:
                params.mac_address = mac_address
                print(f"Connecting to specific device: {mac_address}")

        # Create board and connect
        print("Connecting to Muse 2...")
        self.board = BoardShim(self.board_id, params)
        self.board.prepare_session()

        # Get sampling rate and EEG channels from BrainFlow
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.buffer_size = int(buffer_duration * self.sampling_rate)

        # Initialize buffers for each channel
        self.buffers = [deque(maxlen=self.buffer_size) for _ in range(self.n_channels)]

        # ICA components for artifact removal
        self.ica_weights = None
        self.ica_trained = False

        # Start streaming
        self.board.start_stream()
        print(f"Connected to Muse 2 (sampling rate: {self.sampling_rate} Hz)")
    
    def get_data(self):
        """Pull available data from the board"""
        data = self.board.get_board_data()
        if data.size == 0:
            return None
        # Extract EEG channels
        eeg_data = data[self.eeg_channels, :]
        return eeg_data

    def stop(self):
        """Stop streaming and release the session"""
        self.board.stop_stream()
        self.board.release_session()
        print("Session ended.")
    
    def apply_bandpass_filter(self, data, lowcut=1, highcut=40):
        """
        Apply bandpass filter to remove noise
        
        Args:
            data: Raw EEG data
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
            # If filtering fails, return original data
            filtered = data
        return filtered
    
    def apply_notch_filter(self, data, freq=60):
        """
        Apply notch filter to remove power line interference
        
        Args:
            data: Raw EEG data
            freq: Frequency to remove (Hz, default 60 for US)
        
        Returns:
            Filtered data
        """
        nyquist = self.sampling_rate / 2
        w0 = freq / nyquist
        Q = 30
        
        b, a = signal.iirnotch(w0, Q)
        filtered = signal.filtfilt(b, a, data)
        return filtered
    
    def apply_ica_denoising(self, data, n_components=None):
        """
        Apply ICA-based artifact removal using MNE
        
        Args:
            data: Raw EEG data (n_channels x n_samples)
            n_components: Number of ICA components (default: n_channels)
        
        Returns:
            Denoised data with artifact components removed
        """
        if data.shape[1] < 100:
            return data  # Not enough samples for ICA
        
        if n_components is None:
            n_components = self.n_channels
        
        try:
            # Create MNE Info object
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self.sampling_rate,
                ch_types='eeg'
            )
            
            # Create RawArray from data
            raw = mne.io.RawArray(data, info, verbose=False)
            
            # Apply ICA
            ica = ICA(n_components=n_components, random_state=42, max_iter=200, verbose=False)
            ica.fit(raw)
            
            # Identify artifact components (high variance)
            component_var = np.var(ica.get_sources(raw).get_data(), axis=1)
            threshold = np.mean(component_var) + 1.5 * np.std(component_var)
            artifact_indices = np.where(component_var > threshold)[0].tolist()
            
            # Apply ICA to remove artifacts
            ica.exclude = artifact_indices
            raw_clean = ica.apply(raw.copy())
            
            # Return denoised data
            denoised = raw_clean.get_data()
            
            return denoised
        except Exception as e:
            print(f"ICA denoising failed: {e}. Returning original data.")
            return data
    
    def process_stream(self, duration=None, apply_filter=True, apply_ica=True):
        """
        Stream and process data in real-time

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
    
    def plot_real_time(self, duration=None, apply_ica=True):
        """
        Plot real-time EEG data
        
        Args:
            duration: How long to stream (seconds). None = infinite
            apply_ica: Whether to apply ICA-based artifact removal
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Muse 2 Real-time EEG (with ICA Denoising)' if apply_ica else 'Muse 2 Real-time EEG', fontsize=16)
        
        axes = axes.flatten()
        lines_raw = [ax.plot([], [], label='Raw', color='blue', alpha=0.5)[0] for ax in axes]
        lines_proc = [ax.plot([], [], label='Denoised + Filtered', color='red')[0] for ax in axes]
        
        for i, ax in enumerate(axes):
            ax.set_title(f'{self.channels[i]}')
            ax.set_ylabel('Voltage (µV)')
            ax.set_ylim(-200, 200)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        def update(frame):
            raw, processed = frame
            time_axis = np.arange(raw.shape[1]) / self.sampling_rate
            
            for ch in range(self.n_channels):
                lines_raw[ch].set_data(time_axis, raw[ch])
                lines_proc[ch].set_data(time_axis, processed[ch])
                axes[ch].set_xlim(0, self.buffer_duration)
            
            return lines_raw + lines_proc
        
        # Generator for data
        def data_gen():
            for raw, processed in self.process_stream(duration=duration, apply_ica=apply_ica):
                yield raw, processed
        
        ani = FuncAnimation(fig, update, data_gen, interval=50, blit=True)
        plt.tight_layout()
        plt.show()
    
    def save_data(self, filename, duration=10, apply_ica=True):
        """
        Save streamed data to CSV file
        
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
                    data_list.append(processed[ch, -1])  # Last sample
        
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
        # Option 1: Plot real-time data
        processor.plot_real_time(duration=60)

        # Option 2: Save data to file (uncomment to use)
        # processor.save_data('muse_data.csv', duration=30)
    finally:
        processor.stop()