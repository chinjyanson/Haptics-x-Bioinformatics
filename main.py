"""
Muse 2 EEG Stream Reader with Real-time Processing
Connects directly to Muse 2 using BrainFlow and applies denoising/filtering
"""
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from collections import deque
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

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
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data)
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
    
    def process_stream(self, duration=None, apply_filter=True):
        """
        Stream and process data in real-time

        Args:
            duration: How long to stream (seconds). None = infinite
            apply_filter: Whether to apply bandpass and notch filters

        Yields:
            (raw_data, processed_data) tuples
        """
        import time
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
                    processed_data = np.zeros_like(raw_data)
                    for ch in range(self.n_channels):
                        # Apply notch filter (remove 60 Hz)
                        temp = self.apply_notch_filter(raw_data[ch])
                        # Apply bandpass filter (1-40 Hz)
                        processed_data[ch] = self.apply_bandpass_filter(temp)
                else:
                    processed_data = raw_data

                yield raw_data, processed_data

                # Small delay to prevent busy-waiting
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nStreaming stopped by user")
    
    def plot_real_time(self, duration=None):
        """
        Plot real-time EEG data
        
        Args:
            duration: How long to stream (seconds). None = infinite
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Muse 2 Real-time EEG', fontsize=16)
        
        axes = axes.flatten()
        lines_raw = [ax.plot([], [], label='Raw', color='blue', alpha=0.5)[0] for ax in axes]
        lines_proc = [ax.plot([], [], label='Filtered', color='red')[0] for ax in axes]
        
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
            for raw, processed in self.process_stream(duration=duration):
                yield raw, processed
        
        ani = FuncAnimation(fig, update, data_gen, interval=50, blit=True)
        plt.tight_layout()
        plt.show()
    
    def save_data(self, filename, duration=10):
        """
        Save streamed data to CSV file
        
        Args:
            filename: Output CSV filename
            duration: How long to stream (seconds)
        """
        print(f"Streaming for {duration} seconds...")
        data_list = []
        
        for raw, processed in self.process_stream(duration=duration):
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