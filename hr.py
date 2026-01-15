#!/usr/bin/env python3
"""
Muse 2 Heart Rate Measurement Script

This script connects to a Muse 2 headband and continuously measures heart rate
using the PPG (photoplethysmography) sensor data. It displays real-time heart rate
in beats per minute (BPM) and can optionally save the data to a CSV file.

Requirements:
- BrainFlow library (pip install brainflow)
- numpy (pip install numpy)
- matplotlib (pip install matplotlib) - optional for visualization

Hardware:
- Muse 2 headband
"""

import time
import argparse
import numpy as np
import csv
from datetime import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter

# Global variables for visualization (optional)
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Real-time plotting will be disabled.")

def parse_args():
    parser = argparse.ArgumentParser(description='Measure heart rate with Muse 2 headband')
    parser.add_argument('--serial-port', type=str, help='Serial port for BLED112 dongle (e.g., COM3)')
    parser.add_argument('--mac-address', type=str, help='MAC address of your Muse 2')
    parser.add_argument('--record', action='store_true', help='Record PPG data to CSV file')
    parser.add_argument('--visualize', action='store_true', help='Visualize PPG data in real-time')
    parser.add_argument('--duration', type=int, default=60, help='Recording duration in seconds')
    return parser.parse_args()

def setup_board(args):
    # Enable logging for troubleshooting
    BoardShim.enable_dev_board_logger()
    DataFilter.enable_dev_data_logger()
    
    params = BrainFlowInputParams()
    
    # Configure connection parameters based on what's provided
    if args.serial_port:
        print(f"Using BLED dongle on port: {args.serial_port}")
        params.serial_port = args.serial_port
        board_id = BoardIds.MUSE_2_BLED_BOARD
    else:
        print("Using native Bluetooth connection")
        board_id = BoardIds.MUSE_2_BOARD
        if args.mac_address:
            params.mac_address = args.mac_address
            print(f"Connecting to specific device: {args.mac_address}")
    
    try:
        board = BoardShim(board_id, params)
        return board
    except Exception as e:
        print(f"Error setting up the board: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your Muse 2 is turned on and in pairing mode")
        print("2. Check if the MAC address is correct (if provided)")
        print("3. Verify that the serial port for BLED dongle is correct (if using)")
        print("4. Ensure the Muse 2 is charged")
        exit(1)

def initialize_recording(board):
    print("Preparing session...")
    board.prepare_session()
    
    # Enable PPG data by configuring the board
    # p50 enables both PPG and the 5th EEG channel
    print("Enabling PPG sensor...")
    board.config_board("p50")
    
    print("Starting data stream...")
    board.start_stream()
    
    # Wait a moment for the data stream to stabilize
    time.sleep(2)
    
    return board

def get_ppg_data(board, num_samples=200):
    """Get PPG data from the board"""
    # Using ANCILLARY_PRESET to get PPG data
    data = board.get_board_data(num_samples, preset=BrainFlowPresets.ANCILLARY_PRESET)
    
    # Get PPG channels
    ppg_channels = BoardShim.get_ppg_channels(BoardIds.MUSE_2_BOARD, BrainFlowPresets.ANCILLARY_PRESET)
    
    if len(ppg_channels) < 2:
        print("Warning: Not enough PPG channels available")
        return None, None
    
    # Extract red and IR PPG signals
    ppg_red = data[ppg_channels[0]] if len(data) > ppg_channels[0] else np.array([])
    ppg_ir = data[ppg_channels[1]] if len(data) > ppg_channels[1] else np.array([])
    
    # Print warning if not enough samples collected
    if len(ppg_red) < 40 or len(ppg_ir) < 40:
        print(f"\rCollecting more data... (current samples: {len(ppg_ir)})", end="")
    
    return ppg_red, ppg_ir

def calculate_heart_rate(ppg_ir, ppg_red, sampling_rate):
    """Calculate heart rate from PPG data"""
    if ppg_ir.size == 0 or ppg_red.size == 0:
        return None, None
    
    # Check if we have enough samples for oxygen calculation
    if ppg_ir.size < 40 or ppg_red.size < 40:
        oxygen_level = None
    else:
        # Calculate oxygen level
        try:
            oxygen_level = DataFilter.get_oxygen_level(ppg_ir, ppg_red, sampling_rate)
        except Exception as e:
            oxygen_level = None
            print(f"\nError calculating oxygen level: {e}")
            print("This is often normal during startup or if there's insufficient data.")
    
    # Calculate heart rate from PPG using peak detection
    try:
        # Use IR signal for heart rate calculation as it's typically cleaner
        # First, apply bandpass filter to isolate the heartbeat frequency range (0.5-3.5 Hz)
        if ppg_ir.size >= 4:  # Need at least 4 samples for a 4th order filter
            DataFilter.perform_bandpass(ppg_ir, sampling_rate, 0.5, 3.5, 4, 
                                      0, 0)  # Bandpass filter with order 4
            
            # Find peaks (each peak corresponds to a heartbeat)
            peaks = []
            last_val = ppg_ir[0]
            for i in range(1, len(ppg_ir) - 1):
                if ppg_ir[i] > last_val and ppg_ir[i] > ppg_ir[i + 1] and ppg_ir[i] > 0:
                    peaks.append(i)
                last_val = ppg_ir[i]
            
            # Calculate heart rate if we have enough peaks
            if len(peaks) > 1:
                # Calculate time between peaks
                peak_intervals = np.diff(peaks) / sampling_rate  # in seconds
                
                # Convert to beats per minute
                instantaneous_bpm = 60 / peak_intervals
                
                # Filter out unreasonable values (e.g., due to noise)
                valid_bpm = instantaneous_bpm[(instantaneous_bpm >= 40) & (instantaneous_bpm <= 200)]
                
                if len(valid_bpm) > 0:
                    average_bpm = np.mean(valid_bpm)
                    return average_bpm, oxygen_level
    except Exception as e:
        print(f"\nError calculating heart rate: {e}")
    
    return None, oxygen_level

def save_data_to_csv(timestamp, heart_rate, oxygen_level, filename=None):
    """Save heart rate data to a CSV file"""
    if filename is None:
        # Generate filename with current date and time if not provided
        filename = f"muse2_heart_rate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Append data to the CSV file
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file is empty
        if file.tell() == 0:
            writer.writerow(['Timestamp', 'Heart Rate (BPM)', 'Oxygen Level (%)'])
        
        # Write data
        writer.writerow([timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), 
                         heart_rate if heart_rate is not None else 'N/A',
                         oxygen_level if oxygen_level is not None else 'N/A'])
    
    return filename

def setup_visualization():
    """Set up real-time visualization of heart rate data"""
    if not MATPLOTLIB_AVAILABLE:
        return None, None, None
    
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Initialize line for PPG signal
    ppg_line, = ax1.plot([], [], 'r-', label='PPG Signal')
    ax1.set_title('PPG Signal')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    
    # Initialize text for heart rate display
    heart_rate_text = ax2.text(0.5, 0.5, '', fontsize=24, 
                             horizontalalignment='center',
                             verticalalignment='center')
    ax2.set_title('Heart Rate')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    return fig, (ppg_line, heart_rate_text), (ax1, ax2)

def update_plot(frame, board, ppg_line, heart_rate_text, ax1):
    """Update function for animation"""
    # Get new PPG data
    ppg_red, ppg_ir = get_ppg_data(board, num_samples=100)
    
    if ppg_ir is not None and ppg_ir.size > 0:
        # Update PPG signal plot
        ppg_line.set_data(range(len(ppg_ir)), ppg_ir)
        ax1.relim()
        ax1.autoscale_view()
        
        # Calculate and update heart rate display
        sampling_rate = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD, 
                                                   BrainFlowPresets.ANCILLARY_PRESET)
        heart_rate, _ = calculate_heart_rate(ppg_ir, ppg_red, sampling_rate)
        
        if heart_rate is not None:
            heart_rate_text.set_text(f"{heart_rate:.1f} BPM")
        else:
            heart_rate_text.set_text("Calculating...")
    
    return ppg_line, heart_rate_text

def main():
    args = parse_args()
    
    # Set up the board
    board = setup_board(args)
    
    # Initialize recording
    board = initialize_recording(board)
    
    # Get sampling rate for PPG data
    sampling_rate = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD, 
                                               BrainFlowPresets.ANCILLARY_PRESET)
    
    print(f"PPG sampling rate: {sampling_rate} Hz")
    
    # Setup visualization if requested
    if args.visualize and MATPLOTLIB_AVAILABLE:
        fig, plot_elements, axes = setup_visualization()
        ani = FuncAnimation(fig, update_plot, fargs=(board, *plot_elements, axes[0]),
                           interval=100, blit=True)
        plt.show(block=False)
    
    # CSV filename for data recording
    csv_filename = None
    if args.record:
        csv_filename = f"muse2_heart_rate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"Recording data to {csv_filename}")
    
    # Main loop for data collection
    start_time = time.time()
    accumulated_ppg_ir = np.array([])
    accumulated_ppg_red = np.array([])
    
    try:
        print("\nMeasuring heart rate. Press Ctrl+C to stop...\n")
        print("Waiting for initial data collection...")
        
        while time.time() - start_time < args.duration:
            # Get PPG data
            ppg_red, ppg_ir = get_ppg_data(board)
            
            # Accumulate data if needed
            if ppg_ir is not None and ppg_ir.size > 0 and ppg_red is not None and ppg_red.size > 0:
                accumulated_ppg_ir = np.concatenate((accumulated_ppg_ir, ppg_ir)) if accumulated_ppg_ir.size > 0 else ppg_ir
                accumulated_ppg_red = np.concatenate((accumulated_ppg_red, ppg_red)) if accumulated_ppg_red.size > 0 else ppg_red
                
                # Keep only the most recent 300 samples to avoid too much historical data
                if accumulated_ppg_ir.size > 300:
                    accumulated_ppg_ir = accumulated_ppg_ir[-300:]
                if accumulated_ppg_red.size > 300:
                    accumulated_ppg_red = accumulated_ppg_red[-300:]
                
                # Calculate heart rate using accumulated data
                heart_rate, oxygen_level = calculate_heart_rate(accumulated_ppg_ir, accumulated_ppg_red, sampling_rate)
                
                # Display heart rate
                timestamp = datetime.now()
                if heart_rate is not None:
                    status_msg = f"\r{timestamp.strftime('%H:%M:%S')} - Heart Rate: {heart_rate:.1f} BPM"
                    if oxygen_level is not None:
                        status_msg += f" | O2: {oxygen_level:.1f}%"
                    print(status_msg, end="")
                else:
                    # If we have enough samples but still no heart rate, display calculating message
                    if accumulated_ppg_ir.size >= 100:
                        print(f"\r{timestamp.strftime('%H:%M:%S')} - Heart Rate: Calculating... (samples: {accumulated_ppg_ir.size})", end="")
                    else:
                        print(f"\r{timestamp.strftime('%H:%M:%S')} - Collecting data... (samples: {accumulated_ppg_ir.size})", end="")
                
                # Record data if requested
                if args.record and heart_rate is not None:
                    save_data_to_csv(timestamp, heart_rate, oxygen_level, csv_filename)
            
            # Brief pause to prevent excessive CPU usage
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\nMeasurement stopped by user.")
    finally:
        # Clean up
        print("\nStopping data stream...")
        board.stop_stream()
        board.release_session()
        
        if args.record and csv_filename:
            print(f"Data saved to {csv_filename}")
        
        print("Session ended.")

if __name__ == "__main__":
    main()