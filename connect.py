#!/usr/bin/env python
# Fixed connect.py script for Muse 2 EEG data visualization

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from pylsl import StreamInlet, resolve_byprop
import time
import threading
import mne
from mne.viz import plot_topomap
from mne.channels import make_standard_montage
from mne.io import RawArray

# Define EEG band frequencies
BAND_FREQS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 50)
}

# Muse 2 channel names in correct order
# Note: Muse 2 streams 5 channels - 4 EEG + 1 auxiliary (right ear/PPG for heart rate)
MUSE_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10', 'AUX_R']
MUSE_EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']

def connect_to_muse():
    """
    Connect to the Muse EEG device using LSL protocol.
    Returns the StreamInlet or None if connection fails.
    """
    print("Looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=5)

    if len(streams) == 0:
        print("No EEG stream found. Make sure your device is streaming.")
        return None

    print(f"Found {len(streams)} EEG streams, connecting to the first one.")
    inlet = StreamInlet(streams[0])

    # Print stream info
    info = inlet.info()
    print(f"Connected to EEG stream: {info.name()}")
    print(f"Number of channels: {info.channel_count()}")
    print(f"Sampling rate: {info.nominal_srate()} Hz")

    return inlet

def create_mne_raw_object(data, ch_names=None, sfreq=256.0):
    """
    Create an MNE Raw object from the EEG data.

    Parameters:
    -----------
    data : numpy.ndarray
        EEG data, shape (n_channels, n_samples)
    ch_names : list
        List of channel names
    sfreq : float
        Sampling frequency in Hz

    Returns:
    --------
    raw : mne.io.RawArray
        MNE Raw object
    """
    # Determine channel names based on data shape
    n_channels = data.shape[0]
    if ch_names is None:
        if n_channels == 5:
            ch_names = MUSE_CHANNELS
        elif n_channels == 4:
            ch_names = MUSE_EEG_CHANNELS
        else:
            ch_names = [f'CH{i+1}' for i in range(n_channels)]

    # Set channel types: EEG for first 4 channels, misc for auxiliary
    if n_channels == 5:
        ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'misc']
    elif n_channels == 4:
        ch_types = ['eeg'] * 4
    else:
        ch_types = ['eeg'] * n_channels

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = RawArray(data, info)

    # Set up standard 10-20 montage for EEG channels only
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    return raw

def collect_eeg_data(inlet, duration=5.0):
    """
    Collect EEG data for the specified duration.

    Parameters:
    -----------
    inlet : pylsl.StreamInlet
        LSL inlet to read data from
    duration : float
        Duration to collect data in seconds

    Returns:
    --------
    data : numpy.ndarray
        EEG data, shape (n_channels, n_samples)
    timestamps : numpy.ndarray
        Timestamps for each sample
    """
    # Calculate number of samples to collect
    sfreq = inlet.info().nominal_srate()
    n_samples = int(sfreq * duration)
    n_channels = inlet.info().channel_count()

    # Initialize arrays for data and timestamps
    data = []
    timestamps = []

    print(f"Collecting {duration} seconds of data from {n_channels} channels...")
    start_time = time.time()

    while len(data) < n_samples:
        sample, timestamp = inlet.pull_sample()
        if timestamp is not None:
            # Take all available channels
            data.append(sample[:n_channels])
            timestamps.append(timestamp)

    end_time = time.time()
    print(f"Data collection complete. Collected {len(data)} samples in {end_time - start_time:.2f} seconds.")

    # Convert to numpy arrays with correct shape
    data_array = np.array(data).T  # Transpose to get (n_channels, n_samples)
    timestamps_array = np.array(timestamps)

    print(f"Data shape: {data_array.shape}")

    return data_array, timestamps_array

def visualize_raw_eeg(raw):
    """
    Visualize raw EEG data.
    
    Parameters:
    -----------
    raw : mne.io.RawArray
        MNE Raw object
    """
    # Plot the raw data
    plt.figure(figsize=(15, 10))
    raw.plot(scalings='auto', title='Raw EEG Data', show=False, block=False)
    plt.tight_layout()

def compute_psd(raw):
    """
    Compute power spectral density (PSD) of the EEG data.
    
    Parameters:
    -----------
    raw : mne.io.RawArray
        MNE Raw object
    
    Returns:
    --------
    psd : numpy.ndarray
        PSD data, shape (n_channels, n_freqs)
    freqs : numpy.ndarray
        Frequency bins
    """
    # Use the spectrum method from raw object instead of deprecated psd_welch
    spectrum = raw.compute_psd(method="welch", fmin=0.5, fmax=50, n_fft=512)
    freqs = spectrum.freqs
    psd = spectrum.get_data()
    
    return psd, freqs

def visualize_psd(raw):
    """
    Visualize power spectral density (PSD) of the EEG data.
    
    Parameters:
    -----------
    raw : mne.io.RawArray
        MNE Raw object
    """
    plt.figure(figsize=(15, 10))
    raw.plot_psd(fmax=50, show=False)
    plt.tight_layout()

def compute_band_power(psd, freqs):
    """
    Compute average power in each frequency band.
    
    Parameters:
    -----------
    psd : numpy.ndarray
        PSD data, shape (n_channels, n_freqs)
    freqs : numpy.ndarray
        Frequency bins
    
    Returns:
    --------
    band_power : dict
        Average power in each frequency band for each channel
    """
    band_power = {}
    
    for band, (fmin, fmax) in BAND_FREQS.items():
        # Find frequencies in the specified range
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        
        # Compute average power in the band
        band_power[band] = np.mean(psd[:, idx], axis=1)
    
    return band_power

def visualize_band_power(band_power, ch_names=MUSE_CHANNELS):
    """
    Visualize power in each frequency band for each channel.
    
    Parameters:
    -----------
    band_power : dict
        Average power in each frequency band for each channel
    ch_names : list
        List of channel names
    """
    bands = list(band_power.keys())
    n_bands = len(bands)
    n_channels = len(ch_names)
    
    # Create a bar plot
    plt.figure(figsize=(15, 10))
    
    bar_width = 0.15
    index = np.arange(n_channels)
    
    for i, band in enumerate(bands):
        plt.bar(index + i*bar_width, band_power[band], bar_width, 
                label=band, alpha=0.7)
    
    plt.xlabel('Channels')
    plt.ylabel('Power (µV²/Hz)')
    plt.title('EEG Band Power')
    plt.xticks(index + bar_width * (n_bands-1)/2, ch_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def visualize_topography_snapshot(raw):
    """
    Visualize the topographic distribution of EEG power.

    Parameters:
    -----------
    raw : mne.io.RawArray
        MNE Raw object
    """
    # Get EEG data and compute average amplitude
    data = raw.get_data()
    avg_data = np.mean(np.abs(data), axis=1)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the topographic map
    # Pass the Info object instead of 2D positions for sphere='auto' to work
    im, cn = plot_topomap(
        avg_data,
        raw.info,
        extrapolate='head',
        axes=ax,
        sphere='auto'
    )

    plt.colorbar(im, ax=ax, label='Amplitude (µV)')
    ax.set_title('EEG Amplitude Topography')
    plt.tight_layout()

    return im, cn

def visualize_band_topography(raw, band_power):
    """
    Visualize the topographic distribution of power in each frequency band.

    Parameters:
    -----------
    raw : mne.io.RawArray
        MNE Raw object
    band_power : dict
        Average power in each frequency band for each channel
    """
    # Create a new figure with a subplot for each band
    bands = list(band_power.keys())
    n_bands = len(bands)

    # Calculate grid dimensions
    n_rows = int(np.ceil(n_bands / 2))
    n_cols = min(n_bands, 2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12))
    if n_bands == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, band in enumerate(bands):
        # Plot the topographic map for each band
        # Pass the Info object instead of 2D positions for sphere='auto' to work
        im, cn = plot_topomap(
            band_power[band],
            raw.info,
            extrapolate='head',
            axes=axes[i],
            sphere='auto'
        )

        plt.colorbar(im, ax=axes[i], label='Power (µV²/Hz)')
        axes[i].set_title(f'{band} Band Power')

    plt.tight_layout()

def visualize_filtered_data(raw):
    """
    Visualize the EEG data after filtering in different frequency bands.

    Parameters:
    -----------
    raw : mne.io.RawArray
        MNE Raw object
    """
    plt.figure(figsize=(15, 15))

    # Plot the unfiltered data
    plt.subplot(len(BAND_FREQS) + 1, 1, 1)
    raw.plot(duration=5, scalings='auto', title='Unfiltered EEG', show=False)

    # Plot the filtered data for each band
    for i, (band, (fmin, fmax)) in enumerate(BAND_FREQS.items(), start=2):
        # Filter the data
        raw_filtered = raw.copy().filter(fmin, fmax)

        # Plot the filtered data
        plt.subplot(len(BAND_FREQS) + 1, 1, i)
        raw_filtered.plot(duration=5, scalings='auto', title=f'{band} Band ({fmin}-{fmax} Hz)', show=False)

    plt.tight_layout()

def visualize_auxiliary_data(raw):
    """
    Visualize the auxiliary channel data (PPG/heart rate).

    Parameters:
    -----------
    raw : mne.io.RawArray
        MNE Raw object
    """
    # Check if auxiliary channel exists
    if 'AUX_R' not in raw.ch_names:
        print("No auxiliary channel (AUX_R) found in the data.")
        return

    # Get the auxiliary channel data
    aux_idx = raw.ch_names.index('AUX_R')
    aux_data = raw.get_data()[aux_idx]
    times = raw.times

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Plot the raw auxiliary data
    axes[0].plot(times, aux_data, linewidth=0.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].set_title('Auxiliary Channel Data (PPG/Heart Rate Sensor)')
    axes[0].grid(True, alpha=0.3)

    # Plot a zoomed-in view of a few seconds
    zoom_duration = min(10, len(times) / raw.info['sfreq'])  # 10 seconds or less
    zoom_mask = times < zoom_duration
    axes[1].plot(times[zoom_mask], aux_data[zoom_mask], linewidth=1)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (µV)')
    axes[1].set_title(f'Auxiliary Channel Data - First {zoom_duration:.1f} seconds (Zoomed)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Print some statistics
    print("\nAuxiliary Channel Statistics:")
    print(f"  Mean: {np.mean(aux_data):.2f} µV")
    print(f"  Std: {np.std(aux_data):.2f} µV")
    print(f"  Min: {np.min(aux_data):.2f} µV")
    print(f"  Max: {np.max(aux_data):.2f} µV")

def main():
    # Connect to Muse
    inlet = connect_to_muse()
    if inlet is None:
        print("Failed to connect to Muse. Exiting.")
        return

    # Collect EEG data
    data, timestamps = collect_eeg_data(inlet, duration=10.0)

    # Create MNE Raw object
    raw = create_mne_raw_object(data)

    print(f"\nRecorded channels: {raw.ch_names}")
    print(f"Channel types: {raw.get_channel_types()}")

    # Visualize raw data (all channels)
    visualize_raw_eeg(raw)

    # Visualize auxiliary data if available
    visualize_auxiliary_data(raw)

    # Get only EEG channels for frequency analysis
    raw_eeg = raw.copy().pick_types(eeg=True)

    # Compute and visualize PSD (EEG only)
    psd, freqs = compute_psd(raw_eeg)
    visualize_psd(raw_eeg)

    # Compute and visualize band power (EEG only)
    band_power = compute_band_power(psd, freqs)
    visualize_band_power(band_power, ch_names=raw_eeg.ch_names)

    # Visualize topography (EEG only)
    visualize_topography_snapshot(raw_eeg)

    # Visualize band topography (EEG only)
    visualize_band_topography(raw_eeg, band_power)

    # Visualize filtered data (EEG only)
    visualize_filtered_data(raw_eeg)

    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()