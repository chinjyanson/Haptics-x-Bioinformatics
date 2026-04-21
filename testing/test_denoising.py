"""
test_denoising.py — Record 20 s of live EEG from the Muse 2, run the full
denoising pipeline, and plot raw vs denoised for all 4 channels.

Usage:
    python test_denoising.py                        # native Bluetooth
    python test_denoising.py --serial /dev/tty.usbX # BLED dongle
    python test_denoising.py --duration 30          # longer recording
"""

import argparse
import sys
import os
import time

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from muse import MuseBrainFlowProcessor, MUSE_CHANNELS
from denoising import (
    CHANNELS, SAMPLE_RATE,
    mark_dropouts, interpolate_dropouts,
    notch_filter, wavelet_denoise, emd_drift_remove,
    run_erp_path, run_psd_path,
)


def record(duration: float, serial_port=None) -> pd.DataFrame:
    """Stream EEG for `duration` seconds and return a raw DataFrame."""
    print(f"[test] Connecting to Muse 2...")
    muse = MuseBrainFlowProcessor(buffer_duration=duration + 5,
                                  serial_port=serial_port)
    print(f"[test] Recording for {duration} s — keep still...")
    time.sleep(duration)

    eeg, timestamps = muse.get_data()
    muse.stop()

    if eeg is None or eeg.shape[1] == 0:
        raise RuntimeError("No EEG data received from Muse 2.")

    print(f"[test] Received {eeg.shape[1]} samples across {len(MUSE_CHANNELS)} channels.")

    df = pd.DataFrame(eeg.T, columns=MUSE_CHANNELS)
    df.insert(0, 'time', timestamps)
    return df


def denoise(raw_df: pd.DataFrame):
    """Run the full denoising pipeline. Returns (intermediate_df, erp_df, psd_df)."""
    df = raw_df.copy()

    df, dropout_mask  = mark_dropouts(df)
    df, bad_segments  = interpolate_dropouts(df, dropout_mask)

    print("[test] Notch filter...")
    for ch in CHANNELS:
        df[ch] = notch_filter(df[ch].values)

    print("[test] Wavelet denoising...")
    for ch in CHANNELS:
        df[ch] = wavelet_denoise(df[ch].values)

    print("[test] EMD drift removal...")
    for ch in CHANNELS:
        df[ch] = emd_drift_remove(df[ch].values)

    # Keep the post-core-denoising signal (before path split) for plotting
    intermediate_df = df.copy()

    erp_df = run_erp_path(df, bad_segments)
    psd_df = run_psd_path(df, bad_segments)

    return intermediate_df, erp_df, psd_df


def plot(raw_df: pd.DataFrame, erp_df: pd.DataFrame, psd_df: pd.DataFrame):
    """4-channel figure: raw, ERP-clean, PSD-clean per channel."""
    t_raw = raw_df['time'].values - raw_df['time'].values[0]
    t_erp = erp_df['time'].values - erp_df['time'].values[0]
    t_psd = psd_df['time'].values - psd_df['time'].values[0]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('EEG Denoising Test — Raw vs ERP-clean vs PSD-clean', fontsize=13)

    colors = {'raw': '#888888', 'erp': '#1f77b4', 'psd': '#d62728'}

    for ax, ch in zip(axes, CHANNELS):
        raw_sig = raw_df[ch].values
        erp_sig = erp_df[ch].values
        psd_sig = psd_df[ch].values

        # Compute RMS for annotation
        rms_raw = np.sqrt(np.nanmean(raw_sig ** 2))
        rms_erp = np.sqrt(np.nanmean(erp_sig ** 2))
        rms_psd = np.sqrt(np.nanmean(psd_sig ** 2))

        ax.plot(t_raw, raw_sig, color=colors['raw'], lw=0.6, alpha=0.7, label=f'Raw  (RMS={rms_raw:.1f}µV)')
        ax.plot(t_erp, erp_sig, color=colors['erp'], lw=0.9, alpha=0.9, label=f'ERP-clean (RMS={rms_erp:.1f}µV)')
        ax.plot(t_psd, psd_sig, color=colors['psd'], lw=0.9, alpha=0.9, label=f'PSD-clean (RMS={rms_psd:.1f}µV)', linestyle='--')

        ax.set_ylabel(f'{ch}\n(µV)', fontsize=9)
        ax.legend(loc='upper right', fontsize=7.5, framealpha=0.7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('test_denoising_output.png', dpi=150)
    print("[test] Figure saved to test_denoising_output.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Live EEG denoising test')
    parser.add_argument('--duration', type=float, default=20.0,
                        help='Recording duration in seconds (default: 20)')
    parser.add_argument('--serial', default=None,
                        help='Serial port for BLED dongle (e.g. /dev/tty.usbX)')
    args = parser.parse_args()

    raw_df = record(args.duration, serial_port=args.serial)
    print("[test] Running denoising pipeline...")
    _, erp_df, psd_df = denoise(raw_df)
    print("[test] Plotting...")
    plot(raw_df, erp_df, psd_df)


if __name__ == '__main__':
    main()
