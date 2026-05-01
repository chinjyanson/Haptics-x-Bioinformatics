"""
denoising.py - EEG preprocessing pipeline

Applies notch filtering, EOG regression, wavelet denoising, EMD drift removal,
and bandpass filtering.
Produces two output files: erp_clean.csv (0.1-30Hz + Savitzky-Golay) and psd_clean.csv (1-45Hz).

Usage:
    python denoising.py raw_eeg.csv erp_clean.csv psd_clean.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy import signal
import pywt
import emd

# --- Constants ---
SAMPLE_RATE       = 256
NOTCH_FREQ        = 50.0
NOTCH_Q           = 30.0
DROPOUT_THRESHOLD = 999.0
EMD_DRIFT_IMFS    = 2

# ERP path
ERP_HIGHPASS  = 0.1
ERP_LOWPASS   = 30.0
SG_WINDOW     = 17
SG_POLYORDER  = 3

# PSD path
PSD_HIGHPASS = 1.0
PSD_LOWPASS  = 45.0

CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']


# --- Step 0: Load and validate ---

def load_eeg(filepath):
    """Load raw EEG CSV, sort by time, return DataFrame."""
    df = pd.read_csv(filepath)
    df = df.sort_values('time').reset_index(drop=True)
    return df


def mark_dropouts(df, threshold=DROPOUT_THRESHOLD):
    """
    Replace samples where any channel |value| >= threshold with NaN.
    Returns (df_with_nans, bad_mask) where bad_mask is a boolean Series.
    """
    df = df.copy()
    dropout_rows = (df[CHANNELS].abs() >= threshold).any(axis=1)
    n_dropouts = dropout_rows.sum()
    print(f"[denoising] Dropout samples detected: {n_dropouts}")
    df.loc[dropout_rows, CHANNELS] = np.nan
    return df, dropout_rows


# --- Step 1: Dropout interpolation ---

def interpolate_dropouts(df, bad_mask, max_gap_secs=0.5):
    """
    Linearly interpolate NaN gaps shorter than max_gap_secs.
    Longer gaps are kept as NaN and tracked in bad_segments mask.
    Returns (df_interpolated, bad_segments_mask).
    """
    df = df.copy()
    max_gap_samples = int(max_gap_secs * SAMPLE_RATE)

    # Build bad_segments: runs of NaN longer than max_gap_samples
    bad_segments = pd.Series(False, index=df.index)

    for ch in CHANNELS:
        col = df[ch].copy()
        is_nan = col.isna()

        if not is_nan.any():
            continue

        # Find contiguous NaN runs
        nan_idx = np.where(is_nan)[0]
        if len(nan_idx) == 0:
            continue

        # Group consecutive indices
        gaps = np.split(nan_idx, np.where(np.diff(nan_idx) > 1)[0] + 1)
        for gap in gaps:
            if len(gap) > max_gap_samples:
                bad_segments.iloc[gap] = True
            # else: leave as NaN for interpolation below

        # Interpolate short gaps only (long gaps remain NaN, filled later with ffill/bfill)
        col_interp = col.copy()
        col_interp[bad_segments] = np.nan  # protect long gaps
        col_interp = col_interp.interpolate(method='linear', limit=max_gap_samples)
        # Fill any remaining NaN at edges with nearest valid value
        col_interp = col_interp.ffill().bfill()
        df[ch] = col_interp

    n_bad = bad_segments.sum()
    print(f"[denoising] Samples in long dropout gaps (>= {max_gap_secs}s): {n_bad}")
    return df, bad_segments


# --- Step 2: Notch filter ---

def notch_filter(sig, fs=SAMPLE_RATE, freq=NOTCH_FREQ, Q=NOTCH_Q):
    """Apply zero-phase 2nd-order IIR notch filter."""
    b, a = signal.iirnotch(freq, Q=Q, fs=fs)
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(sig) <= padlen:
        print(f"[denoising] Signal too short for notch filter ({len(sig)} samples), skipping.")
        return sig
    return signal.filtfilt(b, a, sig)


# --- Step 3: Wavelet denoising ---

def wavelet_denoise(sig, wavelet='db4', level=6):
    """
    Soft-threshold wavelet denoising on detail coefficients levels 1-3.
    Approximation and lower detail levels are left unmodified.
    """
    # pywt requires len > 0; also needs enough samples for the decomposition level
    min_len = 2 ** level
    if len(sig) < min_len:
        print(f"[denoising] Signal too short for wavelet denoising ({len(sig)} samples), skipping.")
        return sig
    coeffs = pywt.wavedec(sig, wavelet=wavelet, level=level)
    # Robust noise estimate from finest detail (level 1, index -1)
    finest_detail = coeffs[-1]
    sigma = np.median(np.abs(finest_detail)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(sig)))

    # Soft threshold detail levels 1-3 (indices -1, -2, -3)
    coeffs_thresh = list(coeffs)
    for i in range(1, 4):  # levels 1-3 from finest
        coeffs_thresh[-i] = pywt.threshold(coeffs[-i], value=threshold, mode='soft')
        # coeffs_thresh[-i] = pywt.threshold(coeffs[-i], value=threshold, mode='hard')

    return pywt.waverec(coeffs_thresh, wavelet=wavelet)[:len(sig)]


# --- Step 4: EMD drift removal ---

def emd_drift_remove(sig, n_imfs=EMD_DRIFT_IMFS):
    """
    Remove lowest n_imfs IMFs (slow drift) via EMD.
    Returns cleaned signal. Falls back to original if too few IMFs.
    """
    try:
        imfs = emd.sift.sift(sig)
        if imfs.shape[1] < n_imfs + 1:
            print(f"[denoising] EMD produced {imfs.shape[1]} IMFs (< {n_imfs+1}), skipping drift removal.")
            return sig
        cleaned = imfs[:, :-n_imfs].sum(axis=1)
        return cleaned
    except Exception as e:
        print(f"[denoising] EMD failed: {e}, skipping drift removal.")
        return sig


# --- Step 5: EOG artefact removal (pseudo-EOG from AF7/AF8) ---

# Channels to clean — AF7/AF8 are the reference so excluded
EOG_REFERENCE_CHANNELS  = ['AF7', 'AF8']
EOG_TARGET_CHANNELS     = ['TP9', 'TP10']

# Blink detection on the AF pseudo-EOG
# Blink detection on the AF pseudo-EOG
EOG_BLINK_LP_HZ        = 4.0   # Lowpass cutoff to isolate slow blink envelope
EOG_BLINK_THRESHOLD_UV = 50.0  # AF pseudo-EOG amplitude to declare a blink (µV)
EOG_BLINK_CONTEXT      = 154   # Samples each side of blink peak (±600 ms at 256 Hz)

# Residual artefact interpolation on TP channels after bandpass
EOG_TP_THRESHOLD_UV    = 60.0  # TP amplitude above which residual is interpolated


def eog_regression(df):
    """
    Blink-gated OLS regression to remove EOG component from TP9/TP10.

    Detects blink windows from the AF7/AF8 pseudo-EOG (lowpass + threshold),
    estimates β only from blink-epoch samples, then subtracts β × EOG from
    the full signal. Gating to blink windows avoids attenuating correlated
    neural activity (e.g. frontal-temporal theta) during non-blink periods.
    AF7/AF8 are left untouched.
    """
    df  = df.copy()
    n   = len(df)
    nyq = SAMPLE_RATE / 2.0

    af7     = df['AF7'].values.astype(float)
    af8     = df['AF8'].values.astype(float)
    eog_raw = (af7 + af8) / 2.0

    # Lowpass to isolate blink envelope
    b, a   = signal.butter(4, EOG_BLINK_LP_HZ / nyq, btype='low')
    padlen = 3 * (max(len(a), len(b)) - 1)
    if n <= padlen:
        print("[denoising] EOG: signal too short, skipping.")
        return df
    eog_lp = signal.filtfilt(b, a, eog_raw)

    print(f"[denoising] EOG: pseudo-EOG peak={np.abs(eog_lp).max():.1f}µV, threshold={EOG_BLINK_THRESHOLD_UV}µV")
    blink_mask = np.abs(eog_lp) > EOG_BLINK_THRESHOLD_UV
    if not blink_mask.any():
        print("[denoising] EOG: no blinks detected — skipping regression.")
        return df

    expanded = np.zeros(n, dtype=bool)
    for idx in np.where(blink_mask)[0]:
        lo = max(0, idx - EOG_BLINK_CONTEXT)
        hi = min(n - 1, idx + EOG_BLINK_CONTEXT)
        expanded[lo:hi + 1] = True

    n_blinks = int(np.sum(np.diff(expanded.astype(int)) == 1))
    print(f"[denoising] EOG: {n_blinks} blink(s) detected.")

    eog_blink    = eog_raw[expanded] - eog_raw[expanded].mean()
    denom        = np.dot(eog_blink, eog_blink)
    eog_centered = eog_raw - eog_raw.mean()

    for ch in EOG_TARGET_CHANNELS:
        sig       = df[ch].values.astype(float)
        sig_blink = sig[expanded] - sig[expanded].mean()
        if denom >= 1e-12:
            beta   = np.dot(eog_blink, sig_blink) / denom
            df[ch] = sig - beta * eog_centered
            print(f"[denoising] EOG regression: {ch} β={beta:.4f}")

    return df


def eog_interpolate_tp(df):
    """
    Stage 2 — Interpolate residual blink artefacts in TP9/TP10 after bandpass.
    Any contiguous run of samples exceeding EOG_TP_THRESHOLD_UV is replaced
    with linear interpolation between the nearest clean anchor points.
    Called after bandpass filtering so the threshold operates on the filtered signal.
    """
    df    = df.copy()
    n     = len(df)
    t_all = np.arange(n)

    for ch in EOG_TARGET_CHANNELS:
        sig         = df[ch].values.astype(float)
        artefact    = np.abs(sig) > EOG_TP_THRESHOLD_UV
        if not artefact.any():
            continue

        sig_clean  = sig.copy()
        padded     = np.concatenate([[False], artefact, [False]])
        run_starts = np.where(np.diff(padded.astype(int)) == 1)[0]
        run_ends   = np.where(np.diff(padded.astype(int)) == -1)[0]

        n_interp = 0
        for rs, re in zip(run_starts, run_ends):
            pre = rs - 1
            while pre > 0 and np.abs(sig[pre]) > EOG_TP_THRESHOLD_UV:
                pre -= 1
            post = re
            while post < n - 1 and np.abs(sig[post]) > EOG_TP_THRESHOLD_UV:
                post += 1
            sig_clean[rs:re] = np.interp(t_all[rs:re], [pre, post], [sig[pre], sig[post]])
            n_interp += re - rs

        df[ch] = sig_clean
        print(f"[denoising] EOG interpolation: {ch} — {n_interp} samples replaced.")

    return df


# --- Bandpass FIR ---

def bandpass_fir(sig, fs=SAMPLE_RATE, low=ERP_HIGHPASS, high=ERP_LOWPASS):
    """Zero-phase FIR bandpass filter using firwin + filtfilt."""
    if len(sig) < 30:
        print(f"[denoising] Signal too short for bandpass ({len(sig)} samples), skipping.")
        return sig
    # Order: 3 * fs / low_cutoff, but capped so filtfilt padlen < signal length
    # filtfilt padlen = 3 * (order), so max order = len(sig) // 3 - 1
    ideal_order = int(3 * fs / low)
    max_order = len(sig) // 3 - 1
    order = min(ideal_order, max_order)
    # Order must be even for bandpass firwin
    if order % 2 != 0:
        order -= 1
    order = max(order, 8)  # minimum sensible order
    nyq = fs / 2.0
    taps = signal.firwin(order + 1, [low / nyq, high / nyq], pass_zero=False, window='hamming')
    return signal.filtfilt(taps, 1.0, sig)


# --- Step 6A: Savitzky-Golay (ERP path) ---

def savitzky_golay(sig, window=SG_WINDOW, polyorder=SG_POLYORDER):
    """Smooth signal preserving ERP peak morphology."""
    if len(sig) < window:
        print(f"[denoising] Signal too short for Savitzky-Golay ({len(sig)} samples), skipping.")
        return sig
    return signal.savgol_filter(sig, window_length=window, polyorder=polyorder)


# --- ERP path (Steps 5A-7A) ---

def run_erp_path(df, bad_segments):
    """Apply bandpass 0.1-30Hz, Savitzky-Golay, then TP blink interpolation."""
    out = df.copy()
    for ch in CHANNELS:
        sig = df[ch].values.astype(float)
        sig = bandpass_fir(sig, fs=SAMPLE_RATE, low=ERP_HIGHPASS, high=ERP_LOWPASS)
        sig = savitzky_golay(sig, window=SG_WINDOW, polyorder=SG_POLYORDER)
        out[ch] = sig
    out = eog_interpolate_tp(out)
    out['bad_segment'] = bad_segments.values
    return out[['time'] + CHANNELS + ['bad_segment']]


# --- PSD path (Steps 5B-6B) ---

def run_psd_path(df, bad_segments):
    """Apply bandpass 1-45Hz, then TP blink interpolation."""
    out = df.copy()
    for ch in CHANNELS:
        sig = df[ch].values.astype(float)
        sig = bandpass_fir(sig, fs=SAMPLE_RATE, low=PSD_HIGHPASS, high=PSD_LOWPASS)
        out[ch] = sig
    out = eog_interpolate_tp(out)
    out['bad_segment'] = bad_segments.values
    return out[['time'] + CHANNELS + ['bad_segment']]


# --- Main pipeline ---

def denoise_session(eeg_path, erp_out, psd_out):
    """
    Denoise a single EEG session file and write erp_clean and psd_clean CSVs.
    Returns (erp_df, psd_df).
    """
    print(f"[denoising] Loading {eeg_path}")
    df = load_eeg(eeg_path)

    # Step 0: mark dropouts
    df, dropout_mask = mark_dropouts(df)

    # Step 1: interpolate short gaps, flag long ones
    df, bad_segments = interpolate_dropouts(df, dropout_mask)

    # Step 2: notch filter — runs before EOG regression so that 50 Hz mains
    # noise does not bias the OLS β estimate (shared 50 Hz between AF and TP
    # channels would otherwise be partially absorbed into the regression fit).
    print("[denoising] Applying notch filter (50 Hz)...")
    for ch in CHANNELS:
        df[ch] = notch_filter(df[ch].values)

    # Step 3: EOG artefact removal — after notch so the pseudo-EOG reference
    # is clean, but before wavelet/EMD so blink amplitudes are still large
    # enough to (a) be detected reliably and (b) not inflate the wavelet noise
    # estimate used to set the soft-threshold.
    print("[denoising] Applying EOG artefact removal...")
    df = eog_regression(df)

    # Step 4: wavelet denoising
    print("[denoising] Applying wavelet denoising...")
    for ch in CHANNELS:
        df[ch] = wavelet_denoise(df[ch].values)

    # Step 5: EMD drift removal
    print("[denoising] Applying EMD drift removal...")
    for ch in CHANNELS:
        df[ch] = emd_drift_remove(df[ch].values)

    # Path split
    print("[denoising] Running ERP path...")
    erp_df = run_erp_path(df, bad_segments)

    print("[denoising] Running PSD path...")
    psd_df = run_psd_path(df, bad_segments)

    erp_df.to_csv(erp_out, index=False)
    psd_df.to_csv(psd_out, index=False)
    print(f"[denoising] Saved {erp_out} and {psd_out}")

    return erp_df, psd_df


def main(participant_id, data_dir='data', out_dir='output'):
    """
    Denoise all EEG sessions for a participant.

    Discovers all session_*_eeg.csv files under data/<participant_id>/,
    writes erp_clean and psd_clean CSVs to output/<participant_id>/.

    Returns list of (erp_path, psd_path) tuples, one per session.
    """
    import glob as _glob
    import os

    participant_data_dir = os.path.join(data_dir, participant_id)
    participant_out_dir  = os.path.join(out_dir,  participant_id)

    eeg_files = sorted(_glob.glob(os.path.join(participant_data_dir, 'session_*_eeg.csv')))
    if not eeg_files:
        print(f"[denoising] No EEG files found for participant '{participant_id}' in {participant_data_dir}")
        sys.exit(1)

    os.makedirs(participant_out_dir, exist_ok=True)
    print(f"[denoising] Found {len(eeg_files)} session(s) for participant '{participant_id}'")

    results = []
    for eeg_path in eeg_files:
        # e.g. session_20260303_185045_auditory_eeg.csv -> session_20260303_185045_auditory
        basename = os.path.basename(eeg_path).replace('_eeg.csv', '')
        erp_out = os.path.join(participant_out_dir, f'{basename}_erp_clean.csv')
        psd_out = os.path.join(participant_out_dir, f'{basename}_psd_clean.csv')

        print(f"\n[denoising] === Session: {basename} ===")
        denoise_session(eeg_path, erp_out, psd_out)
        results.append((erp_out, psd_out))

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Denoise EEG sessions for a participant.')
    parser.add_argument('participant_id', help='Participant ID (e.g. 001, test)')
    parser.add_argument('--data-dir', default='data', help='Root data directory (default: data)')
    parser.add_argument('--out-dir',  default='output', help='Root output directory (default: output)')
    args = parser.parse_args()
    main(args.participant_id, args.data_dir, args.out_dir)
