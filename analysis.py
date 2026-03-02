"""
Cognitive Load Analysis for BCI FYP
=====================================
Parses all session data for a given participant and produces plots
showing cognitive load indicators across three device conditions:
  Auditory | Vibrations | Shape Changing

Usage:
    python analysis.py <participant_id>
    python analysis.py test

Plots produced (saved as <participant_id>_analysis.png):
  1. EEG Band Power per Task per Device
       - Theta / Alpha / Beta per task window, per device
  2. Alpha / Theta Ratio (cognitive load proxy) per task
  3. Heart Rate over time with task markers
  4. HRV (RMSSD) per task window
  5. GSR over time with task markers
  6. NASA TLX scores comparison across devices
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import signal as scipy_signal

# ── EEG processing constants ──────────────────────────────────────────────────
EEG_FS          = 256        # Hz — overridden from metadata if present
ARTIFACT_VALUE  = -1000.0    # BrainFlow dropout marker; samples at this value are dropped
ARTIFACT_THRESH = 999.0      # Any |value| >= this is treated as artifact

# Band definitions (Hz)
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4,   8),
    'Alpha': (8,  13),
    'Beta':  (13, 30),
}

# ── Post-hoc EEG filtering parameters ─────────────────────────────────────────
# Note: notch (50 Hz) is already applied by muse.py before saving the CSV.
LIGHT_BP_LOW  = 0.1    # Hz
LIGHT_BP_HIGH = 30.0   # Hz
HEAVY_BP_LOW  = 1.0    # Hz
HEAVY_BP_HIGH = 40.0   # Hz

# ICA parameters
ICA_N_COMPONENTS       = None  # None = use all channels
ICA_RANDOM_STATE       = 42
ICA_MAX_ITER           = 500
ICA_VARIANCE_THRESHOLD = 1.5   # exclude components with variance > mean + threshold*std

# Wavelet denoising parameters
WAVELET_TYPE  = 'db4'
WAVELET_LEVEL = 4

# EOG regression parameters (AF7/AF8 used as frontal proxy)
EOG_PROXY_CHANNELS  = ['AF7', 'AF8']
EOG_REGRESSION_LAGS = list(range(11))  # lags 0–10 samples

# CCA parameters (EMG suppression)
CCA_N_COMPONENTS = 2

# Channels to use for cognitive load (frontal preferred)
FRONTAL_CHANNELS = ['AF7', 'AF8']
ALL_CHANNELS     = ['TP9', 'AF7', 'AF8', 'TP10']

# Device order and colours
DEVICES      = ['Auditory', 'Vibrations', 'Shape Changing']
DEVICE_SLUGS = ['auditory', 'vibrations', 'shape_changing']
DEVICE_COLORS = ['#2196F3', '#FF9800', '#4CAF50']   # blue, orange, green

NASA_DIMS = ['Mental Demand', 'Physical Demand', 'Temporal Demand',
             'Performance', 'Effort', 'Frustration']


# ── Data loading ──────────────────────────────────────────────────────────────

def find_session_files(participant_folder: Path) -> dict:
    """
    Find the most recent session files for each device.
    Returns dict keyed by device slug with paths to each file type.
    """
    sessions = {}
    for slug in DEVICE_SLUGS:
        # Find all files matching this device
        eeg_files     = sorted(participant_folder.glob(f'*_{slug}_eeg.csv'))
        hr_files      = sorted(participant_folder.glob(f'*_{slug}_hr.csv'))
        gsr_files     = sorted(participant_folder.glob(f'*_{slug}_gsr.csv'))
        marker_files  = sorted(participant_folder.glob(f'*_{slug}_markers.csv'))
        tlx_files     = sorted(participant_folder.glob(f'*_{slug}_nasa_tlx.json'))
        meta_files    = sorted(participant_folder.glob(f'*_{slug}.json'))

        if not eeg_files:
            continue  # session not found for this device

        # Use the most recent session for each device
        sessions[slug] = {
            'eeg':     eeg_files[-1],
            'hr':      hr_files[-1]     if hr_files     else None,
            'gsr':     gsr_files[-1]    if gsr_files    else None,
            'markers': marker_files[-1] if marker_files else None,
            'tlx':     tlx_files[-1]    if tlx_files    else None,
            'meta':    meta_files[-1]   if meta_files   else None,
        }

    return sessions


def load_eeg(path: Path, fs: int = EEG_FS) -> tuple[pd.DataFrame, int]:
    """Load EEG CSV, drop artifact rows, return (df, fs)."""
    df = pd.read_csv(path)
    df = df[df['time'] >= 0].copy()   # discard pre-session buffer

    # Drop rows where any channel is at the artifact sentinel value
    channel_cols = [c for c in df.columns if c != 'time']
    mask = (df[channel_cols].abs() < ARTIFACT_THRESH).all(axis=1)
    df = df[mask].reset_index(drop=True)
    return df, fs


def load_hr(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df['time'] >= 0].copy()
    return df


def load_gsr(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df['time'] >= 0].copy()
    return df


def load_markers(path: Optional[Path]) -> list[dict]:
    """Returns list of {time, task_number} dicts sorted by time."""
    if path is None or not path.exists():
        return []
    df = pd.read_csv(path)
    df = df[df['time'] >= 0].sort_values('time')
    return df.to_dict('records')


def load_tlx(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_fs_from_meta(meta_path: Optional[Path]) -> int:
    if meta_path is None or not meta_path.exists():
        return EEG_FS
    with open(meta_path) as f:
        data = json.load(f)
    return data.get('metadata', {}).get('muse_sampling_rate', EEG_FS)


# ── EEG signal processing ────────────────────────────────────────────────────

def bandpass(data: np.ndarray, lo: float, hi: float, fs: int) -> np.ndarray:
    """Zero-phase butterworth bandpass filter, order 4."""
    nyq  = fs / 2
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 0.999)
    if lo_n >= hi_n:
        return data
    b, a = scipy_signal.butter(4, [lo_n, hi_n], btype='band')
    # Need at least 3× filter order samples
    if len(data) < 3 * max(len(b), len(a)):
        return data
    return scipy_signal.filtfilt(b, a, data)


def band_power(data: np.ndarray, lo: float, hi: float, fs: int) -> float:
    """
    Welch PSD band power.  Returns mean power in band [lo, hi] Hz.
    """
    if len(data) < fs:          # need at least 1 second
        return np.nan
    nperseg = min(fs * 2, len(data))
    freqs, psd = scipy_signal.welch(data, fs=fs, nperseg=nperseg)
    idx = (freqs >= lo) & (freqs <= hi)
    if not np.any(idx):
        return np.nan
    return float(np.trapezoid(psd[idx], freqs[idx]))


def apply_light_filter(data: np.ndarray, fs: int) -> np.ndarray:
    """
    Apply light bandpass filter (0.1–30 Hz) to raw EEG array.

    The notch filter (50 Hz) is assumed to have been applied already by muse.py
    before saving, so only a bandpass is needed here.

    Args:
        data: ndarray(n_channels, n_samples)
        fs:   sampling rate in Hz

    Returns:
        Filtered ndarray of same shape.
    """
    nyq  = fs / 2
    lo_n = max(LIGHT_BP_LOW  / nyq, 1e-4)
    hi_n = min(LIGHT_BP_HIGH / nyq, 0.999)
    if lo_n >= hi_n:
        return data
    b, a = scipy_signal.butter(4, [lo_n, hi_n], btype='band')
    out = np.zeros_like(data)
    for ch in range(data.shape[0]):
        seg = data[ch]
        if len(seg) < 3 * max(len(b), len(a)):
            out[ch] = seg
        else:
            out[ch] = scipy_signal.filtfilt(b, a, seg)
    return out


def apply_heavy_filter(data: np.ndarray, fs: int, channels: list[str]) -> np.ndarray:
    """
    Apply full heavy denoising pipeline to EEG array.

    Pipeline:
        1. Bandpass 1–40 Hz
        2. EOG regression (AF7/AF8 as frontal proxy, lags 0–10)
        3. ICA artifact removal (variance-based, via MNE)
        4. Wavelet denoising (db4, level 4, soft threshold via PyWavelets)
        5. CCA for EMG suppression (sklearn)

    Args:
        data:     ndarray(n_channels, n_samples) — already notch-filtered
        fs:       sampling rate in Hz
        channels: list of channel names, e.g. ['TP9', 'AF7', 'AF8', 'TP10']

    Returns:
        Cleaned ndarray of same shape.
    """
    import mne
    from mne.preprocessing import ICA as MNE_ICA
    import pywt
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_decomposition import CCA

    n_ch, n_samp = data.shape
    processed = data.copy()

    # ── Step 1: Bandpass 1–40 Hz ──────────────────────────────────────────────
    nyq  = fs / 2
    lo_n = max(HEAVY_BP_LOW  / nyq, 1e-4)
    hi_n = min(HEAVY_BP_HIGH / nyq, 0.999)
    if lo_n < hi_n:
        b, a = scipy_signal.butter(4, [lo_n, hi_n], btype='band')
        for ch in range(n_ch):
            if n_samp >= 3 * max(len(b), len(a)):
                processed[ch] = scipy_signal.filtfilt(b, a, processed[ch])

    # ── Step 2: EOG regression ────────────────────────────────────────────────
    eog_indices = [channels.index(c) for c in EOG_PROXY_CHANNELS if c in channels]
    max_lag = max(EOG_REGRESSION_LAGS)
    if eog_indices and n_samp > max_lag + 10:
        eog_ref = np.mean(processed[eog_indices, :], axis=0)
        X = np.zeros((n_samp - max_lag, len(EOG_REGRESSION_LAGS)))
        for i, lag in enumerate(EOG_REGRESSION_LAGS):
            X[:, i] = eog_ref[max_lag - lag: n_samp - lag]
        for ch in range(n_ch):
            if ch in eog_indices:
                continue
            y = processed[ch, max_lag:]
            reg = LinearRegression()
            reg.fit(X, y)
            processed[ch, max_lag:] = y - reg.predict(X)

    # ── Step 3: ICA ───────────────────────────────────────────────────────────
    if n_samp >= 100:
        try:
            info = mne.create_info(ch_names=channels, sfreq=fs, ch_types='eeg')
            # MNE expects data in volts; scale µV → V
            data_v = processed * 1e-6 if np.max(np.abs(processed)) > 1 else processed
            raw_mne = mne.io.RawArray(data_v, info, verbose=False)
            n_components = ICA_N_COMPONENTS or n_ch
            ica = MNE_ICA(n_components=n_components, random_state=ICA_RANDOM_STATE,
                          max_iter=ICA_MAX_ITER, verbose=False)
            ica.fit(raw_mne)
            sources = ica.get_sources(raw_mne).get_data()
            comp_var = np.var(sources, axis=1)
            thresh = np.mean(comp_var) + ICA_VARIANCE_THRESHOLD * np.std(comp_var)
            ica.exclude = np.where(comp_var > thresh)[0].tolist()
            raw_clean = ica.apply(raw_mne.copy())
            processed = raw_clean.get_data() * 1e6  # back to µV
        except Exception as e:
            print(f"[heavy_filter] ICA failed: {e}. Skipping.")

    # ── Step 4: Wavelet denoising ─────────────────────────────────────────────
    denoised = np.zeros_like(processed)
    for ch in range(n_ch):
        try:
            coeffs = pywt.wavedec(processed[ch], WAVELET_TYPE, level=WAVELET_LEVEL)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thr   = sigma * np.sqrt(2 * np.log(max(len(processed[ch]), 1)))
            new_coeffs = [coeffs[0]] + [
                pywt.threshold(c, thr, mode='soft') for c in coeffs[1:]
            ]
            denoised[ch] = pywt.waverec(new_coeffs, WAVELET_TYPE)[:n_samp]
        except Exception:
            denoised[ch] = processed[ch]
    processed = denoised

    # ── Step 5: CCA for EMG suppression ──────────────────────────────────────
    if n_samp >= 100:
        try:
            delay = 1
            X_cca = processed[:, :-delay].T
            Y_cca = processed[:, delay:].T
            cca = CCA(n_components=min(CCA_N_COMPONENTS, n_ch))
            cca.fit(X_cca, Y_cca)
            X_c, _ = cca.transform(X_cca, Y_cca)
            X_rec  = cca.inverse_transform(X_c)
            result = np.zeros_like(processed)
            result[:, :-delay] = X_rec.T
            result[:, -delay:] = processed[:, -delay:]
            processed = result
        except Exception as e:
            print(f"[heavy_filter] CCA failed: {e}. Skipping.")

    return processed


def filter_raw_eeg(raw_eeg_path: Path, fs: int = EEG_FS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a raw (notch-filtered) EEG CSV and produce lightly and heavily filtered
    versions, saved alongside the source file.

    Output files:
        <stem>_light.csv  — bandpass 0.1–30 Hz
        <stem>_heavy.csv  — full denoising pipeline

    Args:
        raw_eeg_path: Path to the raw EEG CSV (columns: time, TP9, AF7, AF8, TP10)
        fs:           Sampling rate in Hz (default EEG_FS)

    Returns:
        (light_df, heavy_df) — DataFrames with same columns as input
    """
    raw_eeg_path = Path(raw_eeg_path)
    eeg_df, _ = load_eeg(raw_eeg_path, fs)

    channel_cols = [c for c in eeg_df.columns if c != 'time']
    data = eeg_df[channel_cols].values.T  # (n_channels, n_samples)

    timestamps = eeg_df['time'].values

    # Light filter
    light_data = apply_light_filter(data, fs)
    light_df   = pd.DataFrame(light_data.T, columns=channel_cols)
    light_df.insert(0, 'time', timestamps)
    light_path = raw_eeg_path.with_name(raw_eeg_path.stem + '_light.csv')
    light_df.to_csv(light_path, index=False)
    print(f"Light-filtered EEG saved to: {light_path}")

    # Heavy filter
    heavy_data = apply_heavy_filter(data, fs, channel_cols)
    heavy_df   = pd.DataFrame(heavy_data.T, columns=channel_cols)
    heavy_df.insert(0, 'time', timestamps)
    heavy_path = raw_eeg_path.with_name(raw_eeg_path.stem + '_heavy.csv')
    heavy_df.to_csv(heavy_path, index=False)
    print(f"Heavy-filtered EEG saved to: {heavy_path}")

    return light_df, heavy_df


def task_band_powers(eeg_df: pd.DataFrame, markers: list[dict],
                     channels: list[str], fs: int) -> pd.DataFrame:
    """
    Compute band powers for each task window.

    Task windows are defined by consecutive task_end markers:
      Task 1: t=0  → marker[0].time
      Task 2: marker[0].time → marker[1].time
      …
    If no markers, the whole session is treated as one window.
    """
    if not markers:
        boundaries = [0, eeg_df['time'].max()]
    else:
        boundaries = [0] + [m['time'] for m in markers]

    records = []
    for i in range(len(boundaries) - 1):
        t_start = boundaries[i]
        t_end   = boundaries[i + 1]
        window  = eeg_df[(eeg_df['time'] >= t_start) & (eeg_df['time'] < t_end)]

        if len(window) < 10:
            continue

        row = {'task': i + 1, 't_start': t_start, 't_end': t_end,
               'duration': t_end - t_start}

        for band, (lo, hi) in BANDS.items():
            powers = []
            for ch in channels:
                if ch not in window.columns:
                    continue
                seg = window[ch].values
                seg = seg[np.isfinite(seg)]
                if len(seg) > 10:
                    powers.append(band_power(seg, lo, hi, fs))
            row[band] = float(np.nanmean(powers)) if powers else np.nan

        # Cognitive load proxies
        row['Theta_Alpha_ratio'] = (row['Theta'] / row['Alpha']
                                    if row['Alpha'] and row['Alpha'] > 0 else np.nan)
        row['Alpha_Beta_ratio']  = (row['Alpha'] / row['Beta']
                                    if row['Beta']  and row['Beta']  > 0 else np.nan)
        records.append(row)

    return pd.DataFrame(records)


# ── HRV metrics ──────────────────────────────────────────────────────────────

def parse_rr_series(hr_df: pd.DataFrame) -> np.ndarray:
    """Extract all RR intervals (ms) from the HR dataframe."""
    rrs = []
    for _, row in hr_df.iterrows():
        rr_str = str(row.get('rr_intervals', ''))
        for part in rr_str.split(';'):
            part = part.strip()
            if part:
                try:
                    rrs.append(float(part))
                except ValueError:
                    pass
    return np.array(rrs)


def rmssd(rr_intervals: np.ndarray) -> float:
    if len(rr_intervals) < 2:
        return np.nan
    diffs = np.diff(rr_intervals)
    return float(np.sqrt(np.mean(diffs ** 2)))


def task_hrv(hr_df: pd.DataFrame, markers: list[dict]) -> pd.DataFrame:
    """RMSSD per task window using RR intervals that fall in that window."""
    if hr_df is None or hr_df.empty:
        return pd.DataFrame()

    if not markers:
        rrs = parse_rr_series(hr_df)
        return pd.DataFrame([{'task': 1, 'rmssd': rmssd(rrs),
                               'mean_hr': hr_df['heart_rate'].mean()}])

    boundaries = [0] + [m['time'] for m in markers]
    records = []
    for i in range(len(boundaries) - 1):
        t_start = boundaries[i]
        t_end   = boundaries[i + 1]
        window  = hr_df[(hr_df['time'] >= t_start) & (hr_df['time'] < t_end)]
        rrs     = parse_rr_series(window)
        records.append({
            'task':    i + 1,
            'rmssd':   rmssd(rrs),
            'mean_hr': float(window['heart_rate'].mean()) if not window.empty else np.nan,
        })
    return pd.DataFrame(records)


# ── GSR metrics ──────────────────────────────────────────────────────────────

def task_gsr(gsr_df: pd.DataFrame, markers: list[dict]) -> pd.DataFrame:
    """Mean and std GSR (µS) per task window."""
    if gsr_df is None or gsr_df.empty:
        return pd.DataFrame()

    if not markers:
        return pd.DataFrame([{'task': 1,
                               'mean_gsr': gsr_df['gsr_uS'].mean(),
                               'std_gsr':  gsr_df['gsr_uS'].std()}])

    boundaries = [0] + [m['time'] for m in markers]
    records = []
    for i in range(len(boundaries) - 1):
        t_start = boundaries[i]
        t_end   = boundaries[i + 1]
        window  = gsr_df[(gsr_df['time'] >= t_start) & (gsr_df['time'] < t_end)]
        records.append({
            'task':     i + 1,
            'mean_gsr': float(window['gsr_uS'].mean()) if not window.empty else np.nan,
            'std_gsr':  float(window['gsr_uS'].std())  if not window.empty else np.nan,
        })
    return pd.DataFrame(records)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _task_x(n_tasks: int, device_idx: int, n_devices: int = 3,
            width: float = 0.25) -> np.ndarray:
    """X positions for grouped bar chart."""
    base   = np.arange(1, n_tasks + 1, dtype=float)
    offset = (device_idx - n_devices / 2 + 0.5) * width
    return base + offset


def plot_analysis(participant_id: str, all_data: dict, output_path: Path):
    """Build and save the full analysis figure."""

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(f'Cognitive Load Analysis — Participant: {participant_id}',
                 fontsize=16, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.35,
                           top=0.96, bottom=0.04, left=0.07, right=0.97)

    ax_theta_alpha = fig.add_subplot(gs[0, 0])   # Theta/Alpha ratio per task
    ax_alpha       = fig.add_subplot(gs[0, 1])   # Alpha power per task
    ax_beta        = fig.add_subplot(gs[1, 0])   # Beta power per task
    ax_theta       = fig.add_subplot(gs[1, 1])   # Theta power per task
    ax_hr          = fig.add_subplot(gs[2, 0])   # HR time-series
    ax_gsr         = fig.add_subplot(gs[2, 1])   # GSR time-series
    ax_hrv         = fig.add_subplot(gs[3, 0])   # HRV (RMSSD) per task
    ax_tlx         = fig.add_subplot(gs[3, 1])   # NASA TLX comparison

    legend_patches = [Patch(color=DEVICE_COLORS[i], label=DEVICES[i])
                      for i in range(len(DEVICES))]

    bar_width = 0.25
    max_tasks = max(
        (len(d['bp']) for d in all_data.values() if d.get('bp') is not None
         and not d['bp'].empty),
        default=1
    )

    # ── Per-task band power bars ──────────────────────────────────────────────
    for di, (slug, color) in enumerate(zip(DEVICE_SLUGS, DEVICE_COLORS)):
        d = all_data.get(slug, {})
        bp = d.get('bp')
        if bp is None or bp.empty:
            continue
        n_tasks = len(bp)
        xs      = _task_x(n_tasks, di, n_devices=3, width=bar_width)

        ax_theta_alpha.bar(xs, bp['Theta_Alpha_ratio'], width=bar_width,
                           color=color, alpha=0.85, label=DEVICES[di])
        ax_alpha.bar(xs, bp['Alpha'], width=bar_width, color=color, alpha=0.85)
        ax_beta.bar(xs,  bp['Beta'],  width=bar_width, color=color, alpha=0.85)
        ax_theta.bar(xs, bp['Theta'], width=bar_width, color=color, alpha=0.85)

    for ax, title, ylabel in [
        (ax_theta_alpha, 'Theta / Alpha Ratio per Task\n(higher = more cognitive load)',
         'Theta / Alpha'),
        (ax_alpha, 'Alpha Power per Task\n(lower = more cognitive engagement)',
         'Power (µV²/Hz)'),
        (ax_beta,  'Beta Power per Task\n(higher = more active mental processing)',
         'Power (µV²/Hz)'),
        (ax_theta, 'Theta Power per Task\n(higher = more cognitive load)',
         'Power (µV²/Hz)'),
    ]:
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Task', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(range(1, max_tasks + 1))
        ax.legend(handles=legend_patches, fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    # ── HR time-series ────────────────────────────────────────────────────────
    any_hr = False
    for di, (slug, color) in enumerate(zip(DEVICE_SLUGS, DEVICE_COLORS)):
        d   = all_data.get(slug, {})
        hr  = d.get('hr')
        if hr is None or hr.empty:
            continue
        ax_hr.plot(hr['time'], hr['heart_rate'], color=color,
                   marker='o', markersize=3, linewidth=1.5, label=DEVICES[di])
        # Marker lines
        for m in d.get('markers', []):
            ax_hr.axvline(m['time'], color=color, linestyle='--',
                          alpha=0.4, linewidth=0.8)
        any_hr = True

    ax_hr.set_title('Heart Rate Over Time', fontsize=10)
    ax_hr.set_xlabel('Time (s)', fontsize=9)
    ax_hr.set_ylabel('HR (bpm)', fontsize=9)
    if any_hr:
        ax_hr.legend(handles=legend_patches, fontsize=8)
    ax_hr.grid(alpha=0.3)

    # ── GSR time-series ───────────────────────────────────────────────────────
    any_gsr = False
    for di, (slug, color) in enumerate(zip(DEVICE_SLUGS, DEVICE_COLORS)):
        d   = all_data.get(slug, {})
        gsr = d.get('gsr')
        if gsr is None or gsr.empty:
            continue
        ax_gsr.plot(gsr['time'], gsr['gsr_uS'], color=color,
                    linewidth=1.2, alpha=0.85, label=DEVICES[di])
        for m in d.get('markers', []):
            ax_gsr.axvline(m['time'], color=color, linestyle='--',
                           alpha=0.4, linewidth=0.8)
        any_gsr = True

    ax_gsr.set_title('Galvanic Skin Response Over Time', fontsize=10)
    ax_gsr.set_xlabel('Time (s)', fontsize=9)
    ax_gsr.set_ylabel('GSR (µS)', fontsize=9)
    if any_gsr:
        ax_gsr.legend(handles=legend_patches, fontsize=8)
    ax_gsr.grid(alpha=0.3)

    # ── HRV (RMSSD) per task ──────────────────────────────────────────────────
    for di, (slug, color) in enumerate(zip(DEVICE_SLUGS, DEVICE_COLORS)):
        d   = all_data.get(slug, {})
        hrv = d.get('hrv')
        if hrv is None or hrv.empty:
            continue
        n_tasks = len(hrv)
        xs      = _task_x(n_tasks, di, n_devices=3, width=bar_width)
        vals    = hrv['rmssd'].values
        ax_hrv.bar(xs, vals, width=bar_width, color=color, alpha=0.85,
                   label=DEVICES[di])

    ax_hrv.set_title('HRV (RMSSD) per Task\n(higher = lower cognitive stress)',
                     fontsize=10)
    ax_hrv.set_xlabel('Task', fontsize=9)
    ax_hrv.set_ylabel('RMSSD (ms)', fontsize=9)
    ax_hrv.set_xticks(range(1, max_tasks + 1))
    ax_hrv.legend(handles=legend_patches, fontsize=8)
    ax_hrv.grid(axis='y', alpha=0.3)

    # ── NASA TLX radar / grouped bar ─────────────────────────────────────────
    x_base = np.arange(len(NASA_DIMS))
    tlx_bar_width = 0.25
    any_tlx = False
    for di, (slug, color) in enumerate(zip(DEVICE_SLUGS, DEVICE_COLORS)):
        tlx = all_data.get(slug, {}).get('tlx')
        if tlx is None:
            continue
        scores = [tlx['scores'].get(dim, 50) for dim in NASA_DIMS]
        xs     = x_base + (di - 1) * tlx_bar_width
        ax_tlx.bar(xs, scores, width=tlx_bar_width, color=color,
                   alpha=0.85, label=DEVICES[di])
        # Average line marker
        avg = tlx.get('average', np.mean(scores))
        ax_tlx.hlines(avg, xs[0] - tlx_bar_width / 2, xs[-1] + tlx_bar_width / 2,
                      colors=color, linestyles='dashed', linewidth=1.5)
        any_tlx = True

    ax_tlx.set_title('NASA TLX Scores by Device\n(dashed = average)', fontsize=10)
    ax_tlx.set_xticks(x_base)
    short_labels = ['Mental', 'Physical', 'Temporal', 'Performance', 'Effort', 'Frustration']
    ax_tlx.set_xticklabels(short_labels, fontsize=8, rotation=20, ha='right')
    ax_tlx.set_ylabel('Score (0–100)', fontsize=9)
    ax_tlx.set_ylim(0, 105)
    ax_tlx.legend(handles=legend_patches, fontsize=8)
    ax_tlx.grid(axis='y', alpha=0.3)

    if not any_tlx:
        ax_tlx.text(0.5, 0.5, 'No TLX data', transform=ax_tlx.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Analysis saved to: {output_path}")
    plt.show()


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(participant_id: str, all_data: dict):
    print(f"\n{'='*65}")
    print(f"COGNITIVE LOAD SUMMARY — Participant: {participant_id}")
    print(f"{'='*65}")

    for slug, device_name in zip(DEVICE_SLUGS, DEVICES):
        d = all_data.get(slug, {})
        print(f"\n  [{device_name}]")

        bp = d.get('bp')
        if bp is not None and not bp.empty:
            print(f"    EEG tasks analysed : {len(bp)}")
            print(f"    Theta/Alpha (mean) : {bp['Theta_Alpha_ratio'].mean():.3f}")
            print(f"    Alpha power (mean) : {bp['Alpha'].mean():.2f} µV²/Hz")
            print(f"    Beta  power (mean) : {bp['Beta'].mean():.2f} µV²/Hz")
        else:
            print(f"    EEG data           : not available")

        hr = d.get('hr')
        if hr is not None and not hr.empty:
            print(f"    Mean HR            : {hr['heart_rate'].mean():.1f} bpm")

        hrv = d.get('hrv')
        if hrv is not None and not hrv.empty:
            print(f"    RMSSD (mean)       : {hrv['rmssd'].mean():.1f} ms")

        gsr = d.get('gsr')
        if gsr is not None and not gsr.empty:
            print(f"    Mean GSR           : {gsr['gsr_uS'].mean():.6f} µS")

        tlx = d.get('tlx')
        if tlx:
            print(f"    NASA TLX average   : {tlx.get('average', '?'):.1f}/100")

    print(f"\n{'='*65}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Cognitive load analysis for a BCI FYP participant.'
    )
    parser.add_argument('participant_id',
                        help='Participant ID (e.g. P01, test)')
    parser.add_argument('--data-dir', default='data',
                        help='Root data directory (default: data/)')
    parser.add_argument('--no-show', action='store_true',
                        help='Save plot without displaying it')
    args = parser.parse_args()

    if not args.participant_id:
        parser.error('participant_id is required.')

    participant_id     = args.participant_id
    participant_folder = Path(args.data_dir) / participant_id

    if not participant_folder.exists():
        print(f"ERROR: Participant folder not found: {participant_folder}")
        sys.exit(1)

    sessions = find_session_files(participant_folder)
    if not sessions:
        print(f"ERROR: No session files found under {participant_folder}")
        sys.exit(1)

    print(f"Found sessions for: {list(sessions.keys())}")

    # ── Load and process each device session ──────────────────────────────────
    all_data: dict = {}

    for slug in DEVICE_SLUGS:
        if slug not in sessions:
            print(f"  [{slug}] No data — skipping.")
            all_data[slug] = {}
            continue

        paths = sessions[slug]
        print(f"  [{slug}] Loading data...")

        fs      = get_fs_from_meta(paths['meta'])
        filter_raw_eeg(paths['eeg'], fs)
        eeg, _  = load_eeg(paths['eeg'], fs)
        hr      = load_hr(paths['hr'])
        gsr     = load_gsr(paths['gsr'])
        markers = load_markers(paths['markers'])
        tlx     = load_tlx(paths['tlx'])

        print(f"           EEG samples: {len(eeg)}, HR rows: "
              f"{len(hr) if hr is not None else 0}, "
              f"GSR rows: {len(gsr) if gsr is not None else 0}, "
              f"Markers: {len(markers)}")

        bp  = task_band_powers(eeg, markers, FRONTAL_CHANNELS, fs)
        hrv = task_hrv(hr, markers)
        gsr_task = task_gsr(gsr, markers)

        all_data[slug] = {
            'eeg':     eeg,
            'hr':      hr,
            'gsr':     gsr,
            'markers': markers,
            'tlx':     tlx,
            'bp':      bp,
            'hrv':     hrv,
            'gsr_task': gsr_task,
            'fs':      fs,
        }

    print_summary(participant_id, all_data)

    output_path = participant_folder / f'{participant_id}_analysis.png'

    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')

    plot_analysis(participant_id, all_data, output_path)


if __name__ == '__main__':
    main()
