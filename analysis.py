"""
analysis.py - EEG analysis pipeline

Reads pre-denoised erp_clean.csv and psd_clean.csv (produced by denoising.py)
and the baseline features JSON produced by baseline.py, then runs the full
analysis pipeline.

Usage:
    python denoising.py <participant_id>
    python analysis.py  <participant_id> [--data-dir data] [--out-dir output]
"""

import sys
import os
import json
import math
import warnings
import glob as _glob
from collections import Counter

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal, integrate, stats
from scipy.stats import kurtosis as sp_kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE      = 256
EPOCH_PRE_MS     = 200
EPOCH_POST_MS    = 800
BASELINE_PRE_MS  = 200
ARTIFACT_THRESH  = 100.0
MIN_EPOCHS       = 20
KURTOSIS_THRESH  = 5.0

WINDOW_SEC    = 4
OVERLAP_RATIO = 0.5

TASK_EPOCH_PRE_MS  = 500
TASK_EPOCH_POST_MS = 2000

MORLET_FREQS  = np.linspace(1, 45, 100)
MORLET_CYCLES = 6

DFA_MIN_SCALE       = 16
DFA_MAX_SCALE_RATIO = 0.1

PERMEN_ORDER = 3
PERMEN_TAU   = 1

MULTITAPER_NW = 4

BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4,   8),
    'Alpha': (8,  13),
    'Beta':  (13, 30),
}

ALL_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
ERP_CHANNELS = ['TP_pool', 'AF_pool']
PSD_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10', 'TP_pool', 'AF_pool']

CHANNEL_PAIRS = [
    ('AF7', 'AF8'),
    ('TP9', 'TP10'),
    ('AF7', 'TP9'),
    ('AF8', 'TP10'),
    ('AF7', 'TP10'),
    ('AF8', 'TP9'),
]

_ALL_WITH_POOL = ALL_CHANNELS + ['TP_pool', 'AF_pool']


# ── A0: Data Loading and Channel Pooling ─────────────────────────────────────

def load_data(erp_path, psd_path, events_path):
    erp_df    = pd.read_csv(erp_path)
    psd_df    = pd.read_csv(psd_path)
    events_df = pd.read_csv(events_path)
    return erp_df, psd_df, events_df


def load_baseline_features(participant_dir, session_id):
    """Load baseline features JSON. Returns dict or None if not found."""
    path = os.path.join(participant_dir, f'{session_id}_baseline_features.json')
    if not os.path.exists(path):
        print(f"[analysis] WARNING: Baseline features not found at {path}. "
              "Analyses will use absolute power instead of normalised.")
        return None
    with open(path) as f:
        bf = json.load(f)
    iaf = bf.get('IAF_hz')
    if iaf is not None:
        BANDS['Alpha'] = (iaf - 2, iaf + 2)
        print(f"[analysis] IAF={iaf:.1f}Hz, personalised alpha band: "
              f"[{iaf-2:.1f}, {iaf+2:.1f}]Hz")
    return bf


def add_pooled_channels(df):
    """Add TP_pool and AF_pool virtual channels."""
    df = df.copy()
    for col in ALL_CHANNELS:
        if col not in df.columns:
            return df
    df['TP_pool'] = (df['TP9'] + df['TP10']) / 2.0
    df['AF_pool'] = (df['AF7'] + df['AF8']) / 2.0
    return df


def parse_events(events_df):
    """
    Returns:
        oddball_triggers  : list[float]
        standard_triggers : list[float]
        task_intervals    : list[dict]  {task_number, start, end}
        task_starts       : list[float]
    """
    events_df = events_df.copy().sort_values('time').reset_index(drop=True)

    oddball_triggers  = events_df.loc[events_df['event'] == 'oddball_onset',  'time'].tolist()
    standard_triggers = events_df.loc[events_df['event'] == 'standard_onset', 'time'].tolist()

    task_ends      = events_df[events_df['event'] == 'task_end'].sort_values('time')
    task_starts_ev = events_df[events_df['event'] == 'task_start'].sort_values('time')

    task_intervals = []
    task_starts    = []

    if not task_starts_ev.empty:
        for _, row in task_ends.iterrows():
            tn    = int(row['task_number'])
            match = task_starts_ev[task_starts_ev['task_number'] == tn]
            start = float(match['time'].values[0]) if not match.empty else None
            task_starts.append(start)
            task_intervals.append({'task_number': tn, 'start': start, 'end': float(row['time'])})
    else:
        print("[analysis] WARNING: No task_start markers found; inferring from task_end sequence.")
        prev_end = float(events_df['time'].min())
        for _, row in task_ends.iterrows():
            task_starts.append(prev_end)
            task_intervals.append({
                'task_number': int(row['task_number']),
                'start': prev_end,
                'end': float(row['time']),
            })
            prev_end = float(row['time'])

    return oddball_triggers, standard_triggers, task_intervals, task_starts


def get_condition_at_time(t, task_intervals):
    for iv in task_intervals:
        s = iv['start']
        e = iv['end']
        if s is not None and s <= t < e:
            return f"task_{iv['task_number']}"
    return 'inter_trial'


# ── A1: Epoching ──────────────────────────────────────────────────────────────

def _interp_epoch(df, t_start, t_end, n_samples, channels):
    times = df['time'].values
    mask  = (times >= t_start) & (times < t_end)
    seg   = df[mask]
    if len(seg) < 2:
        return None
    data = np.zeros((n_samples, len(channels)))
    t_interp = np.linspace(t_start, t_end, n_samples)
    for ci, ch in enumerate(channels):
        if ch not in df.columns:
            continue
        data[:, ci] = np.interp(t_interp, seg['time'].values, seg[ch].values)
    return data


def extract_epochs(erp_df, triggers, pre_ms=EPOCH_PRE_MS, post_ms=EPOCH_POST_MS,
                   all_triggers=None):
    """
    Scheme A — stimulus-locked epochs.
    Returns: (epochs, n_bad_seg, n_overlap, n_amplitude, n_kurtosis)
    """
    pre_s     = pre_ms  / 1000.0
    post_s    = post_ms / 1000.0
    n_samples = int((pre_s + post_s) * SAMPLE_RATE)
    time_ms   = np.linspace(-pre_ms, post_ms, n_samples)
    channels  = _ALL_WITH_POOL
    all_other = set(all_triggers or []) - set(triggers)

    epochs = []
    n_bad_seg = n_overlap = n_amplitude = n_kurtosis = 0

    for t in triggers:
        t_start = t - pre_s
        t_end   = t + post_s
        times   = erp_df['time'].values
        mask    = (times >= t_start) & (times < t_end)
        seg     = erp_df[mask]

        if 'bad_segment' in seg.columns and seg['bad_segment'].any():
            n_bad_seg += 1
            continue

        if any(t_start <= ot < t_end for ot in all_other):
            n_overlap += 1
            continue

        data = _interp_epoch(erp_df, t_start, t_end, n_samples, channels)
        if data is None:
            n_bad_seg += 1
            continue

        pk2pk = data.max(axis=0) - data.min(axis=0)
        if np.any(pk2pk > 2 * ARTIFACT_THRESH):
            n_amplitude += 1
            continue

        kurt = sp_kurtosis(data, axis=0)
        if np.any(np.abs(kurt) > KURTOSIS_THRESH):
            n_kurtosis += 1
            continue

        epochs.append({'data': data, 'time_ms': time_ms, 'trigger_time': t,
                       'channels': channels})

    print(f"[analysis] Epochs extracted: {len(epochs)} "
          f"(bad_seg={n_bad_seg}, overlap={n_overlap}, "
          f"amplitude={n_amplitude}, kurtosis={n_kurtosis})")
    return epochs, n_bad_seg, n_overlap, n_amplitude, n_kurtosis


def baseline_correct(epochs, pre_ms=BASELINE_PRE_MS):
    pre_samples = int(pre_ms / 1000.0 * SAMPLE_RATE)
    corrected = []
    for ep in epochs:
        d = ep['data'].copy()
        d -= d[:pre_samples, :].mean(axis=0)
        corrected.append({**ep, 'data': d})
    return corrected


def extract_task_onset_epochs(erp_df, task_starts, task_intervals,
                               pre_ms=TASK_EPOCH_PRE_MS, post_ms=TASK_EPOCH_POST_MS):
    """Scheme B — task-onset locked epochs."""
    pre_s     = pre_ms  / 1000.0
    post_s    = post_ms / 1000.0
    n_samples = int((pre_s + post_s) * SAMPLE_RATE)
    time_ms   = np.linspace(-pre_ms, post_ms, n_samples)
    channels  = _ALL_WITH_POOL
    epochs    = []
    n_rej     = 0

    for i, t in enumerate(task_starts):
        if t is None:
            continue
        data = _interp_epoch(erp_df, t - pre_s, t + post_s, n_samples, channels)
        if data is None:
            n_rej += 1
            continue

        pre_samps = int(pre_ms / 1000.0 * SAMPLE_RATE)
        data -= data[:pre_samps, :].mean(axis=0)

        if np.any((data.max(axis=0) - data.min(axis=0)) > 2 * ARTIFACT_THRESH):
            n_rej += 1
            continue
        if np.any(np.abs(sp_kurtosis(data, axis=0)) > KURTOSIS_THRESH):
            n_rej += 1
            continue

        condition = (f"task_{task_intervals[i]['task_number']}"
                     if i < len(task_intervals) else 'unknown')
        epochs.append({'data': data, 'time_ms': time_ms, 'trigger_time': t,
                       'task_number': i, 'condition': condition, 'channels': channels})

    print(f"[analysis] Task-onset epochs: {len(epochs)} accepted, {n_rej} rejected")
    return epochs


def extract_condition_segments(psd_df, task_intervals):
    """Scheme C — full-length condition segments."""
    segments = {}
    times    = psd_df['time'].values

    if task_intervals and task_intervals[0]['start'] is not None:
        first_start = task_intervals[0]['start']
        pre_mask = times < first_start
        if pre_mask.sum() > 0:
            seg = psd_df[pre_mask].copy()
            dur = float(seg['time'].iloc[-1] - seg['time'].iloc[0])
            segments['pre_task'] = seg
            print(f"[analysis] Segment 'pre_task': {dur:.1f}s")

    for iv in task_intervals:
        label = f"task_{iv['task_number']}"
        s, e  = iv['start'], iv['end']
        if s is None:
            continue
        mask = (times >= s) & (times < e)
        seg  = psd_df[mask].copy()
        if len(seg) == 0:
            continue
        segments[label] = seg
        print(f"[analysis] Segment '{label}': {e - s:.1f}s")

    return segments


# ── A2: ERP Analysis ─────────────────────────────────────────────────────────

def average_epochs(epochs):
    stack = np.stack([ep['data'] for ep in epochs], axis=0)
    return stack.mean(axis=0), stack.std(axis=0) / np.sqrt(stack.shape[0])


def find_peak(erp, time_ms, t_start, t_end, polarity='pos'):
    mask = (time_ms >= t_start) & (time_ms <= t_end)
    seg  = erp[mask]
    t_seg = time_ms[mask]
    if len(seg) == 0:
        return 0.0, 0.0
    idx = np.argmax(seg) if polarity == 'pos' else np.argmin(seg)
    return float(t_seg[idx]), float(seg[idx])


def _ch_idx(ch):
    return _ALL_WITH_POOL.index(ch)


def plot_erp(odd_epochs, std_epochs, out_prefix=''):
    if not odd_epochs or not std_epochs:
        print("[analysis] WARNING: Skipping ERP plot (no epochs).")
        return

    time_ms = odd_epochs[0]['time_ms']
    odd_erp, odd_sem = average_epochs(odd_epochs)
    std_erp, std_sem = average_epochs(std_epochs)
    diff_erp = odd_erp - std_erp

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panel_specs = [
        ('TP_pool', 'P300 channel (TP_pool)', 250, 600, 'pos', axes[0]),
        ('AF_pool', 'N200 channel (AF_pool)', 150, 300, 'neg', axes[1]),
    ]
    for ch, title, ps, pe, pol, ax in panel_specs:
        ci  = _ch_idx(ch)
        odd = odd_erp[:, ci]
        std = std_erp[:, ci]
        dif = diff_erp[:, ci]
        ax.axvline(0, color='k', lw=0.8, linestyle='--')
        ax.plot(time_ms, odd, color='steelblue', lw=1.5, label='Oddball')
        ax.fill_between(time_ms, odd - odd_sem[:, ci], odd + odd_sem[:, ci],
                        alpha=0.2, color='steelblue')
        ax.plot(time_ms, std, color='coral', lw=1.5, label='Standard')
        ax.fill_between(time_ms, std - std_sem[:, ci], std + std_sem[:, ci],
                        alpha=0.2, color='coral')
        ax.plot(time_ms, dif, color='darkgreen', lw=1.2, linestyle='--', label='Difference')
        pt, pa = find_peak(odd, time_ms, ps, pe, polarity=pol)
        label = 'P300' if pol == 'pos' else 'N200'
        ax.annotate(f'{label}\n{pt:.0f}ms', xy=(pt, pa),
                    xytext=(pt + 60, pa + (3 if pol == 'pos' else -3)),
                    arrowprops=dict(arrowstyle='->', color='navy'), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xlim(-EPOCH_PRE_MS, EPOCH_POST_MS)

    ax3 = axes[2]
    for ch, color in [('TP_pool', 'steelblue'), ('AF_pool', 'coral')]:
        ci = _ch_idx(ch)
        ax3.plot(time_ms, diff_erp[:, ci], color=color, lw=1.5, label=ch)
    ax3.axvline(0, color='k', lw=0.8, linestyle='--')
    ax3.axhline(0, color='gray', lw=0.5)
    ax3.invert_yaxis()
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Amplitude (µV)')
    ax3.set_title('Difference Waveform (Oddball − Standard)')
    ax3.legend(fontsize=8)
    ax3.set_xlim(-EPOCH_PRE_MS, EPOCH_POST_MS)

    plt.tight_layout()
    fname = f"{out_prefix}plot1_erp_grand_average.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def plot_epoch_heatmap(odd_epochs, out_prefix=''):
    if not odd_epochs:
        return
    ci      = _ch_idx('TP_pool')
    time_ms = odd_epochs[0]['time_ms']
    stack   = np.stack([ep['data'][:, ci] for ep in odd_epochs], axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = np.percentile(np.abs(stack), 95)
    im   = ax.imshow(stack, aspect='auto', origin='lower',
                     extent=[time_ms[0], time_ms[-1], 0, len(odd_epochs)],
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.axvline(0, color='k', lw=1, linestyle='--')
    plt.colorbar(im, ax=ax, label='Amplitude (µV)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Trial')
    ax.set_title('Single-trial Oddball Epochs — TP_pool')
    plt.tight_layout()
    fname = f"{out_prefix}plot2_epoch_heatmap.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def plot_rejection_summary(counts, out_prefix=''):
    labels   = ['bad_seg', 'overlap', 'amplitude', 'kurtosis']
    odd_vals = [counts.get('odd_bad_seg', 0), counts.get('odd_overlap', 0),
                counts.get('odd_amplitude', 0), counts.get('odd_kurtosis', 0)]
    std_vals = [counts.get('std_bad_seg', 0), counts.get('std_overlap', 0),
                counts.get('std_amplitude', 0), counts.get('std_kurtosis', 0)]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, odd_vals, w, label='Oddball',  color='steelblue')
    ax.bar(x + w/2, std_vals, w, label='Standard', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Epochs rejected')
    ax.set_title('Epoch Rejection Summary')
    ax.legend()
    plt.tight_layout()
    fname = f"{out_prefix}plot3_rejection_summary.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def plot_task_onset_erp(task_epochs, out_prefix=''):
    if not task_epochs:
        print("[analysis] WARNING: Skipping task-onset ERP (no epochs).")
        return
    time_ms  = task_epochs[0]['time_ms']
    grand, sem = average_epochs(task_epochs)
    ci       = _ch_idx('AF_pool')
    erp      = grand[:, ci]
    n200_t, n200_a = find_peak(erp, time_ms, 150, 350, polarity='neg')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvline(0, color='k', lw=0.8, linestyle='--', label='Task onset')
    ax.plot(time_ms, erp, color='steelblue', lw=1.5, label='AF_pool')
    ax.fill_between(time_ms, erp - sem[:, ci], erp + sem[:, ci],
                    alpha=0.2, color='steelblue')
    ax.annotate(f'N200\n{n200_t:.0f}ms\n{n200_a:.1f}µV', xy=(n200_t, n200_a),
                xytext=(n200_t + 100, n200_a - 3),
                arrowprops=dict(arrowstyle='->', color='tomato'), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Task-Onset ERP — AF_pool')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = f"{out_prefix}plot_task_onset_erp.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def plot_p300_by_condition(odd_epochs, task_intervals, out_prefix=''):
    if not odd_epochs:
        print("[analysis] WARNING: Skipping P300 by condition (no epochs).")
        return {}
    time_ms = odd_epochs[0]['time_ms']
    ci      = _ch_idx('TP_pool')
    mask    = (time_ms >= 250) & (time_ms <= 600)
    results = {}
    for ep in odd_epochs:
        cond     = get_condition_at_time(ep['trigger_time'], task_intervals)
        win      = ep['data'][mask, ci]
        results.setdefault(cond, {'mean': [], 'peak': []})
        results[cond]['mean'].append(float(win.mean()))
        results[cond]['peak'].append(float(win.max()))

    if not results:
        return {}
    conditions = sorted(results.keys())
    means_m = [np.mean(results[c]['mean']) for c in conditions]
    sems_m  = [np.std(results[c]['mean']) / np.sqrt(len(results[c]['mean'])) for c in conditions]
    means_p = [np.mean(results[c]['peak']) for c in conditions]
    x = np.arange(len(conditions))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, means_m, w, yerr=sems_m, capsize=4,
           label='Mean 250–600ms', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, means_p, w, label='Peak amp', color='coral', alpha=0.8)
    rng = np.random.default_rng(0)
    for i, c in enumerate(conditions):
        yv = results[c]['mean']
        ax.scatter(rng.uniform(x[i] - w/2 - 0.05, x[i] - w/2 + 0.05, len(yv)),
                   yv, color='navy', s=12, alpha=0.6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('P300 Amplitude by Condition (TP_pool)')
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = f"{out_prefix}plot8_p300_by_condition.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")
    return {c: np.mean(results[c]['mean']) for c in conditions}


# ── A3: ERSP / ITPC ──────────────────────────────────────────────────────────

def morlet_wavelet_transform(sig, fs, freqs, n_cycles=MORLET_CYCLES):
    """Returns complex analytic signal: shape (n_freqs, n_times)."""
    n       = len(sig)
    sig_fft = np.fft.fft(sig, n=n)
    result  = np.zeros((len(freqs), n), dtype=complex)

    for fi, f in enumerate(freqs):
        if f <= 0:
            continue
        sigma = n_cycles / (2.0 * np.pi * f)
        t_wav = np.arange(-3 * sigma, 3 * sigma, 1.0 / fs)
        wav   = (np.exp(2j * np.pi * f * t_wav) *
                 np.exp(-t_wav ** 2 / (2 * sigma ** 2)))
        norm  = np.sqrt(np.sum(np.abs(wav) ** 2))
        if norm > 0:
            wav /= norm
        wav_fft = np.fft.fft(wav, n=n)
        result[fi, :] = np.fft.ifft(sig_fft * wav_fft)

    return result


def compute_ersp(epochs_data, fs, freqs, time_ms,
                 baseline_ms=(-200, 0), n_cycles=MORLET_CYCLES):
    n_epochs, n_times = epochs_data.shape
    power_sum = np.zeros((len(freqs), n_times))
    for i in range(n_epochs):
        tf = morlet_wavelet_transform(epochs_data[i], fs, freqs, n_cycles)
        power_sum += np.abs(tf) ** 2
    mean_power = power_sum / n_epochs
    bl_mask    = (time_ms >= baseline_ms[0]) & (time_ms <= baseline_ms[1])
    baseline   = mean_power[:, bl_mask].mean(axis=1, keepdims=True)
    baseline   = np.where(baseline == 0, 1e-12, baseline)
    ersp       = 10.0 * np.log10(mean_power / baseline)
    return ersp, freqs, time_ms


def compute_itpc(epochs_data, fs, freqs, time_ms, n_cycles=MORLET_CYCLES):
    n_epochs, n_times = epochs_data.shape
    phase_vec = np.zeros((len(freqs), n_times), dtype=complex)
    for i in range(n_epochs):
        tf = morlet_wavelet_transform(epochs_data[i], fs, freqs, n_cycles)
        phase_vec += np.exp(1j * np.angle(tf))
    itpc = np.abs(phase_vec) / n_epochs
    return itpc, freqs, time_ms


def plot_ersp_itpc(ersp, itpc, freqs, time_ms, epochs_data,
                   channel_name, condition, out_prefix=''):
    grand_avg = epochs_data.mean(axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                              gridspec_kw={'height_ratios': [1, 2, 2]})
    axes[0].plot(time_ms, grand_avg, color='steelblue', lw=1.2)
    axes[0].axvline(0, color='k', lw=0.8, linestyle='--')
    axes[0].set_ylabel('µV')
    axes[0].set_title(f'ERP — {channel_name} ({condition})')
    axes[0].invert_yaxis()

    im1 = axes[1].pcolormesh(time_ms, freqs, ersp, cmap='RdBu_r',
                              vmin=-3, vmax=3, shading='auto')
    axes[1].axvline(0, color='k', lw=0.8, linestyle='--')
    for _, (lo, _) in BANDS.items():
        axes[1].axhline(lo, color='white', lw=0.5, linestyle='--', alpha=0.6)
    plt.colorbar(im1, ax=axes[1], label='ERSP (dB)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('ERSP')

    im2 = axes[2].pcolormesh(time_ms, freqs, itpc, cmap='hot',
                              vmin=0, vmax=0.5, shading='auto')
    axes[2].axvline(0, color='k', lw=0.8, linestyle='--')
    for _, (lo, _) in BANDS.items():
        axes[2].axhline(lo, color='white', lw=0.5, linestyle='--', alpha=0.6)
    plt.colorbar(im2, ax=axes[2], label='ITPC')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_title('ITPC')

    plt.tight_layout()
    safe = condition.replace(' ', '_')
    fname = f"{out_prefix}plot_ersp_itpc_{channel_name}_{safe}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def run_ersp_itpc_analysis(odd_epochs, std_epochs, out_prefix=''):
    if not odd_epochs:
        print("[analysis] WARNING: Skipping ERSP/ITPC (no epochs).")
        return
    print("\n[analysis] --- ERSP / ITPC ---")
    time_ms = odd_epochs[0]['time_ms']
    for ch in ['TP_pool', 'AF_pool']:
        ci = _ch_idx(ch)
        for label, epochs in [('oddball', odd_epochs), ('standard', std_epochs)]:
            if not epochs:
                continue
            data = np.stack([ep['data'][:, ci] for ep in epochs], axis=0)
            ersp, fq, tm = compute_ersp(data, SAMPLE_RATE, MORLET_FREQS, time_ms)
            itpc, fq, tm = compute_itpc(data, SAMPLE_RATE, MORLET_FREQS, time_ms)
            plot_ersp_itpc(ersp, itpc, fq, tm, data, ch, label, out_prefix)


# ── A4: ERD/ERS ──────────────────────────────────────────────────────────────

def compute_erd_ers(task_epochs, fs, baseline_features=None):
    if not task_epochs:
        return {}, np.array([])

    pre_ms    = int(task_epochs[0]['time_ms'][0])
    post_ms   = int(task_epochs[0]['time_ms'][-1])
    win_ms    = 500
    step_ms   = 250
    win_samp  = int(win_ms / 1000.0 * fs)
    n_times   = task_epochs[0]['data'].shape[0]
    time_axis = np.arange(pre_ms, post_ms - win_ms + 1, step_ms)
    ci        = _ch_idx('TP_pool')

    result = {b: [] for b in ['Theta', 'Alpha', 'Beta']}

    for ep in task_epochs:
        sig = ep['data'][:, ci]
        ep_pow = {}
        for band in ['Theta', 'Alpha', 'Beta']:
            lo, hi   = BANDS[band]
            row_pow  = []
            for t_ms in time_axis:
                i0 = int((t_ms - pre_ms) / 1000.0 * fs)
                i1 = min(i0 + win_samp, n_times)
                if i1 <= i0:
                    row_pow.append(np.nan)
                    continue
                seg = sig[i0:i1]
                f, psd = scipy_signal.welch(seg, fs=fs,
                                             nperseg=min(win_samp, len(seg)), noverlap=0)
                bm = (f >= lo) & (f <= hi)
                row_pow.append(float(integrate.trapezoid(psd[bm], f[bm])) if bm.sum() > 1 else 0.0)
            ep_pow[band] = row_pow

        for band in ['Theta', 'Alpha', 'Beta']:
            # Always use the pre-stimulus window as the ERD baseline so that the
            # band definition used for baseline and epoch power is identical.
            # Using external baseline_features here causes a mismatch when the
            # Alpha band has been personalised via IAF (the stored baseline power
            # was computed with the original 8-13 Hz band, but BANDS['Alpha'] may
            # now point to a different range).
            pre_idx = time_axis < 0
            pre_v   = np.array(ep_pow[band])[pre_idx]
            bl_pow  = float(np.nanmean(pre_v)) if len(pre_v) > 0 else None
            if bl_pow and bl_pow > 0:
                erd = [(p - bl_pow) / bl_pow * 100 if not np.isnan(p) else np.nan
                       for p in ep_pow[band]]
            else:
                erd = [0.0] * len(ep_pow[band])
            result[band].append(erd)

    for b in result:
        result[b] = np.array(result[b])
    return result, time_axis


def plot_erd_ers(erd_data, time_axis, out_prefix=''):
    bands = [b for b in ['Theta', 'Alpha', 'Beta'] if b in erd_data and len(erd_data[b]) > 0]
    if not bands:
        return
    fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 5))
    if len(bands) == 1:
        axes = [axes]
    for ax, band in zip(axes, bands):
        data = erd_data[band]
        mean = np.nanmean(data, axis=0)
        sem  = np.nanstd(data, axis=0) / np.sqrt(data.shape[0])
        ax.axvline(0, color='k', lw=0.8, linestyle='--', label='Task onset')
        ax.axhline(0, color='gray', lw=0.5)
        ax.fill_between(time_axis, mean - sem, mean + sem, alpha=0.25, color='steelblue')
        ax.plot(time_axis, mean, color='steelblue', lw=1.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('ERD/ERS (%)')
        ax.set_title(f'{band} ERD/ERS')
        ax.legend(fontsize=8)
    plt.tight_layout()
    fname = f"{out_prefix}plot_erd_ers.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def run_erd_ers_analysis(task_epochs, baseline_features, out_prefix=''):
    print("\n[analysis] --- ERD/ERS ---")
    erd_data, time_axis = compute_erd_ers(task_epochs, SAMPLE_RATE, baseline_features)
    plot_erd_ers(erd_data, time_axis, out_prefix)


# ── A5: PSD Analysis ─────────────────────────────────────────────────────────

def compute_multitaper_psd(sig, fs, NW=MULTITAPER_NW):
    n    = len(sig)
    Kmax = 2 * NW - 1
    try:
        tapers, ratios = scipy_signal.windows.dpss(n, NW, Kmax=Kmax, return_ratios=True)
    except Exception:
        return scipy_signal.welch(sig, fs=fs, nperseg=min(n, int(4 * fs)))
    if tapers.ndim == 1:
        tapers = tapers[np.newaxis, :]
        ratios = np.array([ratios])
    freqs   = np.fft.rfftfreq(n, d=1.0 / fs)
    psd_acc = np.zeros(len(freqs))
    w_total = 0.0
    for taper, w in zip(tapers, ratios):
        psd_acc += w * np.abs(np.fft.rfft(sig * taper)) ** 2
        w_total += w
    psd = psd_acc / (w_total * fs)
    psd[1:-1] *= 2
    return freqs, psd


def extract_band_power(psds, baseline_features=None):
    rows = []
    for cond, v in psds.items():
        freqs    = v['freqs']
        mean_psd = v['mean_psd']
        ch_list  = v.get('channels', ALL_CHANNELS)
        for ci, ch in enumerate(ch_list):
            psd_ch     = mean_psd[:, ci]
            total_mask = (freqs >= 0.5) & (freqs <= 45)
            total_p    = float(integrate.trapezoid(psd_ch[total_mask], freqs[total_mask]))
            row        = {'condition': cond, 'channel': ch}
            for band, (lo, hi) in BANDS.items():
                bm    = (freqs >= lo) & (freqs <= hi)
                abs_p = float(integrate.trapezoid(psd_ch[bm], freqs[bm])) if bm.sum() > 1 else 0.0
                rel_p = abs_p / total_p if total_p > 0 else np.nan
                row[f'{band}_abs'] = abs_p
                row[f'{band}_rel'] = rel_p
                if baseline_features is not None:
                    bl_ch  = baseline_features.get('channels', {}).get(ch, {})
                    bl_bp  = bl_ch.get('band_power', {}).get(band)
                    row[f'{band}_norm'] = abs_p / bl_bp if (bl_bp and bl_bp > 0) else rel_p
                else:
                    row[f'{band}_norm'] = rel_p
            rows.append(row)
    return pd.DataFrame(rows)


def compute_theta_alpha_ratio(band_df):
    bd = band_df.copy()
    bd['theta_alpha_ratio'] = bd['Theta_abs'] / bd['Alpha_abs'].replace(0, np.nan)
    return bd


def compute_faa(band_df):
    rows = []
    for cond in band_df['condition'].unique():
        cd  = band_df[band_df['condition'] == cond]
        af7 = cd[cd['channel'] == 'AF7']
        af8 = cd[cd['channel'] == 'AF8']
        if af7.empty or af8.empty:
            rows.append({'condition': cond, 'FAA': np.nan})
            continue
        a7 = af7['Alpha_abs'].values[0]
        a8 = af8['Alpha_abs'].values[0]
        rows.append({'condition': cond,
                     'FAA': np.log(a8) - np.log(a7) if (a7 > 0 and a8 > 0) else np.nan})
    return pd.DataFrame(rows)


def run_psd_analysis(condition_segments, baseline_features, out_prefix=''):
    print("\n[analysis] --- PSD Analysis (multitaper) ---")
    if not condition_segments:
        print("[analysis] WARNING: No condition segments for PSD analysis.")
        return {}, pd.DataFrame(), pd.DataFrame()

    psds = {}
    min_samp = int(WINDOW_SEC * SAMPLE_RATE)
    for cond, seg in condition_segments.items():
        if 'bad_segment' in seg.columns:
            seg = seg[seg['bad_segment'] == False]
        if len(seg) < min_samp:
            print(f"[analysis] WARNING: Segment '{cond}' too short, skipping PSD.")
            continue
        ch_list     = [c for c in _ALL_WITH_POOL if c in seg.columns]
        psd_ch_list = []
        freqs_ref   = None
        for ch in ch_list:
            sig = seg[ch].values.astype(float)
            fq, psd = compute_multitaper_psd(sig, SAMPLE_RATE)
            psd_ch_list.append(psd)
            if freqs_ref is None:
                freqs_ref = fq
        if freqs_ref is None:
            continue
        psds[cond] = {'freqs': freqs_ref,
                      'mean_psd': np.stack(psd_ch_list, axis=1),
                      'n_windows': 1,
                      'channels': ch_list}

    if not psds:
        return {}, pd.DataFrame(), pd.DataFrame()

    band_df = extract_band_power(psds, baseline_features)
    band_df = compute_theta_alpha_ratio(band_df)
    faa_df  = compute_faa(band_df)

    csv_path = f"{out_prefix}band_power_summary.csv"
    band_df.to_csv(csv_path, index=False)
    print(f"[analysis] Saved {csv_path}")

    _plot_psd(psds, out_prefix)
    _plot_band_power(band_df, out_prefix)
    _plot_theta_alpha(band_df, out_prefix)
    _plot_faa(faa_df, out_prefix)
    return psds, band_df, faa_df


def _plot_psd(psds, out_prefix):
    band_colors = {'Delta': '#d4e4f7', 'Theta': '#d4f7d4',
                   'Alpha': '#f7f7d4', 'Beta': '#f7d4d4'}
    for ch in PSD_CHANNELS:
        has_ch = any(ch in v.get('channels', []) for v in psds.values())
        if not has_ch:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for cond, v in psds.items():
            ch_list = v.get('channels', ALL_CHANNELS)
            if ch not in ch_list:
                continue
            ci    = ch_list.index(ch)
            freqs = v['freqs']
            psd   = v['mean_psd'][:, ci]
            mask  = freqs <= 45
            ax.semilogy(freqs[mask], psd[mask], label=cond, lw=1.5)
        for band, (lo, hi) in BANDS.items():
            ax.axvspan(lo, hi, alpha=0.12, color=band_colors.get(band, 'gray'))
            ax.text((lo + hi) / 2, ax.get_ylim()[0] * 2, band,
                    ha='center', fontsize=7, color='gray')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (µV²/Hz)')
        ax.set_title(f'PSD — {ch}')
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, 45)
        plt.tight_layout()
        fname = f"{out_prefix}plot4_psd_{ch}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[analysis] Saved {fname}")


def _plot_band_power(band_df, out_prefix):
    if band_df.empty:
        return
    for ch in PSD_CHANNELS:
        ch_df = band_df[band_df['channel'] == ch]
        if ch_df.empty:
            continue
        conditions = ch_df['condition'].tolist()
        band_names = list(BANDS.keys())
        x = np.arange(len(band_names))
        w = 0.8 / max(len(conditions), 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(conditions), 1)))
        for ci, (cond, row) in enumerate(zip(conditions, ch_df.itertuples())):
            vals = [getattr(row, f'{b}_rel', 0) for b in band_names]
            ax.bar(x + ci * w, vals, w, label=cond, color=colors[ci])
        ax.set_xticks(x + w * (len(conditions) - 1) / 2)
        ax.set_xticklabels(band_names)
        ax.set_ylabel('Relative power')
        ax.set_title(f'Band Power — {ch}')
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = f"{out_prefix}plot5_band_power_{ch}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[analysis] Saved {fname}")


def _plot_theta_alpha(band_df, out_prefix):
    if band_df.empty:
        return
    ch_list = [ch for ch in PSD_CHANNELS if not band_df[band_df['channel'] == ch].empty]
    if not ch_list:
        return
    conditions = band_df[band_df['channel'] == ch_list[0]]['condition'].tolist()
    x = np.arange(len(conditions))
    w = 0.8 / max(len(ch_list), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, ch in enumerate(ch_list):
        ch_df  = band_df[band_df['channel'] == ch]
        ratios = ch_df['theta_alpha_ratio'].tolist()
        if len(ratios) == len(conditions):
            ax.bar(x + i * w, ratios, w, label=ch, alpha=0.8)
    ax.set_xticks(x + w * (len(ch_list) - 1) / 2)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.set_ylabel('Theta / Alpha')
    ax.set_title('Theta/Alpha Ratio by Condition')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = f"{out_prefix}plot6_theta_alpha_ratio.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def _plot_faa(faa_df, out_prefix):
    if faa_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(faa_df))
    ax.bar(x, faa_df['FAA'].fillna(0).tolist(), color='mediumpurple', alpha=0.8)
    ax.axhline(0, color='k', lw=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(faa_df['condition'].tolist(), rotation=15, ha='right')
    ax.set_ylabel('FAA (log AF8α − log AF7α)')
    ax.set_title('Frontal Alpha Asymmetry by Condition')
    plt.tight_layout()
    fname = f"{out_prefix}plot9_faa.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def plot_spectrogram(psd_df, task_intervals, out_prefix=''):
    for ch in ALL_CHANNELS:
        if ch not in psd_df.columns:
            continue
        sig   = psd_df[ch].values.astype(float)
        times = psd_df['time'].values
        f, t, Sxx = scipy_signal.spectrogram(
            sig, fs=SAMPLE_RATE, window='hann',
            nperseg=int(WINDOW_SEC * SAMPLE_RATE),
            noverlap=int(WINDOW_SEC * SAMPLE_RATE * OVERLAP_RATIO))
        t_abs     = times[0] + t
        freq_mask = (f >= 1) & (f <= 45)
        cond_colors = plt.cm.tab10(np.linspace(0, 1, max(len(task_intervals), 1)))

        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.pcolormesh(t_abs, f[freq_mask],
                           10 * np.log10(Sxx[freq_mask] + 1e-12),
                           shading='gouraud', cmap='inferno')
        plt.colorbar(im, ax=ax, label='Power (dB)')
        for idx, iv in enumerate(task_intervals):
            s = iv['start']
            e = iv['end']
            if s is None:
                continue
            label = f"task_{iv['task_number']}"
            color = cond_colors[idx % len(cond_colors)]
            ax.axvspan(s, e, alpha=0.1, color=color)
            ax.axvline(s, color='cyan', lw=1, linestyle='--', alpha=0.7)
            ax.axvline(e, color='lime', lw=1, linestyle='--', alpha=0.7)
            ax.text((s + e) / 2, 43, label, ha='center', fontsize=7,
                    color='white', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Spectrogram — {ch}')
        ax.set_ylim(1, 45)
        plt.tight_layout()
        fname = f"{out_prefix}plot7_spectrogram_{ch}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[analysis] Saved {fname}")


# ── A6: Connectivity ─────────────────────────────────────────────────────────

def _bandpass_butter(sig, fs, lo, hi, order=4):
    nyq  = fs / 2.0
    low  = max(lo / nyq, 1e-4)
    high = min(hi / nyq, 0.999)
    if low >= high:
        return sig
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    return scipy_signal.filtfilt(b, a, sig)


def compute_plv(sig1, sig2, fs, band):
    lo, hi = band
    s1 = _bandpass_butter(sig1, fs, lo, hi)
    s2 = _bandpass_butter(sig2, fs, lo, hi)
    ph1 = np.angle(scipy_signal.hilbert(s1))
    ph2 = np.angle(scipy_signal.hilbert(s2))
    return float(np.abs(np.mean(np.exp(1j * (ph1 - ph2)))))


def compute_coherence_band(sig1, sig2, fs, band):
    lo, hi  = band
    nperseg = min(int(4 * fs), len(sig1))
    f, Cxy  = scipy_signal.coherence(sig1, sig2, fs=fs, nperseg=nperseg)
    mask    = (f >= lo) & (f <= hi)
    return float(Cxy[mask].mean()) if mask.sum() > 0 else 0.0


def compute_connectivity_matrix(condition_segments, fs):
    conn = {}
    for cond, seg in condition_segments.items():
        if 'bad_segment' in seg.columns:
            seg = seg[seg['bad_segment'] == False]
        conn[cond] = {}
        for ch1, ch2 in CHANNEL_PAIRS:
            if ch1 not in seg.columns or ch2 not in seg.columns:
                continue
            s1 = seg[ch1].values.astype(float)
            s2 = seg[ch2].values.astype(float)
            pair = f"{ch1}-{ch2}"
            conn[cond][pair] = {}
            for band_name in ['Theta', 'Alpha', 'Beta']:
                band = BANDS[band_name]
                if len(s1) < int(4 * fs):
                    conn[cond][pair][band_name] = {'plv': np.nan, 'coherence': np.nan}
                    continue
                conn[cond][pair][band_name] = {
                    'plv':       compute_plv(s1, s2, fs, band),
                    'coherence': compute_coherence_band(s1, s2, fs, band),
                }
    return conn


def plot_connectivity_matrix(conn_data, out_prefix=''):
    if not conn_data:
        return
    conditions = list(conn_data.keys())
    ch_order   = ALL_CHANNELS
    for band_name in ['Theta', 'Alpha', 'Beta']:
        n_cond = len(conditions)
        fig, axes = plt.subplots(1, n_cond, figsize=(4 * n_cond, 4))
        if n_cond == 1:
            axes = [axes]
        all_vals = [conn_data[c].get(f"{c1}-{c2}", {}).get(band_name, {}).get('plv', np.nan)
                    for c in conditions for c1 in ch_order for c2 in ch_order]
        vmax = max((v for v in all_vals if not np.isnan(v)), default=1.0)
        for ax, cond in zip(axes, conditions):
            mat = np.zeros((4, 4))
            for i, ch1 in enumerate(ch_order):
                for j, ch2 in enumerate(ch_order):
                    v = conn_data[cond].get(f"{ch1}-{ch2}", {}).get(band_name, {}).get('plv', 0)
                    if np.isnan(v):
                        v = 0
                    mat[i, j] = mat[j, i] = v
            im = ax.imshow(mat, cmap='hot', vmin=0, vmax=vmax)
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(ch_order, fontsize=8, rotation=45)
            ax.set_yticklabels(ch_order, fontsize=8)
            ax.set_title(cond)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.04)
        fig.suptitle(f'PLV Connectivity — {band_name}', fontsize=12)
        plt.tight_layout()
        fname = f"{out_prefix}plot_connectivity_{band_name}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[analysis] Saved {fname}")


def plot_connectivity_by_condition(conn_data, out_prefix=''):
    if not conn_data:
        return
    conditions = list(conn_data.keys())
    ft_pairs   = ['AF7-TP9', 'AF8-TP10']
    x = np.arange(len(conditions))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    for pi, pair in enumerate(['AF7-TP9', 'AF8-TP10']):
        vals = [conn_data.get(c, {}).get(pair, {}).get('Theta', {}).get('plv', 0)
                for c in conditions]
        vals = [0 if np.isnan(v) else v for v in vals]
        ax.bar(x + pi * w, vals, w, label=pair,
               color=['steelblue', 'coral'][pi], alpha=0.8)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.set_ylabel('Theta PLV')
    ax.set_title('Frontotemporal Theta PLV by Condition')
    ax.legend()
    plt.tight_layout()
    fname = f"{out_prefix}plot_frontotemporal_plv.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def run_connectivity_analysis(condition_segments, out_prefix=''):
    print("\n[analysis] --- Connectivity ---")
    if not condition_segments:
        print("[analysis] WARNING: Skipping connectivity (no segments).")
        return {}
    conn = compute_connectivity_matrix(condition_segments, SAMPLE_RATE)
    plot_connectivity_matrix(conn, out_prefix)
    plot_connectivity_by_condition(conn, out_prefix)
    return conn


# ── A7: Complexity ────────────────────────────────────────────────────────────

def permutation_entropy(sig, m=PERMEN_ORDER, tau=PERMEN_TAU):
    sig = np.asarray(sig, dtype=float)
    n   = len(sig)
    n_p = n - (m - 1) * tau
    if n_p <= 0:
        return np.nan
    patterns = [tuple(np.argsort(sig[i: i + m * tau: tau])) for i in range(n_p)]
    counts   = Counter(patterns)
    total    = float(sum(counts.values()))
    probs    = np.array([v / total for v in counts.values()])
    me       = np.log(float(math.factorial(m)))
    if me == 0:
        return np.nan
    pe = -float(np.sum(probs * np.log(probs))) / me
    return float(np.clip(pe, 0.0, 1.0))


def higuchi_fd(sig, kmax=10):
    sig = np.asarray(sig, dtype=float)
    n   = len(sig)
    lk  = []
    ks  = []
    for k in range(1, kmax + 1):
        lmk = []
        for m in range(1, k + 1):
            idxs = np.arange(m - 1, n, k)
            if len(idxs) < 2:
                continue
            s    = sig[idxs]
            l    = np.sum(np.abs(np.diff(s))) * (n - 1) / (k * (len(idxs) - 1) * k)
            lmk.append(l)
        if lmk:
            lk.append(np.mean(lmk))
            ks.append(k)
    if len(ks) < 2:
        return np.nan
    coeffs = np.polyfit(np.log(ks), np.log(np.array(lk) + 1e-12), 1)
    return float(-coeffs[0])


def dfa(sig, min_scale=DFA_MIN_SCALE, max_scale_ratio=DFA_MAX_SCALE_RATIO):
    sig = np.asarray(sig, dtype=float)
    n   = len(sig)
    assert n > 1000, f"DFA requires >1000 samples, got {n}"
    y      = np.cumsum(sig - np.mean(sig))
    max_sc = max(int(n * max_scale_ratio), min_scale + 1)
    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_sc), 20).astype(int))
    flucs  = []
    valid  = []
    for s in scales:
        n_blk = n // s
        if n_blk < 2:
            continue
        rms_list = []
        for b in range(n_blk):
            seg  = y[b * s:(b + 1) * s]
            coef = np.polyfit(np.arange(s, dtype=float), seg, 1)
            rms_list.append(np.sqrt(np.mean((seg - np.polyval(coef, np.arange(s))) ** 2)))
        flucs.append(np.mean(rms_list))
        valid.append(s)
    if len(valid) < 2:
        return np.nan, np.array(valid), np.array(flucs)
    coef = np.polyfit(np.log10(valid), np.log10(np.array(flucs) + 1e-12), 1)
    return float(coef[0]), np.array(valid), np.array(flucs)


def compute_complexity(condition_segments, baseline_features=None):
    rows = []
    for cond, seg in condition_segments.items():
        if 'bad_segment' in seg.columns:
            seg = seg[seg['bad_segment'] == False]
        for ch in ALL_CHANNELS:
            if ch not in seg.columns:
                continue
            sig  = seg[ch].values.astype(float)
            pe   = permutation_entropy(sig)
            hfd  = higuchi_fd(sig)
            dfa_a = np.nan
            if len(sig) > 1000:
                dfa_a, _, _ = dfa(sig)

            pe_norm = dfa_norm = np.nan
            if baseline_features is not None:
                bl_ch  = baseline_features.get('channels', {}).get(ch, {})
                bl_pe  = bl_ch.get('permen_baseline')
                bl_dfa = bl_ch.get('dfa_baseline')
                if bl_pe and bl_pe > 0 and not np.isnan(pe):
                    pe_norm = pe / bl_pe
                if bl_dfa and bl_dfa > 0 and not np.isnan(dfa_a):
                    dfa_norm = dfa_a / bl_dfa

            rows.append({'condition': cond, 'channel': ch,
                         'permen': pe, 'hfd': hfd, 'dfa_alpha': dfa_a,
                         'permen_norm': pe_norm, 'dfa_norm': dfa_norm})
    return pd.DataFrame(rows)


def plot_complexity(complexity_df, out_prefix=''):
    if complexity_df.empty:
        return
    conditions = complexity_df['condition'].unique().tolist()
    channels   = [ch for ch in ALL_CHANNELS if ch in complexity_df['channel'].unique()]
    x      = np.arange(len(conditions))
    w      = 0.8 / max(len(channels), 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(channels)))
    for metric, ylabel, sfx in [
        ('permen',    'Permutation Entropy',      'permen'),
        ('hfd',       'Higuchi Fractal Dimension', 'hfd'),
        ('dfa_alpha', 'DFA Scaling Exponent',      'dfa'),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, ch in enumerate(channels):
            ch_df = complexity_df[complexity_df['channel'] == ch]
            vals  = [ch_df[ch_df['condition'] == c][metric].mean() for c in conditions]
            ax.bar(x + i * w, vals, w, label=ch, color=colors[i], alpha=0.85)
        ax.set_xticks(x + w * (len(channels) - 1) / 2)
        ax.set_xticklabels(conditions, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} by Condition')
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = f"{out_prefix}plot_complexity_{sfx}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[analysis] Saved {fname}")


def run_complexity_analysis(condition_segments, baseline_features, out_prefix=''):
    print("\n[analysis] --- Complexity ---")
    if not condition_segments:
        print("[analysis] WARNING: Skipping complexity (no segments).")
        return pd.DataFrame()
    cx_df = compute_complexity(condition_segments, baseline_features)
    plot_complexity(cx_df, out_prefix)
    return cx_df


# ── A8: Multiscale Entropy ────────────────────────────────────────────────────

def sample_entropy(sig, m, r):
    sig = np.asarray(sig, dtype=float)
    n   = len(sig)

    def _count(length):
        count = 0
        for i in range(n - length):
            tmpl = sig[i:i + length]
            for j in range(i + 1, n - length):
                if np.max(np.abs(sig[j:j + length] - tmpl)) < r:
                    count += 1
        return count

    B = _count(m)
    A = _count(m + 1)
    if B == 0:
        return 0.0
    return float(-np.log(A / B))


def multiscale_entropy(sig, m=2, r_factor=0.2, max_scale=20):
    sig = np.asarray(sig, dtype=float)
    r   = r_factor * float(np.std(sig))
    mse = []
    for scale in range(1, max_scale + 1):
        n_c = len(sig) // scale
        if n_c < 10:
            mse.append(np.nan)
            continue
        coarse = sig[:n_c * scale].reshape(-1, scale).mean(axis=1)
        mse.append(sample_entropy(coarse, m, r))
    return np.array(mse)


def run_mse_analysis(condition_segments, out_prefix=''):
    print("\n[analysis] --- MSE ---")
    if not condition_segments:
        print("[analysis] WARNING: Skipping MSE (no segments).")
        return
    max_scale = 20
    scales    = np.arange(1, max_scale + 1)
    for ch in ['TP_pool', 'AF_pool']:
        mse_by_cond = {}
        for cond, seg in condition_segments.items():
            if 'bad_segment' in seg.columns:
                seg = seg[seg['bad_segment'] == False]
            if ch not in seg.columns:
                continue
            sig = seg[ch].values.astype(float)
            mse_by_cond[cond] = multiscale_entropy(sig, max_scale=max_scale)
        if not mse_by_cond:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(mse_by_cond)))
        for (cond, mse_vals), color in zip(mse_by_cond.items(), colors):
            ax.plot(scales, mse_vals, label=cond, color=color, lw=1.5)
        ax.set_xlabel('Scale')
        ax.set_ylabel('Sample Entropy')
        ax.set_title(f'Multiscale Entropy — {ch}')
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = f"{out_prefix}plot_mse_{ch}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[analysis] Saved {fname}")


# ── A9: Cross-Analysis ────────────────────────────────────────────────────────

def plot_theta_alpha_trajectory(condition_segments, out_prefix=''):
    if not condition_segments:
        return
    win_samp  = int(WINDOW_SEC * SAMPLE_RATE)
    step_samp = int(win_samp * (1 - OVERLAP_RATIO))
    fig, ax   = plt.subplots(figsize=(12, 5))
    colors    = plt.cm.tab10(np.linspace(0, 1, len(condition_segments)))
    for (cond, seg), color in zip(condition_segments.items(), colors):
        if 'bad_segment' in seg.columns:
            seg = seg[seg['bad_segment'] == False]
        if 'TP_pool' not in seg.columns:
            continue
        sig   = seg['TP_pool'].values.astype(float)
        times = seg['time'].values if 'time' in seg.columns else np.arange(len(sig)) / SAMPLE_RATE
        ratios, t_mids = [], []
        i = 0
        while i + win_samp <= len(sig):
            win = sig[i:i + win_samp]
            f, psd = scipy_signal.welch(win, fs=SAMPLE_RATE,
                                         nperseg=win_samp, noverlap=0)
            th_mask = (f >= BANDS['Theta'][0]) & (f <= BANDS['Theta'][1])
            al_mask = (f >= BANDS['Alpha'][0]) & (f <= BANDS['Alpha'][1])
            th = float(integrate.trapezoid(psd[th_mask], f[th_mask]))
            al = float(integrate.trapezoid(psd[al_mask], f[al_mask]))
            ratios.append(th / al if al > 0 else np.nan)
            t_mids.append(float(times[i + win_samp // 2]))
            i += step_samp
        if ratios:
            ax.plot(t_mids, ratios, label=cond, color=color, lw=1.2, alpha=0.85)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Theta / Alpha')
    ax.set_title('Theta/Alpha Trajectory (TP_pool)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = f"{out_prefix}plot_theta_alpha_trajectory.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def compute_trial_correlations(odd_epochs, time_ms):
    if not odd_epochs or time_ms is None:
        return pd.DataFrame()
    ci_tp    = _ch_idx('TP_pool')
    ci_af    = _ch_idx('AF_pool')
    p300_m   = (time_ms >= 250) & (time_ms <= 600)
    n200_m   = (time_ms >= 150) & (time_ms <= 300)
    pre_m    = time_ms < 0
    records  = []
    for ep in odd_epochs:
        d = ep['data']
        pre_sig = d[pre_m, ci_tp]
        if len(pre_sig) > 4:
            f, psd = scipy_signal.welch(pre_sig, fs=SAMPLE_RATE,
                                         nperseg=min(len(pre_sig), 64))
            al_m = (f >= BANDS['Alpha'][0]) & (f <= BANDS['Alpha'][1])
            th_m = (f >= BANDS['Theta'][0]) & (f <= BANDS['Theta'][1])
            alpha_pre = float(integrate.trapezoid(psd[al_m], f[al_m])) if al_m.sum() > 1 else np.nan
            theta_pre = float(integrate.trapezoid(psd[th_m], f[th_m])) if th_m.sum() > 1 else np.nan
        else:
            alpha_pre = theta_pre = np.nan
        records.append({'P300_amp':  float(d[p300_m, ci_tp].mean()),
                        'N200_amp':  float(d[n200_m, ci_af].mean()),
                        'Alpha_pre': alpha_pre,
                        'Theta_pre': theta_pre})
    return pd.DataFrame(records).dropna()


def plot_trial_correlations(corr_df_input, out_prefix=''):
    if corr_df_input.empty:
        print("[analysis] WARNING: Skipping trial correlations (no data).")
        return
    cols  = corr_df_input.columns.tolist()
    n     = len(cols)
    corr  = corr_df_input.corr()
    pvals = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i != j and len(corr_df_input) >= 3:
                _, p = stats.pearsonr(corr_df_input[c1], corr_df_input[c2])
                pvals.loc[c1, c2] = p
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    for i in range(n):
        for j in range(n):
            txt = f'{corr.values[i,j]:.2f}'
            if i != j and pvals.values[i, j] < 0.05:
                txt += '*'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_title('Single-Trial Correlation Matrix')
    plt.tight_layout()
    fname = f"{out_prefix}plot_trial_correlations.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def plot_complexity_vs_p300(odd_epochs, complexity_df, time_ms, task_intervals, out_prefix=''):
    if not odd_epochs or complexity_df.empty or time_ms is None:
        print("[analysis] WARNING: Skipping complexity vs P300 (insufficient data).")
        return
    ci     = _ch_idx('TP_pool')
    p300_m = (time_ms >= 250) & (time_ms <= 600)
    p300_by_cond = {}
    for ep in odd_epochs:
        cond = get_condition_at_time(ep['trigger_time'], task_intervals)
        p300_by_cond.setdefault(cond, []).append(float(ep['data'][p300_m, ci].mean()))
    pe_by_cond = {}
    for cond in p300_by_cond:
        sub = complexity_df[complexity_df['condition'] == cond]['permen']
        pe_by_cond[cond] = float(sub.mean()) if not sub.empty else np.nan
    conditions = [c for c in p300_by_cond if not np.isnan(pe_by_cond.get(c, np.nan))]
    if len(conditions) < 2:
        print("[analysis] WARNING: Not enough conditions for complexity vs P300 scatter.")
        return
    x = [pe_by_cond[c] for c in conditions]
    y = [np.mean(p300_by_cond[c]) for c in conditions]
    r, p = stats.pearsonr(x, y)
    m, b = np.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color='steelblue', s=60, zorder=3)
    for c, xi, yi in zip(conditions, x, y):
        ax.annotate(c, (xi, yi), textcoords='offset points', xytext=(5, 3), fontsize=8)
    xr = np.linspace(min(x), max(x), 100)
    ax.plot(xr, m * xr + b, color='gray', linestyle='--', lw=1)
    ax.set_xlabel('Mean PermEn')
    ax.set_ylabel('Mean P300 Amplitude (µV, TP_pool)')
    ax.set_title(f'Complexity vs P300   r={r:.3f}, p={p:.4f}')
    plt.tight_layout()
    fname = f"{out_prefix}plot_complexity_vs_p300.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def run_cross_analysis(odd_epochs, time_ms, task_intervals,
                       condition_segments, complexity_df, out_prefix=''):
    print("\n[analysis] --- Cross-Analysis ---")
    plot_theta_alpha_trajectory(condition_segments, out_prefix)
    corr_df = compute_trial_correlations(odd_epochs, time_ms)
    plot_trial_correlations(corr_df, out_prefix)
    plot_complexity_vs_p300(odd_epochs, complexity_df, time_ms, task_intervals, out_prefix)


# ── A10: Statistics ───────────────────────────────────────────────────────────

def _cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled != 0 else np.nan


def run_statistics(odd_epochs, std_epochs, band_df, time_ms,
                   complexity_df, conn_data, out_prefix=''):
    print("\n[analysis] --- Statistical Tests ---")
    rows = []

    # A10a: ERP t-tests with Cohen's d
    if odd_epochs and std_epochs and time_ms is not None:
        for ch, t_lo, t_hi, label in [('TP_pool', 250, 600, 'P300'),
                                       ('AF_pool', 150, 300, 'N200')]:
            ci   = _ch_idx(ch)
            mask = (time_ms >= t_lo) & (time_ms <= t_hi)
            odd_a = [float(ep['data'][mask, ci].mean()) for ep in odd_epochs]
            std_a = [float(ep['data'][mask, ci].mean()) for ep in std_epochs]
            if len(odd_a) >= 2 and len(std_a) >= 2:
                t, p = stats.ttest_ind(odd_a, std_a)
                d    = _cohens_d(odd_a, std_a)
                print(f"[stats] {label} t-test {ch}: t={t:.3f}, p={p:.4f}, d={d:.3f}")
                rows.append({'test': f'{label}_ttest', 'channel_or_pair': ch,
                             'band_or_component': label, 'statistic': t,
                             'p_value': p, 'p_corrected': p,
                             'effect_size': d, 'n_epochs': len(odd_a)})

    # A10b: Band power ANOVA with Bonferroni
    if not band_df.empty:
        conditions = band_df['condition'].unique().tolist()
        all_ch     = band_df['channel'].unique().tolist()
        n_tests    = len(BANDS) * len(all_ch)
        for ch in all_ch:
            ch_df = band_df[band_df['channel'] == ch]
            for band in BANDS:
                col    = f'{band}_rel'
                groups = [ch_df[ch_df['condition'] == c][col].dropna().tolist()
                          for c in conditions]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) < 2:
                    continue
                if len(groups) >= 3:
                    stat, p = stats.f_oneway(*groups)
                    test_nm = 'ANOVA'
                else:
                    stat, p = stats.ttest_ind(groups[0], groups[1])
                    test_nm = 'ttest'
                p_corr = min(p * n_tests, 1.0)
                rows.append({'test': test_nm, 'channel_or_pair': ch,
                             'band_or_component': band, 'statistic': stat,
                             'p_value': p, 'p_corrected': p_corr,
                             'effect_size': np.nan,
                             'n_epochs': sum(len(g) for g in groups)})

    # A10c: Complexity Kruskal-Wallis
    if not complexity_df.empty:
        conditions = complexity_df['condition'].unique().tolist()
        for metric in ['permen', 'hfd', 'dfa_alpha']:
            for ch in complexity_df['channel'].unique():
                groups = [complexity_df[(complexity_df['condition'] == c) &
                                        (complexity_df['channel'] == ch)][metric].dropna().tolist()
                          for c in conditions]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) < 2:
                    continue
                try:
                    stat, p = stats.kruskal(*groups)
                    rows.append({'test': 'kruskal', 'channel_or_pair': ch,
                                 'band_or_component': metric, 'statistic': stat,
                                 'p_value': p, 'p_corrected': p,
                                 'effect_size': np.nan,
                                 'n_epochs': sum(len(g) for g in groups)})
                    if p < 0.05 and len(groups) == 2:
                        u, pu = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                        print(f"[stats] {metric} {ch} KW p={p:.4f} → MWU p={pu:.4f}")
                except Exception:
                    pass

    # A10d: Connectivity Kruskal-Wallis
    if conn_data:
        conditions = list(conn_data.keys())
        for ch1, ch2 in CHANNEL_PAIRS:
            pair = f"{ch1}-{ch2}"
            for band in ['Theta', 'Alpha', 'Beta']:
                groups = [[conn_data[c].get(pair, {}).get(band, {}).get('plv', np.nan)]
                          for c in conditions]
                groups = [[v for v in g if not np.isnan(v)] for g in groups]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) < 2:
                    continue
                try:
                    stat, p = stats.kruskal(*groups)
                    rows.append({'test': 'kruskal_plv', 'channel_or_pair': pair,
                                 'band_or_component': band, 'statistic': stat,
                                 'p_value': p, 'p_corrected': p,
                                 'effect_size': np.nan,
                                 'n_epochs': sum(len(g) for g in groups)})
                except Exception:
                    pass

    if rows:
        stats_df = pd.DataFrame(rows)
        csv_path = f"{out_prefix}stats_summary.csv"
        stats_df.to_csv(csv_path, index=False)
        print(f"[analysis] Saved {csv_path}")
    else:
        print("[analysis] No statistical tests ran (insufficient data).")


# ── A11: Output Summary ───────────────────────────────────────────────────────

def save_session_summary(odd_epochs, std_epochs, time_ms, band_df,
                         complexity_df, conn_data, task_intervals,
                         baseline_features, out_prefix=''):
    summary = {
        'p300_amplitude_by_condition':           {},
        'p300_latency_by_condition':             {},
        'n200_amplitude_by_condition':           {},
        'theta_alpha_ratio_by_condition':        {},
        'dfa_alpha_by_condition':                {},
        'permen_by_condition':                   {},
        'frontotemporal_plv_theta_by_condition': {},
        'n_oddball_epochs_accepted':             len(odd_epochs) if odd_epochs else 0,
        'n_standard_epochs_accepted':            len(std_epochs) if std_epochs else 0,
        'iaf_hz':                                baseline_features.get('IAF_hz') if baseline_features else None,
    }

    if odd_epochs and time_ms is not None:
        ci_tp  = _ch_idx('TP_pool')
        ci_af  = _ch_idx('AF_pool')
        p300_m = (time_ms >= 250) & (time_ms <= 600)
        n200_m = (time_ms >= 150) & (time_ms <= 300)
        by_cond = {}
        for ep in odd_epochs:
            cond = get_condition_at_time(ep['trigger_time'], task_intervals)
            by_cond.setdefault(cond, {'p300': [], 'n200': []})
            by_cond[cond]['p300'].append(float(ep['data'][p300_m, ci_tp].mean()))
            by_cond[cond]['n200'].append(float(ep['data'][n200_m, ci_af].mean()))
        for cond, vals in by_cond.items():
            summary['p300_amplitude_by_condition'][cond] = float(np.mean(vals['p300']))
            summary['n200_amplitude_by_condition'][cond] = float(np.mean(vals['n200']))
            # Latency: index of max in p300 window of grand average
            grand_p300 = np.mean([ep['data'][p300_m, ci_tp]
                                   for ep in odd_epochs
                                   if get_condition_at_time(ep['trigger_time'], task_intervals) == cond],
                                  axis=0)
            lat_ms = float(time_ms[p300_m][np.argmax(grand_p300)]) if len(grand_p300) > 0 else np.nan
            summary['p300_latency_by_condition'][cond] = lat_ms

    if not band_df.empty:
        for _, row in band_df.iterrows():
            cond = row['condition']
            ta   = row.get('theta_alpha_ratio', np.nan)
            if not np.isnan(float(ta if ta is not None else np.nan)):
                summary['theta_alpha_ratio_by_condition'][cond] = float(ta)

    if not complexity_df.empty:
        for cond in complexity_df['condition'].unique():
            sub_pe  = complexity_df[complexity_df['condition'] == cond]['permen']
            sub_dfa = complexity_df[complexity_df['condition'] == cond]['dfa_alpha']
            summary['permen_by_condition'][cond]    = float(sub_pe.mean())  if not sub_pe.empty  else None
            summary['dfa_alpha_by_condition'][cond] = float(sub_dfa.mean()) if not sub_dfa.empty else None

    if conn_data:
        for cond in conn_data:
            v = conn_data[cond].get('AF7-TP9', {}).get('Theta', {}).get('plv', np.nan)
            summary['frontotemporal_plv_theta_by_condition'][cond] = \
                float(v) if not np.isnan(v) else None

    def _sanitise(obj):
        if isinstance(obj, dict):
            return {k: _sanitise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitise(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if np.isnan(v) else v
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    json_path = f"{out_prefix}session_summary.json"
    with open(json_path, 'w') as f:
        json.dump(_sanitise(summary), f, indent=2)
    print(f"[analysis] Saved {json_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def analyse_session(erp_path, psd_path, events_path,
                    baseline_features_path=None, out_prefix=''):
    """Full analysis pipeline for a single pre-denoised session."""

    # Stage 0: Load
    print("[analysis] Loading data...")
    erp_df, psd_df, events_df = load_data(erp_path, psd_path, events_path)

    baseline_features = None
    if baseline_features_path and os.path.exists(baseline_features_path):
        bf_dir = os.path.dirname(baseline_features_path)
        bf_sid = os.path.basename(baseline_features_path).replace('_baseline_features.json', '')
        baseline_features = load_baseline_features(bf_dir, bf_sid)

    erp_df = add_pooled_channels(erp_df)
    psd_df = add_pooled_channels(psd_df)

    # Stage 1: Parse events
    oddball_triggers, standard_triggers, task_intervals, task_starts = parse_events(events_df)

    # Stage 2: Epoching
    print("\n[analysis] --- Epoching ---")
    all_triggers = oddball_triggers + standard_triggers
    odd_raw, *odd_rej = extract_epochs(erp_df, oddball_triggers, all_triggers=all_triggers)
    std_raw, *std_rej = extract_epochs(erp_df, standard_triggers, all_triggers=all_triggers)
    odd_epochs = baseline_correct(odd_raw)
    std_epochs = baseline_correct(std_raw)

    if len(odd_epochs) < MIN_EPOCHS:
        print(f"[analysis] WARNING: Only {len(odd_epochs)} clean oddball epochs "
              f"(min {MIN_EPOCHS}). ERP SNR may be poor.")

    task_onset_epochs  = extract_task_onset_epochs(erp_df, task_starts, task_intervals)
    condition_segments = extract_condition_segments(psd_df, task_intervals)
    condition_segments = {k: add_pooled_channels(v) for k, v in condition_segments.items()}

    rejection_counts = dict(
        odd_bad_seg=odd_rej[0], odd_overlap=odd_rej[1],
        odd_amplitude=odd_rej[2], odd_kurtosis=odd_rej[3],
        std_bad_seg=std_rej[0], std_overlap=std_rej[1],
        std_amplitude=std_rej[2], std_kurtosis=std_rej[3],
    )
    time_ms = odd_epochs[0]['time_ms'] if odd_epochs else None

    # Stage 3: ERP
    print("\n[analysis] --- ERP Analysis ---")
    plot_erp(odd_epochs, std_epochs, out_prefix)
    plot_epoch_heatmap(odd_epochs, out_prefix)
    plot_rejection_summary(rejection_counts, out_prefix)
    plot_task_onset_erp(task_onset_epochs, out_prefix)
    plot_p300_by_condition(odd_epochs, task_intervals, out_prefix)

    # Stage 4: ERSP + ITPC
    run_ersp_itpc_analysis(odd_epochs, std_epochs, out_prefix)

    # Stage 5: ERD/ERS
    run_erd_ers_analysis(task_onset_epochs, baseline_features, out_prefix)

    # Stage 6: PSD
    psds, band_df, faa_df = run_psd_analysis(condition_segments, baseline_features, out_prefix)
    plot_spectrogram(psd_df, task_intervals, out_prefix)

    # Stage 7: Connectivity
    conn_data = run_connectivity_analysis(condition_segments, out_prefix)

    # Stage 8: Complexity
    complexity_df = run_complexity_analysis(condition_segments, baseline_features, out_prefix)

    # Stage 9: MSE
    run_mse_analysis(condition_segments, out_prefix)

    # Stage 10: Cross-analysis
    run_cross_analysis(odd_epochs, time_ms, task_intervals,
                       condition_segments, complexity_df, out_prefix)

    # Stage 11: Statistics
    run_statistics(odd_epochs, std_epochs, band_df, time_ms,
                   complexity_df, conn_data, out_prefix)

    # Stage 12: Summary
    save_session_summary(odd_epochs, std_epochs, time_ms, band_df,
                         complexity_df, conn_data, task_intervals,
                         baseline_features, out_prefix)

    print("\n[analysis] Session complete.")


def main(participant_id, data_dir='data', out_dir='output'):
    participant_out_dir  = os.path.join(out_dir,  participant_id)
    participant_data_dir = os.path.join(data_dir, participant_id)

    # Look for denoised files in output dir first, then data dir
    erp_files = sorted(_glob.glob(os.path.join(participant_out_dir, 'session_*_erp_clean.csv')))
    if not erp_files:
        erp_files = sorted(_glob.glob(os.path.join(participant_data_dir, 'session_*_erp_clean.csv')))

    if not erp_files:
        print(f"[analysis] No denoised files found for participant '{participant_id}'.")
        print(f"[analysis] Run denoising first: python denoising.py {participant_id}")
        sys.exit(1)

    os.makedirs(participant_out_dir, exist_ok=True)
    print(f"[analysis] Found {len(erp_files)} session(s) for '{participant_id}'")

    for erp_path in erp_files:
        erp_dir  = os.path.dirname(erp_path)
        basename = os.path.basename(erp_path).replace('_erp_clean.csv', '')
        psd_path = os.path.join(erp_dir, f'{basename}_psd_clean.csv')

        events_path = os.path.join(participant_data_dir, f'{basename}_markers.csv')
        out_prefix  = os.path.join(participant_out_dir, f'{basename}_')

        # Baseline JSON: session_YYYYMMDD_HHMMSS prefix
        parts       = basename.split('_')
        session_stem = '_'.join(parts[:3]) if len(parts) >= 3 else basename
        bl_path = os.path.join(participant_data_dir,
                               f'{session_stem}_baseline_features.json')
        if not os.path.exists(bl_path):
            bl_path = os.path.join(participant_out_dir,
                                   f'{session_stem}_baseline_features.json')
        if not os.path.exists(bl_path):
            bl_path = None

        if not os.path.exists(psd_path):
            print(f"[analysis] WARNING: psd_clean not found for {basename}, skipping.")
            continue
        if not os.path.exists(events_path):
            print(f"[analysis] WARNING: markers not found for {basename}, skipping.")
            continue

        print(f"\n[analysis] === Session: {basename} ===")
        analyse_session(erp_path, psd_path, events_path, bl_path, out_prefix)

    print(f"\n[analysis] All sessions complete. Outputs in {participant_out_dir}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyse EEG sessions for a participant.')
    parser.add_argument('participant_id', help='Participant ID (e.g. 001, test)')
    parser.add_argument('--data-dir', default='data',
                        help='Root data directory (default: data)')
    parser.add_argument('--out-dir',  default='output',
                        help='Root output directory (default: output)')
    args = parser.parse_args()
    main(args.participant_id, args.data_dir, args.out_dir)
