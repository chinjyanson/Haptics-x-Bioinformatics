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
import warnings
import glob as _glob

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

WINDOW_SEC    = 2
OVERLAP_RATIO = 0.5

TASK_EPOCH_PRE_MS  = 500
TASK_EPOCH_POST_MS = 2000

MORLET_FREQS  = np.linspace(1, 45, 100)
MORLET_CYCLES = 6

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

DEVICE_SLUGS = ['auditory', 'vibrations', 'shape_changing']
DEVICE_PRETTY = {
    'auditory':       'Auditory',
    'vibrations':     'Vibrations',
    'shape_changing': 'Shape Changing',
}
DEVICE_COLORS = {
    'auditory':       'steelblue',
    'vibrations':     'coral',
    'shape_changing': 'mediumseagreen',
}


def _infer_device_slug(basename):
    """Infer device slug from a session basename (handles shape_changing underscore)."""
    for slug in DEVICE_SLUGS:
        if basename.endswith(f'_{slug}'):
            return slug
    return None


def _devices_present(session_results):
    """Return list of (slug, pretty_label, color, result) in canonical device order."""
    return [(s, DEVICE_PRETTY[s], DEVICE_COLORS[s], session_results[s])
            for s in DEVICE_SLUGS if s in session_results]


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





def plot_task_onset_erp(task_epochs, out_prefix=''):
    if not task_epochs:
        print("[analysis] WARNING: Skipping task-onset ERP (no epochs).")
        return
    time_ms    = task_epochs[0]['time_ms']
    grand, sem = average_epochs(task_epochs)

    channels = [
        ('TP_pool', 'steelblue',  250, 600, 'pos', 'P300'),
        ('AF_pool', 'darkorange', 150, 350, 'neg', 'N200'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, (ch, color, t0, t1, pol, label) in zip(axes, channels):
        ci  = _ch_idx(ch)
        erp = grand[:, ci]
        t_pk, a_pk = find_peak(erp, time_ms, t0, t1, polarity=pol)
        ax.axvline(0, color='k', lw=0.8, linestyle='--', label='Task onset')
        ax.plot(time_ms, erp, color=color, lw=1.5, label=ch)
        ax.fill_between(time_ms, erp - sem[:, ci], erp + sem[:, ci],
                        alpha=0.2, color=color)
        ax.annotate(f'{label}\n{t_pk:.0f}ms\n{a_pk:.1f}µV', xy=(t_pk, a_pk),
                    xytext=(t_pk + 100, a_pk + (3 if pol == 'neg' else -3)),
                    arrowprops=dict(arrowstyle='->', color='tomato'), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'Task-Onset ERP — {ch}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fname = f"{out_prefix}plot_task_onset_erp.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")




# ── A4: ERD/ERS ──────────────────────────────────────────────────────────────

def morlet_wavelet_transform(sig, fs, freqs, n_cycles=MORLET_CYCLES):  # used by _compare_tfr_diff
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
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude (µV)')
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


def run_ersp_itpc_analysis(task_onset_epochs, out_prefix=''):
    """Compute and plot ERSP/ITPC pooled across all task-onset epochs."""
    if not task_onset_epochs:
        print("[analysis] WARNING: Skipping ERSP/ITPC (no task-onset epochs).")
        return {}
    print("\n[analysis] --- ERSP / ITPC ---")
    time_ms = task_onset_epochs[0]['time_ms']
    ersp_itpc_cache = {}

    for ch in ['TP_pool', 'AF_pool']:
        ci = _ch_idx(ch)
        data = np.stack([ep['data'][:, ci] for ep in task_onset_epochs], axis=0)
        ersp, fq, tm = compute_ersp(data, SAMPLE_RATE, MORLET_FREQS, time_ms)
        itpc, fq, tm = compute_itpc(data, SAMPLE_RATE, MORLET_FREQS, time_ms)
        plot_ersp_itpc(ersp, itpc, fq, tm, data, ch, 'all_tasks', out_prefix)
        ersp_itpc_cache[(ch, 'all_tasks')] = {'ersp': ersp, 'itpc': itpc,
                                               'freqs': fq, 'time_ms': tm}
    return ersp_itpc_cache


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
    fig.suptitle('Event-Related Desynchronisation / Synchronisation (TP_pool)', fontweight='bold')
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
        ax.set_title(f'{band} Band')
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
    return erd_data, time_axis


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

    csv_path = f"{out_prefix}band_power_summary.csv"
    band_df.to_csv(csv_path, index=False)
    print(f"[analysis] Saved {csv_path}")

    _plot_psd(psds, out_prefix)
    _plot_band_power(band_df, out_prefix)
    _plot_theta_alpha(band_df, out_prefix)
    return psds, band_df


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
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Relative Power')
        ax.set_title(f'Relative Band Power — {ch}')
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
    ax.set_xlabel('Task Condition')
    ax.set_ylabel('Theta / Alpha Ratio')
    ax.set_title('Theta/Alpha Ratio by Condition\n(Higher = Greater Cognitive Load)', fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = f"{out_prefix}plot6_theta_alpha_ratio.png"
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



# ── A7: Complexity ────────────────────────────────────────────────────────────



# ── A8: Multiscale Entropy ────────────────────────────────────────────────────

def sample_entropy(sig, m, r):
    """Vectorised sample entropy — O(n·m) instead of O(n²)."""
    sig = np.asarray(sig, dtype=float)
    n   = len(sig)

    def _count(length):
        # Build template matrix via strided view: shape (n-length, length)
        shape   = (n - length, length)
        strides = (sig.strides[0], sig.strides[0])
        templates = np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides)
        count = 0
        for i in range(len(templates) - 1):
            diffs = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += int(np.sum(diffs < r))
        return count

    B = _count(m)
    A = _count(m + 1)
    if B == 0:
        return 0.0
    return float(-np.log(A / B))


def multiscale_entropy(sig, m=2, r_factor=0.2, max_scale=20, max_samples=2000):
    sig = np.asarray(sig, dtype=float)
    # Cap length to keep sample_entropy tractable (O(n^2) template matching)
    if len(sig) > max_samples:
        sig = sig[:max_samples]
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
        return {}
    max_scale = 20
    scales    = np.arange(1, max_scale + 1)
    mse_cache = {}
    for ch in ['TP_pool', 'AF_pool']:
        mse_by_cond = {}
        for cond, seg in condition_segments.items():
            if 'bad_segment' in seg.columns:
                seg = seg[seg['bad_segment'] == False]
            if ch not in seg.columns:
                continue
            sig = seg[ch].values.astype(float)
            mse_by_cond[cond] = multiscale_entropy(sig, max_scale=max_scale)
        mse_cache[ch] = mse_by_cond
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
    return mse_cache


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


def run_cross_analysis(condition_segments, out_prefix=''):
    print("\n[analysis] --- Cross-Analysis ---")
    plot_theta_alpha_trajectory(condition_segments, out_prefix)


# ── A10: Statistics ───────────────────────────────────────────────────────────

def _cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled != 0 else np.nan


def run_statistics(band_df, out_prefix=''):
    print("\n[analysis] --- Statistical Tests ---")
    rows = []

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



    if rows:
        stats_df = pd.DataFrame(rows)
        csv_path = f"{out_prefix}stats_summary.csv"
        stats_df.to_csv(csv_path, index=False)
        print(f"[analysis] Saved {csv_path}")
    else:
        print("[analysis] No statistical tests ran (insufficient data).")


# ── A11: Output Summary ───────────────────────────────────────────────────────

def save_session_summary(band_df, task_onset_epochs,
                         baseline_features, out_prefix=''):
    summary = {
        'theta_alpha_ratio_by_condition':    {},
        'task_onset_erp_peak_by_condition':  {},
        'n_task_onset_epochs_accepted':      len(task_onset_epochs) if task_onset_epochs else 0,
        'iaf_hz':                            baseline_features.get('IAF_hz') if baseline_features else None,
    }

    if not band_df.empty:
        for _, row in band_df.iterrows():
            cond = row['condition']
            ta   = row.get('theta_alpha_ratio', np.nan)
            if not np.isnan(float(ta if ta is not None else np.nan)):
                summary['theta_alpha_ratio_by_condition'][cond] = float(ta)

    if task_onset_epochs:
        ci_tp   = _ch_idx('TP_pool')
        time_ms = task_onset_epochs[0]['time_ms']
        p300_m  = (time_ms >= 250) & (time_ms <= 600)
        by_cond = {}
        for ep in task_onset_epochs:
            cond = ep.get('condition', 'unknown')
            by_cond.setdefault(cond, []).append(float(ep['data'][p300_m, ci_tp].mean()))
        for cond, vals in by_cond.items():
            summary['task_onset_erp_peak_by_condition'][cond] = float(np.mean(vals))


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


# ── GSR analysis ─────────────────────────────────────────────────────────────

# Tonic / phasic decomposition constants
GSR_TONIC_LP_HZ   = 0.05   # Lowpass cutoff for SCL (tonic) extraction
GSR_SCR_MIN_AMP   = 0.01   # µS — minimum peak amplitude to count as an SCR
GSR_SCR_MIN_DIST  = 1.0    # seconds — minimum distance between SCR peaks


def compute_gsr_metrics(gsr_df, fs=None):
    """
    Decompose GSR into tonic (SCL) and phasic (SCR) components and compute:
      - Mean SCL (µS)       : mean of the lowpass-filtered tonic signal
      - NS-SCR Rate (/min)  : non-specific SCR count per minute
      - SCR Amplitude (µS)  : mean peak amplitude of detected SCRs

    Decomposition:
      Tonic  = zero-phase Butterworth lowpass at GSR_TONIC_LP_HZ (0.05 Hz)
      Phasic = raw GSR − tonic

    SCR detection on phasic component:
      Peaks with amplitude >= GSR_SCR_MIN_AMP µS and separated by
      >= GSR_SCR_MIN_DIST seconds are counted as NS-SCRs.

    Returns a dict with keys: mean_scl, scr_rate_per_min, mean_scr_amplitude,
    and arrays: tonic, phasic, scr_indices (sample indices of detected peaks).
    Returns None if signal is too short.
    """
    sig   = gsr_df['gsr_uS'].values.astype(float)
    times = gsr_df['time'].values.astype(float)
    n     = len(sig)

    if n < 10:
        return None

    # Estimate sampling rate from time axis if not provided
    if fs is None:
        dt = np.median(np.diff(times))
        fs = 1.0 / dt if dt > 0 else 50.0

    duration_min = (times[-1] - times[0]) / 60.0

    # ── Tonic extraction (SCL) ────────────────────────────────────────────────
    nyq    = fs / 2.0
    cutoff = min(GSR_TONIC_LP_HZ, nyq * 0.9)
    order  = 2
    padlen = 3 * order * max(1, int(np.ceil(fs / cutoff)))
    if n > padlen:
        b, a   = scipy_signal.butter(order, cutoff / nyq, btype='low')
        tonic  = scipy_signal.filtfilt(b, a, sig)
    else:
        tonic = np.full(n, np.mean(sig))

    mean_scl = float(np.mean(tonic))

    # ── Phasic extraction (SCR) ───────────────────────────────────────────────
    phasic = sig - tonic

    # Peak detection on phasic signal
    min_dist_samples = max(1, int(GSR_SCR_MIN_DIST * fs))
    peaks, props = scipy_signal.find_peaks(
        phasic,
        height=GSR_SCR_MIN_AMP,
        distance=min_dist_samples,
    )

    n_scr          = len(peaks)
    scr_rate       = n_scr / duration_min if duration_min > 0 else np.nan
    mean_scr_amp   = float(np.mean(props['peak_heights'])) if n_scr > 0 else 0.0

    return {
        'mean_scl':           mean_scl,
        'scr_rate_per_min':   scr_rate,
        'mean_scr_amplitude': mean_scr_amp,
        'n_scr':              n_scr,
        'tonic':              tonic,
        'phasic':             phasic,
        'scr_indices':        peaks,
    }


def plot_gsr_decomposition(gsr_df, metrics, out_prefix=''):
    """
    4-panel plot showing:
      1. Raw GSR + tonic (SCL) overlay
      2. Phasic component with detected SCR peaks marked
      3. SCL trajectory
      4. Summary text box with all three metrics
    """
    times  = gsr_df['time'].values.astype(float)
    sig    = gsr_df['gsr_uS'].values.astype(float)
    tonic  = metrics['tonic']
    phasic = metrics['phasic']
    peaks  = metrics['scr_indices']

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Raw + tonic
    axes[0].plot(times, sig,   color='steelblue', lw=0.8, alpha=0.7, label='Raw GSR')
    axes[0].plot(times, tonic, color='red',       lw=1.5,             label='Tonic (SCL)')
    axes[0].set_ylabel('GSR (µS)')
    axes[0].set_title('GSR Signal with Tonic (SCL) Component')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Phasic + SCR peaks
    axes[1].plot(times, phasic, color='darkorange', lw=0.8, label='Phasic (SCR)')
    if len(peaks) > 0:
        axes[1].scatter(times[peaks], phasic[peaks],
                        color='red', zorder=5, s=25, label=f'SCRs (n={len(peaks)})')
    axes[1].axhline(GSR_SCR_MIN_AMP, color='gray', lw=0.8, linestyle='--',
                    label=f'Min amp ({GSR_SCR_MIN_AMP} µS)')
    axes[1].set_ylabel('Phasic GSR (µS)')
    axes[1].set_title('Phasic Component with Detected SCRs')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: SCL trajectory
    axes[2].plot(times, tonic, color='red', lw=1.5)
    axes[2].fill_between(times, tonic.min(), tonic, alpha=0.2, color='red')
    axes[2].set_ylabel('SCL (µS)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Skin Conductance Level (Tonic) Trajectory')
    axes[2].grid(True, alpha=0.3)

    scl  = metrics['mean_scl']
    rate = metrics['scr_rate_per_min']
    amp  = metrics['mean_scr_amplitude']
    fig.text(0.99, 0.01,
             f"Mean SCL: {scl:.4f} µS   |   NS-SCR Rate: {rate:.2f} /min   |   "
             f"Mean SCR Amplitude: {amp:.4f} µS",
             ha='right', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.07)
    fname = f"{out_prefix}gsr_decomposition.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def run_gsr_analysis(gsr_path, events_path, out_prefix=''):
    """
    Load GSR CSV and markers CSV, plot the GSR signal with task_end markers
    overlaid, plus the rolling-statistics analysis plot.
    Saves two PNGs alongside other session outputs.
    """
    print("\n[analysis] --- GSR Analysis ---")
    if not os.path.exists(gsr_path):
        print(f"[analysis] WARNING: GSR file not found: {gsr_path}, skipping.")
        return

    import pandas as pd

    gsr_df = pd.read_csv(gsr_path)
    if 'timestamp' in gsr_df.columns:
        gsr_df['time'] = gsr_df['timestamp'] - gsr_df['timestamp'].iloc[0]

    # Load task_end markers
    task_ends = []
    if events_path and os.path.exists(events_path):
        ev = pd.read_csv(events_path)
        task_ends = ev[ev['event'] == 'task_end'][['time', 'task_number']].values.tolist()

    is_calibrated = gsr_df['gsr_uS'].abs().max() > 1e-3
    unit      = "µS" if is_calibrated else "a.u."
    gsr_label = f"GSR ({unit})"

    # ── Plot 1: GSR signal with task markers ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(gsr_df['time'], gsr_df['gsr_uS'], 'r-', linewidth=0.8, label='GSR')

    colors = plt.cm.tab10.colors
    for t, task_num in task_ends:
        color = colors[int(task_num) % len(colors)]
        ax.axvline(x=t, color=color, linestyle='--', linewidth=1.2,
                   label=f'Task {int(task_num)} end')

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc='upper right')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel(gsr_label)
    ax.set_title('GSR Signal with Task Markers')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"{out_prefix}gsr_signal.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved {fname}")

    # ── Plot 2: rolling-statistics analysis ───────────────────────────────────
    plot_gsr_analysis(gsr_path, save_path=f"{out_prefix}gsr_analysis.png")

    # ── GSR metrics: SCL, NS-SCR Rate, SCR Amplitude ─────────────────────────
    metrics = compute_gsr_metrics(gsr_df)
    if metrics:
        plot_gsr_decomposition(gsr_df, metrics, out_prefix)

        metrics_csv = f"{out_prefix}gsr_metrics.csv"
        pd.DataFrame([{
            'mean_scl_uS':          metrics['mean_scl'],
            'scr_rate_per_min':     metrics['scr_rate_per_min'],
            'mean_scr_amplitude_uS': metrics['mean_scr_amplitude'],
            'n_scr':                metrics['n_scr'],
        }]).to_csv(metrics_csv, index=False)
        print(f"[analysis] GSR metrics — Mean SCL: {metrics['mean_scl']:.4f} µS | "
              f"NS-SCR Rate: {metrics['scr_rate_per_min']:.2f}/min | "
              f"Mean SCR Amp: {metrics['mean_scr_amplitude']:.4f} µS")
        print(f"[analysis] Saved {metrics_csv}")
    else:
        print("[analysis] WARNING: GSR signal too short for metrics computation.")


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
    _, _, task_intervals, task_starts = parse_events(events_df)

    # Stage 2: Epoching
    print("\n[analysis] --- Epoching ---")
    task_onset_epochs  = extract_task_onset_epochs(erp_df, task_starts, task_intervals)
    condition_segments = extract_condition_segments(psd_df, task_intervals)
    condition_segments = {k: add_pooled_channels(v) for k, v in condition_segments.items()}

    if not task_onset_epochs:
        print("[analysis] WARNING: No task-onset epochs found.")

    # Stage 3: ERP
    print("\n[analysis] --- ERP Analysis ---")
    plot_task_onset_erp(task_onset_epochs, out_prefix)

    # Stage 4: ERSP + ITPC
    ersp_itpc_cache = run_ersp_itpc_analysis(task_onset_epochs, out_prefix)

    # Stage 5: ERD/ERS
    erd_data, erd_time_axis = run_erd_ers_analysis(task_onset_epochs, baseline_features, out_prefix)

    # Stage 6: PSD
    psds, band_df = run_psd_analysis(condition_segments, baseline_features, out_prefix)
    plot_spectrogram(psd_df, task_intervals, out_prefix)

    # Stage 9: MSE
    mse_cache = run_mse_analysis(condition_segments, out_prefix)

    # Stage 10: Cross-analysis
    run_cross_analysis(condition_segments, out_prefix)

    # Stage 11: Statistics
    run_statistics(band_df, out_prefix)

    # Stage 12: Summary
    save_session_summary(band_df, task_onset_epochs,
                         baseline_features, out_prefix)

    # Stage 13: GSR analysis
    # GSR CSV lives alongside the markers CSV: same dir, _gsr.csv suffix
    gsr_path = events_path.replace('_markers.csv', '_gsr.csv')
    run_gsr_analysis(gsr_path, events_path, out_prefix)

    print("\n[analysis] Session complete.")
    return {
        'task_onset_epochs': task_onset_epochs,
        'band_df':           band_df,
        'ersp_itpc':         ersp_itpc_cache if ersp_itpc_cache else {},
        'erd_data':          erd_data,
        'erd_time_axis':     erd_time_axis,
        'mse_data':          mse_cache if mse_cache else {},
    }


# ── GSR plotting ──────────────────────────────────────────────────────────────

def plot_gsr_data(filepath: str, save_path: str | None = None):
    """
    Plot GSR data from a CSV file.

    Creates a 3-panel figure showing:
    1. Raw audio signal
    2. Filtered signal
    3. GSR conductance (µS)

    Args:
        filepath: Path to GSR CSV file
        save_path: Optional path to save the figure
    """
    import pandas as pd

    df = pd.read_csv(filepath)

    if 'timestamp' in df.columns:
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    elif 'time' not in df.columns:
        raise ValueError("CSV must have 'timestamp' or 'time' column")

    is_calibrated = df['gsr_uS'].abs().max() > 1e-3
    unit = "µS" if is_calibrated else "a.u."
    gsr_label = f"GSR ({unit})"
    calibration_note = "" if is_calibrated else "\n⚠ Uncalibrated — values are raw audio amplitude, not µS"

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('eSense GSR Recording Analysis', fontsize=14, fontweight='bold')

    axes[0].plot(df['time'], df['raw_audio'], 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Raw Audio\nAmplitude')
    axes[0].set_title('Raw Audio Signal (Microphone Input)')
    axes[0].grid(True, alpha=0.3)
    axes[0].margins(y=0.1)

    axes[1].plot(df['time'], df['filtered_signal'], 'g-', linewidth=0.8)
    axes[1].set_ylabel('Filtered Signal\nAmplitude')
    axes[1].set_title('Lowpass Filtered Signal (5 Hz cutoff)')
    axes[1].grid(True, alpha=0.3)
    axes[1].margins(y=0.1)

    axes[2].plot(df['time'], df['gsr_uS'], 'r-', linewidth=1)
    axes[2].set_ylabel(gsr_label)
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title('Galvanic Skin Response (Conductance)')
    axes[2].grid(True, alpha=0.3)

    duration   = df['time'].iloc[-1]
    n_samples  = len(df)
    sample_rate = n_samples / duration if duration > 0 else 0
    gsr_mean, gsr_std = df['gsr_uS'].mean(), df['gsr_uS'].std()
    gsr_min,  gsr_max = df['gsr_uS'].min(),  df['gsr_uS'].max()
    stats_text = (
        f"Duration: {duration:.1f}s | Samples: {n_samples:,} | Rate: {sample_rate:.1f} Hz\n"
        f"GSR - Mean: {gsr_mean:.2e} | Std: {gsr_std:.2e} | Range: [{gsr_min:.2e}, {gsr_max:.2e}]"
        f"{calibration_note}"
    )
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat' if is_calibrated else 'lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[analysis] GSR plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_gsr_analysis(filepath: str, window_size: float = 5.0, save_path: str | None = None):
    """
    Plot detailed GSR analysis with rolling statistics.

    Creates a 4-panel figure showing:
    1. GSR signal with rolling mean
    2. Rolling standard deviation (variability)
    3. Rate of change (derivative)
    4. Distribution histogram

    Args:
        filepath: Path to GSR CSV file
        window_size: Rolling window size in seconds
        save_path: Optional path to save the figure
    """
    import pandas as pd

    df = pd.read_csv(filepath)

    if 'timestamp' in df.columns:
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]

    duration      = df['time'].iloc[-1]
    n_samples     = len(df)
    sample_rate   = n_samples / duration if duration > 0 else 50
    window_samples = int(window_size * sample_rate)

    df['rolling_mean'] = df['gsr_uS'].rolling(window=window_samples, center=True).mean()
    df['rolling_std']  = df['gsr_uS'].rolling(window=window_samples, center=True).std()
    df['derivative']   = np.gradient(df['gsr_uS'], df['time'])

    is_calibrated    = df['gsr_uS'].abs().max() > 1e-3
    unit             = "µS" if is_calibrated else "a.u."
    gsr_label        = f"GSR ({unit})"
    calibration_note = "" if is_calibrated else "⚠ Uncalibrated — values are raw audio amplitude, not µS"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('eSense GSR Detailed Analysis', fontsize=14, fontweight='bold')

    axes[0, 0].plot(df['time'], df['gsr_uS'], 'lightblue', linewidth=0.5, alpha=0.7, label='Raw GSR')
    axes[0, 0].plot(df['time'], df['rolling_mean'], 'b-', linewidth=1.5, label=f'Rolling Mean ({window_size}s)')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel(gsr_label)
    axes[0, 0].set_title('GSR Signal with Trend')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df['time'], df['rolling_std'], 'orange', linewidth=1)
    axes[0, 1].fill_between(df['time'], 0, df['rolling_std'], alpha=0.3, color='orange')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel(f'Std Dev ({unit})')
    axes[0, 1].set_title(f'GSR Variability (Rolling Std, {window_size}s window)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df['time'], df['derivative'], 'green', linewidth=0.5)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel(f'dGSR/dt ({unit}/s)')
    axes[1, 0].set_title('Rate of Change (Phasic Activity Indicator)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(df['gsr_uS'], bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=df['gsr_uS'].mean(), color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {df['gsr_uS'].mean():.2e}")
    axes[1, 1].axvline(x=df['gsr_uS'].median(), color='orange', linestyle='--', linewidth=2,
                       label=f"Median: {df['gsr_uS'].median():.2e}")
    axes[1, 1].set_xlabel(gsr_label)
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('GSR Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if calibration_note:
        fig.text(0.5, 0.01, calibration_note, ha='center', fontsize=10,
                 color='darkorange', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.subplots_adjust(bottom=0.07)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[analysis] GSR analysis plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig


# ── Cross-Device Comparison ───────────────────────────────────────────────────

def _compare_erp(devs, out_prefix):
    """Grand-average task-onset ERP waveform overlay across devices."""
    ci = _ch_idx('TP_pool')
    fig, ax = plt.subplots(figsize=(9, 5))
    for slug, label, color, res in devs:
        task_epochs = res.get('task_onset_epochs', [])
        if not task_epochs:
            continue
        time_ms = task_epochs[0].get('time_ms')
        if time_ms is None:
            continue
        stack = np.stack([ep['data'][:, ci] for ep in task_epochs])
        grand = stack.mean(axis=0)
        sem   = stack.std(axis=0) / np.sqrt(len(task_epochs))
        ax.plot(time_ms, grand, color=color, lw=1.5, label=label)
        ax.fill_between(time_ms, grand - sem, grand + sem, alpha=0.15, color=color)
    ax.axvline(0, color='k', lw=0.8, linestyle='--')
    ax.invert_yaxis()
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Grand-Average Task-Onset ERP — TP_pool')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = f"{out_prefix}compare_erp.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def _compare_band_power(devs, out_prefix):
    """Grouped band-power bar chart per channel across devices."""
    band_names = list(BANDS.keys())
    x = np.arange(len(band_names))
    for ch in ['TP_pool', 'AF_pool']:
        fig, ax = plt.subplots(figsize=(10, 5))
        n = len(devs)
        w = 0.8 / max(n, 1)
        for i, (slug, label, color, res) in enumerate(devs):
            bdf = res.get('band_df')
            if bdf is None or bdf.empty:
                continue
            ch_df = bdf[bdf['channel'] == ch]
            if ch_df.empty:
                continue
            vals = [ch_df[f'{b}_rel'].mean() for b in band_names]
            ax.bar(x + i * w, vals, w, label=label, color=color, alpha=0.85)
        ax.set_xticks(x + w * (n - 1) / 2)
        ax.set_xticklabels(band_names)
        ax.set_ylabel('Relative Power')
        ax.set_title(f'Band Power — {ch} — Cross-Device')
        ax.legend(fontsize=9)
        plt.tight_layout()
        fname = f"{out_prefix}compare_band_power_{ch}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[analysis] Saved {fname}")


def _compare_ersp_itpc(devs, out_prefix):
    """Side-by-side ERSP/ITPC pcolormesh grid across devices (all tasks pooled)."""
    n = len(devs)
    if n == 0:
        return
    for ch in ['TP_pool', 'AF_pool']:
        fig, axes = plt.subplots(2, n, figsize=(6 * n, 8),
                                 sharey=True, squeeze=False)
        last_im_ersp = last_im_itpc = None
        for col, (slug, label, color, res) in enumerate(devs):
            cache = res.get('ersp_itpc', {}).get((ch, 'all_tasks'))
            axes[0, col].set_title(label, fontsize=10)
            if cache is None:
                axes[0, col].set_visible(False)
                axes[1, col].set_visible(False)
                continue
            last_im_ersp = axes[0, col].pcolormesh(
                cache['time_ms'], cache['freqs'], cache['ersp'],
                cmap='RdBu_r', vmin=-3, vmax=3, shading='auto')
            axes[0, col].axvline(0, color='k', lw=0.8, linestyle='--')
            last_im_itpc = axes[1, col].pcolormesh(
                cache['time_ms'], cache['freqs'], cache['itpc'],
                cmap='hot', vmin=0, vmax=0.5, shading='auto')
            axes[1, col].axvline(0, color='k', lw=0.8, linestyle='--')
            axes[1, col].set_xlabel('Time (ms)', fontsize=9)
        axes[0, 0].set_ylabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Frequency (Hz)')
        if last_im_ersp is not None:
            fig.colorbar(last_im_ersp, ax=axes[0, :].tolist(), label='ERSP (dB)')
        if last_im_itpc is not None:
            fig.colorbar(last_im_itpc, ax=axes[1, :].tolist(), label='ITPC')
        fig.suptitle(f'ERSP (top) / ITPC (bottom) — {ch} — All Tasks Pooled', fontsize=12)
        plt.tight_layout()
        fname = f"{out_prefix}compare_ersp_itpc_{ch}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[analysis] Saved {fname}")


def _compare_erd(devs, out_prefix):
    """ERD/ERS trajectory per band across devices."""
    for band in ['Theta', 'Alpha', 'Beta']:
        fig, ax = plt.subplots(figsize=(10, 5))
        for slug, label, color, res in devs:
            erd_data   = res.get('erd_data', {})
            time_axis  = res.get('erd_time_axis', np.array([]))
            band_data  = erd_data.get(band)
            if band_data is None or len(band_data) == 0 or len(time_axis) == 0:
                continue
            arr = np.array(band_data)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            mean = np.nanmean(arr, axis=0)
            sem  = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
            ax.plot(time_axis, mean, color=color, lw=1.5, label=label)
            ax.fill_between(time_axis, mean - sem, mean + sem, alpha=0.15, color=color)
        ax.axvline(0, color='k', lw=0.8, linestyle='--', label='Task onset')
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('ERD/ERS (%)')
        ax.set_title(f'{band} ERD/ERS — Cross-Device')
        ax.legend(fontsize=9)
        plt.tight_layout()
        fname = f"{out_prefix}compare_erd_{band}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[analysis] Saved {fname}")





def _compare_mse(devs, out_prefix):
    """Multiscale entropy curves across devices per channel."""
    max_scale = 20
    scales = np.arange(1, max_scale + 1)
    for ch in ['TP_pool', 'AF_pool']:
        fig, ax = plt.subplots(figsize=(9, 5))
        for slug, label, color, res in devs:
            cond_dict = res.get('mse_data', {}).get(ch, {})
            if not cond_dict:
                continue
            arrays = [v for v in cond_dict.values()
                      if v is not None and len(v) == max_scale]
            if not arrays:
                continue
            mean_mse = np.nanmean(np.stack(arrays, axis=0), axis=0)
            ax.plot(scales, mean_mse, color=color, lw=1.5, label=label)
        ax.set_xlabel('Scale')
        ax.set_ylabel('Sample Entropy')
        ax.set_title(f'Multiscale Entropy — {ch} — Cross-Device')
        ax.legend(fontsize=9)
        plt.tight_layout()
        fname = f"{out_prefix}compare_mse_{ch}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[analysis] Saved {fname}")



def _compare_theta_alpha(devs, out_prefix):
    """Theta/Alpha ratio bar chart across devices."""
    fig, ax = plt.subplots(figsize=(8, 5))
    vals, errs, labels, colors = [], [], [], []
    for slug, label, color, res in devs:
        bdf = res.get('band_df')
        if bdf is not None and not bdf.empty:
            ch_df = bdf[bdf['channel'] == 'TP_pool']
            v = ch_df['theta_alpha_ratio'].dropna() if 'theta_alpha_ratio' in ch_df.columns else pd.Series(dtype=float)
            vals.append(v.mean() if len(v) else np.nan)
            errs.append(v.std() if len(v) > 1 else 0)
        else:
            vals.append(np.nan); errs.append(0)
        labels.append(label)
        colors.append(color)
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Theta / Alpha Ratio')
    ax.set_title('Theta/Alpha Ratio — Cross-Device')
    plt.tight_layout()
    fname = f"{out_prefix}compare_theta_alpha.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def _compare_tfr_diff(devs, out_prefix):
    """
    For each pair of devices, plot the difference in mean Morlet TFR power
    (device A - device B) for each condition shared between them, with one
    subplot per raw EEG channel (TP9, AF7, AF8, TP10).

    Output: compare_tfr_diff_{slugA}_vs_{slugB}_{condition}.png
    """
    import itertools

    def _mean_power(epochs, ch_idx, freqs, fs):
        """Mean Morlet power across epochs: shape (n_freqs, n_times)."""
        if not epochs:
            return None
        power_sum = None
        for ep in epochs:
            sig = ep['data'][:, ch_idx].astype(float)
            tf  = morlet_wavelet_transform(sig, fs, freqs)
            p   = np.abs(tf) ** 2
            power_sum = p if power_sum is None else power_sum + p
        return power_sum / len(epochs)

    freqs = MORLET_FREQS

    for (slugA, labelA, _, resA), (slugB, labelB, _, resB) in itertools.combinations(devs, 2):
        # Find conditions present in both sessions (by task condition)
        task_epochsA = resA.get('task_onset_epochs', [])
        task_epochsB = resB.get('task_onset_epochs', [])
        if not task_epochsA or not task_epochsB:
            continue

        time_ms = task_epochsA[0].get('time_ms') if task_epochsA else None
        if time_ms is None:
            time_ms = task_epochsB[0].get('time_ms') if task_epochsB else None
        if time_ms is None:
            continue

        time_s = time_ms / 1000.0

        # Group task-onset epochs by condition for each device
        def _by_cond(epochs):
            groups = {}
            for ep in epochs:
                cond = ep.get('condition', 'all')
                groups.setdefault(str(cond), []).append(ep)
            return groups

        groupsA = _by_cond(task_epochsA)
        groupsB = _by_cond(task_epochsB)
        shared_conds = sorted(set(groupsA) & set(groupsB))

        if not shared_conds:
            # Fall back: use all epochs pooled together
            shared_conds = ['all']
            groupsA = {'all': task_epochsA}
            groupsB = {'all': task_epochsB}

        for cond in shared_conds:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
            axes_flat = axes.flatten()

            vmax = 0.0
            power_diffs = {}
            for ci, ch in enumerate(ALL_CHANNELS):
                ch_idx = _ch_idx(ch)
                pA = _mean_power(groupsA[cond], ch_idx, freqs, SAMPLE_RATE)
                pB = _mean_power(groupsB[cond], ch_idx, freqs, SAMPLE_RATE)
                if pA is None or pB is None:
                    power_diffs[ch] = None
                    continue
                diff = pA - pB
                power_diffs[ch] = diff
                vmax = max(vmax, float(np.abs(diff).max()))

            if vmax == 0:
                plt.close()
                continue

            for ci, ch in enumerate(ALL_CHANNELS):
                ax   = axes_flat[ci]
                diff = power_diffs.get(ch)
                if diff is None:
                    ax.set_visible(False)
                    continue
                im = ax.pcolormesh(time_s, freqs, diff,
                                   cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                                   shading='auto')
                ax.axvline(0, color='k', lw=0.8, linestyle='--')
                ax.set_title(ch, fontsize=10)
                ax.set_ylabel('Frequency (Hz)', fontsize=8)
                ax.set_xlabel('Time from Event (s)', fontsize=8)
                plt.colorbar(im, ax=ax, label='Power')

            safe_cond = str(cond).replace(' ', '_')
            fig.suptitle(
                f'Difference in Time-Frequency spectrum between '
                f'{labelA} and {labelB} - event \'{safe_cond}\'',
                fontsize=12)
            plt.tight_layout()
            fname = f"{out_prefix}compare_tfr_diff_{slugA}_vs_{slugB}_{safe_cond}.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"[analysis] Saved {fname}")


def compare_devices(session_results, out_prefix):
    """Produce cross-device comparison plots for all available devices."""
    devs = _devices_present(session_results)
    if not devs:
        print("[analysis] No recognisable device slugs — skipping cross-device comparison.")
        return
    if len(devs) < 3:
        print(f"[analysis] WARNING: Only {len(devs)}/3 devices found "
              f"({[d[0] for d in devs]}). Plots will include available devices only.")
    print(f"\n[analysis] === Cross-Device Comparison ({len(devs)} devices) ===")
    _compare_erp(devs, out_prefix)
    _compare_band_power(devs, out_prefix)
    _compare_ersp_itpc(devs, out_prefix)
    _compare_erd(devs, out_prefix)
    _compare_mse(devs, out_prefix)
    _compare_theta_alpha(devs, out_prefix)
    _compare_tfr_diff(devs, out_prefix)
    print(f"[analysis] Cross-device comparison plots saved with prefix {out_prefix}compare_*")


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

    device_results = {}

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
        result = analyse_session(erp_path, psd_path, events_path, bl_path, out_prefix)

        slug = _infer_device_slug(basename)
        if slug is not None and result is not None:
            device_results[slug] = result

    participant_prefix = os.path.join(participant_out_dir, f'{participant_id}_')
    compare_devices(device_results, participant_prefix)

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
