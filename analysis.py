"""
analysis.py - EEG analysis pipeline

Reads pre-denoised erp_clean.csv and psd_clean.csv (produced by denoising.py),
then performs all ERP and PSD analyses and generates plots.

Run denoising first:
    python denoising.py <participant_id>

Then analyse:
    python analysis.py <participant_id>
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy import signal, integrate, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

# --- Constants ---
SAMPLE_RATE      = 256
EPOCH_PRE_MS     = 200
EPOCH_POST_MS    = 800
BASELINE_PRE_MS  = 200
ARTIFACT_THRESH  = 100.0
MIN_EPOCHS       = 20

WINDOW_SEC    = 4
OVERLAP_RATIO = 0.5

BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4,   8),
    'Alpha': (8,  12),
    'Beta':  (12, 30),
}

ERP_CHANNELS = ['AF7', 'AF8']
PSD_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']  # REQ-3: all four channels
ALL_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']


# --- Step 0: Load data ---

def load_data(erp_path, psd_path, events_path):
    """Load pre-denoised ERP and PSD CSVs plus event markers."""
    erp_df    = pd.read_csv(erp_path)
    psd_df    = pd.read_csv(psd_path)
    events_df = pd.read_csv(events_path)
    return erp_df, psd_df, events_df


def parse_events(events_df):
    """
    Returns:
        oddball_triggers: list of times for oddball_onset events
        standard_triggers: list of times for standard_onset events
        task_intervals: list of dicts {task_number, start, end}
    """
    events_df = events_df.copy().sort_values('time').reset_index(drop=True)

    oddball_triggers  = events_df.loc[events_df['event'] == 'oddball_onset',  'time'].tolist()
    standard_triggers = events_df.loc[events_df['event'] == 'standard_onset', 'time'].tolist()

    task_ends = events_df[events_df['event'] == 'task_end'].copy()
    task_intervals = []
    prev_end = events_df['time'].min()
    for _, row in task_ends.iterrows():
        task_intervals.append({
            'task_number': int(row['task_number']),
            'start': prev_end,
            'end': row['time'],
        })
        prev_end = row['time']

    return oddball_triggers, standard_triggers, task_intervals


def get_condition_at_time(t, task_intervals):
    """Return task condition label for a given time."""
    for interval in task_intervals:
        if interval['start'] <= t < interval['end']:
            return f"task_{interval['task_number']}"
    return 'inter_trial'


# ============================================================
# ERP Analysis
# ============================================================

def extract_epochs(erp_df, triggers, pre_ms=EPOCH_PRE_MS, post_ms=EPOCH_POST_MS,
                   all_triggers=None):
    """
    Extract fixed-length epochs around each trigger time.
    Rejects epochs with bad_segment flags or overlapping events.

    Returns list of dicts: {data: array(samples, channels), time_ms: array, trigger_time: float}
    """
    pre_s  = pre_ms  / 1000.0
    post_s = post_ms / 1000.0
    n_samples = int((pre_s + post_s) * SAMPLE_RATE)
    time_ms = np.linspace(-pre_ms, post_ms, n_samples)

    times = erp_df['time'].values
    all_other_triggers = set(all_triggers) - set(triggers) if all_triggers else set()

    epochs = []
    reject_bad_seg = 0
    reject_overlap = 0

    for t in triggers:
        t_start = t - pre_s
        t_end   = t + post_s

        mask = (times >= t_start) & (times < t_end)
        seg = erp_df[mask]

        if seg['bad_segment'].any():
            reject_bad_seg += 1
            continue

        overlap = any(t_start <= ot < t_end for ot in all_other_triggers)
        if overlap:
            reject_overlap += 1
            continue

        data = np.zeros((n_samples, len(ALL_CHANNELS)))
        seg_times = seg['time'].values
        if len(seg_times) < 2:
            continue
        for ci, ch in enumerate(ALL_CHANNELS):
            data[:, ci] = np.interp(
                np.linspace(t_start, t_end, n_samples),
                seg_times,
                seg[ch].values
            )

        epochs.append({'data': data, 'time_ms': time_ms, 'trigger_time': t})

    print(f"[analysis] Epochs extracted: {len(epochs)}  "
          f"(rejected bad_seg={reject_bad_seg}, overlap={reject_overlap})")
    return epochs


def baseline_correct(epochs, pre_ms=BASELINE_PRE_MS):
    """Subtract pre-stimulus mean from each epoch/channel."""
    corrected = []
    pre_samples = int(pre_ms / 1000.0 * SAMPLE_RATE)
    for ep in epochs:
        d = ep['data'].copy()
        baseline = d[:pre_samples, :].mean(axis=0)
        d -= baseline
        corrected.append({**ep, 'data': d})
    return corrected


def reject_artifacts(epochs, threshold=ARTIFACT_THRESH):
    """Reject epochs where any channel exceeds ±threshold µV."""
    accepted, rejected = [], 0
    for ep in epochs:
        pk2pk = ep['data'].max(axis=0) - ep['data'].min(axis=0)
        if np.any(pk2pk > 2 * threshold):
            rejected += 1
        else:
            accepted.append(ep)
    print(f"[analysis] Artifact rejection: {len(accepted)} accepted, {rejected} rejected")
    return accepted


def average_epochs(epochs):
    """Stack epochs and compute grand average. Returns (mean_array, sem_array)."""
    stack = np.stack([ep['data'] for ep in epochs], axis=0)
    return stack.mean(axis=0), stack.std(axis=0) / np.sqrt(stack.shape[0])


def find_peak(erp, time_ms, t_start, t_end, polarity='pos'):
    """Find peak amplitude and latency in a time window."""
    mask = (time_ms >= t_start) & (time_ms <= t_end)
    seg = erp[mask]
    t_seg = time_ms[mask]
    if polarity == 'pos':
        idx = np.argmax(seg)
    else:
        idx = np.argmin(seg)
    return t_seg[idx], seg[idx]


# --- Plot 1: Grand Average ERP (REQ-2: SEM shading) ---
def plot_erp(oddball_erp, standard_erp, diff_erp, time_ms,
             oddball_sem, standard_sem, out_prefix=''):
    fig, axes = plt.subplots(1, len(ERP_CHANNELS), figsize=(14, 5), sharey=False)
    if len(ERP_CHANNELS) == 1:
        axes = [axes]

    for ax, ch in zip(axes, ERP_CHANNELS):
        ci = ALL_CHANNELS.index(ch)
        odd  = oddball_erp[:, ci]
        std  = standard_erp[:, ci]
        diff = diff_erp[:, ci]
        odd_sem = oddball_sem[:, ci]
        std_sem = standard_sem[:, ci]

        ax.axvline(0, color='k', lw=0.8, linestyle='--', label='Stimulus onset')
        ax.axhspan(-BASELINE_PRE_MS, 0, alpha=0.08, color='gray')

        ax.plot(time_ms, odd,  label='Oddball',    color='steelblue', lw=1.5)
        ax.fill_between(time_ms, odd - odd_sem, odd + odd_sem,
                        alpha=0.2, color='steelblue')

        ax.plot(time_ms, std,  label='Standard',   color='coral',     lw=1.5)
        ax.fill_between(time_ms, std - std_sem, std + std_sem,
                        alpha=0.2, color='coral')

        ax.plot(time_ms, diff, label='Difference', color='darkgreen', lw=1.5, linestyle='--')

        p300_t, p300_a = find_peak(odd, time_ms, 250, 600, polarity='pos')
        ax.annotate(f'P300\n{p300_t:.0f}ms', xy=(p300_t, p300_a),
                    xytext=(p300_t + 50, p300_a + 2),
                    arrowprops=dict(arrowstyle='->', color='steelblue'),
                    fontsize=8, color='steelblue')

        n200_t, n200_a = find_peak(odd, time_ms, 150, 300, polarity='neg')
        ax.annotate(f'N200\n{n200_t:.0f}ms', xy=(n200_t, n200_a),
                    xytext=(n200_t + 50, n200_a - 2),
                    arrowprops=dict(arrowstyle='->', color='tomato'),
                    fontsize=8, color='tomato')

        ax.invert_yaxis()
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'Grand Average ERP — {ch}')
        ax.legend(fontsize=8)
        ax.set_xlim(-EPOCH_PRE_MS, EPOCH_POST_MS)

    plt.tight_layout()
    fname = f"{out_prefix}plot1_erp_grand_average.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


# --- Plot 2: Single-trial heatmap ---
def plot_epoch_heatmap(oddball_epochs, time_ms, out_prefix=''):
    ch = 'AF7'
    ci = ALL_CHANNELS.index(ch)
    stack = np.stack([ep['data'][:, ci] for ep in oddball_epochs], axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = np.percentile(np.abs(stack), 95)
    im = ax.imshow(stack, aspect='auto', origin='lower',
                   extent=[time_ms[0], time_ms[-1], 0, len(oddball_epochs)],
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.axvline(0, color='k', lw=1, linestyle='--')
    plt.colorbar(im, ax=ax, label='Amplitude (µV)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Trial number')
    ax.set_title(f'Single-trial Oddball Epochs — {ch}')
    plt.tight_layout()
    fname = f"{out_prefix}plot2_epoch_heatmap.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


# --- Plot 3: Epoch rejection summary ---
def plot_rejection_summary(n_odd_accepted, n_odd_rejected, n_std_accepted, n_std_rejected,
                           out_prefix=''):
    fig, ax = plt.subplots(figsize=(7, 5))
    conditions = ['Oddball', 'Standard']
    accepted = [n_odd_accepted, n_std_accepted]
    rejected = [n_odd_rejected, n_std_rejected]
    x = np.arange(len(conditions))
    w = 0.35
    ax.bar(x - w/2, accepted, w, label='Accepted', color='steelblue')
    ax.bar(x + w/2, rejected, w, label='Rejected', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Epoch count')
    ax.set_title('Epoch Rejection Summary')
    ax.legend()
    plt.tight_layout()
    fname = f"{out_prefix}plot3_rejection_summary.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def run_erp_analysis(erp_df, events_df, out_prefix=''):
    oddball_triggers, standard_triggers, task_intervals = parse_events(events_df)
    all_triggers = oddball_triggers + standard_triggers

    print("\n[analysis] --- ERP Analysis ---")
    print(f"[analysis] Oddball triggers: {len(oddball_triggers)}, Standard: {len(standard_triggers)}")

    odd_epochs_raw = extract_epochs(erp_df, oddball_triggers,  all_triggers=all_triggers)
    std_epochs_raw = extract_epochs(erp_df, standard_triggers, all_triggers=all_triggers)

    n_odd_raw = len(oddball_triggers)
    n_std_raw = len(standard_triggers)

    odd_epochs = baseline_correct(odd_epochs_raw)
    std_epochs = baseline_correct(std_epochs_raw)

    odd_epochs = reject_artifacts(odd_epochs)
    std_epochs = reject_artifacts(std_epochs)

    n_odd_rejected = n_odd_raw - len(odd_epochs)
    n_std_rejected = n_std_raw - len(std_epochs)

    # REQ-6: warn if below minimum recommended epoch count
    if len(odd_epochs) < MIN_EPOCHS:
        print(f"[analysis] WARNING: Only {len(odd_epochs)} clean oddball epochs "
              f"(minimum recommended: {MIN_EPOCHS}). ERP SNR may be poor.")
    if len(odd_epochs) == 0 or len(std_epochs) == 0:
        print("[analysis] WARNING: Not enough epochs for ERP averaging.")
        return None, None, None, None, oddball_triggers, task_intervals

    time_ms = odd_epochs[0]['time_ms']
    oddball_erp,  oddball_sem  = average_epochs(odd_epochs)
    standard_erp, standard_sem = average_epochs(std_epochs)
    diff_erp = oddball_erp - standard_erp

    plot_erp(oddball_erp, standard_erp, diff_erp, time_ms,
             oddball_sem, standard_sem, out_prefix)
    plot_epoch_heatmap(odd_epochs, time_ms, out_prefix)
    plot_rejection_summary(len(odd_epochs), n_odd_rejected,
                           len(std_epochs), n_std_rejected, out_prefix)

    # REQ-5: return std_epochs too
    return odd_epochs, std_epochs, oddball_erp, time_ms, oddball_triggers, task_intervals


# ============================================================
# PSD Analysis
# ============================================================

def sliding_windows(psd_df, window_sec=WINDOW_SEC, overlap=OVERLAP_RATIO, task_intervals=None):
    """
    Segment EEG into overlapping windows, label each by condition.
    Returns list of dicts: {data: raw array(samples, channels), condition, t_mid}
    The Hann window is NOT applied here; signal.welch applies it internally (REQ-1).
    """
    win_samples  = int(window_sec * SAMPLE_RATE)
    step_samples = int(win_samples * (1 - overlap))
    times = psd_df['time'].values
    bad   = psd_df['bad_segment'].values.astype(bool)

    windows = []
    i = 0
    while i + win_samples <= len(psd_df):
        if bad[i:i + win_samples].any():
            i += step_samples
            continue

        t_mid = times[i + win_samples // 2]
        condition = get_condition_at_time(t_mid, task_intervals or [])
        data = psd_df.iloc[i:i + win_samples][ALL_CHANNELS].values.astype(float)

        windows.append({'data': data, 'condition': condition, 't_mid': t_mid})
        i += step_samples

    print(f"[analysis] PSD windows: {len(windows)}")
    return windows


def compute_welch(windows):
    """
    Compute Welch PSD per condition per channel using signal.welch (REQ-1).
    Groups raw window data by condition, concatenates, then calls signal.welch once
    per condition per channel so averaging is handled correctly by Welch's method.

    Returns dict: condition -> {freqs, mean_psd (freqs x channels), sem_psd, n_windows}
    """
    nperseg  = int(WINDOW_SEC * SAMPLE_RATE)
    noverlap = int(nperseg * OVERLAP_RATIO)

    # Group raw sample data by condition
    by_condition = {}
    for win in windows:
        cond = win['condition']
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(win['data'])

    result = {}
    for cond, data_list in by_condition.items():
        n_windows = len(data_list)
        # Concatenate all windows for this condition: (total_samples, channels)
        concat = np.concatenate(data_list, axis=0)

        psds_per_ch = []
        for ci in range(len(ALL_CHANNELS)):
            freqs, psd = signal.welch(concat[:, ci], fs=SAMPLE_RATE,
                                      window='hann', nperseg=nperseg,
                                      noverlap=noverlap)
            psds_per_ch.append(psd)

        mean_psd = np.stack(psds_per_ch, axis=1)  # (freqs, channels)
        result[cond] = {
            'freqs':     freqs,
            'mean_psd':  mean_psd,
            'sem_psd':   np.zeros_like(mean_psd),  # Welch gives one estimate; SEM not applicable
            'n_windows': n_windows,
        }
    return result


def extract_band_power(psds, bands=BANDS):
    """
    Compute absolute and relative band power per condition per channel.
    Returns DataFrame: condition x channel x band
    """
    rows = []
    for cond, v in psds.items():
        freqs    = v['freqs']
        mean_psd = v['mean_psd']
        for ci, ch in enumerate(ALL_CHANNELS):
            psd_ch = mean_psd[:, ci]
            total_mask  = (freqs >= 0.5) & (freqs <= 45)
            total_power = integrate.trapezoid(psd_ch[total_mask], freqs[total_mask])
            band_row = {'condition': cond, 'channel': ch}
            for band_name, (f_lo, f_hi) in bands.items():
                band_mask = (freqs >= f_lo) & (freqs <= f_hi)
                abs_power = integrate.trapezoid(psd_ch[band_mask], freqs[band_mask])
                band_row[f'{band_name}_abs'] = abs_power
                band_row[f'{band_name}_rel'] = abs_power / total_power if total_power > 0 else np.nan
            rows.append(band_row)
    return pd.DataFrame(rows)


def compute_theta_alpha_ratio(band_df):
    """Add theta/alpha ratio column to band_df."""
    bd = band_df.copy()
    bd['theta_alpha_ratio'] = bd['Theta_abs'] / bd['Alpha_abs'].replace(0, np.nan)
    return bd


# --- REQ-4: Frontal Alpha Asymmetry ---
def compute_faa(band_df):
    """
    Compute FAA = log(Alpha_abs_AF8) - log(Alpha_abs_AF7) per condition.
    Returns DataFrame with columns: condition, FAA.
    """
    rows = []
    conditions = band_df['condition'].unique()
    for cond in conditions:
        cond_df = band_df[band_df['condition'] == cond]
        af7_row = cond_df[cond_df['channel'] == 'AF7']
        af8_row = cond_df[cond_df['channel'] == 'AF8']
        if af7_row.empty or af8_row.empty:
            rows.append({'condition': cond, 'FAA': np.nan})
            continue
        alpha_af7 = af7_row['Alpha_abs'].values[0]
        alpha_af8 = af8_row['Alpha_abs'].values[0]
        if alpha_af7 <= 0 or alpha_af8 <= 0 or np.isnan(alpha_af7) or np.isnan(alpha_af8):
            rows.append({'condition': cond, 'FAA': np.nan})
        else:
            rows.append({'condition': cond, 'FAA': np.log(alpha_af8) - np.log(alpha_af7)})
    return pd.DataFrame(rows)


def plot_faa(faa_df, out_prefix=''):
    fig, ax = plt.subplots(figsize=(8, 5))
    conditions = faa_df['condition'].tolist()
    faas = faa_df['FAA'].tolist()
    x = np.arange(len(conditions))
    ax.bar(x, faas, color='mediumpurple', alpha=0.8)
    ax.axhline(0, color='k', lw=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.set_ylabel('FAA (log AF8α − log AF7α)')
    ax.set_title('Frontal Alpha Asymmetry by Condition')
    plt.tight_layout()
    fname = f"{out_prefix}plot9_faa.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


# --- Plot 4: PSD by condition (REQ-7: band labels after semilogy) ---
def plot_psd(psds, out_prefix=''):
    band_colors = {'Delta': '#d4e4f7', 'Theta': '#d4f7d4', 'Alpha': '#f7f7d4', 'Beta': '#f7d4d4'}

    for ch in PSD_CHANNELS:
        ci = ALL_CHANNELS.index(ch)
        fig, ax = plt.subplots(figsize=(10, 5))

        for cond, v in psds.items():
            freqs = v['freqs']
            psd   = v['mean_psd'][:, ci]
            mask  = freqs <= 45
            ax.semilogy(freqs[mask], psd[mask], label=cond, lw=1.5)

        # REQ-7: add band shading and labels AFTER semilogy so ylim is in log scale
        for band_name, (f_lo, f_hi) in BANDS.items():
            ax.axvspan(f_lo, f_hi, alpha=0.15, color=band_colors[band_name])
            ax.text((f_lo + f_hi) / 2, ax.get_ylim()[0] * 3, band_name,
                    ha='center', fontsize=7, color='gray')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (µV²/Hz)')
        ax.set_title(f'Power Spectral Density — {ch}')
        ax.legend(fontsize=8)
        ax.set_xlim(1, 45)
        plt.tight_layout()
        fname = f"{out_prefix}plot4_psd_{ch}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[analysis] Saved {fname}")


# --- Plot 5: Relative band power bar chart ---
def plot_band_power(band_df, psds, out_prefix=''):
    for ch in PSD_CHANNELS:
        ch_df = band_df[band_df['channel'] == ch]
        conditions = ch_df['condition'].tolist()
        band_names = list(BANDS.keys())
        x = np.arange(len(band_names))
        w = 0.8 / max(len(conditions), 1)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
        for ci, (cond, row) in enumerate(zip(conditions, ch_df.itertuples())):
            rel_powers = [getattr(row, f'{b}_rel') for b in band_names]
            ax.bar(x + ci * w, rel_powers, w, label=cond, color=colors[ci], capsize=3)

        ax.set_xticks(x + w * (len(conditions) - 1) / 2)
        ax.set_xticklabels(band_names)
        ax.set_ylabel('Relative power')
        ax.set_title(f'Relative Band Power — {ch}')
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = f"{out_prefix}plot5_band_power_{ch}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[analysis] Saved {fname}")


# --- Plot 6: Theta/alpha ratio ---
def plot_theta_alpha(band_df, out_prefix=''):
    fig, ax = plt.subplots(figsize=(8, 5))
    n_ch = len(PSD_CHANNELS)
    w = 0.8 / max(n_ch, 1)
    # Use first channel's conditions for x-axis ticks
    first_ch_df = band_df[band_df['channel'] == PSD_CHANNELS[0]]
    conditions = first_ch_df['condition'].tolist()
    x = np.arange(len(conditions))

    for i, ch in enumerate(PSD_CHANNELS):
        ch_df = band_df[band_df['channel'] == ch].copy()
        ratios = ch_df['theta_alpha_ratio'].tolist()
        ax.bar(x + i * w, ratios, w, label=ch, alpha=0.8)

    ax.set_xticks(x + w * (n_ch - 1) / 2)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.set_ylabel('Theta / Alpha ratio')
    ax.set_title('Theta/Alpha Ratio by Condition')
    ax.legend()
    plt.tight_layout()
    fname = f"{out_prefix}plot6_theta_alpha_ratio.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


# --- Plot 7: Spectrogram ---
def plot_spectrogram(psd_df, task_intervals, out_prefix=''):
    for ch in PSD_CHANNELS:
        sig   = psd_df[ch].values.astype(float)
        times = psd_df['time'].values

        f, t, Sxx = signal.spectrogram(sig, fs=SAMPLE_RATE,
                                        window='hann',
                                        nperseg=int(WINDOW_SEC * SAMPLE_RATE),
                                        noverlap=int(WINDOW_SEC * SAMPLE_RATE * OVERLAP_RATIO))
        t_abs     = times[0] + t
        freq_mask = (f >= 1) & (f <= 45)

        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.pcolormesh(t_abs, f[freq_mask],
                           10 * np.log10(Sxx[freq_mask] + 1e-12),
                           shading='gouraud', cmap='inferno')
        plt.colorbar(im, ax=ax, label='Power (dB)')

        for interval in task_intervals:
            ax.axvline(interval['start'], color='cyan', lw=1, linestyle='--', alpha=0.7)
            ax.axvline(interval['end'],   color='lime',  lw=1, linestyle='--', alpha=0.7)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Spectrogram — {ch}')
        ax.set_ylim(1, 45)
        plt.tight_layout()
        fname = f"{out_prefix}plot7_spectrogram_{ch}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[analysis] Saved {fname}")


def run_psd_analysis(psd_df, events_df, out_prefix=''):
    _, _, task_intervals = parse_events(events_df)

    print("\n[analysis] --- PSD Analysis ---")
    windows  = sliding_windows(psd_df, task_intervals=task_intervals)
    psds     = compute_welch(windows)
    band_df  = extract_band_power(psds)
    band_df  = compute_theta_alpha_ratio(band_df)

    # REQ-4: FAA
    faa_df = compute_faa(band_df)

    print(band_df.to_string())

    # REQ-8: save band power summary CSV
    csv_path = f"{out_prefix}band_power_summary.csv"
    band_df.to_csv(csv_path, index=False)
    print(f"[analysis] Saved {csv_path}")

    plot_psd(psds, out_prefix)
    plot_band_power(band_df, psds, out_prefix)
    plot_theta_alpha(band_df, out_prefix)
    plot_spectrogram(psd_df, task_intervals, out_prefix)
    plot_faa(faa_df, out_prefix)

    return psds, band_df, faa_df, task_intervals


# ============================================================
# Cross-Analysis
# ============================================================

def plot_p300_vs_condition(odd_epochs, time_ms, task_intervals, out_prefix=''):
    """Plot mean P300 amplitude (250-600ms) per concurrent main task condition."""
    if odd_epochs is None or len(odd_epochs) == 0:
        print("[analysis] Skipping P300 vs condition: no oddball epochs.")
        return

    results = {}
    ci_af7  = ALL_CHANNELS.index('AF7')
    for ep in odd_epochs:
        cond = get_condition_at_time(ep['trigger_time'], task_intervals)
        mask = (time_ms >= 250) & (time_ms <= 600)
        p300_amp = ep['data'][mask, ci_af7].mean()
        results.setdefault(cond, []).append(p300_amp)

    if not results:
        print("[analysis] No P300 data per condition.")
        return

    conditions = sorted(results.keys())
    means = [np.mean(results[c]) for c in conditions]
    sems  = [np.std(results[c]) / np.sqrt(len(results[c])) for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conditions))
    ax.bar(x, means, yerr=sems, capsize=5, color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.set_ylabel('Mean P300 Amplitude (µV)')
    ax.set_title('P300 Amplitude vs Main Task Condition (AF7)')
    ax.axhline(0, color='k', lw=0.5)
    plt.tight_layout()
    fname = f"{out_prefix}plot8_p300_vs_condition.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[analysis] Saved {fname}")


def run_cross_analysis(odd_epochs, time_ms, task_intervals, out_prefix=''):
    print("\n[analysis] --- Cross Analysis ---")
    plot_p300_vs_condition(odd_epochs, time_ms, task_intervals, out_prefix)


# ============================================================
# REQ-5: Statistical Tests
# ============================================================

def run_statistics(odd_epochs, std_epochs, band_df, time_ms, out_prefix=''):
    print("\n[analysis] --- Statistical Tests ---")
    stat_rows = []

    # 5a: P300 amplitude t-test per ERP channel
    if odd_epochs and std_epochs and time_ms is not None:
        p300_mask = (time_ms >= 250) & (time_ms <= 600)
        for ch in ERP_CHANNELS:
            ci = ALL_CHANNELS.index(ch)
            odd_amps = [ep['data'][p300_mask, ci].mean() for ep in odd_epochs]
            std_amps = [ep['data'][p300_mask, ci].mean() for ep in std_epochs]
            if len(odd_amps) >= 2 and len(std_amps) >= 2:
                t, p = stats.ttest_ind(odd_amps, std_amps)
                print(f"[stats] P300 t-test {ch}: t={t:.3f}, p={p:.4f}")
                stat_rows.append({'test': 'P300_ttest', 'channel': ch,
                                   'band_or_component': 'P300',
                                   'statistic': t, 'p_value': p})
            else:
                print(f"[stats] P300 t-test {ch}: insufficient epochs, skipped.")

    # 5b: Band power ANOVA/t-test across conditions
    conditions = band_df['condition'].unique().tolist()
    for ch in PSD_CHANNELS:
        ch_df = band_df[band_df['channel'] == ch]
        for band_name in BANDS:
            col = f'{band_name}_rel'
            groups = [ch_df.loc[ch_df['condition'] == c, col].dropna().tolist()
                      for c in conditions]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                continue
            if len(groups) >= 3:
                stat, p = stats.f_oneway(*groups)
                test_name = 'ANOVA'
            else:
                stat, p = stats.ttest_ind(groups[0], groups[1])
                test_name = 'ttest_ind'
            print(f"[stats] {ch} {band_name} across conditions: "
                  f"F/t={stat:.3f}, p={p:.4f}")
            stat_rows.append({'test': test_name, 'channel': ch,
                               'band_or_component': band_name,
                               'statistic': stat, 'p_value': p})

    # 5c: Save summary CSV
    if stat_rows:
        stats_df = pd.DataFrame(stat_rows)
        csv_path = f"{out_prefix}stats_summary.csv"
        stats_df.to_csv(csv_path, index=False)
        print(f"[analysis] Saved {csv_path}")
    else:
        print("[analysis] No statistical tests ran (insufficient data).")


# ============================================================
# Main
# ============================================================

def analyse_session(erp_path, psd_path, events_path, out_prefix=''):
    """Run the full analysis pipeline for a single pre-denoised session."""
    erp_df, psd_df, events_df = load_data(erp_path, psd_path, events_path)

    erp_result = run_erp_analysis(erp_df, events_df, out_prefix)
    odd_epochs, std_epochs, oddball_erp, time_ms, oddball_triggers, task_intervals = erp_result

    psds, band_df, faa_df, task_intervals = run_psd_analysis(psd_df, events_df, out_prefix)

    run_cross_analysis(odd_epochs, time_ms, task_intervals, out_prefix)

    # REQ-5: statistics (pass std_epochs and band_df)
    run_statistics(odd_epochs, std_epochs, band_df, time_ms, out_prefix)

    print("\n[analysis] Session complete.")


def main(participant_id, out_dir='output'):
    """
    Run full analysis pipeline for all sessions of a participant.

    Expects denoising.py to have already been run, producing:
        output/<participant_id>/session_*_erp_clean.csv
        output/<participant_id>/session_*_psd_clean.csv

    Reads event markers from the original data directory.
    Writes all plots and CSVs to output/<participant_id>/.
    """
    import glob as _glob
    import os

    participant_out_dir = os.path.join(out_dir, participant_id)

    erp_files = sorted(_glob.glob(os.path.join(participant_out_dir, 'session_*_erp_clean.csv')))
    if not erp_files:
        print(f"[analysis] No denoised files found in {participant_out_dir}/")
        print(f"[analysis] Run denoising first: python denoising.py {participant_id}")
        sys.exit(1)

    print(f"[analysis] Found {len(erp_files)} session(s) for participant '{participant_id}'")

    for erp_path in erp_files:
        basename    = os.path.basename(erp_path).replace('_erp_clean.csv', '')
        psd_path    = os.path.join(participant_out_dir, f'{basename}_psd_clean.csv')
        events_path = os.path.join('data', participant_id, f'{basename}_markers.csv')
        out_prefix  = os.path.join(participant_out_dir, f'{basename}_')

        if not os.path.exists(psd_path):
            print(f"[analysis] WARNING: psd_clean not found for {basename}, skipping.")
            continue
        if not os.path.exists(events_path):
            print(f"[analysis] WARNING: markers file not found for {basename}, skipping.")
            continue

        print(f"\n[analysis] === Session: {basename} ===")
        analyse_session(erp_path, psd_path, events_path, out_prefix)

    print(f"\n[analysis] All sessions complete. Outputs in {participant_out_dir}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyse EEG sessions for a participant.')
    parser.add_argument('participant_id', help='Participant ID (e.g. 001, test)')
    parser.add_argument('--out-dir', default='output', help='Root output directory (default: output)')
    args = parser.parse_args()
    main(args.participant_id, args.out_dir)
