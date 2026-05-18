"""
baseline.py - Baseline EEG Recording, Denoising, and Feature Extraction

Standalone pre-experiment step. Run this BEFORE main.py:

  1. Asks for the participant ID via GUI.
  2. Opens the EEG calibration screen so the operator can verify electrode
     contact.
  3. Records 60 s of eyes-open resting EEG via the baseline GUI screen.
  4. Denoises the recording (in-process call to denoising.denoise_session).
  5. Extracts reference features (band power, DFA, PermEn, theta/alpha) used
     downstream by analysis.py for ERD/ERS normalisation and ERSP baseline
     correction.

Usage:
    python baseline.py
    python baseline.py --participant-id 001          # skip the GUI prompt
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.integrate import trapezoid

from denoising import SAMPLE_RATE, CHANNELS, denoise_session

# ── Constants ─────────────────────────────────────────────────────────────────

EYES_OPEN_DURATION_S  = 60  # seconds

BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4,   8),
    'Alpha': (8,  12),
    'Beta':  (12, 30),
}


# ── REQ-B1: Baseline Recording ────────────────────────────────────────────────

def _countdown():
    """Print 3-2-1 countdown to terminal."""
    for n in (3, 2, 1):
        print(f"[baseline] Recording in {n}...", flush=True)
        time.sleep(1)


def _record_phase(muse, duration_s, label):
    """
    Record EEG for duration_s seconds from an already-started MuseBrainFlowProcessor.

    Returns a DataFrame with columns: time, TP9, AF7, AF8, TP10
    Time is relative (0 = start of this phase).
    """
    # Drain any buffered data before recording starts
    muse.get_data()

    rows = []
    phase_start = time.time()

    while True:
        elapsed = time.time() - phase_start
        if elapsed >= duration_s:
            break

        eeg_data, hw_timestamps = muse.get_data()
        if eeg_data is not None and eeg_data.shape[1] > 0:
            for i in range(eeg_data.shape[1]):
                rel_time = float(hw_timestamps[i]) - phase_start
                row = {'time': rel_time}
                for ch_idx, ch in enumerate(muse.channels):
                    row[ch] = float(eeg_data[ch_idx, i])
                rows.append(row)

        time.sleep(0.01)

    df = pd.DataFrame(rows, columns=['time'] + CHANNELS)
    return df


def record_baseline(participant_id, session_id, data_dir='data', muse=None):
    """
    REQ-B1: Record eyes-open (60 s) resting EEG.

    Parameters:
        participant_id : str
        session_id     : str
        data_dir       : root data directory
        muse           : an already-connected MuseBrainFlowProcessor instance.
                         When provided the existing connection is reused and NOT
                         stopped on return.  When None a new connection is opened
                         and stopped after recording.

    Saves:
        data/<participant_id>/<session_id>_baseline_eyes_open.csv

    Returns:
        eyes_open_path
    """
    out_dir = os.path.join(data_dir, participant_id)
    os.makedirs(out_dir, exist_ok=True)

    eyes_open_path = os.path.join(out_dir, f'{session_id}_baseline_eyes_open.csv')

    _owns_muse = muse is None
    if _owns_muse:
        from muse import MuseBrainFlowProcessor
        print("[baseline] Connecting to Muse 2...")
        muse = MuseBrainFlowProcessor()

    try:
        print()
        print("[baseline] === Eyes Open Rest ===")
        print("[baseline] Please relax and look straight ahead.")
        print("[baseline] Do not blink excessively or move your head.")
        _countdown()
        print(f"[baseline] Recording... ({EYES_OPEN_DURATION_S}s)", flush=True)

        eo_df = _record_phase(muse, EYES_OPEN_DURATION_S, 'eyes_open')
        eo_df.to_csv(eyes_open_path, index=False)
        print(f"[baseline] Recording complete. {len(eo_df)} samples recorded.")

    finally:
        if _owns_muse:
            muse.stop()

    print()
    print(f"[baseline] Saved: {eyes_open_path}")

    return eyes_open_path


# ── REQ-B2: Baseline Preprocessing ───────────────────────────────────────────

def denoise_baseline(participant_id, session_id, data_dir='data', out_dir='data'):
    """
    REQ-B2: Run the shared denoising pipeline on the eyes-open baseline CSV.

    Reads:
        <data_dir>/<participant_id>/<session_id>_baseline_eyes_open.csv

    Writes to <out_dir>/<participant_id>/:
        <session_id>_baseline_eyes_open_erp_clean.csv
        <session_id>_baseline_eyes_open_psd_clean.csv

    Returns dict with keys 'eo_erp', 'eo_psd' mapping to file paths.
    """
    in_dir   = os.path.join(data_dir, participant_id)
    proc_dir = os.path.join(out_dir,  participant_id)
    os.makedirs(proc_dir, exist_ok=True)

    raw_path = os.path.join(in_dir,   f'{session_id}_baseline_eyes_open.csv')
    erp_out  = os.path.join(proc_dir, f'{session_id}_baseline_eyes_open_erp_clean.csv')
    psd_out  = os.path.join(proc_dir, f'{session_id}_baseline_eyes_open_psd_clean.csv')

    print("\n[baseline] === Denoising eyes_open ===")
    denoise_session(raw_path, erp_out, psd_out)

    return {'eo_erp': erp_out, 'eo_psd': psd_out}


# ── REQ-B3: Baseline Feature Extraction ──────────────────────────────────────

def _welch_band_power(sig, fs, bands, nperseg=None, noverlap=None):
    """
    Compute absolute band power for each band via Welch PSD + trapezoidal integration.

    Returns:
        freqs, psd, band_powers dict
    """
    if nperseg is None:
        nperseg = 4 * fs
    if noverlap is None:
        noverlap = 2 * fs

    nperseg = min(nperseg, len(sig))
    noverlap = min(noverlap, nperseg - 1)

    freqs, psd = scipy_signal.welch(sig, fs=fs, window='hann',
                                    nperseg=nperseg, noverlap=noverlap)
    band_powers = {}
    for band_name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() < 2:
            band_powers[band_name] = 0.0
        else:
            band_powers[band_name] = float(trapezoid(psd[mask], freqs[mask]))

    return freqs, psd, band_powers


def _dfa(sig, scales=None):
    """
    Detrended Fluctuation Analysis (DFA).

    Computes the scaling exponent alpha by fitting a line to
    log(scale) vs log(F(scale)) using log-spaced scales.

    Parameters:
        sig    : 1-D signal array
        scales : array of window sizes (samples). Default: log-spaced 16 to len(sig)//10

    Returns:
        alpha (float), or np.nan on failure
    """
    sig = np.asarray(sig, dtype=float)
    n = len(sig)

    if scales is None:
        min_scale = 16
        max_scale = max(min_scale + 1, n // 10)
        scales = np.unique(
            np.logspace(np.log10(min_scale), np.log10(max_scale), num=20).astype(int)
        )

    # Cumulative sum of mean-centred signal
    y = np.cumsum(sig - np.mean(sig))

    fluctuations = []
    valid_scales = []

    for s in scales:
        s = int(s)
        if s < 4 or s > n:
            continue
        n_blocks = n // s
        if n_blocks < 2:
            continue

        rms_list = []
        for b in range(n_blocks):
            segment = y[b * s: (b + 1) * s]
            x_seg   = np.arange(s, dtype=float)
            # Linear detrend of segment
            coeffs  = np.polyfit(x_seg, segment, 1)
            trend   = np.polyval(coeffs, x_seg)
            rms_list.append(np.sqrt(np.mean((segment - trend) ** 2)))

        if rms_list:
            fluctuations.append(np.mean(rms_list))
            valid_scales.append(s)

    if len(valid_scales) < 2:
        return np.nan

    log_s = np.log10(valid_scales)
    log_f = np.log10(fluctuations)
    coeffs = np.polyfit(log_s, log_f, 1)
    return float(coeffs[0])


def _permutation_entropy(sig, m=3, tau=1):
    """
    Permutation Entropy (PermEn) with embedding dimension m and time delay tau.

    Returns:
        permen (float, normalised to [0, 1]), or np.nan on failure
    """
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    n_patterns = n - (m - 1) * tau
    if n_patterns <= 0:
        return np.nan

    # Build permutation patterns
    patterns = []
    for i in range(n_patterns):
        indices = np.argsort(sig[i: i + m * tau: tau])
        patterns.append(tuple(indices))

    # Frequency of each pattern
    from collections import Counter
    counts = Counter(patterns)
    total  = float(sum(counts.values()))

    probabilities = np.array([v / total for v in counts.values()])
    # Shannon entropy, normalised by log(m!)
    import math as _math
    max_entropy = np.log(float(_math.factorial(m)))
    if max_entropy == 0:
        return np.nan

    pe = -np.sum(probabilities * np.log(probabilities)) / max_entropy
    return float(pe)


def extract_baseline_features(participant_id, session_id, out_dir='output'):
    """
    REQ-B3: Compute all reference metrics from the preprocessed eyes-open baseline CSVs.

    Reads from output/<participant_id>/:
        <session_id>_baseline_eyes_open_erp_clean.csv
        <session_id>_baseline_eyes_open_psd_clean.csv

    Saves:
        output/<participant_id>/<session_id>_baseline_features.json

    Returns:
        features dict
    """
    proc_dir = os.path.join(out_dir, participant_id)

    eo_erp_path = os.path.join(proc_dir, f'{session_id}_baseline_eyes_open_erp_clean.csv')
    eo_psd_path = os.path.join(proc_dir, f'{session_id}_baseline_eyes_open_psd_clean.csv')

    eo_erp_df = pd.read_csv(eo_erp_path)
    eo_psd_df = pd.read_csv(eo_psd_path)

    # ── B3a: Band Power Reference (from eo_psd) ───────────────────────────────
    # Individual channels + pooled
    channel_features = {ch: {} for ch in CHANNELS}
    channel_features['TP_pool'] = {}
    channel_features['AF_pool'] = {}

    # Eyes-open PSD — use good segments only
    eo_good = eo_psd_df[eo_psd_df['bad_segment'] == False] if 'bad_segment' in eo_psd_df.columns else eo_psd_df

    for ch in CHANNELS:
        sig = eo_good[ch].values.astype(float)
        _, _, bp = _welch_band_power(sig, SAMPLE_RATE, BANDS)
        channel_features[ch]['band_power'] = bp

    # Pooled channels
    tp9  = eo_good['TP9'].values.astype(float)
    tp10 = eo_good['TP10'].values.astype(float)
    af7  = eo_good['AF7'].values.astype(float)
    af8  = eo_good['AF8'].values.astype(float)

    tp_pool = (tp9 + tp10) / 2.0
    af_pool = (af7 + af8) / 2.0

    _, _, tp_bp = _welch_band_power(tp_pool, SAMPLE_RATE, BANDS)
    _, _, af_bp = _welch_band_power(af_pool, SAMPLE_RATE, BANDS)
    channel_features['TP_pool']['band_power'] = tp_bp
    channel_features['AF_pool']['band_power'] = af_bp

    # ── B3b: DFA Reference (from eo_psd) ─────────────────────────────────────
    for ch in CHANNELS:
        sig = eo_good[ch].values.astype(float)
        channel_features[ch]['dfa_baseline'] = _dfa(sig)

    # ── B3c: PermEn Reference (from eo_erp) ──────────────────────────────────
    eo_erp_good = eo_erp_df[eo_erp_df['bad_segment'] == False] if 'bad_segment' in eo_erp_df.columns else eo_erp_df

    for ch in CHANNELS:
        sig = eo_erp_good[ch].values.astype(float)
        channel_features[ch]['permen_baseline'] = _permutation_entropy(sig, m=3, tau=1)

    # ── B3d: Theta/Alpha Ratio ────────────────────────────────────────────────
    for ch in CHANNELS:
        bp = channel_features[ch]['band_power']
        theta = bp.get('Theta', 0.0)
        alpha = bp.get('Alpha', 0.0)
        if alpha > 0:
            channel_features[ch]['theta_alpha_baseline'] = float(theta / alpha)
        else:
            channel_features[ch]['theta_alpha_baseline'] = float('nan')

    # ── Assemble output ───────────────────────────────────────────────────────
    features = {
        'participant_id': participant_id,
        'session_id': session_id,
        'channels': channel_features,
    }

    # Serialise (convert any np.nan to null via custom encoder)
    json_path = os.path.join(proc_dir, f'{session_id}_baseline_features.json')

    class _NaNEncoder(json.JSONEncoder):
        def iterencode(self, obj, _one_shot=False):
            # Replace nan/inf with null
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                yield 'null'
                return
            yield from super().iterencode(obj, _one_shot)

        def default(self, obj):
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            return super().default(obj)

    # Use standard json with a sanitise pass instead (simpler and more reliable)
    def _sanitise(obj):
        if isinstance(obj, dict):
            return {k: _sanitise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitise(v) for v in obj]
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(json_path, 'w') as f:
        json.dump(_sanitise(features), f, indent=2)

    print(f"[baseline] Saved baseline features to {json_path}")
    return features


# ── REQ-B4: CLI Entry Point ───────────────────────────────────────────────────

def main():
    """
    Standalone baseline runner — driven by the same GUI screens main.py uses.

    Flow: participant ID → EEG calibration → 60 s eyes-open recording →
    denoise → feature extraction. Output is written under <data-dir>/<pid>/.
    """
    from gui import (
        show_participant_screen, show_calibration_screen, show_baseline_screen,
        show_error_popup,
    )

    parser = argparse.ArgumentParser(
        description='Record, denoise, and extract features from baseline EEG '
                    '(eyes-open rest). Run this before main.py.'
    )
    parser.add_argument('--participant-id', default=None,
                        help='Participant ID. If omitted, the GUI prompts for it.')
    parser.add_argument('--data-dir', default='data',
                        help='Root data directory (default: data)')
    parser.add_argument('--out-dir', default='data',
                        help='Root output directory (default: data)')
    args = parser.parse_args()

    # Step 1: Participant ID — GUI prompt, unless supplied on the CLI.
    participant_id = args.participant_id
    if not participant_id:
        participant_id = show_participant_screen()
        if not participant_id:
            print('[baseline] Cancelled at participant registration.')
            sys.exit(1)

    session_id = f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    print(f'[baseline] Participant: {participant_id}')
    print(f'[baseline] Session ID:  {session_id}')

    # Step 2: EEG calibration — live signal check. Returns a live Muse session
    # that we hand to the recording screen so we don't reconnect twice.
    calib_muse = show_calibration_screen(proceed_label='Proceed to Baseline')
    if calib_muse is None:
        print('[baseline] Cancelled at EEG calibration.')
        sys.exit(1)

    # Step 3: Record 60 s eyes-open via the existing GUI screen.
    if not show_baseline_screen(
        participant_id=participant_id,
        session_id=session_id,
        data_dir=args.data_dir,
        muse=calib_muse,
    ):
        print('[baseline] Cancelled at baseline recording.')
        sys.exit(1)

    # Step 4: Denoise the recording (in-process, no subprocess).
    try:
        denoise_baseline(
            participant_id=participant_id,
            session_id=session_id,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
        )
    except Exception as e:
        show_error_popup(f'Baseline denoising failed:\n{e}')
        print(f'[baseline] Denoising failed: {e}')
        sys.exit(1)

    # Step 5: Extract reference features for analysis.py.
    extract_baseline_features(
        participant_id=participant_id,
        session_id=session_id,
        out_dir=args.out_dir,
    )

    feat_path = os.path.join(args.out_dir, participant_id,
                             f'{session_id}_baseline_features.json')
    print(f'[baseline] Baseline complete. Features saved to {feat_path}')


if __name__ == '__main__':
    main()
