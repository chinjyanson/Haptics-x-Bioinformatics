"""
grand_analysis.py — Grand average analysis across all participants.

Aggregates data from data/{pid}/ and output/{pid}/ for each participant
and each device (auditory, shape_changing, vibrations), then produces
cross-device comparison plots saved to output/grand/.
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import interpolate

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICES = ["auditory", "shape_changing", "vibrations"]
DEVICE_LABELS = {
    "auditory": "Auditory",
    "shape_changing": "Shape Changing",
    "vibrations": "Vibrations",
}
DEVICE_COLORS = {
    "auditory": "#2196F3",
    "shape_changing": "#4CAF50",
    "vibrations": "#FF9800",
}

BANDS = ["Delta", "Theta", "Alpha", "Beta"]
TASKS = ["task_1", "task_2", "task_3", "task_4", "task_5"]

# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def discover_participants(data_dir: str = "data") -> list:
    """Return sorted list of participant IDs (subdirectory names in data_dir)."""
    if not os.path.isdir(data_dir):
        return []
    pids = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    return pids


# ---------------------------------------------------------------------------
# Individual loaders (each returns None on failure)
# ---------------------------------------------------------------------------

def _glob_first(pattern: str):
    """Return the first file matching glob pattern, or None."""
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def load_session_summary(output_dir: str, pid: str, device: str) -> dict | None:
    path = _glob_first(os.path.join(output_dir, pid, f"session_*_{device}_session_summary.json"))
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def load_band_power(output_dir: str, pid: str, device: str) -> pd.DataFrame | None:
    path = _glob_first(os.path.join(output_dir, pid, f"session_*_{device}_band_power_summary.csv"))
    if not path:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_markers(data_dir: str, pid: str, device: str) -> pd.DataFrame | None:
    """Return only task_end rows from markers CSV."""
    path = _glob_first(os.path.join(data_dir, pid, f"session_*_{device}_markers.csv"))
    if not path:
        return None
    try:
        df = pd.read_csv(path)
        task_end = df[df["event"] == "task_end"].copy().reset_index(drop=True)
        return task_end if not task_end.empty else None
    except Exception:
        return None


def load_arduino(data_dir: str, pid: str, device: str) -> pd.DataFrame | None:
    """
    Load arduino CSV and return a DataFrame with columns [time, position, target]
    by joining encoder events with task boundaries from markers.
    Each row gets the target for the task it belongs to.
    """
    arduino_path = _glob_first(os.path.join(data_dir, pid, f"session_*_{device}_arduino.csv"))
    markers_path = _glob_first(os.path.join(data_dir, pid, f"session_*_{device}_markers.csv"))
    if not arduino_path or not markers_path:
        return None
    try:
        arduino = pd.read_csv(arduino_path)
        markers = pd.read_csv(markers_path)
    except Exception:
        return None

    # Parse encoder rows
    enc = arduino[arduino["event_type"] == "encoder"].copy()
    if enc.empty:
        return None
    enc["position"] = enc["data_json"].apply(lambda x: json.loads(x)["position"])
    enc = enc[["time", "position"]].reset_index(drop=True)

    # task_end rows give us task boundaries and targets
    task_ends = markers[markers["event"] == "task_end"][["time", "task_number", "target"]].copy()
    task_ends = task_ends.sort_values("time").reset_index(drop=True)
    if task_ends.empty:
        return None

    # Assign each encoder sample to a task based on time boundaries
    # task N spans: (task_ends[N-1].time, task_ends[N].time]
    # task 1 spans: (0, task_ends[0].time]
    boundaries = [0.0] + task_ends["time"].tolist()
    task_numbers = task_ends["task_number"].tolist()
    targets = task_ends["target"].tolist()

    rows = []
    for _, row in enc.iterrows():
        t = row["time"]
        for i, (t_start, t_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if t_start < t <= t_end:
                rows.append({
                    "time":        t,
                    "position":    row["position"],
                    "task_number": int(task_numbers[i]),
                    "target":      float(targets[i]),
                    "t_start":     t_start,
                    "t_end":       t_end,
                })
                break

    if not rows:
        return None
    return pd.DataFrame(rows)


def load_gsr(data_dir: str, pid: str, device: str) -> pd.DataFrame | None:
    path = _glob_first(os.path.join(data_dir, pid, f"session_*_{device}_gsr.csv"))
    if not path:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_hr(data_dir: str, pid: str, device: str) -> pd.DataFrame | None:
    path = _glob_first(os.path.join(data_dir, pid, f"session_*_{device}_hr.csv"))
    if not path:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_nasa_tlx(data_dir: str, pid: str, device: str) -> dict | None:
    path = _glob_first(os.path.join(data_dir, pid, f"session_*_{device}_nasa_tlx.json"))
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def load_gsr_baseline(data_dir: str, pid: str) -> float | None:
    path = os.path.join(data_dir, pid, "gsr_baseline.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get("amplitude_baseline")


# ---------------------------------------------------------------------------
# Master loader
# ---------------------------------------------------------------------------

def load_all_data(data_dir: str = "data", output_dir: str = "output") -> dict:
    """
    Returns nested dict: data[pid][device] = {
        "session_summary": dict | None,
        "band_power":      DataFrame | None,
        "encoder":         DataFrame | None,  # task_end rows only
        "arduino":         DataFrame | None,  # continuous encoder positions per task
        "gsr":             DataFrame | None,
        "hr":              DataFrame | None,
        "nasa_tlx":        dict | None,
        "gsr_baseline":    float | None,
    }
    """
    pids = discover_participants(data_dir)
    if not pids:
        print(f"[grand_analysis] No participants found in {data_dir!r}")
        return {}

    data = {}
    for pid in pids:
        data[pid] = {}
        gsr_baseline = load_gsr_baseline(data_dir, pid)
        for device in DEVICES:
            data[pid][device] = {
                "session_summary": load_session_summary(output_dir, pid, device),
                "band_power":      load_band_power(output_dir, pid, device),
                "encoder":         load_markers(data_dir, pid, device),
                "arduino":         load_arduino(data_dir, pid, device),
                "gsr":             load_gsr(data_dir, pid, device),
                "hr":              load_hr(data_dir, pid, device),
                "nasa_tlx":        load_nasa_tlx(data_dir, pid, device),
                "gsr_baseline":    gsr_baseline,
            }
        print(f"[grand_analysis] Loaded participant {pid}")
    return data


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def safe_mean(values) -> float:
    """Mean of a list/array, ignoring None and NaN. Returns NaN if empty."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(clean)) if clean else np.nan


def safe_sem(values) -> float:
    """Standard error of the mean, ignoring None/NaN."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(clean) < 2:
        return 0.0
    return float(np.std(clean, ddof=1) / np.sqrt(len(clean)))


def _bar_group(ax, device_values: dict, ylabel: str, title: str,
               colors: dict = None, ylim=None):
    """
    Draw a simple grouped bar chart with one bar per device.
    device_values: {device: (mean, sem)}
    """
    if colors is None:
        colors = DEVICE_COLORS
    devs = list(device_values.keys())
    means = [device_values[d][0] for d in devs]
    sems  = [device_values[d][1] for d in devs]
    x = np.arange(len(devs))
    bars = ax.bar(x, means, yerr=sems, capsize=5,
                  color=[colors[d] for d in devs],
                  error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels([DEVICE_LABELS[d] for d in devs])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    return bars


# ---------------------------------------------------------------------------
# Plot A: Average band power across entire session
# ---------------------------------------------------------------------------

def plot_band_power_avg(data: dict, out_dir: str):
    """Grouped bar chart: mean absolute band power per device, split by channel pool."""
    channels = ["TP_pool", "AF_pool"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Grand Average Band Power (Absolute) Across All Tasks", fontsize=13, fontweight="bold")

    for ax, ch in zip(axes, channels):
        x = np.arange(len(BANDS))
        width = 0.25
        offsets = np.linspace(-(len(DEVICES)-1)*width/2, (len(DEVICES)-1)*width/2, len(DEVICES))

        for i, device in enumerate(DEVICES):
            per_participant = []
            for pid in data:
                bp = data[pid][device]["band_power"]
                if bp is None:
                    continue
                ch_rows = bp[bp["channel"] == ch]
                if ch_rows.empty:
                    continue
                # average across all task rows for this channel
                vals = [ch_rows[f"{b}_abs"].mean() for b in BANDS]
                per_participant.append(vals)

            if not per_participant:
                continue
            arr = np.array(per_participant)  # shape (n_participants, n_bands)
            means = arr.mean(axis=0)
            sems  = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(len(BANDS))

            ax.bar(x + offsets[i], means, width, label=DEVICE_LABELS[device],
                   color=DEVICE_COLORS[device], yerr=sems, capsize=4,
                   error_kw={"elinewidth": 1.5})

        ax.set_xticks(x)
        ax.set_xticklabels(BANDS)
        ax.set_xlabel("Frequency Band")
        ax.set_ylabel("Absolute Power (µV²)")
        ax.set_title(f"Channel: {ch}")
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "plot_band_power_avg.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grand_analysis] Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot B: Band power difference — beginning vs end of session
# ---------------------------------------------------------------------------

def plot_band_power_diff(data: dict, out_dir: str):
    """Bar chart: (late tasks mean) - (early tasks mean) per device per band."""
    early_tasks = {"task_1", "task_2"}
    late_tasks  = {"task_4", "task_5"}
    channels = ["TP_pool", "AF_pool"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Band Power Change: Late Tasks − Early Tasks (Absolute Power)", fontsize=13, fontweight="bold")

    for ax, ch in zip(axes, channels):
        x = np.arange(len(BANDS))
        width = 0.25
        offsets = np.linspace(-(len(DEVICES)-1)*width/2, (len(DEVICES)-1)*width/2, len(DEVICES))

        for i, device in enumerate(DEVICES):
            per_participant = []
            for pid in data:
                bp = data[pid][device]["band_power"]
                if bp is None:
                    continue
                ch_rows = bp[bp["channel"] == ch]
                if ch_rows.empty:
                    continue
                early_rows = ch_rows[ch_rows["condition"].isin(early_tasks)]
                late_rows  = ch_rows[ch_rows["condition"].isin(late_tasks)]
                if early_rows.empty or late_rows.empty:
                    continue
                diff = [late_rows[f"{b}_abs"].mean() - early_rows[f"{b}_abs"].mean() for b in BANDS]
                per_participant.append(diff)

            if not per_participant:
                continue
            arr = np.array(per_participant)
            means = arr.mean(axis=0)
            sems  = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(len(BANDS))

            ax.bar(x + offsets[i], means, width, label=DEVICE_LABELS[device],
                   color=DEVICE_COLORS[device], yerr=sems, capsize=4,
                   error_kw={"elinewidth": 1.5})

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(BANDS)
        ax.set_xlabel("Frequency Band")
        ax.set_ylabel("Δ Absolute Power (µV²)")
        ax.set_title(f"Channel: {ch}")
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "plot_band_power_diff.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grand_analysis] Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot C: Knob rotation average — 3 subplots (one per device)
# ---------------------------------------------------------------------------

def plot_knob_rotation(data: dict, out_dir: str):
    """
    3 subplots (one per device). For each device, every encoder sample from
    the arduino CSV is assigned to a task. Time is kept as real elapsed seconds
    from task start. Each task trace is padded (held at its last value) to the
    full task duration so shorter traces don't shrink the average. All task
    traces from all participants are resampled onto a common time grid (0 s to
    max task duration across all participants) and then averaged, producing a
    grand-average trace with a shaded SEM band.

    Y-axis shows encoder_position − target (the actual target per task, NOT
    always 0), so y=0 means the knob is exactly on its target.
    """
    DT = 0.1  # time resolution in seconds for the common grid

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle(
        "Knob Rotation: Grand Average Absolute Distance from Target\n"
        "(All tasks × all participants — real time, padded at task end)",
        fontsize=13, fontweight="bold",
    )

    for ax, device in zip(axes, DEVICES):
        # First pass: collect all task traces as (elapsed_times, delta) pairs
        # and find the maximum task duration to set the common grid.
        task_traces = []  # list of (t_elapsed_array, delta_array) — real seconds

        for pid in data:
            ard = data[pid][device]["arduino"]
            if ard is None or ard.empty:
                continue
            for task_num, grp in ard.groupby("task_number"):
                grp = grp.sort_values("time")
                t_start  = grp["t_start"].iloc[0]
                t_end    = grp["t_end"].iloc[0]
                duration = t_end - t_start
                if duration <= 0:
                    continue
                t_elapsed = grp["time"].values - t_start   # seconds from task start
                delta     = np.abs(grp["position"].values.astype(float) - grp["target"].values.astype(float))
                if len(t_elapsed) < 1:
                    continue
                task_traces.append((t_elapsed, delta, duration))

        if not task_traces:
            ax.set_title(DEVICE_LABELS[device])
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        max_duration = max(dur for _, _, dur in task_traces)
        common_t = np.arange(0, max_duration + DT, DT)

        # Second pass: resample each trace onto the common grid, padding with
        # last known value beyond the trace's own duration.
        resampled = []
        for t_elapsed, delta, duration in task_traces:
            # interp1d with fill_value=last value for times beyond last sample
            if len(t_elapsed) == 1:
                # single sample — hold constant for whole task duration
                resampled.append(np.full(len(common_t), delta[0]))
            else:
                f = interpolate.interp1d(
                    t_elapsed, delta, kind="linear",
                    bounds_error=False,
                    fill_value=(delta[0], delta[-1]),   # pad left=first, right=last
                )
                resampled.append(f(common_t))

        arr = np.array(resampled)  # (n_traces, len(common_t))
        mean_trace = arr.mean(axis=0)
        sem_trace  = (arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                      if arr.shape[0] > 1 else np.zeros(len(common_t)))

        ax.plot(common_t, mean_trace, color=DEVICE_COLORS[device], linewidth=2,
                label=f"Grand avg (n={arr.shape[0]} task-trials)")
        ax.fill_between(common_t,
                        mean_trace - sem_trace,
                        mean_trace + sem_trace,
                        color=DEVICE_COLORS[device], alpha=0.2, label="±SEM")
        ax.axhline(0, color="black", linewidth=1, linestyle="--", label="On target (0)")

        ax.set_xlabel("Time from task start (s)")
        ax.set_ylabel("|Encoder − Target| (position units)")
        ax.set_title(DEVICE_LABELS[device])
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "plot_knob_rotation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grand_analysis] Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot D: Summary table — encoder MSE + task time (CSV + PNG)
# ---------------------------------------------------------------------------

def _compute_encoder_mse(data: dict, device: str) -> float:
    errors_sq = []
    for pid in data:
        enc = data[pid][device]["encoder"]
        if enc is None or enc.empty:
            continue
        for _, row in enc.iterrows():
            err = row.get("encoder_error")
            if not pd.isna(err):
                errors_sq.append(float(err) ** 2)
    return float(np.mean(errors_sq)) if errors_sq else np.nan


def _compute_mean_task_time(data: dict, device: str) -> float:
    times = []
    for pid in data:
        enc = data[pid][device]["encoder"]
        if enc is None or enc.empty:
            continue
        ts = enc["time"].values.astype(float)
        if len(ts) == 0:
            continue
        # time for task 1 = ts[0] (from session start ~0)
        # time for task N = ts[N] - ts[N-1]
        diffs = [ts[0]] + [ts[i] - ts[i-1] for i in range(1, len(ts))]
        times.extend(diffs)
    return float(np.mean(times)) if times else np.nan


def _compute_mean_hr(data: dict, device: str) -> float:
    vals = []
    for pid in data:
        hr_df = data[pid][device]["hr"]
        if hr_df is None or hr_df.empty:
            continue
        vals.append(hr_df["heart_rate"].mean())
    return safe_mean(vals)


def _compute_mean_gsr(data: dict, device: str) -> float:
    vals = []
    for pid in data:
        gsr_df = data[pid][device]["gsr"]
        if gsr_df is None or gsr_df.empty:
            continue
        vals.append(gsr_df["gsr_uS"].mean())
    return safe_mean(vals)


def _compute_mean_p300(data: dict, device: str) -> float:
    vals = []
    for pid in data:
        ss = data[pid][device]["session_summary"]
        if ss is None:
            continue
        p300 = ss.get("task_onset_erp_peak_by_condition", {})
        task_vals = [v for v in p300.values() if v is not None]
        if task_vals:
            vals.append(np.mean(task_vals))
    return safe_mean(vals)


def _compute_nasa_avg(data: dict, device: str) -> float:
    vals = []
    for pid in data:
        tlx = data[pid][device]["nasa_tlx"]
        if tlx is None:
            continue
        avg = tlx.get("average")
        if avg is not None:
            vals.append(avg)
    return safe_mean(vals)


def plot_summary_table(data: dict, out_dir: str):
    """Save grand_stats_summary.csv and a matplotlib table PNG."""
    rows = []
    for device in DEVICES:
        rows.append({
            "Device":              DEVICE_LABELS[device],
            "Encoder MSE":         round(_compute_encoder_mse(data, device), 2),
            "Mean Task Time (s)":  round(_compute_mean_task_time(data, device), 2),
            "Mean HR (bpm)":       round(_compute_mean_hr(data, device), 2),
            "Mean GSR (µS)":       round(_compute_mean_gsr(data, device), 6),
            "P300 Amp (µV)":       round(_compute_mean_p300(data, device), 3),
            "NASA-TLX Avg":        round(_compute_nasa_avg(data, device), 2),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "grand_stats_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[grand_analysis] Saved {csv_path}")

    # PNG table
    fig, ax = plt.subplots(figsize=(13, 2 + 0.6 * len(df)))
    ax.axis("off")
    col_labels = df.columns.tolist()
    cell_text  = df.values.tolist()
    # colour header cells by device colour
    col_colors = ["#E0E0E0"] * len(col_labels)
    # rowColours: one colour per row (the device colour for each row)
    row_colors = [DEVICE_COLORS[d] for d in DEVICES]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colColours=col_colors,
        rowColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    ax.set_title("Grand Summary: Key Metrics per Device", fontsize=12, fontweight="bold", pad=12)

    out_path = os.path.join(out_dir, "plot_summary_table.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grand_analysis] Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot E: ERD/ERS grand average
# ---------------------------------------------------------------------------

def plot_erd_ers_grand(data: dict, out_dir: str):
    """
    3-panel grouped bar chart: normalised band power (session/baseline) per task condition.
    X-axis = task, grouped bars per task coloured by device. One panel per band (Theta/Alpha/Beta).
    Values < 1 = ERD (desync), > 1 = ERS (sync).
    """
    norm_bands = ["Theta", "Alpha", "Beta"]

    # Collect all task conditions
    all_conds = set()
    for pid in data:
        for device in DEVICES:
            bp = data[pid][device]["band_power"]
            if bp is None:
                continue
            for cond in bp["condition"].unique():
                if str(cond).startswith("task_"):
                    all_conds.add(str(cond))
    task_conds = sorted(all_conds)

    if not task_conds:
        print("[grand_analysis] WARNING: No task conditions found for ERD/ERS plot.")
        return

    n_tasks   = len(task_conds)
    n_devices = len(DEVICES)
    width     = 0.8 / n_devices
    offsets   = np.linspace(-(n_devices - 1) * width / 2,
                             (n_devices - 1) * width / 2, n_devices)
    x = np.arange(n_tasks)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle("ERD/ERS per Task — TP_pool (Normalised to Baseline)", fontsize=12, fontweight="bold")

    for ax, band in zip(axes, norm_bands):
        col = f"{band}_norm"
        for i, device in enumerate(DEVICES):
            means, sems = [], []
            for cond in task_conds:
                vals = []
                for pid in data:
                    bp = data[pid][device]["band_power"]
                    if bp is None:
                        continue
                    tp_rows = bp[(bp["channel"] == "TP_pool") & (bp["condition"] == cond)]
                    if tp_rows.empty or col not in tp_rows.columns:
                        continue
                    v = tp_rows[col].mean()
                    if not np.isnan(v):
                        vals.append(v)
                means.append(safe_mean(vals))
                sems.append(safe_sem(vals))

            ax.bar(x + offsets[i], means, width,
                   label=DEVICE_LABELS[device],
                   color=DEVICE_COLORS[device],
                   yerr=sems, capsize=3,
                   error_kw={"elinewidth": 1.2})

        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("task_", "T") for c in task_conds])
        ax.set_xlabel("Task")
        ax.set_ylabel("Normalised Power (session / baseline)")
        ax.set_title(f"{band} Band")
        if ax is axes[0]:
            ax.legend(fontsize=8)

    axes[0].text(0.01, 0.02, "< 1.0 = ERD   > 1.0 = ERS",
                 transform=axes[0].transAxes, fontsize=7, color="gray")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "plot_erd_ers_grand.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grand_analysis] Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot F: GSR grand average — normalised time line graph
# ---------------------------------------------------------------------------

def plot_gsr_grand(data: dict, out_dir: str):
    """Interpolate each participant's GSR to 1000 common time points (0–1), then average."""
    N_POINTS = 1000
    common_t = np.linspace(0, 1, N_POINTS)

    fig, ax = plt.subplots(figsize=(11, 5))

    for device in DEVICES:
        traces = []
        for pid in data:
            gsr_df = data[pid][device]["gsr"]
            if gsr_df is None or gsr_df.empty:
                continue
            times = gsr_df["time"].values.astype(float)
            vals  = gsr_df["gsr_uS"].values.astype(float)
            if times.max() == 0:
                continue
            t_norm = times / times.max()
            # interpolate
            try:
                f = interpolate.interp1d(t_norm, vals, kind="linear",
                                         bounds_error=False, fill_value=(vals[0], vals[-1]))
                traces.append(f(common_t))
            except Exception:
                continue

        if not traces:
            continue
        arr = np.array(traces)  # (n_participants, N_POINTS)
        mean_trace = arr.mean(axis=0)
        sem_trace  = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(N_POINTS)

        ax.plot(common_t * 100, mean_trace, label=DEVICE_LABELS[device],
                color=DEVICE_COLORS[device], linewidth=2)
        ax.fill_between(common_t * 100,
                        mean_trace - sem_trace,
                        mean_trace + sem_trace,
                        color=DEVICE_COLORS[device], alpha=0.2)

    ax.set_xlabel("Session Progress (%)")
    ax.set_ylabel("GSR (µS)")
    ax.set_title("Grand Average GSR Timeseries (Normalised Time)", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "plot_gsr_grand.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grand_analysis] Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot G: NASA TLX
# ---------------------------------------------------------------------------

def plot_nasa_tlx(data: dict, out_dir: str):
    """Grouped bar chart: NASA TLX subscales per device."""
    subscales = ["Mental Demand", "Physical Demand", "Temporal Demand",
                 "Performance", "Effort", "Frustration", "Average"]

    # collect per-participant scores
    device_scores = {d: {s: [] for s in subscales} for d in DEVICES}
    for pid in data:
        for device in DEVICES:
            tlx = data[pid][device]["nasa_tlx"]
            if tlx is None:
                continue
            scores = tlx.get("scores", {})
            for s in subscales[:-1]:
                v = scores.get(s)
                if v is not None:
                    device_scores[device][s].append(v)
            avg = tlx.get("average")
            if avg is not None:
                device_scores[device]["Average"].append(avg)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(subscales))
    width = 0.25
    offsets = np.linspace(-(len(DEVICES)-1)*width/2, (len(DEVICES)-1)*width/2, len(DEVICES))

    for i, device in enumerate(DEVICES):
        means = [safe_mean(device_scores[device][s]) for s in subscales]
        sems  = [safe_sem(device_scores[device][s])  for s in subscales]
        ax.bar(x + offsets[i], means, width, label=DEVICE_LABELS[device],
               color=DEVICE_COLORS[device], yerr=sems, capsize=4,
               error_kw={"elinewidth": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(subscales, rotation=20, ha="right")
    ax.set_ylabel("NASA TLX Score (0–100)")
    ax.set_title("NASA TLX Workload Scores per Device", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "plot_nasa_tlx.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grand_analysis] Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot H: Task-onset ERP peak amplitude (TP_pool, 250–600 ms)
# ---------------------------------------------------------------------------

def plot_p300_grand(data: dict, out_dir: str):
    """Bar chart: mean task-onset ERP peak amplitude (TP_pool, 250–600 ms) per device."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Task-Onset ERP Peak Amplitude per Device\n(TP_pool, 250–600 ms)",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(DEVICES))
    means, sems, all_pts = [], [], []
    for device in DEVICES:
        vals = []
        for pid in data:
            ss = data[pid][device]["session_summary"]
            if ss is None:
                continue
            task_vals = [v for v in ss.get("task_onset_erp_peak_by_condition", {}).values()
                         if v is not None]
            if task_vals:
                vals.append(np.mean(task_vals))
        means.append(safe_mean(vals))
        sems.append(safe_sem(vals))
        all_pts.append(vals)

    ax.bar(x, means, yerr=sems, capsize=5,
           color=[DEVICE_COLORS[d] for d in DEVICES],
           error_kw={"elinewidth": 1.5}, zorder=2)
    for i, pts in enumerate(all_pts):
        jitter = np.random.uniform(-0.08, 0.08, len(pts))
        ax.scatter(i + jitter, pts, color="black", s=25, zorder=3, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([DEVICE_LABELS[d] for d in DEVICES])
    ax.set_ylabel("Mean Amplitude (µV)")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "plot_p300_grand.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grand_analysis] Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot I: Heart rate & HRV
# ---------------------------------------------------------------------------

def _compute_rmssd(rr_series: pd.Series) -> float:
    """Compute RMSSD from a column of possibly semicolon-separated RR strings."""
    all_rr = []
    for v in rr_series.dropna():
        parts = str(v).split(";")
        for p in parts:
            try:
                all_rr.append(float(p.strip()))
            except ValueError:
                pass
    if len(all_rr) < 2:
        return np.nan
    diffs = np.diff(all_rr)
    return float(np.sqrt(np.mean(diffs ** 2)))


def plot_hr_grand(data: dict, out_dir: str):
    """Two-panel: mean HR and RMSSD per device with participant scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Heart Rate & HRV Grand Average per Device", fontsize=13, fontweight="bold")

    for ax, metric, ylabel, title in [
        (ax1, "hr",    "Heart Rate (bpm)",  "Mean Heart Rate"),
        (ax2, "rmssd", "RMSSD (ms)",        "HRV — RMSSD"),
    ]:
        x = np.arange(len(DEVICES))
        means, sems, all_pts = [], [], []
        for device in DEVICES:
            vals = []
            for pid in data:
                hr_df = data[pid][device]["hr"]
                if hr_df is None or hr_df.empty:
                    continue
                if metric == "hr":
                    vals.append(hr_df["heart_rate"].mean())
                else:
                    rmssd = _compute_rmssd(hr_df["rr_intervals"])
                    if not np.isnan(rmssd):
                        vals.append(rmssd)
            means.append(safe_mean(vals))
            sems.append(safe_sem(vals))
            all_pts.append(vals)

        ax.bar(x, means, yerr=sems, capsize=5,
               color=[DEVICE_COLORS[d] for d in DEVICES],
               error_kw={"elinewidth": 1.5}, zorder=2)
        # scatter individual participants
        for i, pts in enumerate(all_pts):
            jitter = np.random.uniform(-0.08, 0.08, len(pts))
            ax.scatter(i + jitter, pts, color="black", s=20, zorder=3, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([DEVICE_LABELS[d] for d in DEVICES])
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "plot_hr_grand.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grand_analysis] Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot J: Theta/Alpha ratio grand average
# ---------------------------------------------------------------------------

def plot_theta_alpha_grand(data: dict, out_dir: str):
    """Line plot: theta/alpha ratio per task condition, one line per device."""
    # Collect all task conditions present across all participants/devices
    all_conds = set()
    for pid in data:
        for device in DEVICES:
            ss = data[pid][device]["session_summary"]
            if ss is None:
                continue
            all_conds.update(ss.get("theta_alpha_ratio_by_condition", {}).keys())
    task_conds = sorted(c for c in all_conds if c.startswith("task_"))

    if not task_conds:
        print("[grand_analysis] WARNING: No task conditions found for theta/alpha plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(task_conds))

    for device in DEVICES:
        means, sems = [], []
        for cond in task_conds:
            vals = []
            for pid in data:
                ss = data[pid][device]["session_summary"]
                if ss is None:
                    continue
                tar = ss.get("theta_alpha_ratio_by_condition", {})
                v = tar.get(cond)
                if v is not None:
                    vals.append(v)
            means.append(safe_mean(vals))
            sems.append(safe_sem(vals))

        means = np.array(means, dtype=float)
        sems  = np.array(sems,  dtype=float)
        ax.plot(x, means, marker="o", lw=2, color=DEVICE_COLORS[device],
                label=DEVICE_LABELS[device])
        ax.fill_between(x, means - sems, means + sems,
                        color=DEVICE_COLORS[device], alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("task_", "Task ") for c in task_conds])
    ax.set_xlabel("Task")
    ax.set_ylabel("Theta / Alpha Ratio")
    ax.set_title("Theta/Alpha Ratio per Task — Grand Average by Device\n(Higher = Greater Cognitive Load)",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "plot_theta_alpha_grand.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grand_analysis] Saved {out_path}")




# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plot M: Oddball detection accuracy
# ---------------------------------------------------------------------------

def plot_detection_accuracy(data: dict, out_dir: str):
    """Bar chart: detected / actual red circles (%) per device."""
    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(DEVICES))
    means, sems, all_pts = [], [], []
    for device in DEVICES:
        vals = []
        for pid in data:
            tlx = data[pid][device]["nasa_tlx"]
            if tlx is None:
                continue
            detected = tlx.get("red_circle_count")
            actual   = tlx.get("actual_red_circle_count")
            if detected is not None and actual and actual > 0:
                vals.append(detected / actual * 100)
        means.append(safe_mean(vals))
        sems.append(safe_sem(vals))
        all_pts.append(vals)

    ax.bar(x, means, yerr=sems, capsize=5,
           color=[DEVICE_COLORS[d] for d in DEVICES],
           error_kw={"elinewidth": 1.5}, zorder=2)
    for i, pts in enumerate(all_pts):
        jitter = np.random.uniform(-0.08, 0.08, len(pts))
        ax.scatter(i + jitter, pts, color="black", s=25, zorder=3, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([DEVICE_LABELS[d] for d in DEVICES])
    ax.set_ylabel("Detection Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.axhline(100, color="green", linewidth=1, linestyle="--", label="Perfect (100%)")
    ax.set_title("Oddball Detection Accuracy per Device", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "plot_detection_accuracy.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grand_analysis] Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir   = "data"
    output_dir = "output"
    out_dir    = os.path.join(output_dir, "grand")
    os.makedirs(out_dir, exist_ok=True)

    print("[grand_analysis] Loading data...")
    data = load_all_data(data_dir, output_dir)

    if not data:
        print("[grand_analysis] No data found. Exiting.")
        return

    print("[grand_analysis] Generating plots...")
    plot_band_power_avg(data, out_dir)
    plot_band_power_diff(data, out_dir)
    plot_knob_rotation(data, out_dir)
    plot_summary_table(data, out_dir)
    plot_erd_ers_grand(data, out_dir)
    plot_gsr_grand(data, out_dir)
    plot_nasa_tlx(data, out_dir)
    plot_p300_grand(data, out_dir)
    plot_hr_grand(data, out_dir)
    plot_theta_alpha_grand(data, out_dir)

    plot_detection_accuracy(data, out_dir)

    print(f"\n[grand_analysis] Done. All outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
