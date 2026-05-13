"""gsr_io.py — Load Mindfield eSense iPad CSV exports.

The iPad app exports CSV files with ~24 lines of metadata, a questionnaire
section, then a `STATISTICS` header, then a `;`-delimited time-series block
sampled at 5 Hz. This module hides that format.

A single iPad recording spans the entire participant visit (started before
the GUI runs, stopped after it). For each device session, the iPad rows are
sliced to that session's window using `session_start_unix` /
`session_end_unix` from the session's `_markers.json`, and the resulting
DataFrame's `time` column is re-anchored so `t=0` is the session start.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd


# Substrings unique to the iPad export — used by `is_ipad_export` to decide
# whether a CSV came from the iPad app or from some other source.
IPAD_HEADER_MARKERS = ("RECORDING NAME", "DESCRIPTION;VALUE", "STATISTICS")

# The exact column-header line that precedes the time-series block.
IPAD_DATA_HEADER = "SECOND;MICROSIEMENS;TIMESTAMP;SCR;SCR/MIN;MARKER"


def is_ipad_export(path: str) -> bool:
    """True if the CSV's first 2 KB matches the iPad eSense export signature."""
    try:
        with open(path, encoding="utf-8") as f:
            head = f.read(2048)
    except OSError:
        return False
    return any(m in head for m in IPAD_HEADER_MARKERS)


def _parse_ipad_metadata(path: str) -> Tuple[dict, int]:
    """Read the metadata block above the time-series.

    Returns (metadata_dict, data_header_line_index). The data header line
    index is 0-based and points at the `SECOND;MICROSIEMENS;...` row, so a
    subsequent `pd.read_csv(skiprows=index)` lands on it.
    """
    meta: dict = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if line.startswith(IPAD_DATA_HEADER):
                return meta, i
            if ";" in line:
                key, _, val = line.partition(";")
                key = key.strip()
                val = val.split(";")[0].strip()
                if key and val:
                    meta[key] = val
    raise ValueError(f"No `{IPAD_DATA_HEADER}` row found in {path}")


def _parse_ipad_start_unix(meta: dict) -> Optional[float]:
    """Convert the `DATE (HH:MM:SS)` metadata field to a Unix timestamp.

    The iPad export formats the field as DD.MM.YY HH:MM:SS, e.g.
    `05.05.26 08:54:08`. Returns None if the field is missing or unparseable.
    """
    date_str = meta.get("DATE (HH:MM:SS)")
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%d.%m.%y %H:%M:%S").timestamp()
    except ValueError:
        return None


def _read_ipad_raw(path: str) -> Tuple[pd.DataFrame, dict]:
    """Parse the full iPad export into a DataFrame with absolute Unix time.

    Returns (df, metadata):
      df columns:
        unix_time    Unix timestamp of the sample (ipad_start + SECOND)
        gsr_uS       skin conductance in microsiemens
        scr          cumulative SCR count from the iPad's onboard detector
        scr_per_min  the iPad's rolling SCR rate
    """
    meta, data_header_idx = _parse_ipad_metadata(path)
    df = pd.read_csv(
        path,
        sep=";",
        skiprows=data_header_idx,
        usecols=range(6),
        engine="python",
    )
    df.columns = ["second", "gsr_uS", "timestamp", "scr", "scr_per_min", "marker"]

    df = df[df["timestamp"] != "Paused"].copy()
    for col in ("second", "gsr_uS", "scr", "scr_per_min"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["second", "gsr_uS"]).reset_index(drop=True)

    ipad_start = _parse_ipad_start_unix(meta)
    if ipad_start is None:
        raise ValueError(
            f"Could not parse iPad start time from {path}. "
            "The `DATE (HH:MM:SS)` metadata field is required to anchor GSR "
            "samples to study time."
        )
    df["unix_time"] = df["second"] + ipad_start
    return df[["unix_time", "gsr_uS", "scr", "scr_per_min"]], meta


def load_session_window(markers_json_path: str) -> Tuple[float, float]:
    """Read `session_start_unix` and `session_end_unix` from a session's markers JSON.

    Raises FileNotFoundError if the file is missing and ValueError if
    `session_start_unix` is absent. If `session_end_unix` is absent (older
    recordings of the final device session, before that bug was fixed),
    fall back to start + last task_end's session-relative time.
    """
    if not os.path.exists(markers_json_path):
        raise FileNotFoundError(markers_json_path)
    with open(markers_json_path) as f:
        m = json.load(f)
    start = m.get("session_start_unix")
    if start is None:
        raise ValueError(
            f"{markers_json_path} is missing session_start_unix — "
            "re-run the experiment so the GUI writes it."
        )
    end = m.get("session_end_unix")
    if end is None:
        markers = m.get("markers") or []
        rel_times = [mk.get("time") for mk in markers
                     if isinstance(mk.get("time"), (int, float))]
        if not rel_times:
            raise ValueError(
                f"{markers_json_path} has no session_end_unix and no markers "
                "to derive it from."
            )
        end = float(start) + max(rel_times)
    return float(start), float(end)


def participant_gsr_path(participant_dir: str, participant_id: str) -> str:
    """Canonical path of the single per-participant GSR CSV."""
    return os.path.join(participant_dir, f"{participant_id}_gsr.csv")


def load_session_gsr(participant_gsr_csv: str,
                     markers_json_path: str) -> pd.DataFrame:
    """Slice the participant-level iPad GSR CSV to a single session's window.

    Returns a DataFrame with columns (time, gsr_uS, scr, scr_per_min) where
    `time` is seconds since `session_start_unix`. Empty DataFrame if no iPad
    samples fall inside the window.

    Raises:
        FileNotFoundError  if the iPad CSV or markers JSON is missing
        ValueError         if the file isn't an iPad export, lacks a parseable
                           start time, or the markers JSON has no window
    """
    if not os.path.exists(participant_gsr_csv):
        raise FileNotFoundError(participant_gsr_csv)
    if not is_ipad_export(participant_gsr_csv):
        raise ValueError(f"{participant_gsr_csv} is not an iPad eSense export")

    start_unix, end_unix = load_session_window(markers_json_path)
    raw, _meta = _read_ipad_raw(participant_gsr_csv)

    sl = raw[(raw["unix_time"] >= start_unix) & (raw["unix_time"] <= end_unix)].copy()
    sl["time"] = sl["unix_time"] - start_unix
    return sl[["time", "gsr_uS", "scr", "scr_per_min"]].reset_index(drop=True)
