"""gsr_io.py — Load Mindfield eSense iPad CSV exports.

The iPad app exports CSV files with ~24 lines of metadata, a questionnaire
section, then a `STATISTICS` header, then a `;`-delimited time-series block
sampled at 5 Hz. This module hides that format and returns a clean DataFrame
with columns (time, gsr_uS, scr, scr_per_min) so that downstream analysis
code can treat the iPad CSV like any other physiological time-series.

The iPad clock and the host clock have independent origins, so when a
markers.json (produced by main.py at session start) is available the iPad
SECOND axis is re-anchored to study time using the wall-clock difference
between the iPad's recording-start datetime and `session_start_unix`.
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
                # Trailing empty fields appear because the CSV has 6 columns
                # but most metadata rows only fill the first two.
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


def load_ipad_gsr_csv(path: str,
                      session_start_unix: Optional[float] = None
                      ) -> Tuple[pd.DataFrame, dict]:
    """Read an iPad eSense export.

    Returns (df, metadata):
      df columns:
        time         seconds, optionally re-anchored to study time
        gsr_uS       skin conductance in microsiemens
        scr          cumulative SCR count from the iPad's onboard detector
        scr_per_min  the iPad's rolling SCR rate

    If `session_start_unix` is given, alignment is computed as
    `time = SECOND + (ipad_start_unix − session_start_unix)`. Otherwise the
    iPad's relative SECOND axis is returned directly.
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

    # Drop "Paused" rows (no real measurement) and any non-numeric debris.
    df = df[df["timestamp"] != "Paused"].copy()
    for col in ("second", "gsr_uS", "scr", "scr_per_min"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["second", "gsr_uS"]).reset_index(drop=True)

    ipad_start = _parse_ipad_start_unix(meta)
    if session_start_unix is not None and ipad_start is not None:
        offset = ipad_start - session_start_unix
        df["time"] = df["second"] + offset
    else:
        if session_start_unix is not None:
            print(f"[gsr_io] WARNING: could not parse iPad start time from {path}; "
                  f"GSR will not be re-anchored to study time.")
        df["time"] = df["second"]

    return df[["time", "gsr_uS", "scr", "scr_per_min"]], meta


def load_session_start_unix(participant_dir: str, basename: str) -> Optional[float]:
    """Read `session_start_unix` from `<participant_dir>/<basename>_markers.json`.

    Returns None if the markers file is missing or has no anchor.
    """
    markers_path = os.path.join(participant_dir, f"{basename}_markers.json")
    if not os.path.exists(markers_path):
        return None
    try:
        with open(markers_path) as f:
            v = json.load(f).get("session_start_unix")
    except (json.JSONDecodeError, OSError):
        return None
    return float(v) if v else None
