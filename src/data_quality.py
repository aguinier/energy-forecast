"""
Training-data invariants for energy_renewable-sourced targets (ABL-188).

energy_renewable's per-column mapper (_map_renewable_columns in
energy-data-gathering/src/entsoe_client.py) initialises every renewable
column to 0.0 before checking whether ENTSO-E's response actually contained
that production type, and its .fillna(0) calls give it no way to say
"unknown" -- unlike energy_generation's NaN-preserving twin mapper. ABL-188
found this let DE's real solar_mw generation (proven present and non-zero in
energy_generation.solar_mw for the same fetch) get silently zero-filled in
energy_renewable.solar_mw for 6,408 consecutive quarter-hours (2025-09-08
22:00 through 2025-11-14 15:45 UTC). energy_renewable is frozen and cannot
be re-derived (CLAUDE.md), so this module guards the training-data boundary
instead: it is called just after energy_renewable is read and before a
renewable target is resampled/trained on.
"""

import logging
from dataclasses import dataclass
from typing import List

import pandas as pd

logger = logging.getLogger('energy_forecast')


@dataclass
class SuspectConstantRun:
    """A maximal contiguous run where a target column held one exact value
    for longer than any real, continuously weather- or dispatch-driven
    generation series plausibly would."""
    start: pd.Timestamp
    end: pd.Timestamp
    value: float
    n_rows: int
    duration_hours: float


def find_suspect_constant_runs(
    df: pd.DataFrame,
    value_col: str,
    timestamp_col: str = "timestamp_utc",
    min_run_hours: float = 24.0,
) -> List[SuspectConstantRun]:
    """
    Find maximal contiguous runs where value_col holds a bit-identical value
    for at least min_run_hours.

    A live meter or weather-driven dispatch series essentially never repeats
    the exact same float for a full day straight -- a run this long (0.0
    included) is the signature of a missing/unavailable source defaulted to
    a constant, not a measurement. Rows are sorted by timestamp and rows with
    a NaN value are dropped first; NaN is already the correct "not measured"
    encoding and needs no flagging.
    """
    if df.empty:
        return []

    d = df[[timestamp_col, value_col]].dropna(subset=[value_col]).sort_values(timestamp_col)
    if d.empty:
        return []

    values = d[value_col].to_numpy()
    times = d[timestamp_col].to_numpy()

    runs: List[SuspectConstantRun] = []
    run_start_idx = 0
    n = len(values)
    for i in range(1, n + 1):
        if i == n or values[i] != values[run_start_idx]:
            run_len = i - run_start_idx
            if run_len >= 2:
                start_t = pd.Timestamp(times[run_start_idx])
                end_t = pd.Timestamp(times[i - 1])
                duration_hours = (end_t - start_t).total_seconds() / 3600.0
                if duration_hours >= min_run_hours:
                    runs.append(SuspectConstantRun(
                        start=start_t,
                        end=end_t,
                        value=float(values[run_start_idx]),
                        n_rows=run_len,
                        duration_hours=duration_hours,
                    ))
            run_start_idx = i

    return runs


def exclude_suspect_constant_runs(
    df: pd.DataFrame,
    value_col: str,
    timestamp_col: str = "timestamp_utc",
    min_run_hours: float = 24.0,
    context: str = "",
) -> pd.DataFrame:
    """
    Null out value_col for rows inside any suspect constant run so they read
    as not-measured (matching this codebase's NULL-is-not-0 rule) instead of
    silently entering training as if they were a real, corroborated value.

    Returns a copy; the caller's existing dropna() (load_training_data
    resamples to hourly, then drops NaN rows) removes the nulled rows same
    as any other genuinely-missing interval -- no separate exclusion path is
    needed downstream.
    """
    runs = find_suspect_constant_runs(df, value_col, timestamp_col, min_run_hours)
    if not runs:
        return df

    d = df.copy()
    mask = pd.Series(False, index=d.index)
    for run in runs:
        mask |= (d[timestamp_col] >= run.start) & (d[timestamp_col] <= run.end)

    for run in runs:
        logger.warning(
            "training-data invariant (ABL-188): excluding suspect constant "
            "run %s=%.6g from %s to %s (%d rows, %.1fh) -- held one exact "
            "value too long to be a real measurement; treated as "
            "unadjudicated-missing, not zero.%s",
            value_col, run.value, run.start, run.end, run.n_rows,
            run.duration_hours, f" [{context}]" if context else "",
        )

    d.loc[mask, value_col] = float("nan")
    return d
