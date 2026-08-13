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
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
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


def _observed_series(
    df: pd.DataFrame,
    value_col: str,
    timestamp_col: str = "timestamp_utc",
):
    """
    Sorted (values, times) for the rows that actually carry an observation.

    Rows with a NaN value are dropped -- NaN is already the correct "not
    measured" encoding -- as are rows whose timestamp will not parse. Returns
    two aligned numpy arrays, empty when there is nothing observed.
    """
    if df.empty:
        return np.empty(0), np.empty(0, dtype="datetime64[ns]")

    d = df[[timestamp_col, value_col]].dropna(subset=[value_col]).sort_values(timestamp_col)
    if d.empty:
        return np.empty(0), np.empty(0, dtype="datetime64[ns]")

    values = d[value_col].to_numpy()
    parsed_times = pd.to_datetime(d[timestamp_col], format="mixed", errors="coerce")
    valid_time = parsed_times.notna()
    values = values[valid_time.to_numpy()]
    times = parsed_times.loc[valid_time].to_numpy()
    return values, times


def infer_cadence(times) -> Optional[pd.Timedelta]:
    """
    Median positive step between consecutive observations.

    "Contiguous" is about observation cadence, not merely adjacency after
    sorting, so every run- and gap-detector here needs the same notion of how
    far apart two neighbouring observations are *supposed* to be. Returns None
    when there are fewer than two distinct timestamps.
    """
    if len(times) < 2:
        return None
    positive_steps = pd.Series(times).diff().dropna()
    positive_steps = positive_steps[positive_steps > pd.Timedelta(0)]
    if positive_steps.empty:
        return None
    return positive_steps.median()


def _scan_constant_runs(values, times, max_contiguous_step):
    """
    Yield every maximal contiguous run of a bit-identical value (length >= 2).

    A run is terminated by a different value or by a step larger than
    max_contiguous_step -- without that boundary, two ordinary solar nights
    separated by a missing daytime block become one apparently multi-day zero
    run.
    """
    n = len(values)
    run_start_idx = 0
    for i in range(1, n + 1):
        gap_break = (
            i < n and max_contiguous_step is not None
            and pd.Timestamp(times[i]) - pd.Timestamp(times[i - 1]) > max_contiguous_step
        )
        if i == n or values[i] != values[run_start_idx] or gap_break:
            run_len = i - run_start_idx
            if run_len >= 2:
                start_t = pd.Timestamp(times[run_start_idx])
                end_t = pd.Timestamp(times[i - 1])
                yield SuspectConstantRun(
                    start=start_t,
                    end=end_t,
                    value=float(values[run_start_idx]),
                    n_rows=run_len,
                    duration_hours=(end_t - start_t).total_seconds() / 3600.0,
                )
            run_start_idx = i


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
    values, times = _observed_series(df, value_col, timestamp_col)
    if len(times) == 0:
        return []

    max_contiguous_step = infer_cadence(times)
    if max_contiguous_step is not None:
        max_contiguous_step = max_contiguous_step * 1.5

    return [
        run for run in _scan_constant_runs(values, times, max_contiguous_step)
        if run.duration_hours >= min_run_hours
    ]


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


# ---------------------------------------------------------------------------
# Availability census (ABL-318)
#
# exclude_suspect_constant_runs above answers "may this row enter training?".
# The census below answers the prior question ABL-316 needs per country and
# stream: "is there anything here worth training on at all?" -- same run
# detector, same NULL-is-not-0 rule, reported instead of applied.
# ---------------------------------------------------------------------------


@dataclass
class SeriesGap:
    """A break between consecutive observations wider than the series cadence."""
    start: pd.Timestamp   # last observation before the gap
    end: pd.Timestamp     # first observation after the gap
    duration_hours: float


@dataclass
class StreamQuality:
    """Availability and contamination census for one country/stream/table."""
    source_table: str
    n_rows: int                       # rows present for this country in the table
    n_nonnull: int                    # rows carrying an observation (NULL != 0)
    first_ts: Optional[pd.Timestamp]  # first observed (non-null) timestamp
    last_ts: Optional[pd.Timestamp]   # last observed (non-null) timestamp
    cadence_minutes: Optional[float]
    n_exact_zero: int
    pct_exact_zero: float             # of non-null observations
    max_value: Optional[float]
    longest_zero_run_hours: float
    longest_zero_run: Optional[SuspectConstantRun]
    longest_gap_hours: float
    longest_gap: Optional[SeriesGap]
    suspect_runs: List[SuspectConstantRun] = field(default_factory=list)
    suspect_rows: int = 0
    suspect_hours: float = 0.0

    @property
    def span_days(self) -> float:
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        return (self.last_ts - self.first_ts).total_seconds() / 86400.0

    @property
    def all_zero(self) -> bool:
        """Reported, but never once non-zero -- the signature of no fleet."""
        return self.n_nonnull > 0 and self.n_exact_zero == self.n_nonnull


def find_gaps(
    df: pd.DataFrame,
    value_col: str,
    timestamp_col: str = "timestamp_utc",
    min_gap_hours: float = 3.0,
) -> List[SeriesGap]:
    """
    Breaks between consecutive *observations* longer than min_gap_hours.

    Deliberately computed over non-null values, not over rows: a block of rows
    whose value_col is NULL is just as absent from training as a block of
    missing rows, and the FR 2023 renewable outage shows up as both depending
    on the table you ask.
    """
    _, times = _observed_series(df, value_col, timestamp_col)
    if len(times) < 2:
        return []

    steps = pd.Series(times).diff()
    gaps: List[SeriesGap] = []
    for i, step in enumerate(steps):
        if pd.isna(step):
            continue
        hours = step.total_seconds() / 3600.0
        if hours > min_gap_hours:
            gaps.append(SeriesGap(
                start=pd.Timestamp(times[i - 1]),
                end=pd.Timestamp(times[i]),
                duration_hours=hours,
            ))
    return gaps


def summarize_stream(
    df: pd.DataFrame,
    value_col: str,
    source_table: str,
    timestamp_col: str = "timestamp_utc",
    min_run_hours: float = 24.0,
    min_gap_hours: float = 3.0,
) -> StreamQuality:
    """
    Census one country/stream series without modifying it.

    Every count separates NULL from 0.0: n_nonnull counts observations,
    n_exact_zero counts observations that are exactly 0.0. A stream with
    n_nonnull == 0 was never reported; a stream with n_exact_zero == n_nonnull
    was reported and is flat -- those are different verdicts and the caller
    needs to tell them apart.
    """
    values, times = _observed_series(df, value_col, timestamp_col)
    n_rows = len(df)

    if len(times) == 0:
        return StreamQuality(
            source_table=source_table, n_rows=n_rows, n_nonnull=0,
            first_ts=None, last_ts=None, cadence_minutes=None,
            n_exact_zero=0, pct_exact_zero=0.0, max_value=None,
            longest_zero_run_hours=0.0, longest_zero_run=None,
            longest_gap_hours=0.0, longest_gap=None,
        )

    cadence = infer_cadence(times)
    max_contiguous_step = cadence * 1.5 if cadence is not None else None

    zero_runs = [
        run for run in _scan_constant_runs(values, times, max_contiguous_step)
        if run.value == 0.0
    ]
    longest_zero = max(zero_runs, key=lambda r: r.duration_hours) if zero_runs else None

    suspect = find_suspect_constant_runs(df, value_col, timestamp_col, min_run_hours)
    gaps = find_gaps(df, value_col, timestamp_col, min_gap_hours)
    longest_gap = max(gaps, key=lambda g: g.duration_hours) if gaps else None

    n_zero = int((values == 0.0).sum())
    return StreamQuality(
        source_table=source_table,
        n_rows=n_rows,
        n_nonnull=len(values),
        first_ts=pd.Timestamp(times[0]),
        last_ts=pd.Timestamp(times[-1]),
        cadence_minutes=(cadence.total_seconds() / 60.0) if cadence is not None else None,
        n_exact_zero=n_zero,
        pct_exact_zero=100.0 * n_zero / len(values),
        max_value=float(np.nanmax(values)),
        longest_zero_run_hours=longest_zero.duration_hours if longest_zero else 0.0,
        longest_zero_run=longest_zero,
        longest_gap_hours=longest_gap.duration_hours if longest_gap else 0.0,
        longest_gap=longest_gap,
        suspect_runs=suspect,
        suspect_rows=sum(r.n_rows for r in suspect),
        suspect_hours=sum(r.duration_hours for r in suspect),
    )
