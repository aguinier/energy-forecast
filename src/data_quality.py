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
# Cross-table zero disproof (ABL-200)
#
# exclude_suspect_constant_runs above adjudicates a zero by asking how LONG it
# lasted. That question has no good answer for wind: BE's wind_offshore_mw
# carries 105 flat-zero runs of 6h or longer (re-measured on the replica
# 2026-08-14) and only 9 of them reach the 24h default, so the threshold either
# misses the rest or, lowered, starts deleting the genuine calm and curtailment
# spells a duration test cannot tell them apart from.
#
# The rule below asks a different question, the one ABL-42 settled the formally
# identical MD hydro case with: not "is this zero long enough to be suspicious?"
# but "is this zero DISPROVED?". `energy_generation` is the NaN-preserving twin
# of the same fetch, so a 0.0 in `energy_renewable` is disproved when the twin
# reports real generation at the identical instant -- at any run length, with no
# duration argument at all.
#
# Three measured facts shape it, and each is a way the naive form is wrong.
#
#   1. The disproof must be ONE-SIDED. `energy_generation` is signed
#      net-of-consumption (A75); `energy_renewable` is floored at zero and holds
#      no negative value in any of the 120 country/stream pairs. So a BE
#      wind_offshore instant reading 0.0 against a sibling of -25 MW is not a
#      hidden 25 MW, it is an idle farm drawing house load, and the 0.0 is
#      right. 2,158 of BE wind_offshore's 2,214 exact zeros are of exactly that
#      shape. A magnitude test on |sibling| would null all of them.
#
#   2. The two tables ALSO disagree by revision vintage, in both directions, and
#      that disagreement is enormous on some pairs and literally zero on others.
#      Measured over instants where `energy_renewable` is positive: 32 of 100
#      comparable pairs agree bit-for-bit at least 99% of the time, while NL
#      wind_onshore has `energy_generation` HIGHER at 83.5% of instants (median
#      +311.8 MW, the ABL-439 vintage seam). A fixed MW floor, or a floor keyed
#      to fleet size, cannot serve both.
#
#   3. There is NO empty band in the disproof magnitudes to hang a global
#      threshold on. Sibling value over fleet p99.5, across all 18,900 raw
#      candidates, runs continuously across four decades: q05 0.000065, q50
#      0.000478, q95 0.1395, max 1.018, with no gap wider than 4x anywhere and
#      that one down at a ratio of 1e-7. This is the opposite of ABL-431's
#      tolerance, which sits inside a measured empty band 2.3x wide, and it is
#      why a fleet-scaled floor here would be a convention wearing a
#      measurement's clothes.
#
# So the floor is calibrated PER PAIR against fact 2: it is a high quantile of
# |renewable - generation| over the instants where the renewable side is
# positive -- how much these two tables routinely disagree with each other on
# this very series, when the zero-fill defect provably is not what is being
# looked at. A sibling value below that is not evidence of anything; a sibling
# value above it is a disagreement larger than the pair has ever normally shown.
# Bit-identical pairs get a floor of exactly 0.0 and any positive sibling
# disproves; vintage-divergent pairs set their own high bar automatically and
# the rule falls quiet on them, which is the correct outcome for a pair whose
# two tables are known to hold different vintages.
#
# Nothing is repaired. A disproved zero becomes NaN -- unadjudicated-missing,
# the same encoding exclude_suspect_constant_runs uses and the same encoding
# load_renewable_type_data already gives two duplicate spellings that disagree.
# The sibling's value is NOT written in: the twin holds a different vintage and
# a possibly different netting convention, so it is good enough to refute a zero
# and not good enough to become the target.
# ---------------------------------------------------------------------------

#: Quantile of the per-pair inter-table disagreement that a sibling value must
#: exceed to disprove a 0.0. Not a knife edge: over all pairs the rule nulls
#: 896 / 739 / 564 / 416 rows at q = 0.90 / 0.95 / 0.99 / 1.00, and no
#: acceptance case changes verdict anywhere in that range. 0.99 is registered
#: because the tail this floor exists to exclude is exactly the 1% of
#: instants where one table has been revised and the other has not, and because
#: the conservative direction here is refusing to null a row -- an unexcluded
#: zero-fill is a bounded cost, deleting a real calm period is the failure mode
#: ABL-431 declined to risk in the same situation. q = 1.00 was considered and
#: rejected: one contaminated calibration row would set an unreachable floor and
#: silently disable the rule for that pair for good (DE wind_onshore's largest
#: inter-table disagreement is 21,364 MW).
SIBLING_DISPROOF_QUANTILE = 0.99

#: Below this many calibration instants the floor is not estimated and the rule
#: does not fire, carrying the reason -- ABL-431's `evaluable` pattern, for the
#: same reason: no reference is better than a guessed one. This is where the
#: empty band is. Measured over the 120 pairs, 20 have a calibration population
#: of exactly **0** -- the all-zero series, landlocked countries whose
#: `wind_offshore_mw` is 0.0 forever, which have no positive instant to
#: calibrate on and where an unguarded floor of 0.0 would let any sibling value
#: at all delete a new fleet's first output. The smallest non-zero population is
#: **2,559**. Anything in (0, 2559) is the same rule; 1000 is inside it with
#: room, and puts >= 10 observations in the tail the q0.99 is read from.
SIBLING_DISPROOF_MIN_CALIBRATION_ROWS = 1000


@dataclass
class SiblingDisproof:
    """What a cross-table zero adjudication found for one series."""
    evaluable: bool
    reason: str                        # why not evaluable, or how the floor was set
    floor: Optional[float]             # MW a sibling must exceed to disprove a 0.0
    floor_quantile: float
    calibration_n: int                 # instants the floor was estimated from
    n_zero: int                        # exact 0.0 rows in the series
    n_zero_with_sibling: int           # of those, instants the twin also reports
    n_disproved: int
    max_disproving_value: Optional[float]
    mask: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))


def _collapse_duplicate_instants(
    df: pd.DataFrame, value_col: str, timestamp_col: str
) -> pd.DataFrame:
    """One instant, one value; contradictory spellings become NaN.

    `energy_generation` has no duplicate instants today (`energy_renewable`'s
    78,510 are the ABL-321 finding), so this is a no-op on the sibling in
    practice -- but a disprover that silently picked one of two contradictory
    spellings would decide a training row on row order.
    """
    if not df[timestamp_col].duplicated().any():
        return df
    grouped = df.groupby(timestamp_col)[value_col]
    disagreeing = grouped.nunique(dropna=False) > 1
    collapsed = grouped.last()
    collapsed[disagreeing] = float("nan")
    return collapsed.reset_index()[[timestamp_col, value_col]]


def align_sibling(
    df: pd.DataFrame,
    sibling: pd.DataFrame,
    value_col: str = "target_value",
    timestamp_col: str = "timestamp_utc",
) -> pd.Series:
    """
    The twin table's value at each of `df`'s instants, index-aligned to `df`.

    Aligned on **parsed instants**, never on the stored string. That is not
    defensive: `energy_renewable` stores BE's 2025-11-09 -> 2025-11-25 rows in
    the ISO `2025-11-14T16:00:00` form while `energy_generation` stores every
    row in the `2025-11-14 16:00:00` form, so a SQL join on `timestamp_utc`
    returns NULL for all 540 of them -- including every row of the worked
    example this rule was written for, which would have read as "no sibling,
    nothing to adjudicate" instead of as 424 MW of hidden generation.
    """
    if df.empty or sibling is None or sibling.empty:
        return pd.Series(float("nan"), index=df.index)
    s = sibling[[timestamp_col, value_col]].dropna(subset=[timestamp_col])
    s = _collapse_duplicate_instants(s, value_col, timestamp_col)
    return df[timestamp_col].map(s.set_index(timestamp_col)[value_col])


def sibling_disagreement_floor(
    df: pd.DataFrame,
    sibling_values: pd.Series,
    value_col: str = "target_value",
    quantile: float = SIBLING_DISPROOF_QUANTILE,
) -> tuple:
    """
    (floor, n) -- how far apart these two tables routinely sit on this series.

    Estimated only over instants where `df`'s own value is **strictly
    positive**, so the zero-fill defect being adjudicated cannot raise the bar
    that would catch it, and only where the sibling reports something.
    """
    observed = df[value_col]
    comparable = observed.notna() & (observed > 0) & sibling_values.notna()
    n = int(comparable.sum())
    if n == 0:
        return None, 0
    gap = (observed[comparable] - sibling_values[comparable]).abs()
    return float(gap.quantile(quantile)), n


def adjudicate_zeros_against_sibling(
    df: pd.DataFrame,
    sibling: pd.DataFrame,
    value_col: str = "target_value",
    timestamp_col: str = "timestamp_utc",
    quantile: float = SIBLING_DISPROOF_QUANTILE,
    min_calibration_rows: int = SIBLING_DISPROOF_MIN_CALIBRATION_ROWS,
) -> SiblingDisproof:
    """
    Which exact-0.0 rows of `df` the twin table disproves. Measures; changes
    nothing. `exclude_zeros_disproved_by_sibling` is the applying half, and
    `scripts/abl200_cross_table_zero_census.py` the reporting one.
    """
    empty_mask = pd.Series(False, index=df.index)
    if df.empty or value_col not in df.columns:
        return SiblingDisproof(
            evaluable=False, reason="empty series", floor=None,
            floor_quantile=quantile, calibration_n=0, n_zero=0,
            n_zero_with_sibling=0, n_disproved=0, max_disproving_value=None,
            mask=empty_mask,
        )

    is_zero = df[value_col] == 0.0
    sibling_values = align_sibling(df, sibling, value_col, timestamp_col)
    n_zero = int(is_zero.sum())
    n_zero_with_sibling = int((is_zero & sibling_values.notna()).sum())

    floor, calibration_n = sibling_disagreement_floor(
        df, sibling_values, value_col=value_col, quantile=quantile
    )
    if calibration_n < min_calibration_rows:
        return SiblingDisproof(
            evaluable=False,
            reason=(
                f"only {calibration_n} instants carry a positive value and a "
                f"sibling, below the {min_calibration_rows} needed to estimate "
                f"how far the two tables routinely disagree; refusing to "
                f"adjudicate rather than guess a floor"
            ),
            floor=None, floor_quantile=quantile, calibration_n=calibration_n,
            n_zero=n_zero, n_zero_with_sibling=n_zero_with_sibling,
            n_disproved=0, max_disproving_value=None, mask=empty_mask,
        )

    # One-sided by construction: a negative sibling is A75 netting, not hidden
    # generation, and `energy_renewable` cannot represent it at all.
    disproved = is_zero & sibling_values.notna() & (sibling_values > floor)
    n_disproved = int(disproved.sum())
    return SiblingDisproof(
        evaluable=True,
        reason=(
            f"floor = q{quantile:g} of |value - sibling| over "
            f"{calibration_n} positive-value instants = {floor:.6g}"
        ),
        floor=floor, floor_quantile=quantile, calibration_n=calibration_n,
        n_zero=n_zero, n_zero_with_sibling=n_zero_with_sibling,
        n_disproved=n_disproved,
        max_disproving_value=(
            float(sibling_values[disproved].max()) if n_disproved else None
        ),
        mask=disproved,
    )


def exclude_zeros_disproved_by_sibling(
    df: pd.DataFrame,
    sibling: pd.DataFrame,
    value_col: str = "target_value",
    timestamp_col: str = "timestamp_utc",
    quantile: float = SIBLING_DISPROOF_QUANTILE,
    min_calibration_rows: int = SIBLING_DISPROOF_MIN_CALIBRATION_ROWS,
    context: str = "",
) -> pd.DataFrame:
    """
    Null `value_col` on every exact 0.0 the twin table disproves.

    Returns a copy when anything is excluded, `df` itself otherwise. Call this
    **after** `exclude_suspect_constant_runs`, never before: that guard infers a
    run's extent from the observations present, so punching NaN holes in a long
    flat run first would split it at the gap boundary and drop both halves under
    `min_run_hours` -- weakening the older guard instead of adding to it. In the
    order used here the two are strictly additive.
    """
    verdict = adjudicate_zeros_against_sibling(
        df, sibling, value_col=value_col, timestamp_col=timestamp_col,
        quantile=quantile, min_calibration_rows=min_calibration_rows,
    )
    suffix = f" [{context}]" if context else ""
    if not verdict.evaluable:
        logger.info(
            "training-data invariant (ABL-200): not adjudicating %s zeros "
            "against the sibling table -- %s.%s",
            value_col, verdict.reason, suffix,
        )
        return df
    if not verdict.n_disproved:
        return df

    d = df.copy()
    d.loc[verdict.mask, value_col] = float("nan")
    excluded = d.loc[verdict.mask, timestamp_col]
    logger.warning(
        "training-data invariant (ABL-200): excluding %d of %d exact-0.0 %s "
        "rows disproved by the sibling table (%s to %s; largest disproving "
        "sibling value %.6g MW; %s). Treated as unadjudicated-missing, not "
        "zero, and not repaired from the sibling.%s",
        verdict.n_disproved, verdict.n_zero, value_col,
        excluded.min(), excluded.max(), verdict.max_disproving_value,
        verdict.reason, suffix,
    )
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
