"""
Plausibility guard on TSO day-ahead forecast reads (ABL-431).

ENTSO-E's published day-ahead forecasts are ingested verbatim, and on
2026-02-04 HU's ``wind_onshore_mw`` came through at **140,996 MW** against a
fleet whose p99.5 over five years is 283 MW. Dividing those 96 quarter-hours
by 1000 reproduces HU's own measured generation for the same day to within
the day's forecast error (35.8-141.0 MW predicted against 36.8-133.0 MW
observed, both rising together through the day), so the signature is a
kW-published-as-MW unit error, not noise -- the shape is right and only the
scale is wrong.

That is inert today because nothing fits on this table. ABL-247 proposes
feeding exactly this series to load/solar/wind models, and a single value
three orders of magnitude out does not degrade a squared-error fit
gracefully; it dominates it. This module is the precondition.

**What it does.** Nulls a read value that exceeds ``PLAUSIBILITY_TOLERANCE``
times a per-country, per-column reference scale, and logs what it nulled.
NULL is this codebase's "not measured" encoding, so a guarded value drops out
of a fit the same way a genuinely absent one does. **No stored row is
touched**: the guard lives on the read path, the table is never mutated, and
a value that looks impossible today is still there for whoever re-reads it.

**Why the reference is derived, not registered.** The issue asked for a scale
anchored on installed capacity rather than a hard constant, and there is no
installed-capacity table on the replica. A committed capacity table would
also go stale in the one direction that matters: NL solar grew from nothing to
7.9 GW inside this history, and a frozen bound would start rejecting real
growth. So the reference is recomputed from the full history of the series
itself at read time (cached per process), which can only move with the fleet.

    reference_mw = max( q(actuals), q(day-ahead forecasts) )     q = p99.5

Both sides, because neither alone is sound. The actuals table under-reports
whole technologies for some countries -- NL's ``energy_generation.solar_mw``
tops out at 428.8 MW while NL's own published solar forecast routinely reaches
7,871 MW -- so an actuals-only anchor would reject 18x of legitimate NL solar.
The forecast table is the defect's own home, so a forecast-only anchor could be
inflated by the very rows it is meant to catch; ``q`` is a high quantile rather
than a maximum for that reason. **Bound worth knowing:** the guard is robust to
a contaminated cluster covering less than 1 - q of the series. At p99.5 on a
five-year quarter-hourly series that is about 985 rows (~10 days). HU's
incident is 96 rows, 0.0487% of its series -- an order of magnitude inside the
bound, but not infinitely so.

**Why 3x, measured.** Across all 146 evaluable (country, column) pairs on the
replica (2026-08-14), ``max / reference`` separates cleanly with no pair in
between:

    anomalous   HU wind_onshore 497.7x   HU total 37.3x   SK wind_onshore 8.70x
                MK wind_onshore 6.05x    MK total 4.12x
    ---------------------------------- 3.0x -----------------------------------
    healthy     PT solar 1.82x   PT wind_offshore 1.77x   NL load 1.60x
                p90 of all pairs 1.41x   p50 1.11x

3.0 sits inside a measured empty band running from 1.82 to 4.12, so the
tolerance is not a guess and the highest healthy pair keeps 1.65x of headroom.
At 3.0 the guard flags **213 of 14,610,819 column-observations (0.0015%)**
across both tables -- 192 HU (``wind_onshore_mw`` and the ``total_forecast_mw``
it dominates, one CET market day), 20 MK on 2022-04-10, 1 SK on 2022-09-25, and
**0 rows in ``energy_load_forecast``**, whose worst pair is NL at 1.60x.
``scripts/abl431_tso_plausibility_census.py`` regenerates all of it.

**Why it is not a blanket filter.** This repo has twice shipped a deliberately
narrow guard to avoid discarding legitimately published values (ABL-71 keeps
published zeros; ABL-109's is not a blanket ``> 0`` because DE solar has 56
real overnight zeros). The same rule holds here and is load-bearing in three
places:

- **Only an upper bound.** A published 0.0 is never flagged, at any tolerance
  -- ``0 <= 3 * reference`` always. Both prior guards survive untouched.
- **A zero reference refuses to evaluate rather than rejecting everything.**
  28 of the 174 (country, column) pairs are all-zero series -- landlocked
  countries reporting ``wind_offshore_mw = 0.0`` forever. Their reference is
  0.0, and ``value > 3 * 0`` would flag every non-zero value a new fleet ever
  published. ``ReferenceScale.evaluable`` is False there and the series passes
  through unguarded, carrying the reason.
- **A fleet's first days are unguarded, by the same mechanism.** A country
  standing up its first wind farm has a mostly-zero history and so no usable
  reference until the new plant is in its own p99.5. The guard fails open there
  and says so, rather than rejecting the new fleet's real output.

The limitation that direction leaves: a step change in capacity large enough to
exceed 3x the incumbent p99.5 *would* be flagged for as long as it takes the
new level to reach that quantile. Nothing in five years of this replica does
that, and the guard logs every drop by country, column and window, so a country
flagging persistently reads as a fleet change rather than disappearing quietly.
"""

import logging
import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger('energy_forecast')


#: Quantile used as the robust upper reference for a series. See module
#: docstring for why this is a quantile and not a maximum, and for the size of
#: contaminated cluster it tolerates.
REFERENCE_QUANTILE = 0.995

#: A read value above this many times the reference is refused. Chosen from the
#: measured 1.82x-4.12x empty band across all evaluable pairs, not by convention.
PLAUSIBILITY_TOLERANCE = 3.0

#: Every TSO day-ahead forecast column this module knows how to scale, mapped to
#: the actuals series that measures the same fleet. There is deliberately no
#: default: an unregistered column raises rather than being guarded against a
#: guessed reference, for the same reason NIGHT_GENERATION_POSSIBLE has no
#: default -- silently guessing is how a guard deletes real MW.
#:
#: The actuals side is an SQL expression, not just a column, because
#: total_forecast_mw has no single counterpart. COALESCE there matches
#: db.RENEWABLE_TYPE_COLUMNS' null-aware form: a country reporting only solar
#: must still contribute its solar, not be erased by a strict `+`.
TSO_FORECAST_SOURCES: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("energy_generation_forecast", "solar_mw"):
        ("energy_generation", "solar_mw"),
    ("energy_generation_forecast", "wind_onshore_mw"):
        ("energy_generation", "wind_onshore_mw"),
    ("energy_generation_forecast", "wind_offshore_mw"):
        ("energy_generation", "wind_offshore_mw"),
    ("energy_generation_forecast", "total_forecast_mw"):
        ("energy_generation",
         "COALESCE(solar_mw, 0) + COALESCE(wind_onshore_mw, 0) "
         "+ COALESCE(wind_offshore_mw, 0)"),
    ("energy_load_forecast", "forecast_value_mw"):
        ("energy_load", "load_mw"),
}

#: Which timestamp column each table bounds an as-of read on.
_TIMESTAMP_COLUMN = {
    "energy_generation_forecast": "target_timestamp_utc",
    "energy_load_forecast": "target_timestamp_utc",
    "energy_generation": "timestamp_utc",
    "energy_load": "timestamp_utc",
}


class UnknownTsoSourceError(KeyError):
    """A (table, column) pair with no registered actuals counterpart.

    Raised rather than defaulted. A guard that guesses which fleet it is
    scaling against can silently null real generation, which is the failure
    mode this module exists to avoid causing.
    """


@dataclass(frozen=True)
class ReferenceScale:
    """The per-(country, column) scale a read is held to, and its provenance.

    ``evaluable`` is the load-bearing field. False means "there is no fleet
    scale to anchor on here", which is a different statement from "everything
    here is plausible" -- and both are different from "this value is too
    large". A caller that treats a non-evaluable reference as a pass is
    correct; one that treats it as a rejection would be the blanket filter
    this module refuses to become.
    """

    country_code: str
    table: str
    column: str
    reference_mw: Optional[float]
    quantile: float
    tolerance: float
    n_forecast: int
    n_actual: int
    forecast_quantile_mw: Optional[float]
    actual_quantile_mw: Optional[float]
    as_of: Optional[str]
    reason: str

    @property
    def evaluable(self) -> bool:
        return self.reference_mw is not None and self.reference_mw > 0.0

    @property
    def threshold_mw(self) -> Optional[float]:
        """The value above which a read is refused, or None when not evaluable."""
        if not self.evaluable:
            return None
        return self.tolerance * self.reference_mw


@dataclass(frozen=True)
class GuardOutcome:
    """What one guarded read actually did, for logging and telemetry."""

    reference: ReferenceScale
    n_observed: int
    n_flagged: int
    max_value_mw: Optional[float]
    max_ratio: Optional[float]
    first_flagged: Optional[pd.Timestamp]
    last_flagged: Optional[pd.Timestamp]

    @property
    def applied(self) -> bool:
        return self.reference.evaluable


def _nearest_rank_quantile(
    conn: sqlite3.Connection,
    table: str,
    expression: str,
    country_code: str,
    quantile: float,
    as_of: Optional[str],
    extra_where: str = "",
) -> Tuple[Optional[float], int]:
    """Nearest-rank quantile of ``expression`` over one country's whole history.

    Nearest-rank rather than an interpolated percentile so the reference is a
    value the series actually published, and so it can be taken with an OFFSET
    query instead of pulling ~200k floats per call into Python. The difference
    between the two definitions is under 3% here and the tolerance band it
    feeds is a factor of 2.3 wide, so the choice is not load-bearing -- but it
    has to be stated, because two callers computing the reference differently
    would be the defect one level up.

    Absent, and only absent, is tolerated: a caller whose database has no such
    table (a fixture, a partial snapshot) gets ``(None, 0)`` rather than an
    exception, because this runs on a read path and a guard that turns a
    missing side into a crash is worse than the value it was watching for. The
    other side still anchors the reference, and the caller records which sides
    answered.
    """
    where = ["country_code = ?", f"({expression}) IS NOT NULL"]
    params: list = [country_code]
    if extra_where:
        where.append(extra_where)
    if as_of is not None:
        where.append(f"{_TIMESTAMP_COLUMN[table]} <= ?")
        params.append(as_of)
    clause = " AND ".join(where)

    try:
        n = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {clause}", params
        ).fetchone()[0]
        if not n:
            return None, 0

        # Rank counted from the top so a 0.995 quantile is the 0.5%-th largest.
        offset = int(round((1.0 - quantile) * (n - 1)))
        row = conn.execute(
            f"SELECT ({expression}) AS v FROM {table} WHERE {clause} "
            f"ORDER BY v DESC LIMIT 1 OFFSET ?",
            [*params, offset],
        ).fetchone()
    except sqlite3.OperationalError as exc:
        logger.debug("TSO plausibility guard (ABL-431): %s unavailable for %s "
                     "-- %s", table, country_code, exc)
        return None, 0
    return (float(row[0]) if row is not None and row[0] is not None else None), n


def _database_key(conn: sqlite3.Connection) -> str:
    """The file this connection is actually attached to.

    Part of the cache key on purpose. This box carries a 3.0 GB stale partial
    snapshot next to the live replica (CLAUDE.md), and a reference cached from
    one must never be served to a read of the other -- the snapshot's HU wind
    history stops in 2023 and would yield a reference for a different fleet.
    """
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except sqlite3.Error:
        return "<unknown>"
    for _, name, path in rows:
        if name == "main":
            return path or "<memory>"
    return "<unknown>"


_REFERENCE_CACHE: Dict[tuple, ReferenceScale] = {}


def clear_reference_cache() -> None:
    """Drop every memoised reference. Tests and long-lived processes only."""
    _REFERENCE_CACHE.clear()


def reference_scale(
    conn: sqlite3.Connection,
    country_code: str,
    table: str,
    column: str,
    as_of: Optional[str] = None,
    quantile: float = REFERENCE_QUANTILE,
    tolerance: float = PLAUSIBILITY_TOLERANCE,
) -> ReferenceScale:
    """Resolve the fleet scale ``(country_code, table, column)`` is held to.

    ``as_of`` bounds both the forecast and the actuals read on their own
    timestamp column, and defaults to None -- the whole history. None is the
    serve-faithful default because at serving time the whole history *is*
    everything available. A backtest reconstructing a past vintage should pass
    the run's observation cutoff, exactly as ``build_for_country`` does: the
    reference is a slowly-varying fleet property rather than a target-correlated
    signal, so the leak is small, but "small" is not "absent" and the caller
    should get to decide.
    """
    key = (_database_key(conn), country_code, table, column, as_of, quantile, tolerance)
    cached = _REFERENCE_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        actual_table, actual_expression = TSO_FORECAST_SOURCES[(table, column)]
    except KeyError:
        raise UnknownTsoSourceError(
            f"no registered actuals counterpart for {table}.{column}; add one to "
            f"TSO_FORECAST_SOURCES rather than guarding against a guessed scale"
        ) from None

    forecast_q, n_forecast = _nearest_rank_quantile(
        conn, table, column, country_code, quantile, as_of,
        extra_where="forecast_type = 'day_ahead'",
    )
    actual_q, n_actual = _nearest_rank_quantile(
        conn, actual_table, actual_expression, country_code, quantile, as_of,
    )

    candidates = [q for q in (forecast_q, actual_q) if q is not None]
    if not candidates:
        reference, reason = None, "no history in either the forecast or the actuals table"
    else:
        reference = max(candidates)
        if reference > 0.0:
            reason = "ok"
        else:
            reason = (
                "reference quantile is 0.0 on both sides -- the series reports no "
                "fleet, so there is no scale to hold a value to"
            )

    scale = ReferenceScale(
        country_code=country_code,
        table=table,
        column=column,
        reference_mw=reference,
        quantile=quantile,
        tolerance=tolerance,
        n_forecast=n_forecast,
        n_actual=n_actual,
        forecast_quantile_mw=forecast_q,
        actual_quantile_mw=actual_q,
        as_of=as_of,
        reason=reason,
    )
    _REFERENCE_CACHE[key] = scale
    return scale


def implausible_mask(values: pd.Series, reference: ReferenceScale) -> pd.Series:
    """Boolean mask of the read values this reference refuses.

    All-False when the reference is not evaluable, and all-False for NaN. The
    test is a one-sided ``>``: a published zero, or any value at or below the
    threshold, is kept.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if not reference.evaluable:
        return pd.Series(False, index=values.index)
    return (numeric > reference.threshold_mw).fillna(False)


def guard_series(
    values: pd.Series,
    reference: ReferenceScale,
    context: str = "",
) -> Tuple[pd.Series, GuardOutcome]:
    """Null every implausible entry of ``values``; return the copy and a report.

    The input is never mutated and no database row is written. Flagged entries
    become NaN, which downstream code already reads as not-measured.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.notna()
    mask = implausible_mask(values, reference)

    flagged_index = values.index[mask]
    outcome = GuardOutcome(
        reference=reference,
        n_observed=int(observed.sum()),
        n_flagged=int(mask.sum()),
        max_value_mw=float(numeric.max()) if observed.any() else None,
        max_ratio=(
            float(numeric.max()) / reference.reference_mw
            if observed.any() and reference.evaluable else None
        ),
        first_flagged=flagged_index.min() if len(flagged_index) else None,
        last_flagged=flagged_index.max() if len(flagged_index) else None,
    )

    if not reference.evaluable:
        if reference.n_forecast or reference.n_actual:
            logger.debug(
                "TSO plausibility guard (ABL-431) not applied to %s %s.%s: %s%s",
                reference.country_code, reference.table, reference.column,
                reference.reason, f" [{context}]" if context else "",
            )
        return values, outcome

    if outcome.n_flagged:
        logger.warning(
            "TSO plausibility guard (ABL-431): nulling %d of %d observed "
            "%s %s.%s values above %.1f MW (%.3gx the p%.4g reference of "
            "%.1f MW); largest %.1f MW = %.0fx; %s .. %s. The stored rows are "
            "untouched -- this read treats them as not-measured.%s",
            outcome.n_flagged, outcome.n_observed, reference.country_code,
            reference.table, reference.column, reference.threshold_mw,
            reference.tolerance, 100.0 * reference.quantile,
            reference.reference_mw, outcome.max_value_mw or float("nan"),
            outcome.max_ratio or float("nan"),
            outcome.first_flagged, outcome.last_flagged,
            f" [{context}]" if context else "",
        )

    guarded = values.copy()
    guarded[mask] = float("nan")
    return guarded, outcome


def guard_tso_series(
    values: pd.Series,
    conn: sqlite3.Connection,
    country_code: str,
    table: str,
    column: str,
    as_of: Optional[str] = None,
    context: str = "",
) -> pd.Series:
    """Resolve the reference and guard ``values`` in one call.

    This is the read-site entry point: call it on the series a TSO day-ahead
    read is about to hand back, before any resample, lag or merge, so that a
    refused value cannot be averaged into a neighbour first.
    """
    reference = reference_scale(conn, country_code, table, column, as_of=as_of)
    guarded, _ = guard_series(values, reference, context=context)
    return guarded


def guard_tso_frame(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    country_code: str,
    table: str,
    column: str,
    frame_column: Optional[str] = None,
    timestamp_column: str = "timestamp_utc",
    as_of: Optional[str] = None,
    context: str = "",
) -> pd.DataFrame:
    """``guard_tso_series`` for a column of a frame, aliased or not.

    ``frame_column`` names the column as the caller's SELECT aliased it;
    ``column`` always names it as the table declares it, because that is what
    the reference is registered under.

    ``timestamp_column`` is used only to index the series while guarding, so
    that the warning names the window in timestamps rather than in row numbers.
    A frame without it still guards correctly; the log just reads worse.
    """
    target = frame_column or column
    if df.empty or target not in df.columns:
        return df

    out = df.copy()
    values = out[target]
    if timestamp_column in out.columns:
        values = pd.Series(values.to_numpy(),
                           index=pd.DatetimeIndex(out[timestamp_column]))
    guarded = guard_tso_series(values, conn, country_code, table, column,
                               as_of=as_of, context=context)
    out[target] = guarded.to_numpy()
    return out
