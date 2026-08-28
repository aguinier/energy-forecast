#!/usr/bin/env python
"""ABL-247 -- does a TSO day-ahead vintage earn its place as a model *feature*?

Executes the pre-registration stored on ABL-247 as document key
``abl247-prereg`` (written 2026-08-14, eleven days before the data existed).
Every rule below was fixed before any outcome could be seen; deviations are
recorded as deviations in the emitted record, never applied silently.

What is being asked
-------------------
Not "is the TSO better than us" -- that is ABL-246, and it is answered. This
asks whether *our* forecast plus the TSO's, combined, beats our forecast alone
on the same rows. The estimator is registered:

    y ~= a + b * f_ours + c * f_tso        fit per (forecast_type, band)

with the null hypothesis **c = 0**. A 13-day archive cannot carry a
gradient-boosted refit against production models trained on months -- such an
arm measures the training window, not the feature -- so the affine combiner is
the primary estimator and the retrain is explicitly secondary and underpowered
(prereg section 3).

Why the archive and nothing else
--------------------------------
Before ABL-184 went live at 2026-08-11T19:16:13Z, ``energy_load_forecast``
retained one vintage per target: the last one, after every revision. Prereg
section 2 measured what that costs -- 39.6% of load targets carry a revision,
worth ~3.25% of level on the revised ones; wind onshore 10.63%. A feature built
from the retained series therefore carries, on ~40% of rows, information that
did not exist at the model's cutoff. **Training history for a TSO-feature model
is hard-capped at the archive.** There is no proxy path to the pre-archive
months, and the retained-series proxy is refused by measurement, not by
assumption.

The go-live backfill is excluded everywhere. Its ``first_seen_at`` is 2026-08-11
and it carries retained post-revision values for targets back to 2018 -- 13.7M
rows that, counted naively, read as years of vintage history that does not
exist. Every read here starts at ``FIRST_CLEAN_TARGET_DAY``.

Leak-freeness
-------------
For each of our forecast rows the cutoff is its own ``generated_at``: the
instant the forecast existed. The feature is the latest TSO vintage whose
``first_seen_at`` is at or before that instant. Selection happens per target
*instant* and only then averages to the hour, so a cutoff rule can never be
applied to an average of values with different first-seen stamps.

``first_seen_at`` is our poller's stamp, never the TSO's publication time, so
every coverage and lead figure here is a **lower bound**. That bias runs against
the feature, which is the safe direction for a result that favours it.

Bands, and why there is no pooled fit
-------------------------------------
Missingness is structured by horizon, not missing-at-random. A single model
pooled over 0-64h would learn "feature absent => long horizon" and turn
missingness into a horizon proxy. Prereg section 4 forbids it: **band-separate
fits only**. The 48-64h band is re-scoped out of the backtest entirely (0.0%
coverage is a product property of a day-ahead series, not a poll artifact); it
is still *measured* here, because a re-scope that is not re-checked at 14 days
is an assumption.

Scoring
-------
Availability-matched is primary: both arms on the identical rows where the
feature is present. All-rows is reported too with the composition term named --
the gap between them is a coverage effect, not a feature effect. Everything is
aggregated to the hour before any mean (cadence is mixed in-window, so a raw row
mean is cadence-weighted and not comparable across countries).

Read-only on the replica. Writes only to the report directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SUPPORTED_COUNTRIES  # noqa: E402
from src.evaluation.model_free_reference import (  # noqa: E402
    TRAILING_COMPARATORS,
    attach_trailing_references,
)
from src.evaluation.scorecard import ACTUAL_SPECS, PRODUCTION_MODELS  # noqa: E402
from src.tso_plausibility import (  # noqa: E402
    VINTAGE_ARCHIVE_DAY_AHEAD_MODEL,
    VINTAGE_ARCHIVE_TABLE,
    guard_tso_frame,
)

#: ABL-184 archive go-live. Rows first seen at or before this instant are
#: backfill: the stamp measures our ingest, not the TSO.
GO_LIVE = "2026-08-11T19:16:13Z"

#: First target day whose *initial* publication is guaranteed to post-date
#: go-live. A D+1 product published on 08-11 covers targets through 08-12, so
#: 08-13 is the first genuine one. Deliberately one day past the arithmetic
#: minimum -- the same floor the ABL-247 availability probe uses.
FIRST_CLEAN_TARGET_DAY = "2026-08-13"

#: Prereg section 1. Half-open, except the inclusive 64h endpoint, matching
#: `scorecard.HORIZON_BANDS` at the outer edges.
BANDS = (("0-24h", 0.0, 24.0), ("24-48h", 24.0, 48.0), ("48-64h", 48.0, 65.0))

#: The band the feature has to earn its place in.
PRIMARY_BAND = "0-24h"

#: Re-scoped out of the backtest by the CEO on 2026-08-14 and accepted in full.
#: Measured anyway (see module docstring).
NOT_BACKTESTED_BANDS = ("48-64h",)

#: The issue says "load/solar/wind". Wind is two forecast types with two
#: different production algorithms (catboost onshore, xgboost offshore), and a
#: feature can earn its place in one and not the other, so they are separate
#: rows here rather than a summed "wind".
FORECAST_TYPES = ("load", "solar", "wind_onshore", "wind_offshore")

#: Trailing-window length for the causal references. ABL-437's registered value;
#: the actuals series is not archive-bound, so 28 days is available even though
#: the vintage window is ~16.
TRAILING_WINDOW_DAYS = 28

SEASONAL_NAIVE_LAG_HOURS = 168

#: Student-t two-sided 97.5% points, k-1 df. A table beats a scipy dependency
#: for the handful of small k this window can produce.
_T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
           8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
           14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
           20: 2.093}


def t_crit(k: int) -> float:
    """Two-sided 97.5% t point on k-1 degrees of freedom."""
    if k < 2:
        return float("nan")
    return _T_CRIT.get(k, 1.96 if k > 60 else 2.086)


def connect_ro(path: str) -> sqlite3.Connection:
    """Open the replica read-only. The URI form is not optional (AGENTS.md)."""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=120.0)


def normalize_ts(values) -> pd.Series:
    """Parse both SQLite timestamp spellings into naive UTC.

    The two sources spell the separator differently -- `ml` writes an ISO `T`,
    `tso` writes a space -- and joining on the raw text silently matches
    nothing.
    """
    return pd.to_datetime(pd.Series(list(values)), format="mixed", utc=True,
                          errors="coerce").dt.tz_localize(None)


def json_safe(value):
    """Recursively replace non-finite floats with ``None``.

    ``json.dumps`` writes ``NaN`` and ``Infinity``, which are not JSON and which
    a strict reader rejects. More to the point, a NaN here always means the same
    thing -- *this was not measured* -- and ``null`` is how the rest of this
    repo's records say that. Silently emitting ``NaN`` would let a downstream
    reader parse it as a number in some languages and fail in others.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NaT or (value is not None and value is pd.NA):
        return None
    return value


def wape(err, actual) -> float:
    denom = np.abs(np.asarray(actual, dtype=float)).sum()
    if not np.isfinite(denom) or denom == 0:
        return float("nan")
    return float(np.abs(np.asarray(err, dtype=float)).sum() / denom * 100.0)


# --------------------------------------------------------------------------
# Preconditions (prereg section 6)
# --------------------------------------------------------------------------


def replica_state(conn: sqlite3.Connection, db_path: str) -> dict:
    """Bound the replica's staleness before reading a missing burst as a dead clock.

    A refresh holds an exclusive lock for tens of minutes, and a stale replica
    would show the vintage clock as having stopped. The archive's own maximum
    ``first_seen_at`` is the tightest available bound on how current the
    archived vintages are.
    """
    max_seen = conn.execute(
        f"SELECT MAX(first_seen_at) FROM {VINTAGE_ARCHIVE_TABLE}"
    ).fetchone()[0]
    max_target = conn.execute(
        f"SELECT MAX(target_timestamp_utc) FROM {VINTAGE_ARCHIVE_TABLE} "
        f"WHERE source = 'tso' AND model_name = ?",
        (VINTAGE_ARCHIVE_DAY_AHEAD_MODEL,),
    ).fetchone()[0]
    now = pd.Timestamp.utcnow().tz_localize(None)
    seen = normalize_ts([max_seen]).iloc[0]
    return {
        "replica_path": db_path,
        "replica_file_mtime_utc": pd.Timestamp(
            os.path.getmtime(db_path), unit="s").isoformat(),
        "archive_max_first_seen_at": None if pd.isna(seen) else seen.isoformat(),
        "archive_max_tso_target": max_target,
        "read_at_utc": now.isoformat(),
        "archive_staleness_hours": (
            None if pd.isna(seen) else round((now - seen).total_seconds() / 3600, 2)
        ),
    }


def backfill_census(conn: sqlite3.Connection) -> dict:
    """Quantify the go-live backfill bucket that every count here excludes.

    Prereg standing reminder: counting it naively reads as years of vintage
    history that does not exist.
    """
    row = conn.execute(
        f"""SELECT COUNT(*), MIN(target_timestamp_utc), MAX(target_timestamp_utc)
            FROM {VINTAGE_ARCHIVE_TABLE}
            WHERE substr(first_seen_at, 1, 10) = '2026-08-11'"""
    ).fetchone()
    genuine = conn.execute(
        f"""SELECT COUNT(*), COUNT(DISTINCT substr(target_timestamp_utc, 1, 10))
            FROM {VINTAGE_ARCHIVE_TABLE}
            WHERE source = 'tso' AND model_name = ?
              AND target_timestamp_utc >= ?""",
        (VINTAGE_ARCHIVE_DAY_AHEAD_MODEL, FIRST_CLEAN_TARGET_DAY),
    ).fetchone()
    return {
        "backfill_rows_excluded": row[0],
        "backfill_target_span": [row[1], row[2]],
        "genuine_tso_day_ahead_rows": genuine[0],
        "genuine_target_days": genuine[1],
        "genuine_target_day_floor": FIRST_CLEAN_TARGET_DAY,
    }


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


def read_tso_vintages(conn: sqlite3.Connection, forecast_type: str) -> pd.DataFrame:
    """Every genuine TSO day-ahead vintage for one series, guarded per country.

    The ABL-431/458 plausibility guard runs on the raw read, before any
    resample, lag or merge, so a refused value cannot be averaged into a
    neighbour first. The HU ``wind_onshore`` cluster is in-window.
    """
    df = pd.read_sql_query(
        f"""SELECT country_code, target_timestamp_utc, forecast_value, first_seen_at
            FROM {VINTAGE_ARCHIVE_TABLE}
            WHERE source = 'tso' AND model_name = ?
              AND forecast_type = ?
              AND target_timestamp_utc >= ?""",
        conn,
        params=(VINTAGE_ARCHIVE_DAY_AHEAD_MODEL, forecast_type,
                FIRST_CLEAN_TARGET_DAY),
    )
    if df.empty:
        df["target"] = pd.Series(dtype="datetime64[ns]")
        df["first_seen"] = pd.Series(dtype="datetime64[ns]")
        df.attrs["guard_refusals"] = 0
        df.attrs["rows_read"] = 0
        return df

    df = df[df["country_code"].isin(SUPPORTED_COUNTRIES)].copy()
    rows_read = len(df)
    df["target"] = normalize_ts(df["target_timestamp_utc"])

    guarded = []
    for country, grp in df.groupby("country_code", sort=True):
        guarded.append(guard_tso_frame(
            grp.rename(columns={"target": "timestamp_utc"}),
            conn,
            country_code=country,
            table=VINTAGE_ARCHIVE_TABLE,
            column=forecast_type,
            frame_column="forecast_value",
            timestamp_column="timestamp_utc",
            context=f"ABL-247 {country} {forecast_type} vintages",
        ).rename(columns={"timestamp_utc": "target"}))
    out = pd.concat(guarded, ignore_index=True) if guarded else df
    refused = int(out["forecast_value"].isna().sum())
    out = out.dropna(subset=["forecast_value"]).copy()
    out["first_seen"] = normalize_ts(out["first_seen_at"])
    out = out.dropna(subset=["first_seen", "target"])
    out.attrs["guard_refusals"] = refused
    out.attrs["rows_read"] = rows_read
    return out[["country_code", "target", "forecast_value", "first_seen"]]


def read_our_forecasts(conn: sqlite3.Connection, forecast_type: str,
                       start: str, end: str) -> pd.DataFrame:
    """Our production forecast rows, with their own issue instant.

    ``generated_at`` is the cutoff every feature lookup is made against. The
    model is the registered production one for the type
    (``scorecard.PRODUCTION_MODELS``) -- this measures the feature against what
    we actually serve, not against a convenient challenger.
    """
    model = PRODUCTION_MODELS[forecast_type]
    df = pd.read_sql_query(
        """SELECT country_code, target_timestamp_utc, generated_at,
                  horizon_hours, forecast_value, model_name
           FROM forecasts
           WHERE forecast_type = ? AND model_name = ?
             AND target_timestamp_utc >= ? AND target_timestamp_utc < ?""",
        conn, params=(forecast_type, model, start, end))
    if df.empty:
        return df
    df = df[df["country_code"].isin(SUPPORTED_COUNTRIES)].copy()
    df["target"] = normalize_ts(df["target_timestamp_utc"])
    df["generated_at"] = normalize_ts(df["generated_at"])
    df = df.dropna(subset=["target", "generated_at", "forecast_value"])
    df["band"] = [band_of(h) for h in df["horizon_hours"]]
    return df.dropna(subset=["band"])


def band_of(hours) -> str | None:
    if hours is None or pd.isna(hours):
        return None
    value = float(hours)
    for name, lower, upper in BANDS:
        if lower <= value < upper:
            return name
    return None


def read_actuals(conn: sqlite3.Connection, forecast_type: str,
                 start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Hourly-mean actuals from the registered scoring truth.

    ``ACTUAL_SPECS`` is the one statement of the actual for this repo and the
    dashboard both (ABL-410); restating it here would be the second one.

    Two things are done to the raw rows and both are reported:

    * **ABL-111 / ABL-109.** A ``0.0`` in ``energy_load`` encodes *missing*, not
      zero demand. Dropped, and counted. This touches the load target directly
      and is distinct from the feature-side zero-flips of prereg section 2.
    * **Mixed cadence.** Roughly half the fleet is quarter-hourly over this
      window, so a mean over raw rows is cadence-weighted. Aggregate to the hour
      first (the ABL-332 contract).
    """
    table, expression = ACTUAL_SPECS[forecast_type]
    df = pd.read_sql_query(
        f"""SELECT country_code, timestamp_utc, {expression} AS actual
            FROM {table}
            WHERE timestamp_utc >= ? AND timestamp_utc < ?""",
        conn, params=(start, end))
    notes = {"truth_table": table, "truth_expression": expression,
             "raw_rows": len(df), "zero_rows_dropped": 0}
    if df.empty:
        return pd.DataFrame(columns=["country_code", "target", "actual"]), notes
    df = df[df["country_code"].isin(SUPPORTED_COUNTRIES)].copy()
    df = df.dropna(subset=["actual"])
    if forecast_type == "load":
        zeros = int((df["actual"] == 0).sum())
        notes["zero_rows_dropped"] = zeros
        notes["zero_rule"] = ("ABL-111/ABL-109: 0.0 in energy_load encodes "
                              "missing, not zero demand")
        df = df[df["actual"] != 0]
    df["ts"] = normalize_ts(df["timestamp_utc"])
    hourly = (df.dropna(subset=["ts"]).set_index("ts")
                .groupby("country_code")["actual"]
                .resample("h").mean().reset_index()
                .rename(columns={"ts": "target"})
                .dropna(subset=["actual"]))
    notes["hourly_rows"] = len(hourly)
    return hourly, notes


# --------------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------------


def feature_at_cutoffs(tso: pd.DataFrame, cutoffs: pd.DatetimeIndex) -> pd.DataFrame:
    """The TSO feature as it stood at each cutoff: (cutoff, country, hour) -> value.

    Selection is per target *instant* -- latest vintage with
    ``first_seen <= cutoff`` -- and only then averaged to the hour. Doing it the
    other way round would apply the cutoff rule to a blend of values with
    different first-seen stamps, which is not a value anybody could have held.
    """
    if tso.empty or len(cutoffs) == 0:
        return pd.DataFrame(columns=["generated_at", "country_code", "target",
                                     "f_tso", "tso_first_seen"])
    ordered = tso.sort_values("first_seen")
    frames = []
    for cutoff in pd.DatetimeIndex(cutoffs).unique():
        visible = ordered[ordered["first_seen"] <= cutoff]
        if visible.empty:
            continue
        latest = visible.drop_duplicates(["country_code", "target"], keep="last")
        latest = latest.assign(hour=latest["target"].dt.floor("h"))
        agg = (latest.groupby(["country_code", "hour"], as_index=False)
                     .agg(f_tso=("forecast_value", "mean"),
                          tso_first_seen=("first_seen", "max")))
        agg["generated_at"] = cutoff
        frames.append(agg.rename(columns={"hour": "target"}))
    if not frames:
        return pd.DataFrame(columns=["generated_at", "country_code", "target",
                                     "f_tso", "tso_first_seen"])
    return pd.concat(frames, ignore_index=True)


def build_panel(ours: pd.DataFrame, tso: pd.DataFrame,
                actuals: pd.DataFrame) -> pd.DataFrame:
    """One row per (country, band, target hour) with both arms and the truth.

    Our forecasts are collapsed to the latest issued row per (country, target,
    band) -- the same rule the scorecard applies, so this measures the slice the
    rest of the programme scores. The cutoff carried forward is that winning
    row's own ``generated_at``.
    """
    if ours.empty:
        return pd.DataFrame()
    ours = ours.copy()
    ours["target"] = ours["target"].dt.floor("h")
    latest = (ours.sort_values("generated_at")
                  .drop_duplicates(["country_code", "band", "target"], keep="last")
                  .rename(columns={"forecast_value": "f_ours"}))

    feature = feature_at_cutoffs(tso, pd.DatetimeIndex(latest["generated_at"]))
    panel = latest.merge(feature, on=["generated_at", "country_code", "target"],
                         how="left")
    panel = panel.merge(actuals, on=["country_code", "target"], how="inner")
    panel["available"] = panel["f_tso"].notna()
    panel["target_day"] = panel["target"].dt.normalize()
    panel["target_ts"] = panel["target"]
    # How early the feature existed, and how stale it already was when our run
    # picked it up. Both are lower bounds: `first_seen_at` is our poller's
    # stamp, never the TSO's publication time.
    panel["feature_lead_h"] = (
        (panel["target"] - panel["tso_first_seen"]).dt.total_seconds() / 3600.0)
    panel["feature_age_at_cutoff_h"] = (
        (panel["generated_at"] - panel["tso_first_seen"]).dt.total_seconds() / 3600.0)
    return panel


def attach_references(panel: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Seasonal-naive, persistence and the two ABL-437 causal trailing references.

    Prereg section 5 requires a climatology rather than a flat constant for
    solar -- a flat line loses to solar by a margin that certifies nothing.
    ``attach_trailing_references`` supplies both, levelled per issue instant on a
    trailing 28-day window, so the constant sits beside the climatology and the
    difference between them is visible rather than assumed.
    """
    if panel.empty:
        return panel
    truth = (actuals.set_index(["country_code", "target"])["actual"])
    out = []
    for country, grp in panel.groupby("country_code", sort=True):
        grp = grp.copy()
        series = truth.loc[country] if country in truth.index.get_level_values(0) \
            else pd.Series(dtype=float)
        series = series.sort_index()

        grp["seasonal_naive"] = grp["target"].map(
            lambda t: series.get(t - pd.Timedelta(hours=SEASONAL_NAIVE_LAG_HOURS),
                                 np.nan))
        # Persistence for an issued forecast is the last whole hour of truth at
        # or before the issue instant -- `baselines.aligned_point_baselines`'s
        # rule, not the target's own lag, which no live system could hold.
        anchors = grp["generated_at"].dt.floor("h")
        grp["persistence"] = [
            series.loc[:anchor].iloc[-1] if len(series.loc[:anchor]) else np.nan
            for anchor in anchors]

        grp, _ = attach_trailing_references(grp, series,
                                            window_days=TRAILING_WINDOW_DAYS)
        out.append(grp)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Coverage (prereg section 6.1) and availability skew (section 4.4)
# --------------------------------------------------------------------------


def coverage_table(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Re-derive the prereg section 1 coverage table at 14+ days.

    The CEO instruction is explicit that the n=2-3 provisional figures of
    2026-08-14 are not to be carried forward. This is the replacement.
    """
    rows = []
    for forecast_type, panel in panels.items():
        if panel.empty:
            continue
        for band, grp in panel.groupby("band", sort=True):
            present = grp[grp["available"]]
            rows.append({
                "forecast_type": forecast_type,
                "band": band,
                "rows": len(grp),
                "countries": grp["country_code"].nunique(),
                "target_days": grp["target_day"].nunique(),
                "feature_present": int(grp["available"].sum()),
                "coverage_pct": round(100.0 * grp["available"].mean(), 2),
                # Lower bounds -- `first_seen_at` is our fetch, not their publish.
                "median_feature_lead_h": (
                    round(float(present["feature_lead_h"].median()), 2)
                    if len(present) else None),
                "median_feature_age_at_cutoff_h": (
                    round(float(present["feature_age_at_cutoff_h"].median()), 2)
                    if len(present) else None),
                "backtested": band not in NOT_BACKTESTED_BANDS,
            })
    return pd.DataFrame(rows)


def coverage_on_horizon_grid(tso_by_type: dict[str, pd.DataFrame],
                             cutoffs: pd.DatetimeIndex,
                             horizon_hours: int = 64) -> pd.DataFrame:
    """Prereg section 1's coverage question, on its own terms, at 14+ days.

    Section 1's provisional 78.1 / 70.8 / 31.5 / 16.0 / 0.0 came from a
    *horizon-grid* reconstruction: standing at one cutoff, over every target
    hour in ``(cutoff, cutoff + 64h]``, was a TSO value already first-seen? That
    is a different denominator from :func:`coverage_table`, which asks the same
    question only of the target hours our own production runs actually forecast.

    Both belong in the record and neither substitutes for the other. Reporting
    only the panel version would make a definitional change look like a coverage
    change against the section 1 figures the CEO asked to have re-derived;
    reporting only this version would overstate what the backtest can use.
    """
    rows = []
    for forecast_type, tso in tso_by_type.items():
        if tso.empty:
            continue
        ordered = tso.sort_values("first_seen")
        for cutoff in pd.DatetimeIndex(cutoffs).unique():
            window = ordered[
                (ordered["target"] > cutoff)
                & (ordered["target"] <= cutoff + pd.Timedelta(hours=horizon_hours))]
            if window.empty:
                continue
            per_target = window.groupby(["country_code", "target"], as_index=False).agg(
                first_seen=("first_seen", "min"))
            per_target["known"] = per_target["first_seen"] <= cutoff
            per_target["band"] = [
                band_of((t - cutoff).total_seconds() / 3600.0)
                for t in per_target["target"]]
            per_target = per_target.dropna(subset=["band"])
            for band, grp in per_target.groupby("band", observed=True):
                rows.append({"forecast_type": forecast_type, "cutoff": cutoff,
                             "band": band, "target_hours": len(grp),
                             "known": int(grp["known"].sum())})
    if not rows:
        return pd.DataFrame()
    detail = pd.DataFrame(rows)
    return (detail.groupby(["forecast_type", "band"], as_index=False)
                  .agg(cutoffs=("cutoff", "nunique"),
                       target_hours=("target_hours", "sum"),
                       known=("known", "sum"))
                  .assign(coverage_pct=lambda d: (100.0 * d["known"]
                                                  / d["target_hours"]).round(2)))


def availability_skew(panel: pd.DataFrame) -> pd.DataFrame:
    """Training missingness rate per band against the serve-time rate per band.

    Prereg section 4.4. Here the two rates are measured on the same reconstructed
    runs, so they agree by construction; the row exists so that a later
    *serving* proposal has a registered training rate to be checked against,
    rather than an assumed one. A feature trained at one availability rate and
    served at another is the wind-rebuild failure mode repeating.
    """
    if panel.empty:
        return pd.DataFrame()
    rows = []
    for (band, country), grp in panel.groupby(["band", "country_code"], sort=True):
        rows.append({"band": band, "country": country, "rows": len(grp),
                     "present_pct": round(100.0 * grp["available"].mean(), 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Primary estimator: the affine combiner
# --------------------------------------------------------------------------


@dataclass
class AffineFit:
    """One fitted combiner and everything needed to judge it."""
    n: int = 0
    days: int = 0
    countries: int = 0
    coef: list = field(default_factory=list)      # [a, b, c]
    c_hat: float = float("nan")
    c_ci: tuple = (float("nan"), float("nan"))
    c_ci_method: str = ""
    c_fold_spread: list = field(default_factory=list)
    wape_ours_raw: float = float("nan")
    wape_null_cv: float = float("nan")
    wape_combiner_cv: float = float("nan")
    delta_vs_null: float = float("nan")
    delta_vs_raw: float = float("nan")
    delta_ci: tuple = (float("nan"), float("nan"))
    corr_ours_tso: float = float("nan")
    vif: float = float("nan")
    verdict: str = ""


def _ols(y: np.ndarray, x_ours: np.ndarray, x_tso: np.ndarray | None):
    """Least squares on [1, f_ours] or [1, f_ours, f_tso]. None if degenerate."""
    cols = [np.ones_like(y), x_ours] + ([] if x_tso is None else [x_tso])
    design = np.column_stack(cols)
    if design.shape[0] <= design.shape[1]:
        return None
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None
    return beta


def fit_affine(frame: pd.DataFrame) -> AffineFit:
    """Blocked leave-one-day-out CV of the combiner against the c=0 null.

    The independent unit is the target *day*, not the row: within-day errors are
    strongly autocorrelated and European weather correlates across countries on
    top of that. Every prediction scored below is out-of-fold -- the day it
    lands on took no part in the fit that produced it -- so the two CV WAPEs are
    honest out-of-sample numbers and the comparison between them is the
    registered test of c = 0.

    The interval on ``c`` is the delete-one-day cluster jackknife with the
    standard (k-1)/k inflation. A plain t-interval over leave-one-out estimates
    would understate the variance badly, because each fold's estimate reuses
    k-1 of the same days.
    """
    fit = AffineFit()
    frame = frame.dropna(subset=["actual", "f_ours", "f_tso"])
    if frame.empty:
        fit.verdict = "no rows"
        return fit

    y = frame["actual"].to_numpy(float)
    xo = frame["f_ours"].to_numpy(float)
    xt = frame["f_tso"].to_numpy(float)
    days = frame["target_day"].to_numpy()
    unique_days = np.unique(days)

    fit.n = len(frame)
    fit.days = len(unique_days)
    fit.countries = frame["country_code"].nunique()

    # Both arms forecast the same quantity, so they are near-collinear by
    # construction and the *split* of a shared effect between b and c is weakly
    # identified even where the pair jointly fits well. That does not invalidate
    # the registered test -- `c = 0` is still exactly the question "does adding
    # f_tso help", and the CV WAPE delta is measured on predictions, which are
    # stable under collinearity -- but it is why an interval on `c` can be wide
    # beside a real WAPE gain, so it is reported rather than left to be
    # rediscovered by a reader.
    if len(frame) > 2 and np.std(xo) > 0 and np.std(xt) > 0:
        fit.corr_ours_tso = float(np.corrcoef(xo, xt)[0, 1])
        if abs(fit.corr_ours_tso) < 1.0:
            fit.vif = float(1.0 / (1.0 - fit.corr_ours_tso ** 2))

    full = _ols(y, xo, xt)
    if full is None:
        fit.verdict = "degenerate fit"
        return fit
    fit.coef = [float(v) for v in full]
    fit.c_hat = float(full[2])

    if fit.days < 2:
        fit.verdict = "single day -- no blocked CV possible"
        return fit

    pred_comb = np.full(len(y), np.nan)
    pred_null = np.full(len(y), np.nan)
    fold_c = []
    for day in unique_days:
        held = days == day
        train = ~held
        if train.sum() <= 3:
            continue
        beta_c = _ols(y[train], xo[train], xt[train])
        beta_n = _ols(y[train], xo[train], None)
        if beta_c is None or beta_n is None:
            continue
        fold_c.append(float(beta_c[2]))
        pred_comb[held] = beta_c[0] + beta_c[1] * xo[held] + beta_c[2] * xt[held]
        pred_null[held] = beta_n[0] + beta_n[1] * xo[held]

    scored = np.isfinite(pred_comb) & np.isfinite(pred_null)
    if not scored.any():
        fit.verdict = "no fold produced a prediction"
        return fit

    fit.wape_ours_raw = wape(xo[scored] - y[scored], y[scored])
    fit.wape_null_cv = wape(pred_null[scored] - y[scored], y[scored])
    fit.wape_combiner_cv = wape(pred_comb[scored] - y[scored], y[scored])
    fit.delta_vs_null = fit.wape_combiner_cv - fit.wape_null_cv
    fit.delta_vs_raw = fit.wape_combiner_cv - fit.wape_ours_raw

    k = len(fold_c)
    fit.c_fold_spread = [round(v, 5) for v in fold_c]
    if k >= 2:
        arr = np.asarray(fold_c, dtype=float)
        var = (k - 1) / k * float(((arr - arr.mean()) ** 2).sum())
        half = t_crit(k) * float(np.sqrt(var))
        fit.c_ci = (round(fit.c_hat - half, 5), round(fit.c_hat + half, 5))
        fit.c_ci_method = f"delete-one-day cluster jackknife, k={k} days"

    # The same day-block resampling applied to the *decision* quantity: the
    # per-day WAPE difference between the two out-of-fold arms.
    per_day = []
    for day in unique_days:
        held = (days == day) & scored
        if not held.any():
            continue
        per_day.append(wape(pred_comb[held] - y[held], y[held])
                       - wape(pred_null[held] - y[held], y[held]))
    if len(per_day) >= 2:
        arr = np.asarray(per_day, dtype=float)
        se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        half = t_crit(len(arr)) * se
        fit.delta_ci = (round(float(arr.mean()) - half, 4),
                        round(float(arr.mean()) + half, 4))

    lo, hi = fit.c_ci
    if not np.isfinite(lo):
        fit.verdict = "no interval -- underpowered"
    elif lo <= 0.0 <= hi:
        fit.verdict = "c CI includes 0 -- feature has NOT earned its place"
    else:
        fit.verdict = "c CI excludes 0"
    return fit


# --------------------------------------------------------------------------
# Scoring (prereg section 5)
# --------------------------------------------------------------------------


ARMS = ("f_ours", "f_tso", "seasonal_naive", "persistence") + TRAILING_COMPARATORS


def score_by_country(panel: pd.DataFrame, matched: bool) -> pd.DataFrame:
    """WAPE per (country, band) for every arm, with its own n beside it.

    ``matched`` selects the availability-matched intersection -- rows where the
    feature is present -- which prereg section 5 makes primary. Each arm is
    scored on the rows *it* can be scored on, and reports that n: a climatology
    is 24 numbers and can be missing an hour bucket where a forecast is not, so
    a shared-intersection rule would silently delete rows from arms that had
    them.
    """
    frame = panel[panel["available"]] if matched else panel
    rows = []
    for (band, country), grp in frame.groupby(["band", "country_code"], sort=True):
        rec = {"band": band, "country": country, "n_rows": len(grp),
               "n_days": grp["target_day"].nunique(),
               "mean_actual_mw": round(float(grp["actual"].mean()), 2),
               "feature_present_pct": round(100.0 * grp["available"].mean(), 2)}
        for arm in ARMS:
            if arm not in grp.columns:
                continue
            ok = grp[arm].notna() & grp["actual"].notna()
            rec[f"n_{arm}"] = int(ok.sum())
            rec[f"wape_{arm}"] = (
                round(wape(grp.loc[ok, arm] - grp.loc[ok, "actual"],
                           grp.loc[ok, "actual"]), 3) if ok.any() else None)
        rows.append(rec)
    return pd.DataFrame(rows)


def composition_term(panel: pd.DataFrame) -> pd.DataFrame:
    """Our own arm, scored on all rows and on the matched subset.

    The gap between the two is a **coverage effect** -- the rows differ -- and
    not a feature effect. Prereg section 5 requires it named rather than left
    for a reader to infer, because it is the one number that can make a routed
    design look like a modelling win.
    """
    rows = []
    for band, grp in panel.groupby("band", sort=True):
        matched = grp[grp["available"]]
        all_w = wape(grp["f_ours"] - grp["actual"], grp["actual"])
        mat_w = (wape(matched["f_ours"] - matched["actual"], matched["actual"])
                 if len(matched) else float("nan"))
        rows.append({
            "band": band,
            "n_all_rows": len(grp),
            "n_matched": len(matched),
            "wape_ours_all_rows": round(all_w, 3),
            "wape_ours_matched": round(mat_w, 3) if np.isfinite(mat_w) else None,
            "composition_term_pp": (round(mat_w - all_w, 3)
                                    if np.isfinite(mat_w) else None),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run(db_path: str, out_dir: Path) -> dict:
    conn = connect_ro(db_path)
    try:
        record: dict = {
            "issue": "ABL-247",
            "preregistration": "abl247-prereg (written 2026-08-14)",
            "replica": replica_state(conn, db_path),
            "backfill": backfill_census(conn),
            "production_models": {t: PRODUCTION_MODELS[t] for t in FORECAST_TYPES},
            "contamination": {
                "ABL-111/ABL-109": "screened on the load target; count in reads.load",
                "ABL-71": "prod ingest staleness bounded in replica.archive_staleness_hours",
                "ABL-431/ABL-458": "guard_tso_series applied to every archive read; "
                                   "refusals counted per series",
                "ABL-67": "not applicable -- net_position is out of scope here",
            },
            "reads": {}, "fits": {}, "scores": {},
        }

        start = FIRST_CLEAN_TARGET_DAY
        end = (pd.Timestamp(record["replica"]["archive_max_tso_target"] or
                            FIRST_CLEAN_TARGET_DAY).normalize()
               + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        panels = {}
        tso_by_type: dict[str, pd.DataFrame] = {}
        run_hours: list[float] = []
        for forecast_type in FORECAST_TYPES:
            tso = read_tso_vintages(conn, forecast_type)
            tso_by_type[forecast_type] = tso
            ours = read_our_forecasts(conn, forecast_type, start, end)
            if len(ours):
                run_hours.extend(ours["generated_at"].dt.hour
                                 + ours["generated_at"].dt.minute / 60.0)
            actuals, notes = read_actuals(conn, forecast_type, start, end)
            panel = build_panel(ours, tso, actuals)
            if not panel.empty:
                panel = attach_references(panel, actuals)
            panels[forecast_type] = panel
            record["reads"][forecast_type] = {
                "tso_rows_read": tso.attrs.get("rows_read", 0),
                "tso_guard_refusals": tso.attrs.get("guard_refusals", 0),
                "tso_rows_kept": len(tso),
                "our_forecast_rows": len(ours),
                "our_model": PRODUCTION_MODELS[forecast_type],
                "our_generated_at_count": (int(ours["generated_at"].nunique())
                                           if len(ours) else 0),
                "actuals": notes,
                "panel_rows": len(panel),
                "panel_countries": (int(panel["country_code"].nunique())
                                    if len(panel) else 0),
                "panel_days": (int(panel["target_day"].nunique())
                               if len(panel) else 0),
                "window": [start, end],
            }

        record["coverage"] = coverage_table(panels).to_dict("records")

        # The section 1 comparison. The run hour is the *measured* median of our
        # own `generated_at` values, not the 06:00 the probe assumed and not the
        # 18:00 the scheduler file implies -- CLAUDE.md's standing warning about
        # `RUN_HOUR` is that the two disagree, so the grid is anchored on what
        # the rows say rather than on either document.
        grid_hour = float(np.median(run_hours)) if run_hours else 6.0
        days = pd.date_range(FIRST_CLEAN_TARGET_DAY,
                             pd.Timestamp(end) - pd.Timedelta(days=1), freq="D")
        grid_cutoffs = pd.DatetimeIndex(
            [d + pd.Timedelta(hours=grid_hour) for d in days])
        grid = coverage_on_horizon_grid(tso_by_type, grid_cutoffs)
        record["coverage_horizon_grid"] = {
            "run_hour_utc_median_measured": round(grid_hour, 2),
            "cutoff_days": len(grid_cutoffs),
            "note": ("prereg section 1's own denominator -- every target hour in "
                     "(cutoff, cutoff+64h], not only the hours our production "
                     "runs forecast. Directly comparable to the provisional "
                     "78.1 / 70.8 / 31.5 / 16.0 / 0.0."),
            "rows": ([] if grid.empty else grid.to_dict("records")),
        }

        for forecast_type, panel in panels.items():
            if panel.empty:
                record["fits"][forecast_type] = {"status": "no panel"}
                continue
            fits = {}
            for band, _lo, _hi in BANDS:
                grp = panel[panel["band"] == band]
                if band in NOT_BACKTESTED_BANDS:
                    fits[band] = {
                        "status": "re-scoped out (prereg section 1) -- not fitted",
                        "rows": len(grp),
                        "feature_present": int(grp["available"].sum()) if len(grp) else 0,
                    }
                    continue
                if grp.empty:
                    fits[band] = {"status": "no rows"}
                    continue
                pooled = fit_affine(grp)
                per_country = {}
                for country, sub in grp.groupby("country_code", sort=True):
                    cf = fit_affine(sub)
                    per_country[country] = {
                        "n": cf.n, "days": cf.days, "c_hat": round(cf.c_hat, 5),
                        "c_ci": list(cf.c_ci),
                        "wape_ours_raw": round(cf.wape_ours_raw, 3),
                        "wape_null_cv": round(cf.wape_null_cv, 3),
                        "wape_combiner_cv": round(cf.wape_combiner_cv, 3),
                        "delta_vs_null_pp": round(cf.delta_vs_null, 3),
                        "verdict": cf.verdict,
                    }
                fits[band] = {
                    "status": "fitted",
                    "pooled": {
                        "n": pooled.n, "days": pooled.days,
                        "countries": pooled.countries,
                        "coef_a_b_c": [round(v, 5) for v in pooled.coef],
                        "c_hat": round(pooled.c_hat, 5),
                        "c_ci": list(pooled.c_ci),
                        "c_ci_method": pooled.c_ci_method,
                        "wape_ours_raw": round(pooled.wape_ours_raw, 3),
                        "wape_null_cv": round(pooled.wape_null_cv, 3),
                        "wape_combiner_cv": round(pooled.wape_combiner_cv, 3),
                        "delta_vs_null_pp": round(pooled.delta_vs_null, 3),
                        "delta_vs_null_ci_pp": list(pooled.delta_ci),
                        "delta_vs_raw_pp": round(pooled.delta_vs_raw, 3),
                        "corr_ours_tso": round(pooled.corr_ours_tso, 4),
                        "vif": round(pooled.vif, 2),
                        "verdict": pooled.verdict,
                    },
                    "per_country": per_country,
                }
            record["fits"][forecast_type] = fits

            record["scores"][forecast_type] = {
                "availability_matched": score_by_country(panel, matched=True)
                                        .to_dict("records"),
                "all_rows": score_by_country(panel, matched=False).to_dict("records"),
                "composition": composition_term(panel).to_dict("records"),
                "availability_skew": availability_skew(panel).to_dict("records"),
            }
    finally:
        conn.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "abl247_tso_feature_backtest.json").write_text(
        json.dumps(json_safe(record), indent=2, default=str), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=os.environ.get("ENERGY_DB_PATH",
                               r"C:\Code\able\data\energy_dashboard.db"),
        help="Replica path. A worktree has no .env, so pass this explicitly.")
    parser.add_argument("--out", default="reports/abl247",
                        help="Directory for the JSON record.")
    args = parser.parse_args()

    record = run(args.db, Path(args.out))

    print(f"replica            {record['replica']['replica_path']}")
    print(f"archive max seen   {record['replica']['archive_max_first_seen_at']} "
          f"(stale {record['replica']['archive_staleness_hours']} h)")
    print(f"backfill excluded  {record['backfill']['backfill_rows_excluded']:,} rows "
          f"({record['backfill']['backfill_target_span'][0]} .. "
          f"{record['backfill']['backfill_target_span'][1]})")
    print(f"genuine days       {record['backfill']['genuine_target_days']}")
    print()
    print("== coverage, re-derived at 14+ days (prereg section 6.1) ==")
    print(pd.DataFrame(record["coverage"]).to_string(index=False))
    print()
    for forecast_type, fits in record["fits"].items():
        print(f"== {forecast_type} ==")
        for band, entry in fits.items():
            if entry.get("status") != "fitted":
                print(f"  {band:8s} {entry.get('status')}")
                continue
            p = entry["pooled"]
            print(f"  {band:8s} n={p['n']:6d} days={p['days']:3d} "
                  f"c={p['c_hat']:+.4f} CI={p['c_ci']}  "
                  f"WAPE ours={p['wape_ours_raw']:.2f} null={p['wape_null_cv']:.2f} "
                  f"comb={p['wape_combiner_cv']:.2f} "
                  f"d={p['delta_vs_null_pp']:+.3f}pp -> {p['verdict']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
