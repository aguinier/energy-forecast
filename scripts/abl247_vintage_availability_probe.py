#!/usr/bin/env python
"""ABL-247 design probe: is a TSO day-ahead vintage usable as a model feature?

Two questions, both answerable before the 14-day backtest clock expires on
2026-08-25, and both load-bearing for what that backtest should even run.

**Q1 (GATE 1 efficacy).** ABL-458 registered `forecast_vintage_archive` in the
ABL-431 plausibility guard. Registration is not efficacy. This re-reads the
known HU `wind_onshore` contamination *through* the merged guard and reports
whether those rows are actually refused, and whether anything else is.

**Q2 (feature availability).** A TSO series is only a feature if it is published
before the model's cutoff. `first_seen_at` is the archive's first observation of
a (country, type, target, value) tuple -- an *upper bound* on the TSO's true
publication time, loose by at most the archiver's poll gap. For targets whose
first publication post-dates the archive go-live, `target - first_seen` is
therefore a conservative lower bound on the lead time the feature would have
had. Targets retained at go-live are excluded: their `first_seen_at` is the
backfill instant, which measures our ingest, not the TSO.

Read-only against the replica. Writes nothing.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tso_plausibility import (  # noqa: E402
    VINTAGE_ARCHIVE_DAY_AHEAD_MODEL,
    VINTAGE_ARCHIVE_TABLE,
    guard_tso_series,
    reference_scale,
)

#: Archive go-live (ABL-184). Every row whose `first_seen_at` is at or before
#: this instant is backfill: its first-seen stamp is when we started archiving,
#: not when the TSO published. Q2 excludes them by target date rather than by
#: this instant directly, because a target already inside the retained window at
#: go-live was backfilled even though later revisions of it were not.
GO_LIVE = "2026-08-11T19:16:13Z"

#: First target date whose *initial* publication is guaranteed to post-date
#: go-live. A D+1 product published on 08-11 covers targets through 08-12, so
#: 08-13 is the first target day the archive can have seen from its first
#: publication. Deliberately one day past the arithmetic minimum.
FIRST_CLEAN_TARGET_DAY = "2026-08-13"

#: The known contaminated cluster, from ABL-431 / ABL-458. Quoted here so the
#: probe fails loudly if the archive stops matching what the guard was built
#: against, rather than silently reporting "nothing flagged" as a pass.
KNOWN_CONTAMINATION = {
    "country_code": "HU",
    "forecast_type": "wind_onshore",
    "value_mw": 140996.245,
    "expected_rows": 96,
}


def connect(db_path: str) -> sqlite3.Connection:
    """Open the replica read-only. The URI form is not optional here."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def q1_guard_efficacy(conn: sqlite3.Connection) -> pd.DataFrame:
    """Does the merged guard actually refuse the archive contamination?"""
    known = KNOWN_CONTAMINATION
    df = pd.read_sql_query(
        f"""
        SELECT target_timestamp_utc, forecast_value, first_seen_at
        FROM {VINTAGE_ARCHIVE_TABLE}
        WHERE source = 'tso'
          AND model_name = ?
          AND forecast_type = ?
          AND country_code = ?
        """,
        conn,
        params=[VINTAGE_ARCHIVE_DAY_AHEAD_MODEL, known["forecast_type"],
                known["country_code"]],
    )
    if df.empty:
        raise SystemExit(
            f"Q1: no {known['country_code']} {known['forecast_type']} rows in "
            f"the archive at all -- the probe cannot certify the guard."
        )

    reference = reference_scale(
        conn, known["country_code"], VINTAGE_ARCHIVE_TABLE, known["forecast_type"]
    )
    values = pd.Series(
        df["forecast_value"].to_numpy(),
        index=pd.DatetimeIndex(df["target_timestamp_utc"]),
    )
    guarded = guard_tso_series(
        values, conn, known["country_code"], VINTAGE_ARCHIVE_TABLE,
        known["forecast_type"], context="abl247-q1",
    )

    refused = guarded.isna() & values.notna()
    at_known_value = values.eq(known["value_mw"])

    print("== Q1: GATE 1 efficacy -- guard vs the archive contamination ==")
    print(f"  pair                {known['country_code']} {known['forecast_type']} "
          f"({VINTAGE_ARCHIVE_TABLE}, model={VINTAGE_ARCHIVE_DAY_AHEAD_MODEL})")
    print(f"  reference_mw        {reference.reference_mw!r} "
          f"(q={reference.quantile}, evaluable={reference.evaluable})")
    print(f"  threshold_mw        {reference.threshold_mw!r} "
          f"(tolerance={reference.tolerance})")
    print(f"  reference reason    {reference.reason}")
    print(f"  rows read           {len(values)}")
    print(f"  rows at {known['value_mw']} MW  {int(at_known_value.sum())} "
          f"(expected {known['expected_rows']})")
    print(f"  rows refused        {int(refused.sum())}")
    print(f"  refused NOT at the known value  "
          f"{int((refused & ~at_known_value).sum())}")
    print(f"  known-value rows NOT refused    "
          f"{int((at_known_value & ~refused).sum())}")
    if refused.any():
        print(f"  refused span        {refused[refused].index.min()} .. "
              f"{refused[refused].index.max()}")
    survivors = values[~refused]
    print(f"  surviving max_mw    {survivors.max() if len(survivors) else None}")
    return df


def q2_availability(conn: sqlite3.Connection) -> pd.DataFrame:
    """Per (country, series), how much lead time would the feature have had?

    Restricted to targets whose first publication post-dates go-live, so that
    `first_seen_at` measures the TSO rather than our backfill.
    """
    df = pd.read_sql_query(
        f"""
        SELECT country_code,
               forecast_type,
               model_name,
               COUNT(*)                                   AS n_rows,
               COUNT(DISTINCT target_timestamp_utc)       AS n_targets,
               MIN(lead_hours)                            AS lead_min_h,
               MAX(lead_hours)                            AS lead_max_h,
               AVG(lead_hours)                            AS lead_mean_h
        FROM (
            SELECT country_code, forecast_type, model_name,
                   target_timestamp_utc,
                   (julianday(target_timestamp_utc)
                    - julianday(REPLACE(SUBSTR(first_seen_at, 1, 19), 'T', ' ')))
                   * 24.0 AS lead_hours
            FROM {VINTAGE_ARCHIVE_TABLE}
            WHERE source = 'tso'
              AND target_timestamp_utc >= ?
        )
        GROUP BY country_code, forecast_type, model_name
        ORDER BY forecast_type, country_code, model_name
        """,
        conn,
        params=[FIRST_CLEAN_TARGET_DAY],
    )
    return df


def q3_coverage_at_cutoff(conn: sqlite3.Connection, cutoff: str,
                          horizon_hours: int = 64) -> pd.DataFrame:
    """The feature-coverage table a leak-free backtest would actually see.

    Reconstructs one model run: standing at ``cutoff``, for every target hour in
    ``(cutoff, cutoff + horizon_hours]``, was a TSO day-ahead value already
    first-seen? This is the same `first_seen_at <= cutoff` reconstruction the
    backtest will do, run early against real vintages so the shape is known --
    and the code exercised -- before the 14-day clock expires.

    Bucketed by horizon so the answer is per-band rather than an average that
    hides a cliff.
    """
    df = pd.read_sql_query(
        f"""
        SELECT country_code,
               forecast_type,
               target_timestamp_utc,
               MIN(first_seen_at) AS first_seen_at
        FROM {VINTAGE_ARCHIVE_TABLE}
        WHERE source = 'tso'
          AND model_name = ?
          AND target_timestamp_utc > ?
          AND target_timestamp_utc <= datetime(?, '+' || ? || ' hours')
        GROUP BY country_code, forecast_type, target_timestamp_utc
        """,
        conn,
        params=[VINTAGE_ARCHIVE_DAY_AHEAD_MODEL, cutoff, cutoff, horizon_hours],
    )
    if df.empty:
        return df

    target = pd.to_datetime(df["target_timestamp_utc"])
    seen = pd.to_datetime(df["first_seen_at"].str.slice(0, 19).str.replace("T", " "))
    cut = pd.Timestamp(cutoff)

    df["horizon_h"] = (target - cut).dt.total_seconds() / 3600.0
    df["known_at_cutoff"] = seen <= cut
    df["band"] = pd.cut(df["horizon_h"], bins=[0, 24, 48, 64],
                        labels=["0-24h", "24-48h", "48-64h"], right=True)

    return (df.groupby(["forecast_type", "band"], observed=True)
            .agg(target_hours=("known_at_cutoff", "size"),
                 known=("known_at_cutoff", "sum"))
            .assign(coverage_pct=lambda d: 100.0 * d["known"] / d["target_hours"])
            .reset_index())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=os.environ.get("ENERGY_DB_PATH", r"C:\Code\able\data\energy_dashboard.db"),
        help="Replica path. A worktree has no .env, so pass this explicitly.",
    )
    parser.add_argument("--cutoff-hour-utc", type=float, default=6.0,
                        help="Model run hour, UTC. The measured net-position "
                             "rail runs at ~06:00, not the 18:00 the scheduler "
                             "file implies (see CLAUDE.md / RUN_HOUR).")
    parser.add_argument("--cutoff", default="2026-08-13 06:00:00",
                        help="Model-run cutoff to reconstruct for Q3. Must be "
                             "a genuine post-go-live instant, or first_seen_at "
                             "measures the backfill instead of the TSO.")
    args = parser.parse_args()

    print(f"replica: {args.db}")
    print(f"go-live: {GO_LIVE}   clean-target floor: {FIRST_CLEAN_TARGET_DAY}")
    print()

    conn = connect(args.db)
    try:
        q1_guard_efficacy(conn)
        print()
        avail = q2_availability(conn)
        print("== Q2: feature availability from genuine post-go-live vintages ==")
        if avail.empty:
            print("  no post-go-live targets yet -- the clock has not accrued.")
        else:
            summary = (avail.groupby(["forecast_type", "model_name"])
                       .agg(countries=("country_code", "nunique"),
                            targets=("n_targets", "sum"),
                            lead_max_h=("lead_max_h", "max"),
                            lead_max_h_median_over_countries=("lead_max_h", "median"),
                            lead_min_h=("lead_min_h", "min"))
                       .reset_index())
            with pd.option_context("display.width", 200,
                                   "display.max_columns", 20,
                                   "display.max_rows", 200):
                print(summary.to_string(index=False))
                print()
                print("-- per country/series (lead hours, target minus first_seen) --")
                print(avail.to_string(index=False))

        print()
        print(f"== Q3: coverage at a {args.cutoff} cutoff (leak-free reconstruction) ==")
        cov = q3_coverage_at_cutoff(conn, args.cutoff)
        if cov.empty:
            print("  no targets in the horizon window at that cutoff.")
        else:
            with pd.option_context("display.width", 200):
                print(cov.to_string(index=False))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
