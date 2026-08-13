#!/usr/bin/env python3
"""ABL-322 pre-registration probe — read-only, no fitting, no model.

Answers three questions that must be settled *before* DE/NL wind_offshore are
trained, and that do not depend on either blocker (ABL-331, ABL-332):

1. What cadence do DE and NL `wind_offshore` actually carry in
   `energy_generation`, and how many samples does an hourly `:00` read discard?
2. Is the `:00` instant a faithful estimator of the hourly mean for these two
   series, or does sampling alias a volatile signal?  This decides what the
   *actual* is that the ABL-322 gate scores against -- a scoring question, so
   it is settled here rather than deferred to the builder fix.
3. What is the seasonal-naive D-7 bar on the frozen ABL-195 gate window?
   The challenger does not exist yet, so this is the bar stated before any
   result can influence it.

Read-only by construction: the replica is opened with SQLite `mode=ro`,
`uri=True` and nothing is written anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# Frozen, inherited verbatim from experiments/ABL195/config.json.  These bounds
# were fixed by a prior issue, before ABL-322 existed -- they are not chosen
# here and cannot be shopped.
FIT_START = pd.Timestamp("2026-01-14 00:00:00")
GATE_START = pd.Timestamp("2026-07-11 00:00:00")
GATE_END = pd.Timestamp("2026-08-10 00:00:00")

PAIRS = [("DE", "wind_offshore"), ("NL", "wind_offshore")]
# Reference pairs: already served, already gate-read on this exact window by
# ABL-195.  They calibrate whether anything measured here is DE/NL-specific.
REFERENCE = [("BE", "wind_offshore"), ("FR", "wind_offshore")]
COLUMN = "wind_offshore_mw"


def ro_connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)


def load(con: sqlite3.Connection, country: str, start, end) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"SELECT timestamp_utc, {COLUMN} AS value FROM energy_generation "
        "WHERE country_code = ? AND timestamp_utc >= ? AND timestamp_utc < ? "
        f"AND {COLUMN} IS NOT NULL ORDER BY timestamp_utc",
        con,
        params=(country, str(start), str(end)),
    )
    if df.empty:
        return df
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], format="mixed", utc=True
    ).dt.tz_localize(None)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def wape(actual: np.ndarray, pred: np.ndarray) -> float | None:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any() or np.abs(actual[mask]).sum() == 0:
        return None
    return float(100 * np.abs(actual[mask] - pred[mask]).sum() / np.abs(actual[mask]).sum())


def cadence_report(df: pd.DataFrame) -> dict:
    minutes = df["timestamp_utc"].dt.minute.value_counts().sort_index()
    return {
        "rows": int(len(df)),
        "minute_offsets": {int(k): int(v) for k, v in minutes.items()},
        "distinct_minute_offsets": int(len(minutes)),
        "rows_at_00": int(minutes.get(0, 0)),
        "discarded_by_hourly_read": int(len(df) - minutes.get(0, 0)),
    }


def aliasing_report(df: pd.DataFrame) -> dict:
    """Compare the `:00` instant against the mean of its own hour.

    Only hours that carry every expected sub-hourly sample are compared, so a
    partially-observed hour cannot masquerade as disagreement.
    """
    d = df.set_index("timestamp_utc")["value"]
    hour = d.index.floor("h")
    grouped = d.groupby(hour)
    hourly_mean = grouped.mean()
    hourly_count = grouped.count()
    at_00 = d[d.index.minute == 0]
    at_00.index = at_00.index.floor("h")

    expected = int(hourly_count.mode().iloc[0]) if len(hourly_count) else 0
    full = hourly_count[hourly_count == expected].index
    common = hourly_mean.index.intersection(at_00.index).intersection(full)
    if len(common) == 0 or expected <= 1:
        return {"comparable_hours": 0, "expected_samples_per_hour": expected,
                "note": "hourly series; instant and mean coincide"}

    mean_v = hourly_mean.loc[common].to_numpy(float)
    inst_v = at_00.loc[common].to_numpy(float)
    diff = inst_v - mean_v
    return {
        "comparable_hours": int(len(common)),
        "expected_samples_per_hour": expected,
        "mean_mw": float(np.mean(mean_v)),
        "instant_vs_mean_mae_mw": float(np.mean(np.abs(diff))),
        "instant_vs_mean_wape_pct": wape(mean_v, inst_v),
        "instant_vs_mean_bias_mw": float(np.mean(diff)),
        "instant_vs_mean_p95_abs_mw": float(np.percentile(np.abs(diff), 95)),
        "max_abs_mw": float(np.max(np.abs(diff))),
    }


def d7_baseline(df: pd.DataFrame, agg: str) -> dict:
    """Literal seasonal-naive D-7 on the frozen gate window.

    agg='instant' scores the `:00` read (today's builder convention);
    agg='hourly_mean' scores the mean of each hour.  Both use the identical
    finite intersection so the two numbers are comparable.
    """
    d = df.set_index("timestamp_utc")["value"]
    if agg == "hourly_mean":
        series = d.groupby(d.index.floor("h")).mean()
    else:
        series = d[d.index.minute == 0]
        series.index = series.index.floor("h")
    series = series[~series.index.duplicated()].sort_index()

    gate = series[(series.index >= GATE_START) & (series.index < GATE_END)]
    lagged = series.reindex(gate.index - pd.Timedelta(days=7))
    actual = gate.to_numpy(float)
    pred = lagged.to_numpy(float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    return {
        "aggregation": agg,
        "gate_hours_available": int(len(gate)),
        "n_scored": int(mask.sum()),
        "mean_actual_mw": float(np.mean(actual[mask])) if mask.any() else None,
        "d7_wape_pct": wape(actual, pred),
        "d7_mae_mw": float(np.mean(np.abs(actual[mask] - pred[mask]))) if mask.any() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica-db", default=os.environ.get("ENERGY_DB_PATH"))
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if not args.replica_db:
        parser.error("--replica-db or ENERGY_DB_PATH is required")

    replica = Path(args.replica_db)
    out = {
        "replica_db": str(replica),
        "replica_bytes": replica.stat().st_size,
        "access": "sqlite mode=ro, uri=True (read-only)",
        "source_table": "energy_generation",
        "windows": {
            "fit_start": str(FIT_START),
            "gate_start": str(GATE_START),
            "gate_end_exclusive": str(GATE_END),
            "provenance": "inherited verbatim from experiments/ABL195/config.json",
        },
        "pairs": {},
    }

    con = ro_connect(str(replica))
    try:
        for country, ftype in PAIRS + REFERENCE:
            full = load(con, country, FIT_START, GATE_END)
            gate_win = load(con, country, GATE_START - pd.Timedelta(days=8), GATE_END)
            entry = {
                "forecast_type": ftype,
                "role": "ABL-322 scope" if (country, ftype) in PAIRS else "ABL-195 reference (already served)",
                "cadence_fit_to_gate": cadence_report(full) if not full.empty else {"rows": 0},
                "aliasing": aliasing_report(full) if not full.empty else {},
                "d7_baseline": [
                    d7_baseline(gate_win, "instant") if not gate_win.empty else {},
                    d7_baseline(gate_win, "hourly_mean") if not gate_win.empty else {},
                ],
            }
            out["pairs"][f"{country}:{ftype}"] = entry
    finally:
        con.close()

    text = json.dumps(out, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
