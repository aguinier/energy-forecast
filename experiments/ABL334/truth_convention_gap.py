#!/usr/bin/env python3
"""ABL-334 — how far apart are the `:00` sample and the hourly mean, in WAPE?

Registered as "Registered addition 2" in `experiments/ABL321/protocol.md`,
Amendment 3, before any metric was read.

`scripts/evaluate_renewable_source_switch.py:_truth_series` scores against the
**instantaneous `:00` sample** (`frame[frame["h"].dt.minute == 0]`). ABL-332 does
not touch it, which is exactly what makes the three runs comparable. But
post-ABL-332 the builder's target — and so this backtest's fitted target, via
`wind_retrain.build_vintage_frame`'s `actuals = builder._actuals` — is the
**hourly mean**. Pre-fix the two were the same statistic. Post-fix they are not.

This script measures the gap directly, with no model in the loop. For each
serving pair over the registered gate window it scores the hourly mean *as if it
were a forecast* of the `:00` sample. That number is the WAPE a **perfect**
hourly-mean predictor would still be charged by this harness — an irreducible
floor on the post-fix arm's measured error that is a scoring convention, not an
accuracy loss.

Read-only: `mode=ro`, `uri=True`. Writes one JSON under `experiments/ABL334/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.evaluation.scorecard import _ro_connect  # noqa: E402

REPLICA = r"C:\Code\able\data\energy_dashboard.db"
GATE_START, GATE_END = "2026-07-11", "2026-08-10"

SERVED_PAIRS = [
    ("AT", "solar"), ("BE", "solar"), ("DE", "solar"), ("FR", "solar"),
    ("AT", "wind_onshore"), ("BE", "wind_onshore"),
    ("DE", "wind_onshore"), ("FR", "wind_onshore"),
    ("BE", "wind_offshore"), ("FR", "wind_offshore"),
]
TRUTH_COLUMNS = {"solar": "solar_mw", "wind_onshore": "wind_onshore_mw",
                 "wind_offshore": "wind_offshore_mw"}


def wape(actual: np.ndarray, pred: np.ndarray) -> float | None:
    denom = np.abs(actual).sum()
    if denom == 0:
        return None
    return float(100.0 * np.abs(actual - pred).sum() / denom)


def measure(table: str, country: str, stream: str) -> dict:
    con = _ro_connect(REPLICA)
    try:
        frame = pd.read_sql_query(
            f"SELECT timestamp_utc, {TRUTH_COLUMNS[stream]} AS v FROM {table} "
            "WHERE country_code=? AND timestamp_utc>=? AND timestamp_utc<? "
            "AND data_quality='actual'",
            con, params=(country, GATE_START, GATE_END))
    finally:
        con.close()
    if frame.empty:
        return {"table": table, "n_rows": 0}

    ts = pd.to_datetime(frame["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    frame = frame.assign(h=ts).dropna(subset=["v"])
    if frame.empty:
        return {"table": table, "n_rows": 0}

    # Exactly what _truth_series does.
    at_00 = frame[frame["h"].dt.minute == 0].groupby("h")["v"].last().astype(float).sort_index()
    # Exactly what db.aggregate_renewable_to_hourly does.
    hourly_mean = frame.assign(h=frame["h"].dt.floor("h")).groupby("h")["v"].mean().sort_index()

    common = at_00.index.intersection(hourly_mean.index)
    a, m = at_00.loc[common].to_numpy(float), hourly_mean.loc[common].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(m)
    a, m = a[ok], m[ok]
    if not len(a):
        return {"table": table, "n_rows": int(len(frame)), "n_common": 0}

    diff = np.abs(a - m)
    sub_hourly = bool((frame["h"].dt.minute != 0).any())
    return {
        "table": table,
        "sub_hourly": sub_hourly,
        "n_rows": int(len(frame)),
        "n_common_hours": int(len(a)),
        "mean_level_mw": round(float(np.abs(a).mean()), 2),
        "wape_mean_vs_00_pct": None if wape(a, m) is None else round(wape(a, m), 4),
        "median_abs_diff_mw": round(float(np.median(diff)), 2),
        "p90_abs_diff_mw": round(float(np.percentile(diff, 90)), 2),
        "max_abs_diff_mw": round(float(diff.max()), 2),
    }


def main() -> int:
    out = {"window": {"start": GATE_START, "end_exclusive": GATE_END},
           "what": "hourly mean scored as a forecast of the :00 sample "
                   "-- the floor this harness charges a perfect hourly-mean predictor",
           "pairs": []}
    print(f"Gate window {GATE_START} -> {GATE_END} (exclusive), replica {REPLICA}\n")
    header = (f"{'pair':22} {'table':20} {'sub-h':6} {'n':>6} {'WAPE(mean vs :00)':>18} "
              f"{'med |Δ| MW':>11} {'p90 |Δ| MW':>11}")
    print(header)
    print("-" * len(header))
    for country, stream in SERVED_PAIRS:
        row = {"country": country, "stream": stream, "sources": {}}
        for table in ("energy_renewable", "energy_generation"):
            r = measure(table, country, stream)
            row["sources"][table] = r
            if r.get("n_common_hours"):
                print(f"{country + '/' + stream:22} {table:20} "
                      f"{str(r['sub_hourly']):6} {r['n_common_hours']:6d} "
                      f"{r['wape_mean_vs_00_pct']:17.4f}% "
                      f"{r['median_abs_diff_mw']:11.1f} {r['p90_abs_diff_mw']:11.1f}")
            else:
                print(f"{country + '/' + stream:22} {table:20} {'-':6} {'-':>6} "
                      f"{'no data':>18} {'-':>11} {'-':>11}")
        out["pairs"].append(row)

    dest = REPO / "experiments/ABL334/truth_convention_gap.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
