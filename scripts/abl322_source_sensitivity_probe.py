#!/usr/bin/env python3
"""ABL-322 source-sensitivity probe — how much does the ABL-345 defect move *this* gate?

Read-only, no fitting, no model, no writes to either database.

ABL-345 (a first-class blocker on ABL-322) is filed on the finding that both gate
harnesses construct `RenewableFeatureBuilder` with no `actuals_source`, so a run
fits `energy_renewable` whatever the intent. The CEO triage sizes the damage to
this pilot at "13-16% of the window §1 pre-registers", derived from whole-table
spans: 337 d (DE) / 275 d (NL) of `energy_renewable` against 2,049 d of
`energy_generation`.

That ratio is a property of the *tables*, not of the *registered window*. §1 of
`reports/abl_322_preregistration.md` does not register a 2,049-day fit; it
inherits ABL-195's frozen 178-day fit window (2026-01-14 -> 2026-07-11) plus a
14-day lag warm-up, opening 2025-12-31. Both source tables begin before that.

This probe measures the two questions that actually decide the damage:

1. **Coverage** — how many rows does each table carry inside the *registered*
   builder / fit / gate windows, and does either truncate?
2. **Bar movement** — the harness computes its baseline from `builder._actuals`
   (`evaluate_wind_retrain.py:200`), so the source defect moves the D-7 bar as
   well as the fit. §2 registered the bar from `energy_generation`. This
   recomputes it from `energy_renewable` by the identical method and reports the
   delta in percentage points.

The D-7 method is lifted verbatim from `scripts/abl322_preregistration_probe.py`
so the two documents' numbers are comparable.

Usage (the rail interpreter, explicit replica path):

    ENERGY_DB_PATH=C:\\Code\\able\\data\\energy_dashboard.db \\
        .venv\\Scripts\\python.exe scripts/abl322_source_sensitivity_probe.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# Frozen, inherited verbatim from experiments/ABL195/config.json via §1.
FIT_START = pd.Timestamp("2026-01-14 00:00:00")
GATE_START = pd.Timestamp("2026-07-11 00:00:00")
GATE_END = pd.Timestamp("2026-08-10 00:00:00")
#: RenewableFeatureBuilder is constructed with `fit_start - 14d` for lag warm-up
#: (`evaluate_wind_retrain.py:179-180`). This, not FIT_START, is the earliest
#: instant the pilot actually asks either source for.
BUILDER_START = FIT_START - pd.Timedelta(days=14)

PAIRS = ("DE", "NL")
TABLES = ("energy_generation", "energy_renewable")
COLUMN = "wind_offshore_mw"


def ro_connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)


def load(con, table: str, country: str, start, end) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"SELECT timestamp_utc, {COLUMN} AS value FROM {table} "
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


def table_span(con, table: str, country: str) -> dict:
    row = pd.read_sql_query(
        f"SELECT MIN(timestamp_utc) lo, MAX(timestamp_utc) hi, COUNT(*) n "
        f"FROM {table} WHERE country_code = ? AND {COLUMN} IS NOT NULL",
        con,
        params=(country,),
    ).iloc[0]
    lo, hi = pd.Timestamp(str(row.lo)[:19]), pd.Timestamp(str(row.hi)[:19])
    return {
        "rows": int(row.n),
        "first": str(lo),
        "last": str(hi),
        "days": int((hi - lo).days) + 1,
        "covers_builder_start": bool(lo <= BUILDER_START),
    }


def window_stats(df: pd.DataFrame) -> dict:
    """Rows, fabricated zeros and duplicate instants inside one window."""
    if df.empty:
        return {"n": 0, "zeros": 0, "duplicate_instants": 0}
    ts = df["timestamp_utc"]
    return {
        "n": int(len(df)),
        "zeros": int((df["value"] == 0).sum()),
        "duplicate_instants": int(len(ts) - ts.nunique()),
        "first": str(ts.min()),
        "last": str(ts.max()),
    }


def d7_baseline(df: pd.DataFrame, agg: str) -> dict:
    """Literal seasonal-naive D-7 on the frozen gate window.

    Verbatim from `abl322_preregistration_probe.py` so §2's registered bar and
    the counterfactual bar are produced by one method.
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
    actual, pred = gate.to_numpy(float), lagged.to_numpy(float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    return {
        "aggregation": agg,
        "n_scored": int(mask.sum()),
        "mean_actual_mw": float(np.mean(actual[mask])) if mask.any() else None,
        "d7_wape_pct": wape(actual, pred),
        "d7_mae_mw": float(np.mean(np.abs(actual[mask] - pred[mask]))) if mask.any() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replica",
        default=os.environ.get("ENERGY_DB_PATH"),
        help="Live replica path. A worktree has no .env, so pass this explicitly.",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    if not args.replica:
        parser.error("pass --replica or set ENERGY_DB_PATH; refusing to guess a database")
    replica = Path(args.replica)
    if not replica.exists():
        parser.error(f"replica does not exist: {replica}")

    out: dict = {
        "replica": str(replica),
        "replica_bytes": replica.stat().st_size,
        "windows": {
            "builder_start": str(BUILDER_START),
            "fit": [str(FIT_START), str(GATE_START)],
            "gate": [str(GATE_START), str(GATE_END)],
            "fit_days": (GATE_START - FIT_START).days,
        },
        "pairs": {},
    }
    print(f"replica: {replica}  ({replica.stat().st_size:,} bytes)")
    print(f"registered fit window : {FIT_START} -> {GATE_START} "
          f"({(GATE_START - FIT_START).days} d)")
    print(f"builder start (fit-14d): {BUILDER_START}\n")

    con = ro_connect(str(replica))
    try:
        for country in PAIRS:
            entry: dict = {"span": {}, "windows": {}, "d7": {}}
            for table in TABLES:
                entry["span"][table] = table_span(con, table, country)

            for label, lo, hi in (
                ("builder", BUILDER_START, GATE_END),
                ("fit", FIT_START, GATE_START),
                ("gate", GATE_START, GATE_END),
            ):
                entry["windows"][label] = {
                    t: window_stats(load(con, t, country, lo, hi)) for t in TABLES
                }

            # D-7 needs the week preceding the gate window.
            for table in TABLES:
                df = load(con, table, country, GATE_START - pd.Timedelta(days=8), GATE_END)
                entry["d7"][table] = {
                    agg: d7_baseline(df, agg) for agg in ("instant", "hourly_mean")
                }
            out["pairs"][country] = entry

            print(f"--- {country} wind_offshore ---")
            for table in TABLES:
                s = entry["span"][table]
                print(f"  span {table:<18} {s['rows']:>8,} rows  {s['first'][:10]} -> "
                      f"{s['last'][:10]}  {s['days']:>5,} d  "
                      f"covers builder start: {'YES' if s['covers_builder_start'] else 'NO'}")
            for label in ("builder", "fit", "gate"):
                w = entry["windows"][label]
                gen, ren = w["energy_generation"]["n"], w["energy_renewable"]["n"]
                ratio = f"{ren / gen:.1%}" if gen else "n/a"
                print(f"  {label:<8} generation n={gen:>7,} zeros="
                      f"{w['energy_generation']['zeros']:>4,}   renewable n={ren:>7,} "
                      f"zeros={w['energy_renewable']['zeros']:>4,}   renewable/generation {ratio}")
            for agg in ("instant", "hourly_mean"):
                g = entry["d7"][table_g := "energy_generation"][agg]
                e = entry["d7"]["energy_renewable"][agg]
                print(f"  D-7 ({agg:<11}) registered({table_g}) {g['d7_wape_pct']:.2f}% "
                      f"n={g['n_scored']}   counterfactual(energy_renewable) "
                      f"{e['d7_wape_pct']:.2f}%   delta "
                      f"{e['d7_wape_pct'] - g['d7_wape_pct']:+.2f}pp")
            print()
    finally:
        con.close()

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
