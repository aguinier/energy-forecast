#!/usr/bin/env python3
"""Pre-fit precondition check for an ABL-316 tranche, against the ABL-348 registration.

ABL-380 asks for the tranche preconditions to be *verified rather than trusted*,
and the same four questions recur for every one of the 37 pairs:

  1. **Which file did we actually read?**  `.env` is gitignored, so a worktree has
     no `ENERGY_DB_PATH` and `config.DATABASE_PATH` degrades to a bare
     `\\data\\energy_dashboard.db`; the nearest real file to that wrong path is a
     3.0 GB stale partial snapshot whose numbers look fine (CLAUDE.md, ABL-73).
     Path plus byte size is the cheapest thing that distinguishes them.
  2. **Does the registered gate window actually hold its hours?**  The registered
     minimum n is 95% of an intended 720/720/480, and ABL-348 had to declare two
     pairs NOT-EVALUABLE *before* fitting because the source table holds fewer
     gate hours than the bar needs.  That is a property of the pair, knowable
     without a model, and much cheaper to learn before a fit than after one.
  3. **Does ABL-188 bite inside either window?**  `energy_renewable` zero-fills a
     production type ENTSO-E did not return, and the training-boundary screen
     nulls those runs -- which moves n, and therefore can move a cell's
     `enough_pairs` regardless of how the model scored.
  4. **Does the pre-committed D-7 bar reproduce?**  ABL-348 froze a per-pair bar
     measured before any challenger existed.  Recomputing it from the same loader
     on the same file is what makes the frozen number checkable rather than
     merely quoted.

This script answers all four and **fits nothing**.  It is deliberately separate
from `evaluate_wind_retrain.py`: a check that shares the harness's process could
only run after the harness had already decided to fit, which is exactly the
ordering that makes a NOT-EVALUABLE pair expensive to discover.

The bar recomputed here is the *literal* D-7 on the plain hourly target series --
the same protocol ABL-348 recorded.  The harness's own per-band, finite-
intersection D-7 is the authoritative gate number and will differ slightly; a
mismatch here is a signal about the data, not about the model.

Read-only.  Opens the replica through the `mode=ro` URI form and writes nothing
to it.

Usage:
    python scripts/abl380_tranche_precheck.py \
        --pairs BG/wind_onshore,CH/wind_onshore \
        --replica-db C:/Code/able/data/energy_dashboard.db \
        --renewable-source energy_generation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import db
from src.data_quality import find_suspect_constant_runs
from src.evaluation.scorecard import _ro_connect

#: The registration this checks against.  Frozen at ABL-348; not re-derived here.
REGISTERED_INTENDED_N = {"24-36h": 720, "36-48h": 720, "48-64h": 480}
COLUMNS = {"wind_offshore": "wind_offshore_mw", "wind_onshore": "wind_onshore_mw",
           "solar": "solar_mw"}


def _wape(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    denom = np.abs(actual).sum()
    if denom == 0:
        return None
    return float(np.abs(actual - forecast).sum() / denom * 100.0)


def _hourly_series(country: str, stream: str, start, end, source: str,
                   replica: str) -> pd.Series:
    """The plain hourly target series, ABL-188-screened, as the registration read it."""
    frame = db.load_renewable_type_data(country, stream, str(start), str(end),
                                        source=source, db_path=replica)
    if frame.empty:
        return pd.Series(dtype=float)
    stamps = pd.to_datetime(frame["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    return pd.Series(frame["target_value"].to_numpy(dtype=float), index=stamps).sort_index()


def _raw_series(country: str, stream: str, start, end, source: str,
                replica: str) -> pd.DataFrame:
    """The unaggregated rows, so native resolution is observable (ABL-332).

    Through `_ro_connect`, the same `mode=ro` URI the gate harness opens the
    replica with -- `db.get_connection` is a context manager, and this read is
    deliberately the harness's, not a second way of reaching the same file.
    """
    con = _ro_connect(replica)
    try:
        frame = pd.read_sql_query(
            f"SELECT timestamp_utc, {COLUMNS[stream]} AS value FROM {source} "
            "WHERE country_code=? AND timestamp_utc>=? AND timestamp_utc<? "
            "AND data_quality='actual' ORDER BY timestamp_utc",
            con, params=(country, str(start), str(end)),
        )
    finally:
        con.close()
    if not frame.empty:
        frame["timestamp_utc"] = pd.to_datetime(
            frame["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    return frame


def check_pair(country: str, stream: str, source: str, replica: str,
               lookback_start, fit_start, gate_start, gate_end) -> dict:
    gate = _hourly_series(country, stream, gate_start, gate_end, source, replica)
    # D-7 needs the seven days before the gate window to score its first hours.
    d7_hist = _hourly_series(country, stream, gate_start - pd.Timedelta(days=7),
                             gate_end, source, replica)
    fit = _hourly_series(country, stream, fit_start, gate_start, source, replica)

    gate_hours = pd.date_range(gate_start, gate_end, freq="h", inclusive="left")
    intended_gate_hours = len(gate_hours)

    # Literal seasonal-naive D-7: forecast(t) = actual(t - 168h), scored on the
    # hours where both exist.  This is the registration's protocol verbatim.
    shifted = d7_hist.reindex(gate.index - pd.Timedelta(days=7))
    paired = pd.DataFrame({"actual": gate.to_numpy(dtype=float),
                           "d7": shifted.to_numpy(dtype=float)}, index=gate.index)
    scorable = paired.dropna()
    d7_wape = _wape(scorable["actual"].to_numpy(), scorable["d7"].to_numpy())

    # ABL-188 screen, run over each window separately so a hit is attributable.
    runs = {}
    for label, (lo, hi) in {"fit": (fit_start, gate_start),
                            "gate": (gate_start, gate_end),
                            "feature_lookback": (lookback_start, fit_start)}.items():
        raw = _raw_series(country, stream, lo, hi, source, replica)
        runs[label] = [
            {"start": str(r.start), "end": str(r.end), "value": float(r.value),
             "n_rows": int(r.n_rows), "duration_hours": float(r.duration_hours)}
            for r in (find_suspect_constant_runs(raw, "value") if not raw.empty else [])
        ]

    raw_gate = _raw_series(country, stream, gate_start, gate_end, source, replica)
    native_sub_hourly = bool(
        not raw_gate.empty
        and not raw_gate["timestamp_utc"].dt.floor("h").equals(raw_gate["timestamp_utc"]))

    return {
        "country": country,
        "stream": stream,
        "source": source,
        "fit_rows_hourly": int(len(fit)),
        "fit_first": str(fit.index.min()) if len(fit) else None,
        "fit_last": str(fit.index.max()) if len(fit) else None,
        "gate_hours_intended": intended_gate_hours,
        "gate_hours_present": int(len(gate)),
        "gate_hours_missing": int(intended_gate_hours - len(gate)),
        "native_sub_hourly_in_gate_window": native_sub_hourly,
        "n_d7_scorable": int(len(scorable)),
        "d7_wape_pct": None if d7_wape is None else round(d7_wape, 2),
        "mean_actual_mw": round(float(gate.mean()), 1) if len(gate) else None,
        "d7_mae_mw": (round(float(np.abs(scorable["actual"] - scorable["d7"]).mean()), 1)
                      if len(scorable) else None),
        "abl188_constant_runs": runs,
        # The registered minimum n bounds the two 720-bands directly; the 48-64h
        # band selects a subset, so this is a necessary condition, not the cell n.
        "meets_registered_min_n_720_bands": bool(len(scorable) >= int(np.ceil(0.95 * 720))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", required=True,
                        help="comma-separated COUNTRY/stream, e.g. BG/wind_onshore,CH/wind_onshore")
    parser.add_argument("--replica-db", required=True,
                        help="explicit replica file; never defaulted, see ABL-73")
    parser.add_argument("--renewable-source", default="energy_generation",
                        choices=list(db._RENEWABLE_TYPE_SOURCES))
    parser.add_argument("--compare-source", default=None,
                        choices=list(db._RENEWABLE_TYPE_SOURCES),
                        help="second table to cross-check the gate window against")
    parser.add_argument("--feature-lookback-start", default="2025-12-31")
    parser.add_argument("--fit-start", default="2026-01-14")
    parser.add_argument("--gate-start", default="2026-07-11")
    parser.add_argument("--gate-end", default="2026-08-10")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    replica = Path(args.replica_db).resolve()
    if not replica.exists():
        parser.error(f"replica not found: {replica}")
    lookback_start, fit_start, gate_start, gate_end = map(
        pd.Timestamp, (args.feature_lookback_start, args.fit_start,
                       args.gate_start, args.gate_end))

    pairs = []
    for token in args.pairs.split(","):
        country, _, stream = token.strip().partition("/")
        if stream not in COLUMNS:
            parser.error(f"unknown stream in {token!r}")
        pairs.append((country.upper(), stream))

    result = {
        "replica_db": str(replica),
        "replica_bytes": replica.stat().st_size,
        "renewable_source": args.renewable_source,
        "windows": {"feature_lookback_start": str(lookback_start),
                    "fit": [str(fit_start), str(gate_start)],
                    "gate": [str(gate_start), str(gate_end)]},
        "registered_intended_n": REGISTERED_INTENDED_N,
        "pairs": [],
    }

    for country, stream in pairs:
        row = check_pair(country, stream, args.renewable_source, str(replica),
                         lookback_start, fit_start, gate_start, gate_end)
        if args.compare_source and args.compare_source != args.renewable_source:
            other = _hourly_series(country, stream, gate_start, gate_end,
                                   args.compare_source, str(replica))
            mine = _hourly_series(country, stream, gate_start, gate_end,
                                  args.renewable_source, str(replica))
            both = pd.DataFrame({"a": mine, "b": other}).dropna()
            identical = int((both["a"] == both["b"]).sum())
            row["cross_source"] = {
                "compare_source": args.compare_source,
                "n_co_observed_hours": int(len(both)),
                "n_bit_identical": identical,
                "pct_bit_identical": (round(identical / len(both) * 100.0, 2)
                                      if len(both) else None),
                "max_abs_diff_mw": (round(float((both["a"] - both["b"]).abs().max()), 3)
                                    if len(both) else None),
            }
        result["pairs"].append(row)

    text = json.dumps(result, indent=2, allow_nan=False)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
