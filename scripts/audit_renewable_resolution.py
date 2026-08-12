"""ABL-332: does the renewable feature builder silently discard sub-hourly samples?

Two questions, both answered against the real replica, read-only:

1. **Census.** Per (country, stream, source table), how many stored rows sit on a
   non-`:00` minute? Those are the samples an exact `series.loc[ts.floor("h")]`
   lookup can never reach.
2. **Reproduction.** For one pair, build the features `src/wind_features.py`
   actually builds today and the features the same builder would build from an
   hourly-mean series, and print the difference. A difference is the bug; no
   difference would refute it.

Run (the `.env` value points at a path that does not exist — pass the replica
explicitly so a wrong path dies loudly rather than resolving somewhere else):

    ENERGY_DB_PATH=C:\\Code\\able\\data\\energy_dashboard.db \\
        python scripts/audit_renewable_resolution.py

`--census-only` skips the reproduction; `--pair DE:solar` picks the pair to
reproduce (repeatable).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

from src import db  # noqa: E402
from src.wind_features import RenewableFeatureBuilder  # noqa: E402

#: Only the three types `wind_features.SUPPORTED_FORECAST_TYPES` builds.
STREAMS: Dict[str, str] = {
    "solar": "solar_mw",
    "wind_onshore": "wind_onshore_mw",
    "wind_offshore": "wind_offshore_mw",
}

SOURCE_TABLES: Tuple[str, ...] = ("energy_renewable", "energy_generation")


def _minute_expr() -> str:
    """Minutes-of-hour token. Position 15 in every stored spelling: both the
    19-char `YYYY-MM-DD HH:MM:SS` form and the 25-char `...THH:MM:SS+01:00`
    form put the minute there, so this needs no format branch."""
    return "substr(timestamp_utc,15,2)"


def census(conn: sqlite3.Connection) -> pd.DataFrame:
    rows: List[dict] = []
    for table in SOURCE_TABLES:
        for stream, col in STREAMS.items():
            query = f"""
                SELECT country_code,
                       COUNT(*) AS rows_total,
                       SUM(CASE WHEN {_minute_expr()} = '00' THEN 1 ELSE 0 END) AS rows_on_hour,
                       COUNT(DISTINCT {_minute_expr()}) AS distinct_minutes,
                       MIN(timestamp_utc) AS first_ts,
                       MAX(timestamp_utc) AS last_ts
                FROM {table}
                WHERE data_quality = 'actual'
                  AND {col} IS NOT NULL
                GROUP BY country_code
                ORDER BY country_code
            """
            for cc, total, on_hour, n_minutes, first_ts, last_ts in conn.execute(query):
                rows.append({
                    "source": table,
                    "country": cc,
                    "stream": stream,
                    "rows_total": total,
                    "rows_on_hour": on_hour,
                    "rows_dropped": total - on_hour,
                    "pct_dropped": 100.0 * (total - on_hour) / total if total else 0.0,
                    "distinct_minutes": n_minutes,
                    "first_ts": first_ts,
                    "last_ts": last_ts,
                })
    return pd.DataFrame(rows)


def _hourly_mean(series: pd.Series) -> pd.Series:
    """The aggregation `db.load_training_data` already applies (`db.py`'s
    `resample('h').mean()`), reproduced here so the two arms are comparable."""
    if series.empty:
        return series
    return series.resample("h").mean().dropna()


def reproduce(country: str, stream: str, source: str, target_day: str) -> dict:
    """Build one day of features both ways and diff them.

    Arm A is exactly what serving does today. Arm B is the same builder over an
    hourly-mean series. Every feature the builder emits is compared.
    """
    target_start = pd.Timestamp(target_day)
    # A D+2 shape: generation instant the evening before yesterday, matching
    # `Forecaster.predict_d2` (FORECAST_HOUR=18).
    observation_as_of = target_start - pd.Timedelta(days=2) + pd.Timedelta(hours=18)
    span_start = target_start - pd.Timedelta(days=30)
    span_end = target_start + pd.Timedelta(days=1)

    builder = RenewableFeatureBuilder(country, stream, span_start, span_end, actuals_source=source)
    native = builder._actuals
    if native.empty:
        return {"country": country, "stream": stream, "source": source, "error": "no actuals in span"}

    minutes = sorted({int(ts.minute) for ts in native.index})

    arm_a = {}
    for hour in range(24):
        target = target_start + pd.Timedelta(hours=hour)
        arm_a[hour] = {k: v.value for k, v in builder.row(target, observation_as_of).items()}

    # Arm B: identical builder, hourly-mean actuals substituted in place.
    builder_b = RenewableFeatureBuilder(country, stream, span_start, span_end, actuals_source=source)
    builder_b._actuals = _hourly_mean(native)
    builder_b._rolling_cache.clear()
    arm_b = {}
    for hour in range(24):
        target = target_start + pd.Timedelta(hours=hour)
        arm_b[hour] = {k: v.value for k, v in builder_b.row(target, observation_as_of).items()}

    diffs: Dict[str, dict] = {}
    for feature in sorted(arm_a[0]):
        a_vals = pd.Series([arm_a[h][feature] for h in range(24)], dtype=float)
        b_vals = pd.Series([arm_b[h][feature] for h in range(24)], dtype=float)
        if a_vals.equals(b_vals):
            continue
        delta = (b_vals - a_vals).abs()
        denom = a_vals.abs().replace(0.0, float("nan"))
        diffs[feature] = {
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "max_pct_delta": float((delta / denom).max() * 100) if denom.notna().any() else None,
            "hours_differing": int((delta > 0).sum()),
            "example_serving": float(a_vals.iloc[0]),
            "example_hourly_mean": float(b_vals.iloc[0]),
        }

    return {
        "country": country,
        "stream": stream,
        "source": source,
        "target_day": target_day,
        "observation_as_of": str(observation_as_of),
        "native_rows_in_span": int(len(native)),
        "hourly_rows_in_span": int(len(_hourly_mean(native))),
        "distinct_minutes": minutes,
        "features_differing": diffs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census-only", action="store_true")
    parser.add_argument("--pair", action="append", default=[], help="CC:stream, repeatable")
    parser.add_argument("--target-day", default="2026-08-01")
    parser.add_argument("--source", default=None, help="source table; default db.RENEWABLE_TYPE_SOURCE_TABLE")
    parser.add_argument("--out", default=None, help="write JSON here")
    args = parser.parse_args()

    print(f"database: {config.DATABASE_PATH}")
    if not Path(config.DATABASE_PATH).exists():
        print("FATAL: database path does not exist", file=sys.stderr)
        return 2

    with db.get_connection() as conn:
        frame = census(conn)

    supported = set(config.SUPPORTED_COUNTRIES)
    frame["supported"] = frame["country"].isin(supported)

    print("\n=== census: rows a floor('h') lookup can never reach ===")
    for source in SOURCE_TABLES:
        sub = frame[(frame["source"] == source) & frame["supported"]]
        print(f"\n-- {source} (config.SUPPORTED_COUNTRIES only)")
        print(f"{'CC':<4}{'stream':<15}{'rows':>10}{'on :00':>10}{'dropped':>10}{'%drop':>8}  minutes")
        for _, r in sub.sort_values(["country", "stream"]).iterrows():
            print(f"{r['country']:<4}{r['stream']:<15}{r['rows_total']:>10}{r['rows_on_hour']:>10}"
                  f"{r['rows_dropped']:>10}{r['pct_dropped']:>7.1f}%  {r['distinct_minutes']}")

    payload = {"census": frame.to_dict(orient="records"), "reproductions": []}

    if not args.census_only:
        source = args.source or db.RENEWABLE_TYPE_SOURCE_TABLE
        pairs = [tuple(p.split(":", 1)) for p in args.pair] or [("DE", "solar"), ("DE", "wind_onshore")]
        for country, stream in pairs:
            print(f"\n=== reproduction: {country}/{stream} from {source}, target day {args.target_day} ===")
            result = reproduce(country, stream, source, args.target_day)
            payload["reproductions"].append(result)
            if "error" in result:
                print(f"  {result['error']}")
                continue
            print(f"  native rows in 31d span: {result['native_rows_in_span']}"
                  f"  -> hourly rows: {result['hourly_rows_in_span']}"
                  f"  minutes present: {result['distinct_minutes']}")
            if not result["features_differing"]:
                print("  NO FEATURE DIFFERS — the finding does not reproduce for this pair.")
                continue
            print(f"  {len(result['features_differing'])} features differ between "
                  f"serving-today and hourly-mean:")
            for feature, d in result["features_differing"].items():
                pct = f"{d['max_pct_delta']:.1f}%" if d["max_pct_delta"] is not None else "n/a"
                print(f"    {feature:<32} max|Δ|={d['max_abs_delta']:>10.2f}  max%={pct:>8}"
                      f"  hours={d['hours_differing']:>2}")

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
