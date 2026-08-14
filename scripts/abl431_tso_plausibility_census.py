"""Census of implausible TSO day-ahead forecast values on the replica (ABL-431).

Answers the two questions the guard's registration rests on, from the replica
rather than from prose:

1. **Extent.** Which (country, column) pairs carry a value the plausibility
   guard refuses, how many rows, and over what window. ABL-417 found HU's
   140,996 MW wind_onshore and could not say whether it was one row or a
   pattern; this reports the whole table.
2. **Separation.** The distribution of `max / reference` over every evaluable
   pair, which is what makes PLAUSIBILITY_TOLERANCE a measurement rather than
   a convention. If a healthy pair ever climbs into the tolerance, this is the
   script that says so before a fit does.

Read-only: it opens the replica with mode=ro and writes nothing anywhere.

    .venv\\Scripts\\python.exe scripts/abl431_tso_plausibility_census.py \\
        --replica-db C:\\Code\\able\\data\\energy_dashboard.db
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.tso_plausibility import (
    PLAUSIBILITY_TOLERANCE,
    REFERENCE_QUANTILE,
    TSO_FORECAST_SOURCES,
    forecast_read,
    reference_scale,
)


def _countries(conn: sqlite3.Connection, table: str) -> list:
    return [r[0] for r in conn.execute(
        f"SELECT DISTINCT country_code FROM {table} ORDER BY country_code")]


def census(conn: sqlite3.Connection, tolerance: float, quantile: float) -> dict:
    per_table = {t: _countries(conn, t) for t in
                 sorted({table for table, _ in TSO_FORECAST_SOURCES})}

    flagged, ratios = [], []
    n_evaluable = n_zero_reference = n_no_history = 0
    n_observations = n_flagged = 0

    for (table, column), _ in sorted(TSO_FORECAST_SOURCES.items()):
        for country in per_table[table]:
            scale = reference_scale(conn, country, table, column,
                                    quantile=quantile, tolerance=tolerance)
            if scale.n_forecast == 0:
                n_no_history += 1
                continue
            n_observations += scale.n_forecast
            if not scale.evaluable:
                n_zero_reference += 1
                continue
            n_evaluable += 1

            expression, where, params = forecast_read(table, column)
            largest = conn.execute(
                f"SELECT MAX({expression}) FROM {table} WHERE country_code = ? "
                f"AND {where}", (country, *params)).fetchone()[0]
            if largest is not None:
                ratios.append((largest / scale.reference_mw, country, column))

            count, lo, hi = conn.execute(
                f"SELECT COUNT(*), MIN(target_timestamp_utc), MAX(target_timestamp_utc) "
                f"FROM {table} WHERE country_code = ? AND {where} "
                f"AND ({expression}) > ?",
                (country, *params, scale.threshold_mw)).fetchone()
            if count:
                n_flagged += count
                flagged.append({
                    "country": country, "table": table, "column": column,
                    "reference_mw": scale.reference_mw,
                    "threshold_mw": scale.threshold_mw,
                    "n_flagged": count, "n_observations": scale.n_forecast,
                    "max_mw": largest, "ratio": largest / scale.reference_mw,
                    "first": lo, "last": hi,
                })

    ratios.sort(reverse=True)
    return {
        "tolerance": tolerance, "quantile": quantile,
        "n_evaluable": n_evaluable, "n_zero_reference": n_zero_reference,
        "n_no_history": n_no_history, "n_observations": n_observations,
        "n_flagged": n_flagged,
        "flagged": sorted(flagged, key=lambda r: -r["ratio"]),
        "ratios": ratios,
    }


def render(result: dict) -> str:
    out = [
        f"TSO day-ahead plausibility census (ABL-431)",
        f"  reference   = max(p{100 * result['quantile']:.4g} actual, "
        f"p{100 * result['quantile']:.4g} day-ahead forecast), nearest-rank",
        f"  tolerance   = {result['tolerance']}x",
        f"  pairs       = {result['n_evaluable']} evaluable, "
        f"{result['n_zero_reference']} zero-reference (no fleet to scale against), "
        f"{result['n_no_history']} with no forecast history",
        "",
        f"{'cc':<4}{'column':<20}{'reference':>11}{'threshold':>11}"
        f"{'flagged':>9}{'of n':>10}{'max MW':>12}{'x ref':>9}  window",
    ]
    for row in result["flagged"]:
        out.append(
            f"{row['country']:<4}{row['column']:<20}{row['reference_mw']:>11.2f}"
            f"{row['threshold_mw']:>11.2f}{row['n_flagged']:>9}"
            f"{row['n_observations']:>10}{row['max_mw']:>12.1f}"
            f"{row['ratio']:>9.1f}  {row['first']} .. {row['last']}")
    if not result["flagged"]:
        out.append("  (nothing flagged)")

    pct = 100.0 * result["n_flagged"] / result["n_observations"] if result["n_observations"] else 0.0
    out += [
        "",
        f"EXTENT: {result['n_flagged']} of {result['n_observations']} "
        f"column-observations flagged ({pct:.5f}%)",
        "",
        "Separation -- highest max/reference per pair. The tolerance is only a "
        "measurement while there is",
        f"clear air between the last anomaly and the first healthy pair "
        f"(tolerance {result['tolerance']}x):",
    ]
    for ratio, country, column in result["ratios"][:12]:
        marker = "  <-- refused" if ratio > result["tolerance"] else ""
        out.append(f"  {country:<4}{column:<20}{ratio:>9.2f}x{marker}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--replica-db", default=str(config.DATABASE_PATH),
                        help="read-only replica to census")
    parser.add_argument("--tolerance", type=float, default=PLAUSIBILITY_TOLERANCE)
    parser.add_argument("--quantile", type=float, default=REFERENCE_QUANTILE)
    args = parser.parse_args()

    path = Path(args.replica_db)
    if not path.exists():
        print(f"replica not found: {path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        print(render(census(conn, args.tolerance, args.quantile)))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
