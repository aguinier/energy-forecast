#!/usr/bin/env python3
"""ABL-606: re-derive the scheduled matrix and the two runners' write history.

Two independent readings, both cheap, neither needing the container:

  matrix  -- computes Total and the per-external-runner cell counts from
             `config` the way `forecast_daily`'s loops do, before and after the
             ABL-606 selection fix. Reproduces ABL-601's in-container
             `Total: 440 ... Failed: 8` without a container.

  census  -- asks the read-only replica when `chronos-bolt-small`, `tso_raw`
             and `tso_corrected` last wrote a forecast row, next to the
             production model names for the same country and types.

Usage:
    python scripts/abl606_runner_matrix_probe.py
    python scripts/abl606_runner_matrix_probe.py --skip-census

The replica is opened read-only (`mode=ro`). A replica refresh holds an
exclusive lock on it for minutes at a time; `--nolock` reads through that, at
the cost of possibly seeing pages mid-transaction. Coarse census only -- do not
use `--nolock` output for a metric.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import config  # noqa: E402
import forecast_daily  # noqa: E402

#: Verbatim from ABL-601's `docker exec ... forecast_daily.py --dry-run`.
ABL601 = {"total": 440, "success": 154, "skipped": 278, "failed": 8}

#: BE is the only country either runner is registered for.
CENSUS_COUNTRY = "BE"
CENSUS_TYPES = ("price", "solar", "wind_onshore", "wind_offshore")


def parse_args():
    parser = argparse.ArgumentParser(description="ABL-606 runner matrix probe")
    parser.add_argument(
        "--db",
        default=str(config.DATABASE_PATH),
        help="Replica path (default: config.DATABASE_PATH)",
    )
    parser.add_argument(
        "--nolock",
        action="store_true",
        help="Read through an exclusive writer lock (coarse census only)",
    )
    parser.add_argument(
        "--skip-census",
        action="store_true",
        help="Matrix arithmetic only; do not open the replica",
    )
    return parser.parse_args()


def runner_cells(runner, countries, forecast_types):
    """Cells `forecast_daily`'s external-runner loop would visit."""
    runner_countries = runner.get("countries", [])
    runner_types = runner.get("forecast_types", [])
    return sum(
        len(config.get_horizons_for_type(forecast_type))
        for country in countries
        if runner_countries == "all" or country in runner_countries
        for forecast_type in forecast_types
        if runner_types == "all" or forecast_type in runner_types
    )


def report_matrix():
    countries = forecast_daily.get_countries("all")
    forecast_types = forecast_daily.get_forecast_types("all")

    builtin = len(countries) * sum(
        len(config.get_horizons_for_type(t)) for t in forecast_types
    )

    before = [
        r for r in config.MODEL_RUNNERS
        if r.get("type") == "external" and r.get("enabled", False)
    ]
    after = forecast_daily.select_external_runners(config.MODEL_RUNNERS)

    print("== scheduled matrix ==")
    print(f"countries={len(countries)} types={len(forecast_types)} horizons=2")
    print(f"builtin cells            : {builtin}")
    print("external runners, pre-fix selection (type=external and enabled):")
    pre = 0
    for r in before:
        n = runner_cells(r, countries, forecast_types)
        pre += n
        print(
            f"  {r['name']:<20} production={str(r.get('production')):<5} "
            f"cells={n:<3} exe={r.get('python_executable')}"
        )
    post = sum(runner_cells(r, countries, forecast_types) for r in after)
    print(f"external cells pre-fix   : {pre}")
    print(f"external cells post-fix  : {post}  "
          f"(selected: {[r['name'] for r in after] or 'none'})")
    print(f"TOTAL pre-fix            : {builtin + pre}  "
          f"(ABL-601 measured {ABL601['total']})")
    print(f"TOTAL post-fix           : {builtin + post}")
    print(f"expected Failed floor    : pre-fix {pre}  ->  post-fix 0")
    ok = (builtin + pre == ABL601["total"]) and (pre == ABL601["failed"])
    print(f"reproduces ABL-601       : {ok}")
    return ok


def report_census(db_path, nolock):
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    if nolock:
        uri += "&nolock=1"
    print()
    print("== forecast write census ==")
    print(f"db  : {db_path}")
    print(f"uri : {uri}")
    con = sqlite3.connect(uri, uri=True, timeout=30)
    try:
        rows = con.execute(
            """
            SELECT forecast_type, model_name, COUNT(*),
                   MIN(generated_at), MAX(generated_at)
            FROM forecasts
            WHERE country_code = ?
              AND forecast_type IN (?, ?, ?, ?)
            GROUP BY forecast_type, model_name
            ORDER BY forecast_type, COUNT(*) DESC
            """,
            (CENSUS_COUNTRY, *CENSUS_TYPES),
        ).fetchall()
    finally:
        con.close()

    print(f"{'type':<15} {'model_name':<20} {'n':>7}  first_generated_at"
          f"           last_generated_at")
    for forecast_type, model_name, n, first, last in rows:
        print(f"{forecast_type:<15} {model_name:<20} {n:>7}  {first:<28} {last}")
    return rows


def main():
    args = parse_args()
    ok = report_matrix()
    if not args.skip_census:
        try:
            report_census(args.db, args.nolock)
        except sqlite3.OperationalError as exc:
            print(f"\ncensus unavailable: {exc}")
            print("(a replica refresh holds an exclusive lock; retry, or "
                  "re-run with --nolock and treat the result as coarse)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
