#!/usr/bin/env python3
"""Train the V014 per-country net-position challenger (ABL-69).

CPU-only XGBoost, one model per country, fitted on the serve-faithful features
in `src/challengers/v014_features.py`. The twelve backtest weeks are held out by
target day, so `scripts/backtest_v014.py` scores target days this fit never saw.

Reads the replica **readonly** and writes nothing but model artifacts under
`models/net_position/V014/`. It does not touch the sidecar and cannot reach
production.

Usage:
    python scripts/train_v014.py --countries all
    python scripts/train_v014.py --countries BE,NL,AT,FR --start 2023-01-05
    python scripts/train_v014.py --countries DE --dry-run       # fit, report, discard
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.challengers.v014 import (EXPERIMENT_ID, backtest_target_days,
                                  run_days_for_span, save_model, train_country)
from src.challengers.v014_features import TARGET_DAY_OFFSET, build_cache
from src.evaluation.net_position import _ro_connect

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_v014")

#: LU duplicates DE in the A25 document and GR's actuals are fabricated zeros
#: (ABL-35/ABL-67) — the same exclusions the gate and the shadow rail apply, by
#: name rather than by symptom, so a partial upstream resume cannot silently
#: re-admit them.
EXCLUDED_COUNTRIES = ("LU", "GR")

#: Earliest run day worth asking for. `net_position` starts 2023-01-01 and the
#: features need 28 days of trailing history before the first usable run.
DEFAULT_START = "2023-02-01"


def live_countries(conn) -> list[str]:
    """The 19 countries V014 can both fit and serve.

    Two filters, and both are needed. `config.SUPPORTED_COUNTRIES` bounds it
    above because that is the set `forecast_challengers.py` iterates — a model
    outside it would be trained, backtested and reported on while the shadow
    rail never asks it for anything. **IE is the live example**: it has 24,286
    `net_position` rows, so a data-only filter admits it, but it is not a
    supported country and stopped publishing on 2026-07-24 alongside GR.

    The row count bounds it below because five supported countries (CH, IT, NO,
    SE, plus excluded GR) have no live net position at all; fitting them would
    produce a per-country row in every report that reads as a failure rather
    than as an absence.
    """
    df = pd.read_sql_query(
        "SELECT country_code, COUNT(*) n FROM net_position "
        "WHERE net_position_mw IS NOT NULL GROUP BY country_code", conn)
    counts = dict(zip(df["country_code"], df["n"]))
    return sorted(c for c in config.SUPPORTED_COUNTRIES
                  if c not in EXCLUDED_COUNTRIES and counts.get(c, 0) >= 24 * 365)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--countries", default="all")
    p.add_argument("--start", default=DEFAULT_START, help="first run day")
    p.add_argument("--end", default=None, help="last run day (default: D-3, so the target is measured)")
    p.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    p.add_argument("--models-dir", default=str(config.MODELS_DIR))
    p.add_argument("--no-exclude-backtest", action="store_true",
                   help="fit on the backtest weeks too. Only for a serving model "
                        "that will never be backtested; the reported W01-W12 "
                        "numbers become meaningless.")
    p.add_argument("--dry-run", action="store_true", help="fit and report, save nothing")
    p.add_argument("--report", default=None, help="write the per-country report JSON here")
    args = p.parse_args()

    conn = _ro_connect(args.replica_db)
    try:
        countries = (live_countries(conn) if args.countries == "all"
                     else [c.strip().upper() for c in args.countries.split(",")
                           if c.strip().upper() not in EXCLUDED_COUNTRIES])
        end = pd.Timestamp(args.end) if args.end else (
            pd.Timestamp.now("UTC").tz_localize(None).normalize()
            - pd.Timedelta(days=TARGET_DAY_OFFSET + 1))
        held_out = set() if args.no_exclude_backtest else backtest_target_days(config.BACKTEST_WEEKS)
        run_days = run_days_for_span(args.start, end, held_out)
        logger.info("V014 training: %d countries, %d run days %s..%s (%d held out "
                    "for W01-W12)", len(countries), len(run_days), args.start,
                    end.date(), len(pd.date_range(args.start, end, freq="D")) - len(run_days))

        reports, failures = [], []
        for i, cc in enumerate(countries, 1):
            t0 = time.time()
            neighbours = [n for n in config.COUNTRY_NEIGHBORS.get(cc, [])
                          if n not in EXCLUDED_COUNTRIES]
            try:
                cache = build_cache(conn, cc,
                                    pd.Timestamp(args.start) - pd.Timedelta(days=35),
                                    end + pd.Timedelta(days=TARGET_DAY_OFFSET + 1))
                model, report = train_country(conn, cc, run_days, neighbours, cache=cache)
            except Exception as exc:  # noqa: BLE001
                logger.error("%s: %s", cc, exc)
                failures.append({"country": cc, "error": str(exc)})
                continue
            if not args.dry_run:
                report["artifact"] = str(save_model(model, Path(args.models_dir)))
            report["fit_seconds"] = round(time.time() - t0, 1)
            reports.append(report)
            val = report.get("validation") or {}
            logger.info("[%2d/%d] %s  rows=%d  val MAE=%s  slope=%s  (%.0fs)",
                        i, len(countries), cc, report["rows_total"],
                        f"{val['mae']:.0f}" if val.get("mae") is not None else "n/a",
                        f"{val['slope']:.3f}" if val.get("slope") is not None else "n/a",
                        report["fit_seconds"])

        doc = {
            "experiment": EXPERIMENT_ID,
            "trained_at": pd.Timestamp.now("UTC").tz_localize(None).isoformat(timespec="seconds"),
            "replica_db": str(args.replica_db),
            "run_day_span": [args.start, str(end.date())],
            "backtest_weeks_excluded": not args.no_exclude_backtest,
            "countries": reports,
            "failures": failures,
        }
        out = Path(args.report) if args.report else (
            config.EXPERIMENTS_DIR / EXPERIMENT_ID / "training_report.json")
        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(doc, indent=2))
            logger.info("report -> %s", out)

        ok = [r for r in reports if (r.get("validation") or {}).get("mae") is not None]
        logger.info("trained %d/%d countries (%d failed)", len(reports),
                    len(countries), len(failures))
        if ok:
            logger.info("median validation MAE %.0f MW, median slope %.3f",
                        pd.Series([r["validation"]["mae"] for r in ok]).median(),
                        pd.Series([r["validation"]["slope"] for r in ok
                                   if r["validation"]["slope"] is not None]).median())
        return 1 if failures else 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
