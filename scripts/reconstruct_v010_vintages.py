#!/usr/bin/env python3
"""Rebuild historical champion vintages, serve-faithfully, for fitting (ABL-68).

V016 recalibrates the champion, so it has to be fitted on champion forecasts
that look like the ones it will correct at serve time. The as-served sidecar
cannot supply that: only 4 vintages exist since the context-cutoff fix
(`1c5a24f`, 2026-08-04 14:29 UTC) and just 22 paired hours per country. Fitting
a per-country affine plus an AR coefficient on 22 points drawn from one target
day is how a dashboard ends up confidently displaying a wrong number.

So this script re-runs the *fixed* pipeline over a range of past run-days with
every query bounded by the serve-time `as_of`, and stores the result as its own
vintage set.

**`as_of` comes from `net_position.as_of_for_vintage`, not from a local
constant.** `net_position` is day-ahead published (~12:45 CET on D-1), so a
06:00Z run on day D legitimately holds actuals through D 21:00 and its bound is
**D 22:00** -- not the run instant. `compare_experiments.py:178` uses the run
instant for every type, which hands net_position 16h less context than the live
run had; a fit built on that would be calibrating a model that never served.
Sharing the eval's function is what keeps the two definitions from drifting
apart.

Output goes to its own database (`--out-db`), never the as-served sidecar: the
sidecar is the authoritative record of what production actually served, and a
reconstruction is not that. Rows carry the champion's `model_name` so
`evaluate_net_position.py --sidecar-db <out-db>` scores them with no changes.

Usage:
    # Prove the bound first: reproduce a real vintage bit-exactly.
    python scripts/reconstruct_v010_vintages.py --verify 2026-08-06T06:00:44

    # Then build the fitting set.
    python scripts/reconstruct_v010_vintages.py --start 2026-05-01 --end 2026-08-04 \
        --out-db C:\\Code\\able\\data\\forecasts_recon.db
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.evaluation.net_position import as_of_for_vintage

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("reconstruct")

FORECAST_TYPE = "net_position"
RUN_HOUR = 6  # the hour the daily job actually fires (UTC); see CLAUDE.md

# LU duplicates DE byte-for-byte; GR's actuals outage makes its context
# degenerate (24 real hours, 648 zero-filled) so it forecasts a constant zero.
# Both are excluded from the ABL-24 gate, so neither belongs in a fitting set.
EXCLUDED_COUNTRIES = ("LU", "GR")

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code TEXT NOT NULL, forecast_type TEXT NOT NULL,
    target_timestamp_utc TIMESTAMP NOT NULL, generated_at TIMESTAMP NOT NULL,
    horizon_hours INTEGER NOT NULL, forecast_value REAL NOT NULL,
    model_name TEXT NOT NULL, model_version TEXT,
    UNIQUE(country_code, forecast_type, target_timestamp_utc, horizon_hours,
           model_name, generated_at));
CREATE TABLE IF NOT EXISTS forecast_quantiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code TEXT NOT NULL, forecast_type TEXT NOT NULL,
    target_timestamp_utc TIMESTAMP NOT NULL, generated_at TIMESTAMP NOT NULL,
    quantile REAL NOT NULL, forecast_value REAL NOT NULL, model_name TEXT NOT NULL,
    UNIQUE(country_code, forecast_type, target_timestamp_utc, quantile,
           model_name, generated_at));
CREATE TABLE IF NOT EXISTS reconstruction_meta (
    key TEXT PRIMARY KEY, value TEXT);
"""


def synthetic_generated_at(run_day: pd.Timestamp) -> pd.Timestamp:
    """The instant the daily job would have fired on `run_day`."""
    return run_day.normalize() + pd.Timedelta(hours=RUN_HOUR)


def build_engine(experiment_id: str, device: str):
    from src.chronos2.engine import ChronosEngine
    from src.chronos2.input_builder import InputBuilder

    exp_config = json.loads(
        (config.EXPERIMENTS_DIR / experiment_id / "config.json").read_text())
    model_config = exp_config.get("model", {})
    if exp_config.get("training", {}).get("fine_tune", False):
        base = config.MODELS_DIR / "chronos2" / experiment_id / "finetuned-ckpt"
        inner = base / "finetuned-ckpt"
        model_path = str(inner if inner.exists() else base)
    else:
        model_path = None
    context_length = model_config.get("context_length", config.CHRONOS2_CONTEXT_LENGTH)
    engine = ChronosEngine(model_path=model_path, device=device,
                           context_length=context_length)
    return engine, InputBuilder(context_length=context_length)


def forecast_one(engine, builder, country: str, target_date: str,
                 as_of: pd.Timestamp, publication_as_of: pd.Timestamp
                 ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """One serve-faithful vintage-country forecast: (median24, quantiles24, index).

    Mirrors forecast_chronos2.run_forecast exactly — same horizon measurement,
    same last-24 slice, same target-day assertion. A reconstruction that took a
    different tail than production takes would be fitting noise.

    Two bounds: `as_of` is how far observations reach (D 22:00 for a day-ahead
    published target), `publication_as_of` is when the run fired (D 06:00), so
    weather runs issued after the run are not visible. See build_for_country.
    """
    inp = builder.build_for_country(
        country, FORECAST_TYPE, target_date,
        as_of=as_of.strftime("%Y-%m-%d %H:%M:%S"),
        publication_as_of=publication_as_of.strftime("%Y-%m-%d %H:%M:%S"))
    result = engine.forecast(target=inp["target"],
                             past_covariates=inp.get("past_covariates"),
                             future_covariates=inp.get("future_covariates"),
                             prediction_length=inp["prediction_length"])
    day_index = inp["future_index"][-24:]
    expected = pd.date_range(pd.Timestamp(target_date), periods=24, freq="h")
    if not day_index.equals(expected):
        raise ValueError(f"{country}: horizon tail {day_index[0]}..{day_index[-1]} "
                         f"is not target day {expected[0]}..{expected[-1]}")
    return result["median"][-24:], result["quantiles"][:, -24:], day_index


def write_vintage(out_db: Path, country: str, generated_at: pd.Timestamp,
                  index: pd.DatetimeIndex, median: np.ndarray,
                  quantiles: np.ndarray, model_name: str) -> None:
    version = generated_at.strftime("%Y%m%d_%H%M%S")
    con = sqlite3.connect(out_db, timeout=30.0)
    try:
        con.executemany(
            """INSERT OR REPLACE INTO forecasts (country_code, forecast_type,
               target_timestamp_utc, generated_at, horizon_hours, forecast_value,
               model_name, model_version) VALUES (?,?,?,?,?,?,?,?)""",
            [(country, FORECAST_TYPE, str(ts), str(generated_at),
              max(1, int((ts - generated_at).total_seconds() // 3600)),
              float(v), model_name, version)
             for ts, v in zip(index, median)])
        con.executemany(
            """INSERT OR REPLACE INTO forecast_quantiles (country_code, forecast_type,
               target_timestamp_utc, generated_at, quantile, forecast_value, model_name)
               VALUES (?,?,?,?,?,?,?)""",
            [(country, FORECAST_TYPE, str(ts), str(generated_at), float(q),
              float(quantiles[qi, i]), model_name)
             for qi, q in enumerate(config.CHRONOS2_QUANTILE_LEVELS)
             for i, ts in enumerate(index)])
        con.commit()
    finally:
        con.close()


def verify(args, engine, builder, model_name: str) -> int:
    """Reproduce a stored as-served vintage. The bound is right or it is not."""
    con = sqlite3.connect(f"file:{args.sidecar_db}?mode=ro", uri=True)
    try:
        stored = pd.read_sql_query(
            """SELECT country_code, target_timestamp_utc, generated_at, forecast_value
               FROM forecasts WHERE forecast_type = ? AND model_name = ?
                 AND generated_at LIKE ?""",
            con, params=(FORECAST_TYPE, model_name, args.verify.replace("T", " ") + "%"))
    finally:
        con.close()
    if stored.empty:
        logger.error("no stored vintage matching %s", args.verify)
        return 2

    stored["target_ts"] = pd.to_datetime(stored["target_timestamp_utc"], format="mixed")
    generated_at = pd.to_datetime(stored["generated_at"].iloc[0], format="mixed")
    target_date = stored["target_ts"].dt.normalize().max().strftime("%Y-%m-%d")
    as_of = as_of_for_vintage(generated_at)
    logger.info("verifying vintage %s (target %s), as_of=%s",
                generated_at, target_date, as_of)

    countries = sorted(set(stored["country_code"]) - set(EXCLUDED_COUNTRIES))
    if args.countries != "all":
        wanted = set(args.countries.split(","))
        countries = [c for c in countries if c in wanted]
    worst = 0.0
    failures = []
    for cc in countries:
        try:
            median, _, index = forecast_one(engine, builder, cc, target_date,
                                            as_of, generated_at)
        except Exception as exc:  # noqa: BLE001 — report, do not mask
            logger.warning("%s: reconstruction failed - %s", cc, exc)
            failures.append(cc)
            continue
        ref = (stored[stored["country_code"] == cc]
               .set_index("target_ts")["forecast_value"].reindex(index))
        diff = float(np.nanmax(np.abs(ref.to_numpy() - median)))
        # Scale-free: countries differ by an order of magnitude in net position,
        # so a flat MW tolerance would be strict for SI and generous for DE.
        scale = float(np.mean(np.abs(ref.to_numpy())))
        rel = 100.0 * diff / scale if scale > 0 else float("nan")
        worst = max(worst, rel)
        logger.info("%s: max|diff| = %.4g MW (%.3f%% of mean |forecast| %.0f MW)",
                    cc, diff, rel, scale)
        if rel > args.tolerance:
            failures.append(f"{cc} ({rel:.2f}%)")

    logger.info("worst relative max|diff| across %d countries: %.3f%% "
                "(tolerance %.2f%%)", len(countries), worst, args.tolerance)
    if failures:
        logger.error("serve-parity NOT established for: %s", ", ".join(failures))
        return 1
    logger.info("serve-parity established: reconstruction reproduces the "
                "as-served vintage within tolerance")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="V010")
    p.add_argument("--model-name", default=None,
                   help="stored model_name (default: chronos-2-<experiment>)")
    p.add_argument("--start", help="first run-day D (YYYY-MM-DD); targets are D+2")
    p.add_argument("--end", help="last run-day D (YYYY-MM-DD)")
    p.add_argument("--countries", default="all")
    p.add_argument("--out-db", default=None,
                   help="reconstruction DB (never the as-served sidecar)")
    p.add_argument("--sidecar-db", default=config.FORECAST_OUTPUT_DB,
                   help="as-served sidecar, read-only, for --verify")
    p.add_argument("--device", default="cuda")
    p.add_argument("--verify", metavar="GENERATED_AT",
                   help="reproduce this stored vintage and report max|diff|")
    p.add_argument("--tolerance", type=float, default=1.0,
                   help="max|diff| accepted by --verify, %% of mean |forecast|. "
                        "Not 0: `predict_quantiles` is deterministic, so a "
                        "same-day reconstruction matches bit-exactly, but the "
                        "replica moves. Suffix-1 covariates (TSO forecasts, DA "
                        "prices, flows) get revised after the fact and carry no "
                        "usable publication time, so a vintage reconstructed "
                        "days later legitimately differs by a little.")
    args = p.parse_args()

    model_name = args.model_name or f"chronos-2-{args.experiment}"

    if args.countries == "all":
        countries = [c for c in config.SUPPORTED_COUNTRIES
                     if c not in EXCLUDED_COUNTRIES]
    else:
        countries = [c for c in args.countries.split(",")
                     if c not in EXCLUDED_COUNTRIES]

    engine, builder = build_engine(args.experiment, args.device)

    if args.verify:
        if not args.sidecar_db:
            logger.error("--verify needs --sidecar-db (or FORECAST_OUTPUT_DB)")
            return 2
        return verify(args, engine, builder, model_name)

    if not (args.start and args.end and args.out_db):
        logger.error("--start, --end and --out-db are required without --verify")
        return 2

    out_db = Path(args.out_db)
    if config.FORECAST_OUTPUT_DB and out_db.resolve() == Path(config.FORECAST_OUTPUT_DB).resolve():
        logger.error("refusing to write reconstructions into the as-served "
                     "sidecar %s - use a separate --out-db", out_db)
        return 2
    out_db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(out_db)
    con.executescript(SCHEMA)
    con.executemany("INSERT OR REPLACE INTO reconstruction_meta (key, value) VALUES (?,?)",
                    [("as_of_policy", "net_position.as_of_for_vintage (day-ahead: D 22:00)"),
                     ("experiment", args.experiment), ("model_name", model_name),
                     ("run_hour_utc", str(RUN_HOUR)),
                     ("excluded_countries", ",".join(EXCLUDED_COUNTRIES)),
                     ("written_at_utc", datetime.now(timezone.utc).isoformat()),
                     ("range", f"{args.start}..{args.end}")])
    con.commit()
    con.close()

    run_days = pd.date_range(args.start, args.end, freq="D")
    logger.info("reconstructing %d run-days x %d countries -> %s",
                len(run_days), len(countries), out_db)

    ok = failed = 0
    for run_day in run_days:
        generated_at = synthetic_generated_at(run_day)
        as_of = as_of_for_vintage(generated_at)
        target_date = (run_day + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        for cc in countries:
            try:
                median, quantiles, index = forecast_one(
                    engine, builder, cc, target_date, as_of, generated_at)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s %s: skipped - %s", target_date, cc, exc)
                failed += 1
                continue
            write_vintage(out_db, cc, generated_at, index, median, quantiles, model_name)
            ok += 1
        logger.info("run-day %s (target %s): %d ok / %d failed so far",
                    run_day.date(), target_date, ok, failed)

    logger.info("done: %d country-vintages written, %d skipped -> %s", ok, failed, out_db)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
