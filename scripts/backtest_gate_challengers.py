#!/usr/bin/env python3
"""Build the missing serve-faithful W01-W12 gate artefacts (ABL-192).

V010 establishes the current reference with the two-cutoff protocol. V012
replays the exact served baseline definition. V016 reconstructs V010 with
the two serve bounds established on ABL-68 (observations through run-day 22:00,
publications through the 06:00 run), then applies the production-equivalent
correction using only the latest residual observable at that cutoff.

V014 already has its 19-country artefact and generator in backtest_v014.py.

The output is compare_experiments-compatible and is read directly by
evaluate_net_position.py --candidate-backtest. Nothing is fitted or written to
either database.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from scripts.backtest_v014 import EXCLUDED_COUNTRIES, live_countries, week_metrics
from scripts.reconstruct_v010_vintages import (build_engine, forecast_one,
                                                synthetic_generated_at)
from src.challengers.baseline import forecast_baseline_ensemble
from src.challengers.correction import (CountryCorrection, apply_correction,
                                        latest_residual)
from src.challengers.v014_features import load_net_position
from src.evaluation.net_position import _ro_connect, as_of_for_vintage

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_gate_challengers")

REFERENCE_BACKTEST = Path(__file__).parent.parent / "comparison_net_position_servefaithful.json"


def _target_days(weeks: list[tuple[str, str, str]], warmup_days: int = 0
                 ) -> list[pd.Timestamp]:
    days: set[pd.Timestamp] = set()
    for _, start, end in weeks:
        first = pd.Timestamp(start) - pd.Timedelta(days=warmup_days)
        days.update(pd.date_range(first, pd.Timestamp(end), freq="D"))
    return sorted(days)


def backtest_v012(conn, country: str,
                  weeks: list[tuple[str, str, str]]) -> dict:
    span_start = pd.Timestamp(min(w[1] for w in weeks)) - pd.Timedelta(days=40)
    span_end = pd.Timestamp(max(w[2] for w in weeks)) + pd.Timedelta(days=1)
    actuals = load_net_position(conn, country, span_start, span_end)
    out = {}
    for week_id, start, end in weeks:
        preds = []
        for target_day in pd.date_range(start, end, freq="D"):
            generated_at = synthetic_generated_at(target_day - pd.Timedelta(days=2))
            as_of = as_of_for_vintage(generated_at)
            targets = pd.date_range(target_day, periods=24, freq="h")
            preds.append(forecast_baseline_ensemble(actuals, as_of, targets))
        predicted = pd.concat(preds)
        metrics = week_metrics(actuals.reindex(predicted.index), predicted)
        if metrics is not None:
            out[week_id] = metrics
    return out


def reconstruct_champion(engine, builder, country: str,
                         weeks: list[tuple[str, str, str]]) -> pd.DataFrame:
    """V010 target days plus two warm-up days per week for V016's AR residual."""
    rows = []
    for target_day in _target_days(weeks, warmup_days=2):
        run_day = target_day - pd.Timedelta(days=2)
        generated_at = synthetic_generated_at(run_day)
        median, _, index = forecast_one(
            engine, builder, country, target_day.strftime("%Y-%m-%d"),
            as_of_for_vintage(generated_at), generated_at)
        rows.extend({"country_code": country, "target_ts": ts,
                     "generated_at": generated_at, "forecast_value": float(value)}
                    for ts, value in zip(index, median))
    return pd.DataFrame(rows)


def score_champion(conn, raw: pd.DataFrame,
                   weeks: list[tuple[str, str, str]], country: str) -> dict:
    actuals = load_net_position(
        conn, country, raw["target_ts"].min(),
        raw["target_ts"].max() + pd.Timedelta(days=1))
    out = {}
    for week_id, start, end in weeks:
        first, last = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
        g = raw[(raw["target_ts"] >= first) & (raw["target_ts"] < last)]
        predicted = pd.Series(g["forecast_value"].to_numpy(), index=g["target_ts"])
        metrics = week_metrics(actuals.reindex(predicted.index), predicted)
        if metrics is not None:
            out[week_id] = metrics
    return out


def backtest_v016(conn, engine, builder, country: str,
                  weeks: list[tuple[str, str, str]],
                  correction: CountryCorrection) -> dict:
    raw = reconstruct_champion(engine, builder, country, weeks)
    span_start = raw["target_ts"].min() - pd.Timedelta(days=1)
    span_end = raw["target_ts"].max() + pd.Timedelta(days=1)
    actuals = load_net_position(conn, country, span_start, span_end)
    history = raw.copy()
    history["actual"] = actuals.reindex(history["target_ts"]).to_numpy()

    out = {}
    for week_id, start, end in weeks:
        corrected_parts = []
        for target_day in pd.date_range(start, end, freq="D"):
            generated_at = synthetic_generated_at(target_day - pd.Timedelta(days=2))
            as_of = as_of_for_vintage(generated_at)
            g = raw[raw["generated_at"] == generated_at].sort_values("target_ts")
            observable = history[history["target_ts"] < as_of]
            residual, residual_ts = latest_residual(observable, as_of, correction)
            values = apply_correction(
                g["forecast_value"].to_numpy(), pd.DatetimeIndex(g["target_ts"]),
                correction, residual, residual_ts)
            corrected_parts.append(pd.Series(values, index=g["target_ts"]))
        predicted = pd.concat(corrected_parts)
        metrics = week_metrics(actuals.reindex(predicted.index), predicted)
        if metrics is not None:
            out[week_id] = metrics
    return out


def _write(experiment: str, results: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({experiment: results}, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d countries)", out, len(results))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=("V010", "V012", "V016"), required=True)
    p.add_argument("--countries", default=None,
                   help="comma-separated; default all 19 for V012 and the four "
                        "reference countries for V010/V016")
    p.add_argument("--weeks", default="all")
    p.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out")
    args = p.parse_args()

    week_ids = None if args.weeks == "all" else args.weeks.split(",")
    weeks = config.get_backtest_weeks(week_ids)
    if not weeks:
        logger.error("no backtest weeks matched %r", args.weeks)
        return 2

    conn = _ro_connect(args.replica_db)
    try:
        if args.countries:
            countries = [c.strip().upper() for c in args.countries.split(",")
                         if c.strip().upper() not in EXCLUDED_COUNTRIES]
        elif args.experiment == "V012":
            countries = live_countries(conn)
        else:
            reference = json.loads(REFERENCE_BACKTEST.read_text(encoding="utf-8"))
            countries = sorted(next(iter(reference.values())).keys())

        results = {}
        if args.experiment == "V012":
            for i, country in enumerate(countries, 1):
                per_week = backtest_v012(conn, country, weeks)
                if per_week:
                    results[country] = {"net_position": per_week}
                logger.info("[%d/%d] %s: %d weeks", i, len(countries), country,
                            len(per_week))
        else:
            corrections = {}
            if args.experiment == "V016":
                fit_doc = json.loads(
                    (config.EXPERIMENTS_DIR / "V016" / "correction.json")
                    .read_text(encoding="utf-8"))
                corrections = {cc: CountryCorrection(**v)
                               for cc, v in fit_doc["corrections"].items()}
            engine, builder = build_engine("V010", args.device)
            for i, country in enumerate(countries, 1):
                t0 = time.time()
                if args.experiment == "V010":
                    raw = reconstruct_champion(engine, builder, country, weeks)
                    per_week = score_champion(conn, raw, weeks, country)
                else:
                    correction = corrections.get(country)
                    if correction is None:
                        logger.error("%s has no V016 correction; omitted", country)
                        continue
                    per_week = backtest_v016(conn, engine, builder, country, weeks,
                                             correction)
                if per_week:
                    results[country] = {"net_position": per_week}
                logger.info("[%d/%d] %s: %d weeks (%.0fs)", i, len(countries),
                            country, len(per_week), time.time() - t0)

        if args.out:
            out = Path(args.out)
        elif args.experiment == "V010":
            out = REFERENCE_BACKTEST
        else:
            out = config.EXPERIMENTS_DIR / args.experiment / "backtest_W01_W12.json"
        _write(args.experiment, results, out)
        return 0 if results else 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
