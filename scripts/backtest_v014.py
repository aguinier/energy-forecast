#!/usr/bin/env python3
"""Serve-faithful W01-W12 backtest for V014 (ABL-69, gate criterion G5).

Replays each held-out backtest week one run day at a time: for target day T the
run is `T - 2` at 06:00Z, and every feature is bounded by what that run held.
Nothing is fitted here — the models come from `scripts/train_v014.py`, which
excluded these same target days.

Writes a `compare_experiments`-style JSON so it can be handed straight to
`evaluate_net_position.py --candidate-backtest`:

    {"V014": {"FR": {"net_position": {"W01": {"mae": ..., "rmse": ...}, ...}}}}

Two things about the numbers that must be read with the file, not inferred:

- **MAPE is reported but is not a score here.** Net position crosses zero, so a
  near-zero actual makes the percentage explode; the existing V010 file records
  MAPE values of 382% for the same reason. MAE and RMSE are the comparison; MAPE
  is carried only because the reference file carries it.
- **W01-W10 are weather-blind for every model.** The issued-weather archive
  begins 2026-01-11 (`v014_features` docstring), so those ten weeks have NaN
  weather for V014 and the champion's loader filters the same way. The
  comparison is fair, but neither model is in its serving configuration there.

Usage:
    python scripts/backtest_v014.py --countries all
    python scripts/backtest_v014.py --countries BE,NL,AT,FR --weeks W11,W12
"""

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
from src.challengers.v014 import EXPERIMENT_ID, load_model
from src.challengers.v014_features import (TARGET_DAY_OFFSET, ServeWindow,
                                           build_cache, build_features,
                                           load_net_position)
from src.evaluation.net_position import _ro_connect

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_v014")

EXCLUDED_COUNTRIES = ("LU", "GR")


def week_metrics(actual: pd.Series, predicted: pd.Series) -> dict | None:
    """MAE/RMSE/MAPE/SMAPE over the hours both series cover.

    Returns None — never a zeroed dict — when nothing pairs. An empty week must
    read as "not measured", because a 0.0 MAE is indistinguishable from a
    perfect forecast on the report it feeds.
    """
    joined = pd.concat([actual.rename("a"), predicted.rename("p")], axis=1).dropna()
    if joined.empty:
        return None
    err = joined["p"] - joined["a"]
    denom = joined["a"].abs()
    mape = float((err.abs()[denom > 0] / denom[denom > 0]).mean() * 100) if (denom > 0).any() else None
    scale = (joined["a"].abs() + joined["p"].abs()) / 2
    smape = float((err.abs()[scale > 0] / scale[scale > 0]).mean() * 100) if (scale > 0).any() else None
    return {"mae": float(err.abs().mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "mape": mape, "smape": smape, "n": int(len(joined))}


def backtest_country(conn, models_dir: Path, country: str,
                     weeks: list[tuple[str, str, str]]) -> dict:
    model = load_model(models_dir, country)
    span_start = pd.Timestamp(min(w[1] for w in weeks)) - pd.Timedelta(days=40)
    span_end = pd.Timestamp(max(w[2] for w in weeks)) + pd.Timedelta(days=1)
    cache = build_cache(conn, country, span_start, span_end)
    actuals = load_net_position(conn, country, span_start, span_end)

    out = {}
    for week_id, start, end in weeks:
        preds = []
        for target_day in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D"):
            window = ServeWindow.for_target_day(target_day)
            features = build_features(cache, window, neighbours=model.neighbours)
            preds.append(model.predict_frame(features))
        predicted = pd.concat(preds) if preds else pd.Series(dtype=float)
        metrics = week_metrics(actuals.reindex(predicted.index), predicted)
        if metrics is not None:
            out[week_id] = metrics
        else:
            logger.warning("%s %s: no paired hours - week omitted rather than "
                           "scored as 0", country, week_id)
    return out


def live_countries(conn) -> list[str]:
    """The same 19 countries `train_v014.py` fits — see its docstring for why
    both the supported-set and the row-count filter are needed."""
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
    p.add_argument("--weeks", default="all", help="e.g. W01,W11,W12")
    p.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    p.add_argument("--models-dir", default=str(config.MODELS_DIR))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    week_ids = None if args.weeks == "all" else [w.strip() for w in args.weeks.split(",")]
    weeks = config.get_backtest_weeks(week_ids)
    if not weeks:
        logger.error("no backtest weeks matched %r", args.weeks)
        return 2

    conn = _ro_connect(args.replica_db)
    try:
        countries = (live_countries(conn) if args.countries == "all"
                     else [c.strip().upper() for c in args.countries.split(",")
                           if c.strip().upper() not in EXCLUDED_COUNTRIES])
        logger.info("V014 backtest: %d countries x %d weeks (targets D+%d, run 06:00Z)",
                    len(countries), len(weeks), TARGET_DAY_OFFSET)

        results, skipped = {}, []
        for i, cc in enumerate(countries, 1):
            t0 = time.time()
            try:
                per_week = backtest_country(conn, Path(args.models_dir), cc, weeks)
            except FileNotFoundError as exc:
                logger.error("%s: %s", cc, exc)
                skipped.append(cc)
                continue
            if not per_week:
                logger.warning("%s: no week scored", cc)
                skipped.append(cc)
                continue
            results[cc] = {"net_position": per_week}
            maes = [w["mae"] for w in per_week.values()]
            logger.info("[%2d/%d] %s  %d weeks  mean MAE %.0f MW  (%.0fs)",
                        i, len(countries), cc, len(per_week), float(np.mean(maes)),
                        time.time() - t0)

        out = Path(args.out) if args.out else (
            config.EXPERIMENTS_DIR / EXPERIMENT_ID / "backtest_W01_W12.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({EXPERIMENT_ID: results}, indent=2))
        logger.info("wrote %s (%d countries, %d skipped)", out, len(results), len(skipped))
        if skipped:
            logger.warning("not scored: %s", ", ".join(skipped))
        return 0 if results else 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
