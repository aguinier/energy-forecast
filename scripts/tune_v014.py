#!/usr/bin/env python3
"""Bounded hyperparameter search for V014, per country (ABL-69, scope item 4).

The default parameters lose to the champion on exactly the countries this
program cares about — measured on the W01-W12 backtest, V014 is 13-22% worse
than V010 on NL/BE/AT/FR — so `TUNING_ATTENTION` defaults to those four.

**Nothing here touches the backtest weeks.** Candidates are scored on the same
chronological validation split `train_country` uses (the last 12% of run days,
split on run day, never on row), which sits inside the training span; W01-W12
are excluded from the run days before any of this runs. Tuning against the
backtest would make the backtest a training set and its numbers a claim about
data the model had already seen.

Selection is on **validation MAE**, and a candidate has to beat the incumbent by
`MIN_IMPROVEMENT_PCT` to be reported as a win. Anything smaller is noise dressed
as a result: the validation split is ~3,300 rows, and picking the argmin over a
dozen candidates on a split that small will always find *something* lower.

It writes a report and, unless `--adopt` is passed, **saves no model**. Adopting
is a deliberate act — the artifacts are what the daily shadow serves.

Usage:
    python scripts/tune_v014.py                      # the four majors, report only
    python scripts/tune_v014.py --countries BE --adopt
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
from src.challengers.v014 import (DEFAULT_PARAMS, EXPERIMENT_ID,
                                  VALIDATION_FRACTION, V014Model, _metrics,
                                  _split_by_run_day, backtest_target_days,
                                  run_days_for_span, save_model)
from src.challengers.v014_features import build_cache, build_training_frame, feature_columns
from src.evaluation.net_position import _ro_connect

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tune_v014")

#: Where the measured gap against the champion is (ABL-69 backtest vs V010).
TUNING_ATTENTION = ("BE", "NL", "AT", "FR")

#: A candidate must beat the incumbent by this much to count as a win.
MIN_IMPROVEMENT_PCT = 2.0

DEFAULT_START = "2023-02-01"

#: Deliberately a small, hand-read grid rather than a large random search. Each
#: axis is here because it plausibly addresses the observed failure — the model
#: is under-fitting the level on high-variance countries — and a grid a human
#: can read is one whose winner can be sanity-checked.
CANDIDATES = [
    {"name": "default"},
    {"name": "deeper", "max_depth": 8, "min_child_weight": 3},
    {"name": "deeper_slow", "max_depth": 8, "learning_rate": 0.02,
     "n_estimators": 2000},
    {"name": "shallow_slow", "max_depth": 4, "learning_rate": 0.02,
     "n_estimators": 2000},
    {"name": "less_reg", "reg_lambda": 0.5, "min_child_weight": 1},
    {"name": "more_reg", "reg_lambda": 8.0, "min_child_weight": 10},
    {"name": "wide_cols", "colsample_bytree": 1.0, "subsample": 1.0},
    {"name": "slow_deep_reg", "max_depth": 8, "learning_rate": 0.02,
     "n_estimators": 2000, "reg_lambda": 8.0, "subsample": 0.7},
]


def fit_candidate(train, val, cols, overrides):
    """Fit one candidate on a prepared split. Returns (model_or_None, metrics)."""
    from xgboost import XGBRegressor

    cfg = {**DEFAULT_PARAMS, **{k: v for k, v in overrides.items() if k != "name"}}
    fit_kwargs = {}
    if not val.empty:
        cfg = {**cfg, "early_stopping_rounds": 60}
        fit_kwargs["eval_set"] = [(val[cols].to_numpy(dtype=np.float32),
                                   val["target_net_position_mw"].to_numpy(dtype=float))]
        fit_kwargs["verbose"] = False
    booster = XGBRegressor(**cfg)
    booster.fit(train[cols].to_numpy(dtype=np.float32),
                train["target_net_position_mw"].to_numpy(dtype=float), **fit_kwargs)
    if val.empty:
        return booster, {"mae": None, "rmse": None, "slope": None, "n": 0}
    return booster, _metrics(val["target_net_position_mw"].to_numpy(dtype=float),
                             booster.predict(val[cols].to_numpy(dtype=np.float32)))


def tune_country(conn, country: str, run_days, neighbours, models_dir: Path,
                 adopt: bool) -> dict:
    cache = build_cache(conn, country,
                        pd.Timestamp(run_days[0]) - pd.Timedelta(days=35),
                        pd.Timestamp(run_days[-1]) + pd.Timedelta(days=3))
    frame = build_training_frame(conn, country, run_days, neighbours=neighbours,
                                 cache=cache)
    frame = frame[frame["target_net_position_mw"].notna()]
    cols = feature_columns(frame)
    train, val = _split_by_run_day(frame, VALIDATION_FRACTION)
    logger.info("%s: %d rows (%d train / %d val), %d features",
                country, len(frame), len(train), len(val), len(cols))

    results, best = [], None
    for cand in CANDIDATES:
        t0 = time.time()
        booster, m = fit_candidate(train, val, cols, cand)
        row = {"candidate": cand["name"], "overrides": {k: v for k, v in cand.items()
                                                        if k != "name"},
               **{k: m[k] for k in ("mae", "rmse", "slope", "n")},
               "seconds": round(time.time() - t0, 1)}
        results.append(row)
        logger.info("  %-14s val MAE %8.1f  slope %s  (%.0fs)", cand["name"],
                    m["mae"] if m["mae"] is not None else float("nan"),
                    f"{m['slope']:.3f}" if m["slope"] is not None else "n/a",
                    row["seconds"])
        if m["mae"] is not None and (best is None or m["mae"] < best[1]["mae"]):
            best = (cand, m, booster, cols)

    incumbent = next((r for r in results if r["candidate"] == "default"), None)
    out = {"country": country, "n_validation_rows": int(len(val)),
           "incumbent_mae": incumbent["mae"] if incumbent else None,
           "results": sorted(results, key=lambda r: (r["mae"] is None, r["mae"]))}
    if best is None or not incumbent or incumbent["mae"] is None:
        out["winner"] = None
        return out

    gain = 100 * (incumbent["mae"] - best[1]["mae"]) / incumbent["mae"]
    out["best_candidate"] = best[0]["name"]
    out["best_mae"] = best[1]["mae"]
    out["improvement_pct"] = round(gain, 2)
    out["winner"] = best[0]["name"] if gain >= MIN_IMPROVEMENT_PCT else None
    if out["winner"] is None:
        logger.info("  -> no candidate beat default by >=%.1f%% (best %s, %+.2f%%)",
                    MIN_IMPROVEMENT_PCT, best[0]["name"], gain)
        return out

    logger.info("  -> winner %s, %.2f%% better than default", best[0]["name"], gain)
    if adopt:
        model = V014Model(country=country, booster=best[2],
                          feature_columns=best[3], neighbours=list(neighbours),
                          metadata={"tuned": best[0]["name"],
                                    "validation": best[1],
                                    "improvement_pct": out["improvement_pct"]})
        out["artifact"] = str(save_model(model, models_dir))
        logger.info("  -> adopted, wrote %s", out["artifact"])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--countries", default=",".join(TUNING_ATTENTION))
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=None)
    p.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    p.add_argument("--models-dir", default=str(config.MODELS_DIR))
    p.add_argument("--adopt", action="store_true",
                   help="save the winning model. Without this, nothing is written.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    conn = _ro_connect(args.replica_db)
    try:
        end = pd.Timestamp(args.end) if args.end else (
            pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=3))
        run_days = run_days_for_span(args.start, end,
                                     backtest_target_days(config.BACKTEST_WEEKS))
        logger.info("tuning %d countries over %d run days, %d candidates each "
                    "(W01-W12 excluded)", len(countries), len(run_days), len(CANDIDATES))
        report = [tune_country(conn, cc, run_days,
                               [n for n in config.COUNTRY_NEIGHBORS.get(cc, [])
                                if n not in ("LU", "GR")],
                               Path(args.models_dir), args.adopt)
                  for cc in countries]
    finally:
        conn.close()

    doc = {"experiment": EXPERIMENT_ID,
           "tuned_at": pd.Timestamp.now("UTC").tz_localize(None).isoformat(timespec="seconds"),
           "selection": "validation MAE, chronological split on run day",
           "min_improvement_pct": MIN_IMPROVEMENT_PCT,
           "adopted": bool(args.adopt),
           "countries": report}
    out = Path(args.out) if args.out else (
        config.EXPERIMENTS_DIR / EXPERIMENT_ID / "tuning_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))
    logger.info("wrote %s", out)
    winners = [r for r in report if r.get("winner")]
    logger.info("%d of %d countries have a candidate beating default by >=%.1f%%",
                len(winners), len(report), MIN_IMPROVEMENT_PCT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
