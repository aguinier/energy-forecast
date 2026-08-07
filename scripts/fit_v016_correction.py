#!/usr/bin/env python3
"""Fit V016's per-country correction coefficients (ABL-68).

Reads serve-faithful champion forecasts (normally the reconstruction built by
`reconstruct_v010_vintages.py`), joins them to net-position actuals, and fits
per country: an affine recalibration and an AR(1) coefficient on the
recalibrated residual. Writes `experiments/V016/correction.json`.

Three things this refuses to do, because each is a route to a confidently wrong
number:

* **Fit a country it cannot support.** Below the pair and target-day floors, or
  where the champion carries too little signal, the country is written as an
  explicit identity with a reason. V016 then serves V010 for that country.
* **Fit on the backtest weeks.** W11/W12 fall inside the reconstructable window.
  Fitting on them would make the gate's "no regression on the W01-W12 backtest"
  an in-sample claim. Their target days are dropped.
* **Fit on a country whose reconstruction does not match what production
  served.** LT, RO and BG reproduce the as-served 2026-08-06 vintage 38.8%,
  5.9% and 1.4% away from it, because suffix-1 covariates carry no usable
  publication time. Their coefficients would describe a model that never ran.

Usage:
    python scripts/fit_v016_correction.py \
        --pairs-db C:\\Code\\able\\data\\forecasts_recon.db \
        --train-end 2026-06-15
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.challengers.correction import (MIN_PAIRS, MIN_TARGET_DAYS,
                                        fit_country_correction)
from src.challengers.registry import CHAMPION_MODEL_NAME
from src.evaluation.net_position import _parse_ts, _ro_connect

# Measured 2026-08-07 by `reconstruct_v010_vintages.py --verify 2026-08-06T06:00:44`:
# relative max|diff| against the as-served vintage, tolerance 1%.
DEFAULT_UNVERIFIED = "LT,RO,BG"


def load_pairs(pairs_db: str, replica_db: str, model_name: str) -> pd.DataFrame:
    con = _ro_connect(pairs_db)
    try:
        f = pd.read_sql_query(
            """SELECT country_code, target_timestamp_utc, generated_at, forecast_value
               FROM forecasts WHERE forecast_type = 'net_position' AND model_name = ?""",
            con, params=(model_name,))
    finally:
        con.close()
    if f.empty:
        raise SystemExit(f"no '{model_name}' forecasts in {pairs_db}")
    f["target_ts"] = _parse_ts(f["target_timestamp_utc"])
    f["generated_at"] = _parse_ts(f["generated_at"])

    con = _ro_connect(replica_db)
    try:
        a = pd.read_sql_query(
            """SELECT country_code, timestamp_utc, net_position_mw FROM net_position
               WHERE net_position_mw IS NOT NULL""", con)
    finally:
        con.close()
    a["target_ts"] = _parse_ts(a["timestamp_utc"]).dt.floor("h")
    a = (a.sort_values("target_ts").groupby(["country_code", "target_ts"]).tail(1)
          [["country_code", "target_ts", "net_position_mw"]]
          .rename(columns={"net_position_mw": "actual"}))
    return f.merge(a, on=["country_code", "target_ts"], how="left")


def drop_backtest_weeks(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Backtest target days must stay out of sample or the gate is circular."""
    mask = pd.Series(False, index=df.index)
    for _, start, end in config.BACKTEST_WEEKS:
        mask |= (df["target_ts"] >= pd.Timestamp(start)) & \
                (df["target_ts"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    return df[~mask], int(mask.sum())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs-db", required=True,
                   help="serve-faithful champion vintages (the reconstruction)")
    p.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    p.add_argument("--model-name", default=CHAMPION_MODEL_NAME)
    p.add_argument("--train-end", default=None,
                   help="fit on target days before this; later days are the "
                        "held-out validation window")
    p.add_argument("--unverified-countries", default=DEFAULT_UNVERIFIED,
                   help="countries whose reconstruction does not reproduce the "
                        "as-served vintage; passed through uncorrected")
    p.add_argument("--method", default="ols", choices=("ols", "variance"),
                   help="ols minimises error (slope becomes rho**2); variance "
                        "matches sd (slope becomes rho) at a measured 11%% MAE cost")
    p.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    p.add_argument("--min-target-days", type=int, default=MIN_TARGET_DAYS)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = Path(args.out) if args.out else \
        config.EXPERIMENTS_DIR / "V016" / "correction.json"
    unverified = {c for c in args.unverified_countries.split(",") if c}

    pairs = load_pairs(args.pairs_db, args.replica_db, args.model_name)
    total_rows = len(pairs)
    pairs, dropped = drop_backtest_weeks(pairs)
    if args.train_end:
        cutoff = pd.Timestamp(args.train_end)
        held_out = int((pairs["target_ts"] >= cutoff).sum())
        pairs = pairs[pairs["target_ts"] < cutoff]
    else:
        held_out = 0

    corrections, applied = {}, 0
    for country, g in pairs.groupby("country_code"):
        fit = fit_country_correction(
            g, country, min_pairs=args.min_pairs,
            min_target_days=args.min_target_days,
            serve_parity_verified=country not in unverified,
            method=args.method)
        corrections[country] = fit.to_dict()
        applied += bool(fit.applied)

    doc = {
        "experiment": "V016",
        "parent_model": args.model_name,
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "pairs_db": str(args.pairs_db),
        "train_end": args.train_end,
        "method": args.method,
        "rows_loaded": total_rows,
        "rows_dropped_backtest_weeks": dropped,
        "rows_held_out_for_validation": held_out,
        "min_pairs": args.min_pairs,
        "min_target_days": args.min_target_days,
        "unverified_countries": sorted(unverified),
        "countries_applied": applied,
        "countries_total": len(corrections),
        "corrections": corrections,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"  {total_rows:,} rows loaded, {dropped:,} dropped (backtest weeks), "
          f"{held_out:,} held out for validation")
    print(f"  {applied}/{len(corrections)} countries corrected")
    for cc, c in sorted(corrections.items()):
        if c["applied"]:
            print(f"    {cc}: slope {c['slope']:.3f} (was {c['slope_forecast_on_actual']:.3f}"
                  f" forecast-on-actual), intercept {c['intercept_mw']:+.0f} MW, "
                  f"phi {c['ar1_phi']:.3f} -> carries {c['ar1_carry_at_min_lead']:.3f} at 27h")
        else:
            print(f"    {cc}: PASS-THROUGH - {c['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
