"""ABL-200 before/after backtest: does excluding the sibling-disproved zeros
move a model?

Paired A/B on ABL-348's registered windows. Arm A reconstructs the pre-ABL-200
training set, arm B is the rule as landed; both arms fit the same algorithm on
the same protocol and are scored on the same gate rows, and the census reports
**zero** marginal exclusions inside the gate window, so the truth and the scored
rows are byte-identical between arms and only the fit data moves.

Run with several seeds. The effect being measured is a 0.1-0.6% change in the
fit rows of a pair, which is far smaller than the single-seed null this repo has
measured repeatedly (ABL-395: 27.34pp on one CH night-hour statistic), so a
one-seed delta here would be noise wearing a result's clothes.

This is **not** a gate read. It registers no scope, writes no artifact, refits
no serving pair and reads no `SCOPE_OUTPUTS` path. Per ABL-401, re-reading a
registered gate against a changed training set is a new pre-registration and is
not what this script does.

Usage:
  python scripts/abl200_before_after_backtest.py --pairs GR:wind_onshore EE:wind_onshore
  python scripts/abl200_before_after_backtest.py --seeds 5 --json-out reports/abl_200_backtest.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src import data_quality, db
from src.evaluation.scorecard import score_predictions
from src.evaluation.wind_retrain import (
    FEATURE_COLUMNS,
    PRIMARY_BANDS,
    attach_baselines,
    build_vintage_frame,
    common_scores,
    finite_training_rows,
    select_latest_challenger_per_band,
)
from src.wind_features import RenewableFeatureBuilder
from xgboost import XGBRegressor

#: ABL-348's registered windows, unchanged. Named here so the report says which
#: protocol produced it; this script does not re-decide any of them.
FIT_START = "2026-01-14"
GATE_START = "2026-07-11"
GATE_END = "2026-08-10"

#: The pairs where the rule removes the most fit-window rows, per
#: `scripts/abl200_cross_table_zero_census.py` on 2026-08-14. Two of them sit
#: under a registered scope (GR in `abl406-tranche2b`, EE in
#: `abl417-tranche2e`); IT `wind_offshore` sits under none and is the control
#: for "does a pair with no gate move at all".
DEFAULT_PAIRS = ("GR:wind_onshore", "EE:wind_onshore", "IT:wind_offshore")


class _GuardOff:
    """Arm A: `load_renewable_type_data` as it read before ABL-200.

    A context manager rather than a parameter on the loader, deliberately. A
    `disprove_zeros=False` argument on the production read path is a knob that
    can silently disable a data-quality guard from anywhere; reconstructing the
    old behaviour for a measurement is the only legitimate use, and it belongs
    in the measurement, visible, not in the loader.
    """

    def __enter__(self):
        self._real = db.exclude_zeros_disproved_by_sibling
        db.exclude_zeros_disproved_by_sibling = lambda df, sibling, **kw: df
        return self

    def __exit__(self, *exc):
        db.exclude_zeros_disproved_by_sibling = self._real
        return False


def build_arm(country, forecast_type, replica, guard_on):
    """(fit rows, gate rows, builder) for one arm."""
    fit_start, gate_start, gate_end = (
        pd.Timestamp(FIT_START), pd.Timestamp(GATE_START), pd.Timestamp(GATE_END)
    )

    def _build():
        builder = RenewableFeatureBuilder(
            country, forecast_type, fit_start - pd.Timedelta(days=14), gate_end,
            actuals_source=db.RENEWABLE_TYPE_SOURCE_TABLE, db_path=str(replica),
        )
        fit, _ = finite_training_rows(build_vintage_frame(builder, fit_start, gate_start))
        gate, _ = finite_training_rows(build_vintage_frame(builder, gate_start, gate_end))
        return fit, gate, builder

    if guard_on:
        return _build()
    with _GuardOff():
        return _build()


def score_arm(fit, gate, builder, seed):
    params = config.get_default_params("xgboost")
    params.pop("early_stopping_rounds", None)
    params["random_state"] = seed
    model = XGBRegressor(**params)
    model.fit(fit[list(FEATURE_COLUMNS)], fit["actual"])

    scored = gate.copy()
    scored["challenger"] = model.predict(scored[list(FEATURE_COLUMNS)])
    selected = attach_baselines(select_latest_challenger_per_band(scored), builder._actuals)

    out = {}
    for band in PRIMARY_BANDS:
        group = selected[selected["horizon_band"] == band]
        if group.empty:
            out[band] = {"n": 0, "wape_pct": None, "naive_wape_pct": None}
            continue
        _, common = common_scores(group, ("challenger", "seasonal_naive"))
        if common.empty:
            out[band] = {"n": 0, "wape_pct": None, "naive_wape_pct": None}
            continue
        out[band] = {
            "n": int(len(common)),
            "wape_pct": score_predictions(common["actual"], common["challenger"])["wape_pct"],
            "naive_wape_pct": score_predictions(
                common["actual"], common["seasonal_naive"])["wape_pct"],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=(
        "ABL-200 paired before/after backtest on ABL-348's registered windows. "
        "Not a gate read: registers no scope and writes no artifact."
    ))
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS),
                        help="COUNTRY:forecast_type, e.g. GR:wind_onshore")
    parser.add_argument("--seeds", type=int, default=5,
                        help="fits per arm; a one-seed delta is not a result")
    parser.add_argument("--replica-db", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    replica = args.replica_db or config.DATABASE_PATH
    print(f"replica: {replica}")
    print(f"fit {FIT_START} -> {GATE_START}; gate {GATE_START} -> {GATE_END}; "
          f"bands {list(PRIMARY_BANDS)}; {args.seeds} seeds per arm\n")

    results = []
    for spec in args.pairs:
        country, forecast_type = spec.split(":")
        t0 = time.perf_counter()
        fit_a, gate_a, builder_a = build_arm(country, forecast_type, replica, guard_on=False)
        fit_b, gate_b, builder_b = build_arm(country, forecast_type, replica, guard_on=True)
        print(f"{spec}: fit rows {len(fit_a)} (before) -> {len(fit_b)} (after), "
              f"delta {len(fit_b) - len(fit_a)}; gate rows {len(gate_a)} -> {len(gate_b)} "
              f"({time.perf_counter() - t0:.0f}s to build)")

        per_seed = []
        for seed in range(args.seeds):
            before = score_arm(fit_a, gate_a, builder_a, seed)
            after = score_arm(fit_b, gate_b, builder_b, seed)
            per_seed.append({"seed": seed, "before": before, "after": after})

        pair = {"country": country, "forecast_type": forecast_type,
                "fit_rows_before": int(len(fit_a)), "fit_rows_after": int(len(fit_b)),
                "gate_rows_before": int(len(gate_a)), "gate_rows_after": int(len(gate_b)),
                "seeds": per_seed}
        results.append(pair)

        for band in PRIMARY_BANDS:
            b = np.array([s["before"][band]["wape_pct"] for s in per_seed], dtype=float)
            a = np.array([s["after"][band]["wape_pct"] for s in per_seed], dtype=float)
            if np.isnan(b).all() or np.isnan(a).all():
                print(f"    {band:>8}: not measured")
                continue
            delta = a - b
            n = per_seed[0]["after"][band]["n"]
            print(f"    {band:>8}: n={n:5d}  before {np.nanmean(b):6.2f}% "
                  f"(sd {np.nanstd(b, ddof=1):.2f})  after {np.nanmean(a):6.2f}% "
                  f"(sd {np.nanstd(a, ddof=1):.2f})  paired delta "
                  f"{np.nanmean(delta):+.3f}pp  ({int((delta < 0).sum())}/{len(delta)} seeds better)")
        print()

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "meta": {
                "replica_db": str(replica),
                "fit_window": [FIT_START, GATE_START],
                "gate_window": [GATE_START, GATE_END],
                "bands": list(PRIMARY_BANDS),
                "seeds": args.seeds,
                "algorithm": "xgboost",
                "training_source": db.RENEWABLE_TYPE_SOURCE_TABLE,
                "disproof_quantile": data_quality.SIBLING_DISPROOF_QUANTILE,
                "is_gate_read": False,
            },
            "pairs": results,
        }, indent=2, default=float))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
