#!/usr/bin/env python3
"""ABL-395 A/B: what the two ABL-338 geometry features change in a gate fit.

ABL-381 measured that the solar gate harness fits at 25 features where an
ABL-338-current fit is 27, and that CH consequently predicts negative in 80.5%
of night hours. ABL-395 adds `sun_elevation_deg` / `is_night` to
`solar_retrain.FEATURE_COLUMNS`. This script measures the difference.

Protocol
--------
Both arms are fitted from **one** vintage frame, built once at the 27-name
superset, so the only thing that differs between them is the column list handed
to CatBoost:

  * arm `f25` - the 25 names the harness declared through ABL-381 (the control,
    and the number to compare against the published ABL-381 read).
  * arm `f27` - those 25 plus the geometry pair (the treatment).

The geometry features are pure functions of (country, hour) and are never NaN,
so `finite_training_rows` retains the same rows for both arms; the script
asserts that rather than assuming it, because an arm scored on a different row
set is not an A/B.

Windows, source, schedule, bands, seed and algorithm are ABL-348's registered
ones, so the `f25` arm is a like-for-like refit of the published read.

**This is not a gate re-read and dispositions nothing.** It writes no report
under any registered scope's `report_out`/`json_out`, and its artifacts go
wherever `--artifact-dir` points - never into `experiments/ABL348/artifacts`,
which is a registered scope output. ABL-381's PASS 6/6 stands as read, on the
artifacts it was read on.

What is measured, per arm and per country
-----------------------------------------
1. Night-hour behaviour on the rows the gate scored - the fraction predicted
   negative, the minimum prediction, and the mean prediction against the mean
   actual. Night is `solar_features.night_mask`, the same predicate the ABL-337
   serving clamp uses to zero an hour, so this cannot be a mask artefact.
2. Registered-band WAPE against seasonal-naive D-7, and daylight-only WAPE
   beside it. ABL-338's claim for these two features is that they are
   *daylight-safe* (mean -1.0%, worst +2.9% over its eight country-windows);
   this says whether that holds on these pairs.
3. Declared vs produced feature count, read back off the written artifact.

Read-only against the replica (`mode=ro`); writes only its own JSON and, if
asked, its own artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import db
from src.evaluation.gate_artifacts import save_gate_artifact
from src.evaluation.solar_retrain import (
    ALGORITHM, FEATURE_COLUMNS, PRIMARY_BANDS, SOLAR_GEOMETRY_FEATURES,
    attach_baselines, build_vintage_frame, common_scores, finite_training_rows,
    select_latest_challenger_per_band,
)
from src.solar_features import night_mask
from src.wind_features import RenewableFeatureBuilder, to_vector

#: The control arm: exactly what `solar_retrain.FEATURE_COLUMNS` was through
#: ABL-381. Derived by subtraction rather than copied, so it cannot drift from
#: the live list in any way other than the geometry pair itself.
FEATURE_COLUMNS_25 = tuple(c for c in FEATURE_COLUMNS if c not in SOLAR_GEOMETRY_FEATURES)

#: The basis ABL-381's scope gates on. BG and CH have zero rows in `forecasts`,
#: so naming `incumbent` would empty every intersection (ABL-322/ABL-378).
GATE_BASIS = ("challenger", "seasonal_naive")

ARMS = {"f25": FEATURE_COLUMNS_25, "f27": FEATURE_COLUMNS}


def _wape(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    denom = np.abs(actual).sum()
    return None if denom == 0 else float(np.abs(actual - forecast).sum() / denom * 100.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _night_read(selected: pd.DataFrame, country: str) -> dict:
    """The ABL-381 measurement, on the rows the gate would score."""
    preds = selected["challenger"].to_numpy(dtype=float)
    actual = selected["actual"].to_numpy(dtype=float)
    night = night_mask(country, list(pd.to_datetime(selected["target_ts"])))
    negative = preds < 0
    return {
        "n_selected_rows": int(len(preds)),
        "n_negative": int(negative.sum()),
        "pct_negative": round(float(negative.mean() * 100.0), 3),
        "min_prediction_mw": round(float(preds.min()), 2),
        "sum_negative_mwh": round(float(-preds[negative].sum()), 1) if negative.any() else 0.0,
        "n_night_rows": int(night.sum()),
        "n_negative_at_night": int((negative & night).sum()),
        "pct_of_night_rows_negative": (
            round(float((negative & night).sum() / night.sum() * 100.0), 2)
            if night.sum() else None),
        "mean_prediction_at_night_mw": (round(float(preds[night].mean()), 2)
                                        if night.sum() else None),
        "mean_actual_at_night_mw": (round(float(actual[night].mean()), 2)
                                    if night.sum() else None),
    }


def _cells(selected: pd.DataFrame, country: str) -> list[dict]:
    """Registered-band WAPE, and the same cell re-scored on daylight rows only."""
    out = []
    for band, group in selected[selected["horizon_band"].isin(PRIMARY_BANDS)].groupby(
            "horizon_band"):
        _, common = common_scores(group, GATE_BASIS)
        actual = common["actual"].to_numpy(dtype=float)
        chal = common["challenger"].to_numpy(dtype=float)
        d7 = common["seasonal_naive"].to_numpy(dtype=float)
        day = ~night_mask(country, list(pd.to_datetime(common["target_ts"])))
        cell = {
            "horizon_band": band, "n": int(len(common)),
            "challenger_wape_pct": round(_wape(actual, chal), 2),
            "d7_wape_pct": round(_wape(actual, d7), 2),
            "clears_d7": bool(_wape(actual, chal) < _wape(actual, d7)),
            "daylight_n": int(day.sum()),
        }
        cell["daylight_challenger_wape_pct"] = (
            round(_wape(actual[day], chal[day]), 2) if day.any() else None)
        cell["daylight_d7_wape_pct"] = (
            round(_wape(actual[day], d7[day]), 2) if day.any() else None)
        out.append(cell)
    return sorted(out, key=lambda row: row["horizon_band"])


def probe(country: str, replica: str, source: str, fit_start, gate_start, gate_end,
          artifact_dir: Path | None) -> dict:
    builder = RenewableFeatureBuilder(country, "solar", fit_start - pd.Timedelta(days=14),
                                      gate_end, actuals_source=source, db_path=replica)

    # Built once, at the superset, and shared by both arms.
    fit_raw = build_vintage_frame(builder, fit_start, gate_start, FEATURE_COLUMNS)
    gate_raw = build_vintage_frame(builder, gate_start, gate_end, FEATURE_COLUMNS)

    # What the builder actually produces for one row, independent of any list:
    # the "produces 27" half of the evidence. `to_vector` raises on a name it
    # cannot build, so this reaching 27 is a positive result, not an absence.
    probe_row = builder.row(gate_start, gate_start, gate_start)
    produced = to_vector(probe_row, FEATURE_COLUMNS)

    arms = {}
    retained = {}
    for name, columns in ARMS.items():
        fit, audit = finite_training_rows(fit_raw, columns)
        gate_finite, _ = finite_training_rows(gate_raw, columns)
        retained[name] = (audit["retained_rows"], len(gate_finite))

        params = config.get_default_params(ALGORITHM)
        model = CatBoostRegressor(**params)
        model.fit(fit[list(columns)], fit["actual"])

        gate_finite = gate_finite.copy()
        gate_finite["challenger"] = model.predict(gate_finite[list(columns)])
        selected = attach_baselines(select_latest_challenger_per_band(gate_finite),
                                    builder._actuals)

        artifact = None
        if artifact_dir is not None:
            path = save_gate_artifact(
                Path(artifact_dir) / name / country / "solar" / "model.joblib",
                model=model, builder=builder, algorithm=ALGORITHM, params=params,
                feature_columns=columns, fit_window=(fit_start, gate_start))
            bundle = joblib.load(path)
            artifact = {
                "path": str(Path(path).resolve()), "sha256": _sha256(path),
                "declared_feature_columns": list(bundle["feature_columns"]),
                "n_declared": len(bundle["feature_columns"]),
                "training_source": bundle["training_source"],
                "nonneg_objective": bundle.get("nonneg_objective"),
            }

        arms[name] = {
            "n_features_fitted": len(columns),
            "geometry_features_in_fit": [c for c in SOLAR_GEOMETRY_FEATURES if c in columns],
            "fit_rows": audit["retained_rows"], "fit_intended_rows": audit["intended_rows"],
            "degraded_lag_1d_rows": audit["degraded_lag_1d_rows"],
            "night": _night_read(selected, country),
            "cells": _cells(selected, country),
            "artifact": artifact,
        }

    # An arm scored on a different row set is not an A/B. The geometry columns
    # are pure functions of (country, hour) and never NaN, so this must hold.
    assert retained["f25"] == retained["f27"], (
        f"{country}: the two arms retained different rows {retained}; the "
        f"comparison below would confound the feature list with the row set")

    return {
        "country": country,
        "n_features_produced_by_builder": len(produced),
        "geometry_produced_by_builder": {c: round(float(produced[c]), 4)
                                         for c in SOLAR_GEOMETRY_FEATURES},
        "rows_identical_across_arms": True,
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--countries", default="BG,CH")
    parser.add_argument("--replica-db", required=True)
    parser.add_argument("--renewable-source", default="energy_generation",
                        choices=list(db._RENEWABLE_TYPE_SOURCES))
    parser.add_argument("--fit-start", default="2026-01-14")
    parser.add_argument("--gate-start", default="2026-07-11")
    parser.add_argument("--gate-end", default="2026-08-10")
    parser.add_argument("--artifact-dir", default=None,
                        help="Where both arms' artifacts are written. Never a "
                             "registered scope's artifact_dir.")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    replica = Path(args.replica_db).resolve()
    if not replica.exists():
        parser.error(f"replica not found: {replica}")
    fit_start, gate_start, gate_end = map(
        pd.Timestamp, (args.fit_start, args.gate_start, args.gate_end))
    if not fit_start < gate_start < gate_end:
        parser.error("require fit-start < gate-start < gate-end")

    result = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "replica_db": str(replica), "replica_bytes": replica.stat().st_size,
        "renewable_source": args.renewable_source,
        "algorithm": ALGORITHM, "hyperparams": config.get_default_params(ALGORITHM),
        "windows": {"fit": [str(fit_start), str(gate_start)],
                    "gate": [str(gate_start), str(gate_end)]},
        "arms": {name: {"n_features": len(cols), "columns": list(cols)}
                 for name, cols in ARMS.items()},
        "countries": [],
    }
    for country in [c.strip().upper() for c in args.countries.split(",")]:
        result["countries"].append(
            probe(country, str(replica), args.renewable_source, fit_start, gate_start,
                  gate_end, Path(args.artifact_dir) if args.artifact_dir else None))
        print(f"[done] {country}", file=sys.stderr)

    text = json.dumps(result, indent=2, allow_nan=False)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
