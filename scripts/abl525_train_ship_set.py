#!/usr/bin/env python
"""Fit production artifacts for the ABL-316 ship set approved by the Board (ABL-525).

The Board answered `abl316:ship-decision:v4` with `ship8` on 2026-08-22. This
script fits the artifacts that answer authorises, through the *graded* code path
and nothing else.

WHY NOT `scripts/train.py`
--------------------------
ABL-525 item 1 names `scripts/train.py`; item 2 requires the ABL-183 / ABL-191
serve-faithful builders and the feature set the tranches graded. Those are two
different pipelines and only one of them can be served:

    train.py     -> features.create_all_features + select_feature_columns
                    = 28 names (wind) / 31 (solar), including 4 holiday columns
    gate harness -> wind_features.RenewableFeatureBuilder + to_vector
                    = 24 names (wind) / 27 (solar), no holiday columns

`Forecaster.predict_d2` routes wind_onshore/wind_offshore/solar to
`_predict_d2_serve_faithful`, which calls `to_vector(row, artifact.feature_columns)`,
and `to_vector` raises `KeyError` on a column the builder does not produce. The
builder produces no holiday column (`wind_features.py:179`). So a `train.py`
artifact for any of these pairs loads clean and then raises on its first serving
row -- `forecast_daily.py` books a failed result and the pair serves zero rows.
This script therefore fits the way the tranches did: same builder, same
FEATURE_COLUMNS, same algorithm, written through `save_gate_artifact` so
`training_source` (ABL-331/ABL-342) and the ABL-183 intercept witness are
derived from the fit rather than claimed by the caller.

THE FIT WINDOW IS BOUNDED BY WEATHER, NOT BY ACTUALS
----------------------------------------------------
Item 1 asks for full available history. `energy_generation` reaches back to
2021-01-01 for all eight pairs, but a serve-faithful row also needs the weather
*forecast* archive, and `weather_data` with `data_quality='forecast'` begins
2026-01-11 for every one of these countries. An earlier target gets NaN weather
and `finite_training_rows` drops it. So the widest honest window is
2026-01-11 -> 2026-08-22 (223 days) against the gate's registered 178 days, and
the run records what was actually retained rather than what was requested.

Because that window covers ABL-348's gate window, these artifacts have been
fitted on the rows the tranches scored. That is what item 1 asks for and is
correct for production, but it means the tranche gate figures are NOT
out-of-sample for these artifacts. This script scores nothing and grades nothing.

CH SOLAR IS NOT FITTED HERE
---------------------------
ABL-395 moved the solar gate list from 25 names to 27 (adding ABL-338's
`sun_elevation_deg` and `is_night`). CH solar was graded under tranche 1b, and
`evaluate_solar_retrain.SCOPE_FEATURES['abl316-t1b']` is pinned to the legacy 25
for exactly that reason. So the current builder's solar feature set HAS moved
since the tranche read, which is the condition ABL-525 item 2 says to stop and
comment on rather than ship a model nobody graded. CH is carried in the
registration table below with `hold` and a reason, so the record states its
absence instead of silently omitting it.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import config
from src.evaluation.gate_artifacts import save_gate_artifact
from src.evaluation.solar_retrain import FEATURE_COLUMNS as SOLAR_FEATURE_COLUMNS
from src.evaluation.wind_retrain import (
    FEATURE_COLUMNS as WIND_FEATURE_COLUMNS,
    build_vintage_frame,
    finite_training_rows,
)
from src.wind_features import RenewableFeatureBuilder

# The 25-name solar list ABL-395 superseded. Imported rather than restated so
# this file cannot come to disagree with the registration table about what
# tranche 1b was graded on; `scripts/abl402_bg_ch_seed_cv.py` imports it the
# same way.
from scripts.evaluate_solar_retrain import (  # noqa: E402
    LEGACY_FEATURE_COLUMNS as SOLAR_LEGACY_FEATURE_COLUMNS,
)

#: The eight pairs the Board approved, with the tranche each was graded under.
#: Figures live in `reports/abl_444_g23_floor_reread.json` (blob 1e8f37f6, sha256
#: 45fa753f...); nothing here re-derives them. `hold` marks a pair this run
#: refuses to fit, with the reason, so the committed record carries the absence.
SHIP_SET = (
    {"country": "EE", "forecast_type": "wind_onshore", "tranche": "2e", "hold": None},
    {"country": "GR", "forecast_type": "wind_onshore", "tranche": "2b", "hold": None},
    {"country": "SE", "forecast_type": "wind_onshore", "tranche": "2b", "hold": None},
    {"country": "BG", "forecast_type": "wind_onshore", "tranche": "2f", "hold": None},
    {"country": "CZ", "forecast_type": "wind_onshore", "tranche": "2e", "hold": None},
    {"country": "FI", "forecast_type": "wind_onshore", "tranche": "2b", "hold": None},
    {"country": "LT", "forecast_type": "wind_onshore", "tranche": "2e", "hold": None},
    {
        "country": "CH",
        "forecast_type": "solar",
        "tranche": "1b",
        # Pinned to the list tranche 1b was actually graded on, so that a
        # decision to ship CH is a `--include-held` run rather than an edit
        # here. `evaluate_solar_retrain.SCOPE_FEATURES['abl316-t1b']` resolves
        # to the same tuple; this row is the same pin, stated where the fit is.
        "feature_columns": SOLAR_LEGACY_FEATURE_COLUMNS,
        "hold": (
            "ABL-525 item 2. Graded under abl316-t1b at the legacy 25-name solar "
            "list; ABL-395 moved solar.FEATURE_COLUMNS to 27 (adds "
            "sun_elevation_deg, is_night) and SCOPE_FEATURES['abl316-t1b'] pins "
            "the tranche to the legacy 25. Fitting at 27 ships a model nobody "
            "graded; fitting at 25 ships the class ABL-395 superseded for a "
            "measured night defect. Membership is the CEO's call."
        ),
    },
)

#: The source table every one of these pairs was graded on (ABL-321/ABL-348).
RENEWABLE_SOURCE = "energy_generation"

#: Widest window the serve-faithful builder can actually populate; see the module
#: docstring. End is exclusive, as `build_vintage_frame` uses a left-closed range.
FIT_START = "2026-01-11"
FIT_END = "2026-08-22"

#: The builder needs actuals before `FIT_START` for the 14-day point lag and the
#: 168-hour rolling anchors. Same value the gate harnesses use.
LOOKBACK_DAYS = 14

FEATURE_COLUMNS_BY_TYPE = {
    "wind_onshore": WIND_FEATURE_COLUMNS,
    "wind_offshore": WIND_FEATURE_COLUMNS,
    "solar": SOLAR_FEATURE_COLUMNS,
}

#: All three tranches behind this ship set fitted catboost.
ALGORITHM = "catboost"


def columns_for(country, forecast_type):
    """The feature list this pair was graded on.

    Defaults to the type's current gate list, but a SHIP_SET row may pin its
    own -- which is not a per-country fork of the *builder*, the thing ABL-525
    item 2 forbids. It is the opposite: a pin to the list that pair's tranche
    was read on, so the artifact matches the approval instead of drifting with
    a constant that moved afterwards.
    """
    for entry in SHIP_SET:
        if entry["country"] == country and entry["forecast_type"] == forecast_type:
            pinned = entry.get("feature_columns")
            if pinned:
                return tuple(pinned)
            break
    return FEATURE_COLUMNS_BY_TYPE[forecast_type]


def build_model(algorithm):
    """The estimator and the exact params it is fitted with.

    Mirrors `evaluate_wind_retrain._model`: the production xgboost defaults carry
    an early-stopping setting that needs a validation set, and this fit uses every
    row with no tuning, so it is removed rather than left to fail.
    """
    params = config.get_default_params(algorithm)
    if algorithm == "xgboost":
        params.pop("early_stopping_rounds", None)
        return XGBRegressor(**params), params
    return CatBoostRegressor(**params), params


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_one(country, forecast_type, replica_db, models_dir, algorithm=ALGORITHM):
    """Fit one pair and write its artifact. Returns the provenance record."""
    columns = columns_for(country, forecast_type)
    fit_start = pd.Timestamp(FIT_START)
    fit_end = pd.Timestamp(FIT_END)

    t0 = time.perf_counter()
    builder = RenewableFeatureBuilder(
        country,
        forecast_type,
        fit_start - pd.Timedelta(days=LOOKBACK_DAYS),
        fit_end,
        actuals_source=RENEWABLE_SOURCE,
        db_path=str(replica_db),
    )
    frame = build_vintage_frame(builder, fit_start, fit_end, feature_columns=columns)
    fit, audit = finite_training_rows(frame, feature_columns=columns)
    t_build = time.perf_counter() - t0

    if fit.empty:
        raise RuntimeError(
            f"{country}/{forecast_type}: no finite training rows in "
            f"{FIT_START}..{FIT_END} -- refusing to write an artifact"
        )

    model, params = build_model(algorithm)
    t0 = time.perf_counter()
    model.fit(fit[list(columns)], fit["actual"])
    t_fit = time.perf_counter() - t0

    path = save_gate_artifact(
        Path(models_dir) / country / forecast_type / "model.joblib",
        model=model,
        builder=builder,
        algorithm=algorithm,
        params=params,
        feature_columns=columns,
        fit_window=(fit_start, fit_end),
    )

    retained_targets = pd.DatetimeIndex(fit["target_ts"])
    predictions = np.asarray(model.predict(fit[list(columns)]), dtype=float)

    return {
        "country": country,
        "forecast_type": forecast_type,
        "algorithm": algorithm,
        "training_source": builder.actuals_source,
        "fit_window_requested": [FIT_START, FIT_END],
        "fit_window_retained": [
            str(retained_targets.min()),
            str(retained_targets.max()),
        ],
        "n_features": len(columns),
        "feature_columns": list(columns),
        "intended_rows": audit["intended_rows"],
        "retained_rows": audit["retained_rows"],
        "excluded_missing_actual_or_feature": audit["excluded_missing_actual_or_feature"],
        "unique_fit_targets": audit["unique_targets"],
        "degraded_lag_1d_rows": audit["degraded_lag_1d_rows"],
        "hyperparams": params,
        "artifact_path": str(path),
        "artifact_sha256": sha256_of(path),
        "seconds_feature_build": round(t_build, 2),
        "seconds_fit": round(t_fit, 2),
        # ABL-525 item 7: an artifact sha256 cannot witness a refit, because
        # `Forecaster.save` stamps `saved_at`. Predictions can.
        "in_sample_prediction_digest": hashlib.sha256(
            predictions.tobytes()
        ).hexdigest(),
        "in_sample_prediction_mean": float(predictions.mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit production artifacts for the Board-approved ABL-316 ship set "
            "through the graded gate-harness path (ABL-525)."
        )
    )
    parser.add_argument(
        "--replica-db",
        default=config.DATABASE_PATH,
        help="Read-only replica to fit from (default: ENERGY_DB_PATH).",
    )
    parser.add_argument(
        "--models-dir",
        default=str(config.MODELS_DIR),
        help="Artifact root; a pair lands at <root>/<country>/<type>/model.joblib.",
    )
    parser.add_argument(
        "--json-out",
        default="reports/abl_525_ship_set_training.json",
        help=(
            "Committed machine record. Not an experiments/*/results.json path -- "
            "that glob is gitignored (the ABL-440 trap)."
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated COUNTRY/TYPE to restrict the run, e.g. CZ/wind_onshore.",
    )
    parser.add_argument(
        "--include-held",
        action="store_true",
        help="Fit pairs marked hold in SHIP_SET. Off by default and deliberately.",
    )
    args = parser.parse_args()

    replica = Path(args.replica_db)
    if not replica.is_file():
        raise SystemExit(f"replica not found: {replica}")

    only = None
    if args.only:
        only = {item.strip() for item in args.only.split(",") if item.strip()}

    records, held = [], []
    for entry in SHIP_SET:
        key = f"{entry['country']}/{entry['forecast_type']}"
        if only is not None and key not in only:
            continue
        if entry["hold"] and not args.include_held:
            held.append({**entry, "status": "held"})
            print(f"[HOLD] {key}: {entry['hold']}")
            continue
        print(f"[FIT ] {key} ({entry['tranche']}) ...", flush=True)
        record = fit_one(
            entry["country"], entry["forecast_type"], replica, args.models_dir
        )
        record["tranche"] = entry["tranche"]
        records.append(record)
        print(
            f"[OK  ] {key}: {record['retained_rows']}/{record['intended_rows']} rows, "
            f"{record['unique_fit_targets']} targets, {record['n_features']} features, "
            f"build {record['seconds_feature_build']}s fit {record['seconds_fit']}s",
            flush=True,
        )

    payload = {
        "issue": "ABL-525",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "board_decision": "abl316:ship-decision:v4 = ship8, 2026-08-22T08:25Z",
        "evidence_of_record": {
            "path": "reports/abl_444_g23_floor_reread.json",
            "blob": "1e8f37f6b1c4befd2c938363306e664ea58a21e7",
            "sha256_prefix": "45fa753fc356b123",
            "note": "Verified byte-unchanged since the a0b9ffd pin. Not re-derived here.",
        },
        "protocol": {
            "fit_path": "src.wind_features.RenewableFeatureBuilder + "
                        "src.evaluation.wind_retrain.build_vintage_frame",
            "artifact_writer": "src.evaluation.gate_artifacts.save_gate_artifact "
                               "(-> Forecaster.save)",
            "renewable_source": RENEWABLE_SOURCE,
            "vintages_per_target": 8,
            "lookback_days": LOOKBACK_DAYS,
            "algorithm": ALGORITHM,
            "scored_or_graded": False,
            "fitted_on_the_gate_window": True,
            "fit_window_bounded_by": (
                "weather_data data_quality='forecast' begins 2026-01-11 for all "
                "eight countries; energy_generation reaches 2021-01-01 but a "
                "serve-faithful row cannot be built without the weather archive"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "replica_db": str(replica),
            "models_dir": str(args.models_dir),
        },
        "pairs": records,
        "held": held,
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out} ({len(records)} fitted, {len(held)} held)")


if __name__ == "__main__":
    main()
