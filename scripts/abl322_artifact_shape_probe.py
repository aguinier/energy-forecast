"""ABL-322 — does the ABL-195 gate harness's artifact shape survive
`Forecaster.load`, and which table does it resolve to?

Background: `scripts/evaluate_wind_retrain.py:186-191` does not save through
`Forecaster.save`; it writes a bare `joblib.dump` of seven keys. Post-ABL-331,
`Forecaster.load` resolves an absent `training_source` to
`LEGACY_RENEWABLE_TRAINING_SOURCE` ('energy_renewable') — right for the 88
legacy artifacts, wrong for a pair fitted on `energy_generation`.

This probe builds both artifact shapes and round-trips them, to establish
whether that mismatch fails loudly or silently. It reads no database, trains on
synthetic data, and writes only into a temporary directory.

Run on the rail interpreter:

    .venv\\Scripts\\python.exe scripts/abl322_artifact_shape_probe.py

Exit code 1 if the two shapes disagree on the resolved source (the finding).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.db import (  # noqa: E402
    LEGACY_RENEWABLE_TRAINING_SOURCE,
    RENEWABLE_TYPE_SOURCE_TABLE,
)
from src.forecaster import Forecaster  # noqa: E402

COUNTRY, FORECAST_TYPE = "DE", "wind_offshore"
FEATURES = ["f0", "f1"]

# Mirrors scripts/evaluate_wind_retrain.py:188-191 exactly.
HARNESS_KEYS = ("model", "feature_columns", "country_code", "forecast_type",
                "algorithm", "params", "fit_window")


def _fitted_model() -> XGBRegressor:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 2))
    y = 1000.0 + 300.0 * X[:, 0]
    model = XGBRegressor(n_estimators=5, max_depth=2, random_state=42, verbosity=0)
    model.fit(X, y)
    return model


def main() -> int:
    import xgboost

    out = {
        "interpreter": sys.version.split()[0],
        "xgboost_version": xgboost.__version__,
        "LEGACY_RENEWABLE_TRAINING_SOURCE": LEGACY_RENEWABLE_TRAINING_SOURCE,
        "RENEWABLE_TYPE_SOURCE_TABLE": RENEWABLE_TYPE_SOURCE_TABLE,
        "forecast_type_is_renewable": FORECAST_TYPE in config.RENEWABLE_TYPES,
    }
    model = _fitted_model()

    with tempfile.TemporaryDirectory(prefix="abl322_artifact_") as tmp:
        tmpdir = Path(tmp)

        path_a = tmpdir / "harness_shape.joblib"
        joblib.dump(dict(zip(HARNESS_KEYS, (
            model, FEATURES, COUNTRY, FORECAST_TYPE, "xgboost",
            {"n_estimators": 5}, ["2026-01-14", "2026-07-11"],
        ))), path_a)

        saver = Forecaster(COUNTRY, FORECAST_TYPE, algorithm="xgboost",
                           training_source="energy_generation")
        saver.model = model
        saver.feature_columns = FEATURES
        path_b = tmpdir / "forecaster_save.joblib"
        saver.save(str(path_b))

        for label, path in (("A_harness_bare_joblib", path_a),
                            ("B_forecaster_save", path_b)):
            raw = joblib.load(path)
            entry = {
                "keys_on_disk": sorted(raw),
                "has_training_source": "training_source" in raw,
                "has_intercept_witness": "base_score" in raw and "xgboost_version" in raw,
            }
            try:
                loaded = Forecaster.load(COUNTRY, FORECAST_TYPE, path=str(path))
                entry["load"] = "OK"
                entry["resolved_training_source"] = loaded._resolved_training_source()
            except Exception as exc:  # noqa: BLE001
                entry["load"] = f"RAISED {type(exc).__name__}: {exc}"
            out[label] = entry

    a, b = out["A_harness_bare_joblib"], out["B_forecaster_save"]
    skew = a.get("resolved_training_source") != b.get("resolved_training_source")
    out["train_serve_skew_present"] = skew
    out["verdict"] = (
        "The harness shape loads without error and resolves to "
        f"{a.get('resolved_training_source')!r}, while Forecaster.save resolves to "
        f"{b.get('resolved_training_source')!r}. A pair fitted on energy_generation "
        "and written by the harness would be served from the wrong table, silently."
    ) if skew else "No skew: both shapes resolve to the same source."

    print(json.dumps(out, indent=2, default=str))
    return 1 if skew else 0


if __name__ == "__main__":
    raise SystemExit(main())
