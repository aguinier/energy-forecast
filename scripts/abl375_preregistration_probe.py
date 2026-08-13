"""ABL-375: measure the bar before any model exists on the registered window.

ABL-375 asks for a like-for-like XGBoost-vs-CatBoost read on DE solar, with
ABL-338's solar geometry on both arms. The reason it needs its own registration
rather than a reuse of `experiments/ABL348/config.json` is a *contamination of
knowledge*, not a data problem, and this probe is what makes that statement
checkable: it enumerates every committed ABL-338 holdout window and shows that
the one ABL-375 registers has never been fitted or scored for any arm of this
comparison, while ABL-348's gate window has.

Everything here is a **read**. No model is fitted, and none exists for the
registered window when these numbers are taken -- which is the whole point of
taking them now.

What it measures, per country
-----------------------------
- the incumbent artifact's algorithm, version, hyperparameters and feature count
  (read under `.venv`; an xgboost-3.3.0 pickle opened under 2.1.4 silently
  resets the fitted intercept, so the interpreter is part of the measurement)
- n_train / n_holdout and the daylight / shoulder / night split of the holdout,
  against the registered minimum n
- ABL-188 suspect-constant exclusions and ABL-337 impossible-night rows inside
  the fit and holdout windows, so the contamination statement in the
  registration is measured rather than asserted
- the literal seasonal-naive D-7 bar per band -- the free baseline every arm is
  quoted against, fixed before a challenger exists

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl375_preregistration_probe.py \\
        --out reports/abl_375_probe.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src.db import load_training_data  # noqa: E402
from src.features import create_all_features  # noqa: E402
from src.forecaster import Forecaster  # noqa: E402
from src.solar_features import (  # noqa: E402
    NIGHT_ELEVATION_THRESHOLD_DEG,
    night_mask,
    solar_geometry_frame,
)

logger = logging.getLogger("energy_forecast")

#: `models/` is gitignored so a worktree has none; the live artifacts and the
#: scheduled job both read the primary checkout.
LIVE_MODELS_DIR = Path(r"C:\Code\able\energy-forecast\models")

#: The window ABL-375 registers. It is the gap between ABL-338's two committed
#: holdouts (spring ends 2026-04-29, summer starts 2026-06-13), which is exactly
#: why it is available: no arm of the algorithm comparison has been fitted or
#: scored on it.
REGISTERED_HOLDOUT = ("2026-04-30", "2026-06-12")

#: Windows already fitted and scored under ABL-338 (`5cf2296`). Enumerated here
#: so "the registered window is unread" is a checkable claim rather than a
#: recollection. ABL-348's gate window, 2026-07-11 -> 2026-08-10, lies wholly
#: inside the summer entry.
ABL338_READ_WINDOWS = (
    ("2026-06-13", "2026-08-11", "incumbent algorithm, uncleaned"),
    ("2026-06-13", "2026-08-11", "forced xgboost, uncleaned"),
    ("2026-06-13", "2026-08-11", "catboost diagnostic, DE/FR"),
    ("2026-06-13", "2026-08-11", "incumbent algorithm, cleaned"),
    ("2026-06-13", "2026-08-11", "forced xgboost, cleaned"),
    ("2026-03-01", "2026-04-29", "incumbent algorithm, cleaned"),
    ("2026-03-01", "2026-04-29", "forced xgboost, cleaned"),
)


def _overlaps(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
    return not (pd.Timestamp(a_end) < pd.Timestamp(b_start)
                or pd.Timestamp(b_end) < pd.Timestamp(a_start))


def _bands(country_code: str, timestamps: pd.Series) -> np.ndarray:
    """Night / shoulder / daylight, on the serving clamp's own predicate.

    Identical to `scripts/abl338_solar_holdout.py::_bands` by construction —
    both call `solar_features` — so the bar this probe fixes is on the same
    partition the arms are later scored on.
    """
    elevation = solar_geometry_frame(country_code, timestamps)["sun_elevation_deg"].to_numpy()
    night = night_mask(country_code, timestamps)
    return np.where(night, "night", np.where(elevation <= 0.0, "shoulder", "daylight"))


def _band_bar(actual: np.ndarray, predicted: np.ndarray) -> dict:
    n = int(len(actual))
    if n == 0:
        return {"n": 0}
    error = predicted - actual
    out = {
        "n": n,
        "mean_actual_mw": round(float(actual.mean()), 1),
        "d7_mae_mw": round(float(np.abs(error).mean()), 1),
        "d7_bias_mw": round(float(error.mean()), 1),
    }
    total = float(np.abs(actual).sum())
    # Night's denominator is ~0, so a percentage there says only that the
    # denominator is small. MW only in that band.
    if total > 0 and actual.mean() > 1.0:
        out["d7_wape_pct"] = round(100.0 * float(np.abs(error).sum()) / total, 2)
    return out


def probe_country(country_code: str, start_date: str,
                  holdout_start: str, holdout_end: str) -> dict:
    live_path = LIVE_MODELS_DIR / country_code / "solar" / "model.joblib"
    incumbent = Forecaster.load(country_code, "solar", path=str(live_path))
    training_source = incumbent.training_source

    raw = load_training_data(
        country_code, "solar", start_date,
        (pd.Timestamp(holdout_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        source=training_source,
    )
    if raw.empty:
        raise ValueError(f"No solar training data for {country_code}")

    featured = create_all_features(raw, "solar", country_code=country_code).reset_index(drop=True)
    timestamps = pd.to_datetime(featured["timestamp_utc"])
    band = _bands(country_code, timestamps)

    is_holdout = ((timestamps >= pd.Timestamp(holdout_start))
                  & (timestamps <= pd.Timestamp(holdout_end) + pd.Timedelta(hours=23))).to_numpy()
    train = featured.loc[~is_holdout]
    holdout = featured.loc[is_holdout]
    train_band = band[~is_holdout]
    holdout_band = band[is_holdout]

    night_train = train_band == "night"
    impossible = night_train & (train["target_value"].to_numpy() > 1.0)
    night_holdout = holdout_band == "night"
    impossible_holdout = night_holdout & (holdout["target_value"].to_numpy() > 1.0)

    actual = holdout["target_value"].to_numpy(dtype=float)
    d7 = holdout["target_value_lag_7d"].to_numpy(dtype=float)
    bars = {b: _band_bar(actual[holdout_band == b], d7[holdout_band == b])
            for b in ("daylight", "shoulder", "night")}
    bars["all"] = _band_bar(actual, d7)

    return {
        "country_code": country_code,
        "incumbent": {
            "algorithm": incumbent.algorithm,
            "model_version": incumbent.model_version,
            "training_source": training_source,
            "n_feature_columns": len(incumbent.feature_columns),
            "carries_geometry": bool(
                set(("sun_elevation_deg", "is_night")) & set(incumbent.feature_columns)
            ),
            "hyperparams": {k: v for k, v in incumbent.hyperparams.items()},
            "hyperparams_equal_config_default": (
                dict(incumbent.hyperparams)
                == config.get_default_params(incumbent.algorithm)
            ),
        },
        "featured_frame_start": str(timestamps.min()),
        "featured_frame_end": str(timestamps.max()),
        "n_train": int(len(train)),
        "train_end": str(pd.to_datetime(train["timestamp_utc"]).max()),
        "n_holdout": int(len(holdout)),
        "holdout_bands": {b: int((holdout_band == b).sum())
                          for b in ("daylight", "shoulder", "night")},
        "contamination": {
            "abl337_impossible_night_rows_in_fit": int(impossible.sum()),
            "abl337_impossible_night_max_mw_in_fit": round(
                float(train.loc[night_train, "target_value"].max()) if night_train.any() else 0.0, 1
            ),
            "abl337_impossible_night_rows_in_holdout": int(impossible_holdout.sum()),
            "abl337_impossible_night_max_mw_in_holdout": round(
                float(holdout.loc[night_holdout, "target_value"].max())
                if night_holdout.any() else 0.0, 1
            ),
        },
        "baseline_seasonal_naive_d7": bars,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", default="AT,BE,DE,FR")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--holdout", default=":".join(REGISTERED_HOLDOUT),
                        help="START:END, both YYYY-MM-DD, inclusive")
    parser.add_argument("--out", default="reports/abl_375_probe.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    holdout_start, holdout_end = args.holdout.split(":")
    replica = Path(config.DATABASE_PATH)
    payload = {
        "issue": "ABL-375",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "replica_db": str(replica),
        "replica_bytes": os.path.getsize(replica) if replica.exists() else None,
        "registered_holdout": [holdout_start, holdout_end],
        "night_threshold_deg": NIGHT_ELEVATION_THRESHOLD_DEG,
        "registered_window_is_unread": {
            "abl338_committed_windows": [
                {"start": s, "end": e, "arm_set": label,
                 "overlaps_registered_holdout": _overlaps(s, e, holdout_start, holdout_end)}
                for s, e, label in ABL338_READ_WINDOWS
            ],
            "abl348_gate_window": {
                "start": "2026-07-11", "end": "2026-08-10",
                "overlaps_registered_holdout": _overlaps(
                    "2026-07-11", "2026-08-10", holdout_start, holdout_end),
                "inside_an_abl338_read_window": True,
            },
        },
        "countries": {},
    }
    for country in [c.strip().upper() for c in args.countries.split(",") if c.strip()]:
        payload["countries"][country] = probe_country(
            country, args.start, holdout_start, holdout_end)
        r = payload["countries"][country]
        logger.info(
            f"{country}: incumbent {r['incumbent']['algorithm']} "
            f"({r['incumbent']['n_feature_columns']} features, geometry "
            f"{r['incumbent']['carries_geometry']}), n_train {r['n_train']:,}, "
            f"n_holdout {r['n_holdout']:,} "
            f"(daylight {r['holdout_bands']['daylight']:,}), D-7 daylight MAE "
            f"{r['baseline_seasonal_naive_d7']['daylight']['d7_mae_mw']:,.1f} MW"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
