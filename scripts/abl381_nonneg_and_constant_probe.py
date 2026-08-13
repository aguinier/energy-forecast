#!/usr/bin/env python3
"""Two post-gate reads for an ABL-316 solar tranche: ABL-338 non-negativity, and
the constant-predictor reference the CEO authorised on ABL-380's evidence.

Both are questions about a gate read that has already happened, so both are
answered here rather than inside `evaluate_solar_retrain.py`. Putting either in
the harness would change what a dispositioned scope re-reads; this script only
reads the artifacts that harness wrote.

**1. Is ABL-338 actually active?** (ABL-381's first solar-specific check.)
ABL-338 makes a solar model non-negative by two mechanisms, and the gate harness
inherits neither by construction:

  * `Forecaster(nonneg_objective=...)` selects a **link** (Tweedie/Poisson) whose
    inverse is positive everywhere. The harness fits a bare `CatBoostRegressor`
    from `config.get_default_params`, which names no `loss_function`, so CatBoost
    uses RMSE -- an identity link, negative-capable by construction.
  * The geometry features `sun_elevation_deg` / `is_night` tell the model when
    the sun is down. `solar_retrain.FEATURE_COLUMNS` names neither.

Whether that *bites* is an empirical question about the fitted models, not a
reading of the config, so this predicts the registered gate window and counts.
Night is `solar_features.night_mask` -- the same definition ABL-338 uses.

**2. What is the constant-predictor reference worth here?**
ABL-380 found the D-7 bar nearly uninformative on BG/CH wind and reported a
constant predictor beside every cell: *causal* at the fit-window mean (available
at forecast time) and *oracle* at the gate-window median (hindsight, and the
best any flat line can do, since the median minimises sum|a-c|).

Reported per **cell** here rather than per pair, which is what the CEO asked for
on ABL-381 and is strictly the harder test: each constant is scored on exactly
the rows that cell scored, so it is directly comparable to that cell's challenger
and D-7 WAPE. The oracle is taken on the cell's own actuals, so it is a true
upper bound on any constant for that cell rather than a whole-window constant
evaluated on a subset.

This is a **reported reference, not a gate criterion**. Registered bands, bars,
windows and the PASS rule are frozen at `experiments/ABL348/config.json` and
nothing here moves them.

Read-only against the replica (`mode=ro`); writes nothing but its own JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import db
from src.evaluation.solar_retrain import (
    FEATURE_COLUMNS, PRIMARY_BANDS, attach_baselines, build_vintage_frame,
    common_scores, finite_training_rows, select_latest_challenger_per_band,
)
from src.solar_features import night_mask
from src.wind_features import RenewableFeatureBuilder

#: The basis the tranche gates on (`GATE_BASIS['abl316-t1b']`). Scoring the
#: constants on the same intersection is what makes them comparable to the cell.
GATE_BASIS = ("challenger", "seasonal_naive")


def _wape(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    denom = np.abs(actual).sum()
    if denom == 0:
        return None
    return float(np.abs(actual - forecast).sum() / denom * 100.0)


def _hourly(country: str, start, end, source: str, replica: str) -> pd.Series:
    frame = db.load_renewable_type_data(country, "solar", str(start), str(end),
                                        source=source, db_path=replica)
    if frame.empty:
        return pd.Series(dtype=float)
    stamps = pd.to_datetime(frame["timestamp_utc"], format="mixed",
                            utc=True).dt.tz_localize(None)
    return pd.Series(frame["target_value"].to_numpy(dtype=float),
                     index=stamps).sort_index()


def probe(country: str, artifact_dir: Path, source: str, replica: str,
          fit_start, gate_start, gate_end) -> dict:
    bundle = joblib.load(artifact_dir / country / "solar" / "model.joblib")
    model = bundle["model"]

    builder = RenewableFeatureBuilder(country, "solar",
                                      fit_start - pd.Timedelta(days=14), gate_end,
                                      actuals_source=source, db_path=replica)
    gate_raw = build_vintage_frame(builder, gate_start, gate_end, FEATURE_COLUMNS)
    gate_finite, _ = finite_training_rows(gate_raw, FEATURE_COLUMNS)
    gate_finite["challenger"] = model.predict(gate_finite[list(FEATURE_COLUMNS)])
    selected = attach_baselines(select_latest_challenger_per_band(gate_finite),
                                builder._actuals)
    selected["country"] = country

    # ---- 1. ABL-338 non-negativity, on the rows the gate actually scored ----
    preds = selected["challenger"].to_numpy(dtype=float)
    targets = pd.to_datetime(selected["target_ts"])
    night = night_mask(country, list(targets))
    negative = preds < 0
    nonneg = {
        "artifact_nonneg_objective": bundle.get("nonneg_objective"),
        "artifact_loss_function": bundle.get("hyperparams", {}).get("loss_function"),
        "geometry_features_in_fit": [f for f in ("sun_elevation_deg", "is_night")
                                     if f in bundle.get("feature_columns", [])],
        "n_selected_rows": int(len(preds)),
        "n_negative": int(negative.sum()),
        "pct_negative": round(float(negative.mean() * 100.0), 3),
        "min_prediction_mw": round(float(preds.min()), 2),
        "sum_negative_mwh": round(float(-preds[negative].sum()), 1) if negative.any() else 0.0,
        "n_night_rows": int(night.sum()),
        "n_negative_at_night": int((negative & night).sum()),
        "pct_of_night_rows_negative": (round(float((negative & night).sum() / night.sum() * 100.0), 2)
                                       if night.sum() else None),
        "mean_prediction_at_night_mw": (round(float(preds[night].mean()), 2)
                                        if night.sum() else None),
        "mean_actual_at_night_mw": (round(float(selected["actual"].to_numpy(dtype=float)[night].mean()), 2)
                                    if night.sum() else None),
    }

    # ---- 2. Constant-predictor reference, per cell ----
    fit_series = _hourly(country, fit_start, gate_start, source, replica)
    causal_constant = float(fit_series.mean())
    cells = []
    for band, group in selected[selected["horizon_band"].isin(PRIMARY_BANDS)].groupby(
            "horizon_band"):
        _, common = common_scores(group, GATE_BASIS)
        actual = common["actual"].to_numpy(dtype=float)
        chal = common["challenger"].to_numpy(dtype=float)
        d7 = common["seasonal_naive"].to_numpy(dtype=float)
        oracle_constant = float(np.median(actual))
        cells.append({
            "horizon_band": band,
            "n": int(len(common)),
            "challenger_wape_pct": round(_wape(actual, chal), 2),
            "d7_wape_pct": round(_wape(actual, d7), 2),
            # Daylight-only sensitivity. WAPE divides by sum|actual|, so any
            # energy booked while the sun is down inflates the denominator and
            # flatters every forecaster scored on it. Where the target has a
            # night floor (BG does; CH does not) the registered cell is measured
            # partly on hours that are not physically solar, so re-scoring the
            # same challenger and D-7 on daylight rows alone says how much of the
            # margin survives. A **reported reference, not a gate criterion** --
            # the registered cell above is unchanged and is what dispositions.
            **_daylight_only(country, common),
            "causal_constant_mw": round(causal_constant, 1),
            "causal_constant_wape_pct": round(_wape(actual, np.full_like(actual, causal_constant)), 2),
            "oracle_constant_mw": round(oracle_constant, 1),
            "oracle_constant_wape_pct": round(_wape(actual, np.full_like(actual, oracle_constant)), 2),
            # The natural solar generalisation of a flat line: a constant *per
            # hour of day*. A flat line cannot represent a diurnal cycle at all,
            # so on solar it is a far weaker reference than it is on wind; the
            # honest analogue is a climatology, and it costs nothing to add.
            **_climatology(common, fit_series),
        })
    return {"country": country, "nonneg": nonneg,
            "fit_window_mean_mw": round(causal_constant, 1),
            "cells": sorted(cells, key=lambda row: row["horizon_band"])}


def _daylight_only(country: str, common: pd.DataFrame) -> dict:
    """Re-score the cell's own rows with the night hours removed.

    Same challenger, same D-7, same registered band -- only the row set changes,
    so the difference isolates what the night floor was contributing to the cell.
    """
    night = night_mask(country, list(pd.to_datetime(common["target_ts"])))
    day = ~night
    if not day.any():
        return {"daylight_n": 0, "daylight_challenger_wape_pct": None,
                "daylight_d7_wape_pct": None}
    actual = common["actual"].to_numpy(dtype=float)[day]
    chal = common["challenger"].to_numpy(dtype=float)[day]
    d7 = common["seasonal_naive"].to_numpy(dtype=float)[day]
    chal_wape, d7_wape = _wape(actual, chal), _wape(actual, d7)
    return {
        "daylight_n": int(day.sum()),
        "daylight_challenger_wape_pct": round(chal_wape, 2) if chal_wape is not None else None,
        "daylight_d7_wape_pct": round(d7_wape, 2) if d7_wape is not None else None,
        "daylight_clears_d7": (None if chal_wape is None or d7_wape is None
                               else bool(chal_wape < d7_wape)),
    }


def _climatology(common: pd.DataFrame, fit_series: pd.Series) -> dict:
    """Hour-of-day climatology, causal (fit window) and oracle (gate rows)."""
    actual = common["actual"].to_numpy(dtype=float)
    hours = pd.to_datetime(common["target_ts"]).dt.hour.to_numpy()

    causal_profile = fit_series.groupby(fit_series.index.hour).mean()
    causal = np.array([causal_profile.get(h, fit_series.mean()) for h in hours])

    frame = pd.DataFrame({"actual": actual, "hour": hours})
    oracle_profile = frame.groupby("hour")["actual"].median()
    oracle = np.array([oracle_profile.get(h, np.median(actual)) for h in hours])
    return {
        "causal_climatology_wape_pct": round(_wape(actual, causal), 2),
        "oracle_climatology_wape_pct": round(_wape(actual, oracle), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--countries", default="BG,CH")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--replica-db", required=True)
    parser.add_argument("--renewable-source", default="energy_generation",
                        choices=list(db._RENEWABLE_TYPE_SOURCES))
    parser.add_argument("--fit-start", default="2026-01-14")
    parser.add_argument("--gate-start", default="2026-07-11")
    parser.add_argument("--gate-end", default="2026-08-10")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    replica = Path(args.replica_db).resolve()
    if not replica.exists():
        parser.error(f"replica not found: {replica}")
    fit_start, gate_start, gate_end = map(
        pd.Timestamp, (args.fit_start, args.gate_start, args.gate_end))

    result = {
        "replica_db": str(replica),
        "replica_bytes": replica.stat().st_size,
        "renewable_source": args.renewable_source,
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
        "windows": {"fit": [str(fit_start), str(gate_start)],
                    "gate": [str(gate_start), str(gate_end)]},
        "countries": [],
    }
    for country in [c.strip().upper() for c in args.countries.split(",")]:
        result["countries"].append(
            probe(country, Path(args.artifact_dir), args.renewable_source,
                  str(replica), fit_start, gate_start, gate_end))

    text = json.dumps(result, indent=2, allow_nan=False)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
