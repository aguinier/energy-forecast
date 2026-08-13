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

**2. What are the model-free references worth here, per cell?**
ABL-380 found the D-7 bar nearly uninformative on BG/CH wind and reported a
constant predictor beside every cell. This probe originally computed its own
constant *and* its own hour-of-day climatology. It no longer computes either:
ABL-389 (PR #39) put all four references in
`src/evaluation/model_free_reference.py` so the two gate harnesses cannot drift
into computing the same named reference differently, and a third implementation
here would defeat that at one remove. **The levels are the canonical module's,
attached by `attach_model_free_references` from the same ABL-188-filtered
`builder._actuals` the harness scores.** What is left here is the per-cell
*scoring* of those levels, which the harness reports per band already; this file
keeps it only so the daylight-only re-score below sits beside a comparable
reference on the same rows.

That swap is not free, and the direction matters. The deleted local version took
its **oracle** levels on each cell's own rows; the canonical module takes one
level set per pair over the whole gate window and broadcasts it. The two agree
exactly wherever a cell covers the full window and disagree wherever it does not
-- so the 24-36h and 36-48h cells (n=720) are unchanged to the decimal and the
48-64h cells (n=510) move. The canonical number is the weaker, more honest
bound: a single level a forecaster could have picked once, not one re-optimised
per band with hindsight. See `reports/abl_381_tranche1b_findings.md`.

Each reference is scored on **its own intersection** with the gate basis via
`scored_with_comparators`, and each carries its own `n`. A climatology is 24
levels, so unlike every other comparator it can be partially measurable; read
that `n` before setting its WAPE beside a challenger's. Both pairs here cover
24/24 hours in both windows, so no row drops -- that is a property of BG and CH,
not a guarantee for the remaining 33.

These are **reported references, not gate criteria**. Registered bands, bars,
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import db
from src.evaluation.model_free_reference import (
    MODEL_FREE_COMPARATORS, attach_model_free_references, comparator_wape,
)
from src.evaluation.scorecard import score_predictions
from src.evaluation.solar_retrain import (
    FEATURE_COLUMNS, PRIMARY_BANDS, attach_baselines, build_vintage_frame,
    finite_training_rows, scored_with_comparators,
    select_latest_challenger_per_band,
)
from src.solar_features import night_mask
from src.wind_features import RenewableFeatureBuilder

#: The basis the tranche gates on (`GATE_BASIS['abl316-t1b']`). Scoring the
#: references on the same intersection is what makes them comparable to the cell.
GATE_BASIS = ("challenger", "seasonal_naive")

#: The basis plus ABL-389's four model-free references. Same construction as the
#: harness's `REPORTED_COMPARATORS`, so both score the same columns the same way.
REPORTED_COMPARATORS = (*GATE_BASIS, *MODEL_FREE_COMPARATORS)


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
    # ABL-389's four references, from the canonical module and from the same
    # ABL-188-filtered series the harness and the baselines use -- not a second
    # read of the replica, and not a second implementation.
    selected, reference_levels = attach_model_free_references(
        selected, builder._actuals, fit_start, gate_start, gate_end)
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

    # ---- 2. ABL-389's model-free references, scored per cell ----
    cells = []
    for band, group in selected[selected["horizon_band"].isin(PRIMARY_BANDS)].groupby(
            "horizon_band"):
        scores, common, comparator_n = scored_with_comparators(
            group, GATE_BASIS, REPORTED_COMPARATORS)
        cells.append({
            "horizon_band": band,
            "n": int(len(common)),
            "challenger_wape_pct": round(scores["challenger"]["wape_pct"], 2),
            "d7_wape_pct": round(scores["seasonal_naive"]["wape_pct"], 2),
            # Daylight-only sensitivity. WAPE divides by sum|actual|, so any
            # energy booked while the sun is down inflates the denominator and
            # flatters every forecaster scored on it. Where the target has a
            # night floor (BG does; CH does not) the registered cell is measured
            # partly on hours that are not physically solar, so re-scoring the
            # same challenger and D-7 on daylight rows alone says how much of the
            # margin survives. A **reported reference, not a gate criterion** --
            # the registered cell above is unchanged and is what dispositions.
            **_daylight_only(country, common),
            # Each reference carries its own n, because each is scored on its own
            # intersection with the basis. Equal to the cell n on both these
            # pairs; a pair missing an hour of day would show a lower one on the
            # climatology columns alone, and its WAPE would then not be
            # comparable to the challenger's.
            **_references(scores, comparator_n),
        })
    return {"country": country, "nonneg": nonneg,
            "model_free_reference_mw": reference_levels,
            "cells": sorted(cells, key=lambda row: row["horizon_band"])}


def _references(scores: dict, comparator_n: dict) -> dict:
    """The four ABL-389 references for one cell, each with its own n.

    Named exactly as the module names them, so a number in this file and the
    same-named number in the gate report are the same measurement or a bug.
    """
    out = {}
    for name in MODEL_FREE_COMPARATORS:
        wape = comparator_wape(scores, name)
        out[f"{name}_wape_pct"] = None if wape is None else round(wape, 2)
        out[f"{name}_n"] = int(comparator_n.get(name, 0))
    return out


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
    chal_wape = score_predictions(
        actual, common["challenger"].to_numpy(dtype=float)[day])["wape_pct"]
    d7_wape = score_predictions(
        actual, common["seasonal_naive"].to_numpy(dtype=float)[day])["wape_pct"]
    return {
        "daylight_n": int(day.sum()),
        "daylight_challenger_wape_pct": round(chal_wape, 2) if chal_wape is not None else None,
        "daylight_d7_wape_pct": round(d7_wape, 2) if d7_wape is not None else None,
        "daylight_clears_d7": (None if chal_wape is None or d7_wape is None
                               else bool(chal_wape < d7_wape)),
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
