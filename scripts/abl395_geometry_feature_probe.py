#!/usr/bin/env python3
"""ABL-395 A/B: what the two ABL-338 geometry features change in a gate fit.

ABL-381 measured that the solar gate harness fits at 25 features where an
ABL-338-current fit is 27, and that CH consequently predicts negative in 80.5%
of night hours. ABL-395 adds `sun_elevation_deg` / `is_night` to
`solar_retrain.FEATURE_COLUMNS`. This script measures the difference.

Protocol
--------
Both arms are fitted from **one** vintage frame per country, built at the 27-name
superset, so the only thing that differs between them is the column list handed
to CatBoost:

  * arm `f25` - the 25 names the harness declared through ABL-381 (the control,
    and the arm to compare against the published ABL-381 read).
  * arm `f27` - those 25 plus the geometry pair (the treatment).

The geometry features are pure functions of (country, hour) and are never NaN,
so `finite_training_rows` retains the same rows for both arms; the script
asserts that rather than assuming it, because an arm scored on a different row
set is not an A/B.

Windows, source, schedule, bands and algorithm are ABL-348's registered ones, so
the `f25` arm at seed 42 is a like-for-like refit of the published read and
should reproduce its numbers. That reproduction is reported as its own block and
is the check that makes the rest of the run worth reading.

Two reads, because one seed is not a measurement
------------------------------------------------
ABL-385 measured the fleet seed spread and put a number on it: the minimum
readable relative gap between two solar fits is **15% at one seed**, 8.7% at
three. The WAPE movements this A/B produces are around 5% relative, so a
one-seed quote of them would be reporting noise. The script therefore runs:

  * `reproduction` - seed 42 alone, the gate's own seed, against ABL-381.
  * `sweep` - ABL-376's eight registered seeds, which are deliberately disjoint
    from 42, with the comparison taken **paired within each seed** (both arms saw
    the same rows at the same seed, so across-seed variance cancels inside the
    difference) and an **unpaired null** beside it: every control-vs-control seed
    pair, which is what a single-seed gap looks like when nothing changed at all.

What this is not
----------------
**Not a gate re-read, and it dispositions nothing.** It writes no report under
any registered scope's `report_out`/`json_out`, and its artifacts go wherever
`--artifact-dir` points - never into a registered `artifact_dir`. ABL-381's
PASS 6/6 stands as read, on the artifacts it was read on.

Read-only against the replica (`mode=ro`); writes only its own JSON and, if
asked, its own artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
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

logger = logging.getLogger("abl395.geometry_probe")

#: The control arm: exactly what `solar_retrain.FEATURE_COLUMNS` was through
#: ABL-381. Derived by subtraction rather than copied, so it cannot drift from
#: the live list in any way other than the geometry pair itself.
FEATURE_COLUMNS_25 = tuple(c for c in FEATURE_COLUMNS if c not in SOLAR_GEOMETRY_FEATURES)

ARMS = {"f25": FEATURE_COLUMNS_25, "f27": FEATURE_COLUMNS}

#: The basis ABL-381's scope gates on. BG and CH have zero rows in `forecasts`,
#: so naming `incumbent` would empty every intersection (ABL-322/ABL-378).
GATE_BASIS = ("challenger", "seasonal_naive")

#: The gate's own seed. Reported on its own, never pooled into the spread below:
#: a spread anchored on the seed that produced a headline is not a spread.
GATE_SEED = 42

#: ABL-376's registered seed set, reused verbatim rather than freshly invented.
#: They were frozen before that issue's first fit and are disjoint from 42, so
#: nothing here was selected on them; reusing them also makes the two solar seed
#: reads commensurable. `tests/test_gate_feature_list_contract.py` pins the two
#: tuples as equal so they cannot drift apart.
SWEEP_SEEDS = (101, 103, 107, 109, 113, 127, 131, 137)


def _wape(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    denom = np.abs(actual).sum()
    return None if denom == 0 else float(np.abs(actual - forecast).sum() / denom * 100.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _night_read(selected: pd.DataFrame, night: np.ndarray) -> dict:
    """The ABL-381 measurement, on the rows the gate would score."""
    preds = selected["challenger"].to_numpy(dtype=float)
    actual = selected["actual"].to_numpy(dtype=float)
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


def _paired(runs: list[dict], read: callable, label: str) -> dict:
    """Treatment - control at each seed, then summarised, with its own null.

    Pairing is the point: both arms saw the same rows at the same seed, so the
    across-seed variance that swamps an unpaired read cancels inside each
    difference. The null is every control-vs-control pair - what a *one-seed*
    gap looks like when nothing changed at all. Same shape as ABL-376 section 5,
    deliberately, so the two solar seed reads are read the same way.
    """
    seeds = sorted({run["seed"] for run in runs})
    by_arm = {(run["arm"], run["seed"]): read(run) for run in runs}
    control = np.array([by_arm[("f25", s)] for s in seeds], dtype=float)
    treatment = np.array([by_arm[("f27", s)] for s in seeds], dtype=float)
    difference = treatment - control
    scale = float(np.mean(np.abs(control))) or 1.0
    null = np.array([abs(a - b) for a, b in combinations(control, 2)], dtype=float)
    return {
        "metric": label, "seeds": seeds,
        "control": [float(v) for v in control],
        "treatment": [float(v) for v in treatment],
        "control_mean": float(control.mean()),
        "control_sd": float(control.std(ddof=1)) if len(control) > 1 else None,
        "treatment_mean": float(treatment.mean()),
        "treatment_sd": float(treatment.std(ddof=1)) if len(treatment) > 1 else None,
        "paired_mean": float(difference.mean()),
        "paired_mean_pct": 100.0 * float(difference.mean()) / scale,
        "paired_sd": float(difference.std(ddof=1)) if len(difference) > 1 else None,
        "seeds_down": int((difference < 0).sum()), "n_seeds": len(seeds),
        "null_max": float(null.max()) if null.size else None,
        "null_max_pct": 100.0 * float(null.max()) / scale if null.size else None,
        "null_pairs": int(null.size),
        # The reading rule, stated in the record rather than left to a reader:
        # an effect no larger than the largest gap between two identical-arm
        # fits is not distinguishable from seed noise in this design.
        "outside_the_null": (bool(abs(float(difference.mean())) > float(null.max()))
                             if null.size else None),
    }


def _summarise(runs: list[dict]) -> dict:
    """The axes worth a spread: the night defect, and what it costs the bands."""
    summary = {
        "night_pct_negative": _paired(
            runs, lambda r: r["night"]["pct_of_night_rows_negative"],
            "night rows predicted negative (%)"),
        "night_mean_pred_mw": _paired(
            runs, lambda r: r["night"]["mean_prediction_at_night_mw"],
            "mean prediction at night (MW)"),
    }
    for band in PRIMARY_BANDS:
        def _cell(run, band=band):
            return next(c["challenger_wape_pct"] for c in run["cells"]
                        if c["horizon_band"] == band)

        def _day(run, band=band):
            return next(c["daylight_challenger_wape_pct"] for c in run["cells"]
                        if c["horizon_band"] == band)

        summary[f"wape_{band}"] = _paired(runs, _cell, f"challenger WAPE {band} (%)")
        summary[f"daylight_wape_{band}"] = _paired(
            runs, _day, f"daylight challenger WAPE {band} (%)")
    return summary


def probe(country: str, replica: str, source: str, fit_start, gate_start, gate_end,
          seeds: tuple, artifact_dir: Path | None) -> dict:
    builder = RenewableFeatureBuilder(country, "solar", fit_start - pd.Timedelta(days=14),
                                      gate_end, actuals_source=source, db_path=replica)

    started = time.monotonic()
    # Built once, at the superset, and shared by every fit.
    fit_raw = build_vintage_frame(builder, fit_start, gate_start, FEATURE_COLUMNS)
    gate_raw = build_vintage_frame(builder, gate_start, gate_end, FEATURE_COLUMNS)

    # What the builder actually produces for one row, independent of any list:
    # the "produces 27" half of the evidence. `to_vector` raises on a name it
    # cannot build, so this reaching 27 is a positive result, not an absence.
    produced = to_vector(builder.row(gate_start, gate_start, gate_start), FEATURE_COLUMNS)

    frames, gates, retained = {}, {}, {}
    for name, columns in ARMS.items():
        fit, audit = finite_training_rows(fit_raw, columns)
        gate_finite, _ = finite_training_rows(gate_raw, columns)
        frames[name] = (fit, audit)
        gates[name] = gate_finite
        retained[name] = (audit["retained_rows"], len(gate_finite))

    # An arm scored on a different row set is not an A/B. The geometry columns
    # are pure functions of (country, hour) and never NaN, so this must hold.
    assert retained["f25"] == retained["f27"], (
        f"{country}: the two arms retained different rows {retained}; the "
        f"comparison would confound the feature list with the row set")
    logger.info("%s: %d fit rows, %d gate rows in %.1f min", country,
                retained["f25"][0], retained["f25"][1], (time.monotonic() - started) / 60)

    night_by_arm = {}
    runs = []
    for seed in seeds:
        for name, columns in ARMS.items():
            began = time.monotonic()
            params = dict(config.get_default_params(ALGORITHM))
            params["random_seed"] = seed
            model = CatBoostRegressor(**params)
            fit, audit = frames[name]
            model.fit(fit[list(columns)], fit["actual"])

            scored = gates[name].copy()
            scored["challenger"] = model.predict(scored[list(columns)])
            selected = attach_baselines(select_latest_challenger_per_band(scored),
                                        builder._actuals)
            # Sun geometry does not depend on the seed or the arm, so the mask is
            # computed once per (country, selection) and reused: it is the single
            # slowest per-row call in this loop.
            key = len(selected)
            if key not in night_by_arm:
                night_by_arm[key] = night_mask(
                    country, list(pd.to_datetime(selected["target_ts"])))
            night = night_by_arm[key]

            artifact = None
            if artifact_dir is not None and seed == GATE_SEED:
                path = save_gate_artifact(
                    Path(artifact_dir) / name / country / "solar" / "model.joblib",
                    model=model, builder=builder, algorithm=ALGORITHM, params=params,
                    feature_columns=columns, fit_window=(fit_start, gate_start))
                bundle = joblib.load(path)
                artifact = {
                    "path": str(Path(path).resolve()), "sha256": _sha256(path),
                    "n_declared": len(bundle["feature_columns"]),
                    "declared_feature_columns": list(bundle["feature_columns"]),
                    "training_source": bundle["training_source"],
                    "nonneg_objective": bundle.get("nonneg_objective"),
                }

            runs.append({
                "arm": name, "seed": seed, "n_features_fitted": len(columns),
                "geometry_features_in_fit": [c for c in SOLAR_GEOMETRY_FEATURES
                                             if c in columns],
                "fit_rows": audit["retained_rows"],
                "fit_seconds": round(time.monotonic() - began, 1),
                "night": _night_read(selected, night),
                "cells": _cells(selected, country),
                "artifact": artifact,
            })
            logger.info("%s seed=%d arm=%s night-negative %.2f%% (%.0fs)", country, seed,
                        name, runs[-1]["night"]["pct_of_night_rows_negative"],
                        runs[-1]["fit_seconds"])

    reproduction = [r for r in runs if r["seed"] == GATE_SEED]
    sweep = [r for r in runs if r["seed"] != GATE_SEED]
    return {
        "country": country,
        "n_features_produced_by_builder": len(produced),
        "geometry_produced_by_builder": {c: round(float(produced[c]), 4)
                                         for c in SOLAR_GEOMETRY_FEATURES},
        "rows_identical_across_arms": True,
        "fit_rows": retained["f25"][0], "gate_rows": retained["f25"][1],
        "degraded_lag_1d_rows": frames["f25"][1]["degraded_lag_1d_rows"],
        "reproduction": {run["arm"]: run for run in reproduction},
        "runs": runs,
        "summary": _summarise(sweep) if len(sweep) > 1 else None,
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
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seeds. Default: the gate seed 42 plus "
                             "ABL-376's eight registered ones.")
    parser.add_argument("--artifact-dir", default=None,
                        help="Where the seed-42 artifacts of both arms are written. "
                             "Never a registered scope's artifact_dir.")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    replica = Path(args.replica_db).resolve()
    if not replica.exists():
        parser.error(f"replica not found: {replica}")
    fit_start, gate_start, gate_end = map(
        pd.Timestamp, (args.fit_start, args.gate_start, args.gate_end))
    if not fit_start < gate_start < gate_end:
        parser.error("require fit-start < gate-start < gate-end")
    seeds = (tuple(int(s) for s in args.seeds.split(",") if s.strip())
             if args.seeds else (GATE_SEED, *SWEEP_SEEDS))

    result = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "replica_db": str(replica), "replica_bytes": replica.stat().st_size,
        "renewable_source": args.renewable_source,
        "algorithm": ALGORITHM, "hyperparams": config.get_default_params(ALGORITHM),
        "seeds": list(seeds), "gate_seed": GATE_SEED, "sweep_seeds": list(SWEEP_SEEDS),
        "seeds_are_registered": list(seeds) == [GATE_SEED, *SWEEP_SEEDS],
        "windows": {"fit": [str(fit_start), str(gate_start)],
                    "gate": [str(gate_start), str(gate_end)]},
        "arms": {name: {"n_features": len(cols), "columns": list(cols)}
                 for name, cols in ARMS.items()},
        "countries": [],
    }
    out = Path(args.json_out) if args.json_out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
    for country in [c.strip().upper() for c in args.countries.split(",")]:
        result["countries"].append(
            probe(country, str(replica), args.renewable_source, fit_start, gate_start,
                  gate_end, seeds, Path(args.artifact_dir) if args.artifact_dir else None))
        # Written after every country: a run interrupted at the second should
        # still leave the first readable rather than nothing at all.
        if out:
            out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        logger.info("%s done", country)

    for entry in result["countries"]:
        summary = entry["summary"]
        if not summary:
            continue
        night = summary["night_pct_negative"]
        print(f"{entry['country']}: night-negative {night['control_mean']:.2f}% -> "
              f"{night['treatment_mean']:.2f}% (paired {night['paired_mean']:+.2f}pp, "
              f"{night['seeds_down']}/{night['n_seeds']} seeds down, "
              f"null {night['null_max']:.2f}pp)")
    if out:
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
