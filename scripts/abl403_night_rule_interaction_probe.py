#!/usr/bin/env python3
"""ABL-403 2x2: do the geometry features and the ABL-376 night-exclusion rule
need each other?

Two fit-side changes have been measured separately on solar and neither result
is what a reader would predict from the other:

  * **ABL-395** added ABL-338's geometry pair to the gate's feature list and
    found *nothing* on the night-negative axis - CH's control rate is
    77.05% +- 10.11 over eight seeds with a 27.34pp single-seed null, and the
    paired change is -3.85pp at 4/8 seeds. It ran the cell that carries neither
    change: the default `abl253` fit rule is `exclude_impossible_night: False`.
  * **ABL-376** measured its own exclusion rule **27x more effective** on FR's
    night level once `sun_elevation_deg` and `is_night` were in the vector
    (-8.81 MW at 7/8 seeds) than on the legacy 25 columns (-0.33 MW at 5/8),
    and stated the mechanism: nothing in those 25 names distinguishes "0 W/m2
    because the sun is down" from "0 W/m2 at a dark winter dawn", so removing
    the impossible rows leaves the model no handle for what was removed.

If that mechanism is general, the night axis moves only where **both** are
present, and every single-factor read so far has been looking at a main effect
in a design that has an interaction. This script runs the 2x2 that decides it.

Design
------
Four arms, per country, per seed:

    arm         features                                exclude_impossible_night
    f25_off     the 25 names through ABL-381            off   (= ABL-395's f25)
    f27_off     those plus the geometry pair            off   (= ABL-395's f27)
    f25_on      the 25 names                            on
    f27_on      those plus the geometry pair            on

The two `_off` arms are a like-for-like refit of ABL-395 at the same windows,
source, schedule, algorithm and seeds, so their agreement with the published
JSON is a free end-to-end reproduction check on this whole run, and it is
reported before anything else.

BG and CH bracket the question rather than sampling it. ABL-381 section 5
measured 76-85% of BG's night hours carrying 152-246 MW, so the exclusion rule
has a great deal to act on there and `is_night` is telling the model the sun is
down on hours the target books at ~225 MW. CH's night actuals are exactly 0.00,
so the rule is close to a no-op and CH is the control on the exclusion axis.

Exclusion is fit-side and only fit-side
---------------------------------------
`exclude_impossible_night_rows` is applied to the **fit** frame after
`finite_training_rows`, exactly as `evaluate_solar_retrain.py` applies it, so
the two audits partition the dropped rows. The gate frame is never filtered:
all four arms score on identical rows, and that is asserted rather than assumed.
Row identity across the *geometry* axis is asserted within each filter level -
the geometry columns are pure functions of (country, hour) and never NaN. It
cannot hold across the filter axis, which is the treatment itself.

What is measured, and why the night-negative rate is not enough
--------------------------------------------------------------
ABL-403 was written expecting the exclusion to help BG. The physics points the
other way: BG's overnight MW is **real generation** (ABL-396 puts it at 4.98pp
of gate-window energy, the largest of the 24 solar countries), and ABL-405's
PASS survived because the model reproduced that floor. A rule that drops those
rows from the fit may teach the BG model that night is zero when it is 225 MW -
which would *lower* the night-negative rate while making the forecast worse.

So the night axis carries a level metric beside the sign metric: night MAE,
night bias (prediction minus actual) and night WAPE, alongside the percentage
of night rows predicted negative. A run that improves the sign metric and
degrades the level metric is a run that made the model worse, and only one of
those two numbers can see it.

Reading rule, fixed before the run
----------------------------------
Every contrast is taken **paired within seed** - both arms saw the same rows at
the same seed, so across-seed variance cancels inside the difference - and is
reported against a null built from control-vs-control seed pairs, which is what
a one-seed gap looks like when nothing changed at all. The interaction is the
difference of differences, with its own null built from control-arm seed
quadruples. An effect no larger than its null is not distinguishable from seed
noise in this design; the seed count is not extended to chase significance.

What this is not
----------------
**Not a gate read, and it dispositions nothing.** No promotion, no serving
registry change, no ingest change. It writes no report or JSON under any
registered scope's paths and saves no artifacts. Read-only against the replica
(`mode=ro`).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import db
from src.evaluation.solar_retrain import (
    ALGORITHM, FEATURE_COLUMNS, PRIMARY_BANDS, SOLAR_GEOMETRY_FEATURES,
    attach_baselines, build_vintage_frame, common_scores, finite_training_rows,
    select_latest_challenger_per_band,
)
from src.solar_features import (
    IMPOSSIBLE_NIGHT_THRESHOLD_MW, exclude_impossible_night_rows, night_mask,
)
from src.wind_features import RenewableFeatureBuilder

logger = logging.getLogger("abl403.night_rule_interaction")

#: The 25 names the solar gate declared through ABL-381. Derived by subtraction
#: from the live list rather than copied, so the control arm cannot drift from
#: the treatment in any way other than the geometry pair itself. Same
#: construction as `abl395_geometry_feature_probe.py`, deliberately.
FEATURE_COLUMNS_25 = tuple(c for c in FEATURE_COLUMNS if c not in SOLAR_GEOMETRY_FEATURES)

#: (features, exclude_impossible_night) per arm. `f25_off` is the cell that
#: carries neither change and is the reference for every contrast below.
ARMS = {
    "f25_off": (FEATURE_COLUMNS_25, False),
    "f27_off": (FEATURE_COLUMNS, False),
    "f25_on": (FEATURE_COLUMNS_25, True),
    "f27_on": (FEATURE_COLUMNS, True),
}
CONTROL_ARM = "f25_off"

#: ABL-381's registered basis. BG and CH have zero rows in `forecasts`, so
#: naming `incumbent` would empty every intersection (ABL-322/ABL-378).
GATE_BASIS = ("challenger", "seasonal_naive")

#: ABL-376's registered seed set, reused verbatim and disjoint from the gate's
#: 42, so nothing here was selected on them and this read is commensurable with
#: ABL-376's and ABL-395's.
SWEEP_SEEDS = (101, 103, 107, 109, 113, 127, 131, 137)

#: Cap on the control-arm quadruples enumerated for the interaction null. All
#: 1,680 orderings of 4 seeds from 8 fit comfortably under it; the cap exists so
#: a larger `--seeds` cannot silently turn the null into the slow part.
_NULL_QUADRUPLE_CAP = 20000


def _wape(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    denom = np.abs(actual).sum()
    return None if denom == 0 else float(np.abs(actual - forecast).sum() / denom * 100.0)


def _sign_test_p(differences: np.ndarray) -> tuple[float | None, int, int, int]:
    """Two-sided exact sign test. Ties are dropped and reduce n, not counted.

    Returns `(p, n_negative, n_positive, n_tied)`. `p` is None when every
    difference is exactly zero, which is a real outcome for a metric an arm
    cannot move rather than a degenerate one.
    """
    negative = int((differences < 0).sum())
    positive = int((differences > 0).sum())
    tied = int((differences == 0).sum())
    n = negative + positive
    if n == 0:
        return None, negative, positive, tied
    k = min(negative, positive)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n)), negative, positive, tied


def _night_read(selected: pd.DataFrame, night: np.ndarray) -> dict:
    """The night axis: the sign metric ABL-381 reported, plus the level.

    The level half is here because the sign half cannot see the failure this
    2x2 is most likely to produce on BG - a model taught that night is zero
    when BG's night is 225 MW of real generation would show a *better*
    night-negative rate and a worse forecast.
    """
    preds = selected["challenger"].to_numpy(dtype=float)
    actual = selected["actual"].to_numpy(dtype=float)
    negative = preds < 0
    read = {
        "n_selected_rows": int(len(preds)),
        "n_negative": int(negative.sum()),
        "pct_negative": round(float(negative.mean() * 100.0), 3),
        "min_prediction_mw": round(float(preds.min()), 2),
        "n_night_rows": int(night.sum()),
        "n_negative_at_night": int((negative & night).sum()),
        "pct_of_night_rows_negative": None,
        "mean_prediction_at_night_mw": None,
        "mean_actual_at_night_mw": None,
        "night_mae_mw": None,
        "night_bias_mw": None,
        "night_wape_pct": None,
    }
    if night.sum():
        night_pred, night_actual = preds[night], actual[night]
        read.update({
            "pct_of_night_rows_negative": round(
                float((negative & night).sum() / night.sum() * 100.0), 3),
            "mean_prediction_at_night_mw": round(float(night_pred.mean()), 3),
            "mean_actual_at_night_mw": round(float(night_actual.mean()), 3),
            "night_mae_mw": round(float(np.abs(night_pred - night_actual).mean()), 3),
            "night_bias_mw": round(float((night_pred - night_actual).mean()), 3),
            "night_wape_pct": (round(_wape(night_actual, night_pred), 3)
                               if np.abs(night_actual).sum() else None),
        })
    return read


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
            "challenger_wape_pct": round(_wape(actual, chal), 3),
            "d7_wape_pct": round(_wape(actual, d7), 3),
            "clears_d7": bool(_wape(actual, chal) < _wape(actual, d7)),
            "daylight_n": int(day.sum()),
            "daylight_challenger_wape_pct": (round(_wape(actual[day], chal[day]), 3)
                                             if day.any() else None),
            "daylight_d7_wape_pct": (round(_wape(actual[day], d7[day]), 3)
                                     if day.any() else None),
        }
        out.append(cell)
    return sorted(out, key=lambda row: row["horizon_band"])


def _read_metric(run: dict, metric: str):
    """A metric key is either `band|cell field` or a key of the night read."""
    if "|" in metric:
        band, field = metric.split("|", 1)
        return next(c[field] for c in run["cells"] if c["horizon_band"] == band)
    return run["night"][metric]


#: Every axis a contrast is taken on, as (key, human label). Kept as one table
#: so the simple effects, the interaction and the report all iterate the same
#: list and cannot end up covering different metrics.
def _metric_table() -> list[tuple[str, str]]:
    metrics = [
        ("pct_of_night_rows_negative", "night rows predicted negative (%)"),
        ("mean_prediction_at_night_mw", "mean prediction at night (MW)"),
        ("night_mae_mw", "night MAE (MW)"),
        ("night_bias_mw", "night bias, pred - actual (MW)"),
        ("night_wape_pct", "night WAPE (%)"),
    ]
    for band in PRIMARY_BANDS:
        metrics.append((f"{band}|challenger_wape_pct", f"challenger WAPE {band} (%)"))
        metrics.append((f"{band}|daylight_challenger_wape_pct",
                        f"daylight challenger WAPE {band} (%)"))
    return metrics


def _contrast(values: dict, metric: str, seeds: list, treatment: str, control: str,
              null: np.ndarray) -> dict:
    """One paired simple effect: `treatment - control` at each seed.

    `null` is supplied by the caller rather than derived here, because the
    honest null for a contrast depends on how many independent fits it combines
    - two for a simple effect, four for the interaction.
    """
    a = np.array([values[(treatment, s)] for s in seeds], dtype=float)
    b = np.array([values[(control, s)] for s in seeds], dtype=float)
    difference = a - b
    scale = float(np.mean(np.abs(b))) or 1.0
    p, n_negative, n_positive, n_tied = _sign_test_p(difference)
    return {
        "metric": metric, "treatment_arm": treatment, "control_arm": control,
        "seeds": seeds,
        "control": [float(v) for v in b], "treatment": [float(v) for v in a],
        "control_mean": float(b.mean()), "treatment_mean": float(a.mean()),
        "control_sd": float(b.std(ddof=1)) if len(b) > 1 else None,
        "treatment_sd": float(a.std(ddof=1)) if len(a) > 1 else None,
        "paired_mean": float(difference.mean()),
        "paired_mean_pct": 100.0 * float(difference.mean()) / scale,
        "paired_sd": float(difference.std(ddof=1)) if len(difference) > 1 else None,
        "seeds_down": n_negative, "seeds_up": n_positive, "seeds_tied": n_tied,
        "n_seeds": len(seeds), "sign_test_p": p,
        "null_max": float(null.max()) if null.size else None,
        "null_pairs": int(null.size),
        "outside_the_null": (bool(abs(float(difference.mean())) > float(null.max()))
                             if null.size else None),
    }


def _interaction(values: dict, metric: str, seeds: list, null: np.ndarray) -> dict:
    """(f27_on - f27_off) - (f25_on - f25_off), paired within seed.

    This is the quantity ABL-403 exists to measure: how much more (or less) the
    exclusion rule buys once the model has a handle for what was excluded.
    Positive means the rule moves the metric further up with geometry present.
    """
    with_geometry = np.array([values[("f27_on", s)] - values[("f27_off", s)] for s in seeds],
                             dtype=float)
    without = np.array([values[("f25_on", s)] - values[("f25_off", s)] for s in seeds],
                       dtype=float)
    difference = with_geometry - without
    p, n_negative, n_positive, n_tied = _sign_test_p(difference)
    return {
        "metric": metric, "seeds": seeds,
        "exclusion_effect_with_geometry": [float(v) for v in with_geometry],
        "exclusion_effect_without_geometry": [float(v) for v in without],
        "exclusion_effect_with_geometry_mean": float(with_geometry.mean()),
        "exclusion_effect_without_geometry_mean": float(without.mean()),
        "interaction_mean": float(difference.mean()),
        "interaction_sd": float(difference.std(ddof=1)) if len(difference) > 1 else None,
        "seeds_down": n_negative, "seeds_up": n_positive, "seeds_tied": n_tied,
        "n_seeds": len(seeds), "sign_test_p": p,
        "null_max": float(null.max()) if null.size else None,
        "null_samples": int(null.size),
        "outside_the_null": (bool(abs(float(difference.mean())) > float(null.max()))
                             if null.size else None),
    }


def _nulls(control_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seed-noise nulls for a 2-fit contrast and for a 4-fit interaction.

    Both are built from the control arm alone - the same fit, one integer
    apart - so they are what the statistic looks like with nothing changed. The
    4-fit null is deliberately conservative: it combines four *independent*
    control fits, where the real interaction combines four fits that share a
    seed and are therefore positively correlated. An effect that clears this
    null has cleared more than it strictly had to.
    """
    pair = np.array([abs(a - b) for a, b in combinations(control_values, 2)], dtype=float)
    quads = []
    for i, (a, b, c, d) in enumerate(permutations(control_values, 4)):
        if i >= _NULL_QUADRUPLE_CAP:
            break
        quads.append(abs((a - b) - (c - d)))
    return pair, np.array(quads, dtype=float)


def _effects(runs: list[dict]) -> dict:
    """Every simple effect, both main effects and the interaction, per metric."""
    seeds = sorted({run["seed"] for run in runs})
    out = {}
    for metric, label in _metric_table():
        values = {(run["arm"], run["seed"]): _read_metric(run, metric) for run in runs}
        if any(values[k] is None for k in values):
            out[metric] = {"metric": label, "measured": False,
                           "reason": "metric is None in at least one arm/seed"}
            continue
        pair_null, quad_null = _nulls(
            np.array([values[(CONTROL_ARM, s)] for s in seeds], dtype=float))
        out[metric] = {
            "metric": label, "measured": True,
            # The two simple effects of geometry, one at each level of the rule.
            "geometry_rule_off": _contrast(values, label, seeds, "f27_off", "f25_off",
                                           pair_null),
            "geometry_rule_on": _contrast(values, label, seeds, "f27_on", "f25_on",
                                          pair_null),
            # The two simple effects of the rule, one at each feature list.
            "exclusion_at_f25": _contrast(values, label, seeds, "f25_on", "f25_off",
                                          pair_null),
            "exclusion_at_f27": _contrast(values, label, seeds, "f27_on", "f27_off",
                                          pair_null),
            # Both changes at once, against the cell that carries neither.
            "both_vs_neither": _contrast(values, label, seeds, "f27_on", "f25_off",
                                         pair_null),
            "interaction": _interaction(values, label, seeds, quad_null),
            "arm_means": {arm: float(np.mean([values[(arm, s)] for s in seeds]))
                          for arm in ARMS},
        }
    return out


def probe(country: str, replica: str, source: str, fit_start, gate_start, gate_end,
          seeds: tuple) -> dict:
    builder = RenewableFeatureBuilder(country, "solar", fit_start - pd.Timedelta(days=14),
                                      gate_end, actuals_source=source, db_path=replica)

    started = time.monotonic()
    # Built once, at the 27-name superset, and shared by every arm and seed.
    fit_raw = build_vintage_frame(builder, fit_start, gate_start, FEATURE_COLUMNS)
    gate_raw = build_vintage_frame(builder, gate_start, gate_end, FEATURE_COLUMNS)

    frames, gates, night_audits = {}, {}, {}
    for name, (columns, exclude) in ARMS.items():
        fit, audit = finite_training_rows(fit_raw, columns)
        # Same order as `evaluate_solar_retrain.py:727` -- after the missingness
        # filter, so the two audits partition the dropped rows rather than
        # double-counting a missing actual as impossible.
        if exclude:
            fit, night_audit = exclude_impossible_night_rows(fit, country)
        else:
            night_audit = None
        night_audits[name] = night_audit
        frames[name] = (fit, audit)
        # The gate frame is never filtered, at any arm. That asymmetry is the
        # ABL-376 rule: we refuse to fit on values the sun says are impossible
        # and still score against whatever the source reports.
        gates[name], _ = finite_training_rows(gate_raw, columns)

    # An arm scored on different rows is not an A/B. Geometry columns are pure
    # functions of (country, hour) and never NaN, so fit-row identity must hold
    # across the geometry axis within each filter level -- and gate-row identity
    # must hold across all four arms, since nothing filters a gate frame.
    for level, (a, b) in {"off": ("f25_off", "f27_off"), "on": ("f25_on", "f27_on")}.items():
        assert len(frames[a][0]) == len(frames[b][0]), (
            f"{country}: geometry axis changed the fit row count at rule={level} "
            f"({len(frames[a][0])} vs {len(frames[b][0])}); the contrast would "
            f"confound the feature list with the row set")
    gate_sizes = {name: len(frame) for name, frame in gates.items()}
    assert len(set(gate_sizes.values())) == 1, (
        f"{country}: the four arms scored different gate rows {gate_sizes}; the "
        f"exclusion rule must never reach a gate frame")

    logger.info("%s: fit rows off=%d on=%d (rule dropped %d), gate rows %d, built in %.1f min",
                country, len(frames["f25_off"][0]), len(frames["f25_on"][0]),
                len(frames["f25_off"][0]) - len(frames["f25_on"][0]),
                gate_sizes["f25_off"], (time.monotonic() - started) / 60)

    night_by_key, runs = {}, []
    for seed in seeds:
        for name, (columns, _) in ARMS.items():
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
            # Sun geometry depends on neither seed nor arm, and this is the
            # slowest per-row call in the loop, so the mask is computed once.
            key = len(selected)
            if key not in night_by_key:
                night_by_key[key] = night_mask(
                    country, list(pd.to_datetime(selected["target_ts"])))

            runs.append({
                "arm": name, "seed": seed, "n_features_fitted": len(columns),
                "exclude_impossible_night": ARMS[name][1],
                "fit_rows": int(len(fit)),
                "fit_seconds": round(time.monotonic() - began, 1),
                "night": _night_read(selected, night_by_key[key]),
                "cells": _cells(selected, country),
            })
            logger.info("%s seed=%d arm=%-7s night-neg %.2f%% night-MAE %.1f MW (%.0fs)",
                        country, seed, name,
                        runs[-1]["night"]["pct_of_night_rows_negative"],
                        runs[-1]["night"]["night_mae_mw"], runs[-1]["fit_seconds"])

    return {
        "country": country,
        "fit_rows": {name: int(len(frame)) for name, (frame, _) in frames.items()},
        "gate_rows": gate_sizes[CONTROL_ARM],
        "missingness_audit": frames[CONTROL_ARM][1],
        # What the rule actually removed here. BG and CH are expected to differ
        # by orders of magnitude, and that difference is the design.
        "night_exclusion_audit": night_audits["f25_on"],
        "runs": runs,
        "effects": _effects(runs) if len(seeds) > 1 else None,
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
                        help="Comma-separated seeds. Default: ABL-376's eight "
                             "registered ones, which ABL-395 also used.")
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
             if args.seeds else SWEEP_SEEDS)

    result = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "replica_db": str(replica), "replica_bytes": replica.stat().st_size,
        "renewable_source": args.renewable_source,
        "algorithm": ALGORITHM, "hyperparams": config.get_default_params(ALGORITHM),
        "seeds": list(seeds), "registered_seeds": list(SWEEP_SEEDS),
        "seeds_are_registered": list(seeds) == list(SWEEP_SEEDS),
        "impossible_night_threshold_mw": IMPOSSIBLE_NIGHT_THRESHOLD_MW,
        "windows": {"fit": [str(fit_start), str(gate_start)],
                    "gate": [str(gate_start), str(gate_end)]},
        "gate_basis": list(GATE_BASIS),
        "arms": {name: {"n_features": len(cols), "exclude_impossible_night": exclude,
                        "columns": list(cols)}
                 for name, (cols, exclude) in ARMS.items()},
        "countries": [],
    }
    out = Path(args.json_out) if args.json_out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
    for country in [c.strip().upper() for c in args.countries.split(",")]:
        result["countries"].append(
            probe(country, str(replica), args.renewable_source, fit_start, gate_start,
                  gate_end, seeds))
        # Written after every country: a run interrupted at the second should
        # still leave the first readable rather than nothing at all.
        if out:
            out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        logger.info("%s done", country)

    for entry in result["countries"]:
        effects = entry["effects"]
        if not effects:
            continue
        for metric in ("pct_of_night_rows_negative", "night_mae_mw"):
            block = effects[metric]
            if not block["measured"]:
                continue
            inter = block["interaction"]
            print(f"{entry['country']} {block['metric']}: rule alone "
                  f"{block['exclusion_at_f25']['paired_mean']:+.3f}, rule+geometry "
                  f"{block['exclusion_at_f27']['paired_mean']:+.3f}, interaction "
                  f"{inter['interaction_mean']:+.3f} "
                  f"({inter['seeds_down']}/{inter['n_seeds']} down, "
                  f"sign p={inter['sign_test_p']}, null {inter['null_max']:.3f})")
    if out:
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
