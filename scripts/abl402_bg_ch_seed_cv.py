#!/usr/bin/env python3
"""ABL-402: the per-fit seed CV of ABL-381's challenger on BG and CH solar.

Why this exists
---------------
ABL-381 reported BG and CH solar beating a hindsight hour-of-day climatology by
0.26pp and 0.86pp, and then read those two gaps against **ABL-385's fleet
percentile** -- because BG and CH are not among ABL-385's 14 served pairs, and
cannot be: having no served model is exactly why they are in the ABL-316
rollout.  ABL-385 says in terms to prefer a pair-specific CV where one exists.
This measures the one that did not exist.

Two things about that fleet read are approximations this script removes:

- it is a **percentile over a modest number of units**, for which ABL-385
  claims no parametric interval;
- it is drawn on **daylight MAE**, while the two margins are whole-window
  **WAPE**.  A relative CV transfers between the two better than an absolute pp
  figure would, but they are not the same metric, so both are reported here.

What is measured
----------------
One arm, refitted at each of `SEEDS`, on frames built **once** per country and
shared by every fit.  Everything but `random_seed` is the registered gate's own
configuration, and the scoring path is the gate's own functions rather than a
second implementation of them -- `select_latest_challenger_per_band`,
`attach_baselines`, `attach_model_free_references`, `scored_with_comparators`.

**One arm and not ABL-376's two.**  ABL-376's design is a *paired* A/B, and the
pairing is what makes a rule effect readable through the seed noise.  Here the
comparison ABL-381 needs an error bar for is challenger-vs-climatology, and the
climatology is deterministic arithmetic on the actuals -- there is no second
arm to pair against and nothing for the pairing to cancel.  What the margin
needs is the *marginal* per-fit spread of the single fitted arm, which is the
`c_A` in `delta_min` with `c_B = 0`.  Taking ABL-376's protocol here means its
build-once-refit-around design, not its two arms.

The reproduction control
------------------------
Seed 42 -- the gate's pinned seed -- is fitted **once, separately, and is
excluded from every CV**.  It is not a 21st draw: it is the check that this rig
reproduces ABL-381's published cells rather than measuring the spread of some
neighbouring quantity.  A CV anchored on the arm that produced the headline is
not a spread, which is why the 20 that *are* averaged are disjoint from it.

`SEEDS` is frozen in this file and committed before the first fit, so the
ABL-322 property ABL-381 section 1 holds here too and is checkable in git
rather than asserted.

(ASCII throughout this docstring on purpose: it is passed as
`description=__doc__`, which ABL-364's sweep reads as help text.  The comments
below keep their typography, as `abl376_night_seed_spread.py` does.)

What this is not
----------------
Not a gate read, and not a re-read of one.  No artifact is written, no
registered scope is touched, and the six dispositioned cells are not re-scored
-- `--artifact-dir` has no analogue here because nothing is saved.  This
measures the variance *around* ABL-381's read.

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl402_bg_ch_seed_cv.py \\
        --replica-db C:\\Code\\able\\data\\energy_dashboard.db
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402
from src.evaluation.model_free_reference import (  # noqa: E402
    MODEL_FREE_COMPARATORS, attach_model_free_references, comparator_wape,
)
from src.evaluation.scorecard import score_predictions  # noqa: E402
from src.evaluation.solar_retrain import (  # noqa: E402
    ALGORITHM, PRIMARY_BANDS,
    attach_baselines, build_vintage_frame, finite_training_rows,
    scored_with_comparators, select_latest_challenger_per_band,
)
from src.solar_features import SOLAR_BANDS, solar_bands  # noqa: E402
from src.wind_features import RenewableFeatureBuilder  # noqa: E402

# The CV's error bar and the decision margin come from ABL-385's reader rather
# than from a second copy here.  Two implementations of `delta_min` are free to
# drift, and this issue exists because a margin was quoted from a remembered
# number; `tests/test_abl385_margin.py` pins the chi-square approximation these
# use against scipy.
#
# `scripts.` and not a second `sys.path` entry pointing at `scripts/`: that would
# make `abl385_read_margin` reachable as both `abl385_read_margin` and
# `scripts.abl385_read_margin`, which is the one-module-two-names bug class
# `tests/test_script_imports.py` (ABL-340/ABL-354) exists to forbid.  The repo
# root is already on the path from the line above, and this is the form
# `attest_net_position_serve_faithfulness.py` and `backtest_gate_challengers.py`
# already use to import a sibling script.
from scripts.abl385_read_margin import cv_interval, delta_min  # noqa: E402
from scripts.evaluate_solar_retrain import LEGACY_FEATURE_COLUMNS  # noqa: E402

logger = logging.getLogger("abl402.seed_cv")

#: Frozen before the first fit and committed, and **disjoint from 42** -- the
#: seed `config.CATBOOST_PARAMS` pins and the one every ABL-381 cell was fitted
#: at.  Twenty rather than ABL-376's eight or ABL-385's twelve, because the
#: question here is decided at a *threshold*: CH's margin is 10.5% of its own
#: error, and `delta_min(1) = 1.96 c` crosses that at c = 5.36%.  A CV whose
#: confidence interval straddles 5.36% answers nothing.  At 12 seeds a single
#: cell's CV is uncertain by -29%/+70% (ABL-385 §1); at 20 it is -24%/+46%, and
#: the cost of the extra eight fits is about a minute per pair.
SEEDS = (211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
         269, 271, 277, 281, 283, 293, 307, 311, 313, 317)

#: The gate's pinned seed.  Fitted as a reproduction control and never averaged.
CONTROL_SEED = 42

#: ABL-381's scope `abl316-t1b`.  Registered in `scripts/evaluate_solar_retrain.py`.
COUNTRIES = ("BG", "CH")

#: **ABL-381's challenger is a 25-feature fit, and this pins it.**
#:
#: ABL-395 added ABL-338's two geometry features to the solar gate's
#: `FEATURE_COLUMNS`, so that constant is now 27 and, in ABL-395's own words, "a
#: re-run of `abl253` or `abl316-t1b` no longer reproduces its published read".
#: Importing `FEATURE_COLUMNS` here would therefore have measured the spread of a
#: *different challenger* than the one whose margins this issue exists to
#: re-read -- and it would have changed answer silently the moment `origin/main`
#: was merged, which happened mid-run.
#:
#: `LEGACY_FEATURE_COLUMNS` is ABL-395's own anchor for exactly this, derived by
#: subtraction from the live list rather than hand-copied, and pinned at 25 by
#: `tests/test_gate_feature_list_contract.py`.  Note that `features_for` is
#: *not* what is called here: `SCOPE_FEATURES` does not carry an `abl316-t1b`
#: row, so `features_for("abl316-t1b")` returns the 27 (ABL-404).
#:
#: The empirical check that this is the right vector is the seed-42 control
#: reproducing all six published cells, not this comment.
FEATURE_COLUMNS = LEGACY_FEATURE_COLUMNS

#: ABL-348's registration, unchanged and deliberately not re-derived here.
#:
#: **tz-naive on purpose.**  `experiments/ABL348/config.json` writes these as
#: `...Z`, but the gate reaches the builder through
#: `map(pd.Timestamp, (args.fit_start, ...))` on bare `YYYY-MM-DD` strings, and
#: `RenewableFeatureBuilder` works in naive UTC throughout -- an aware Timestamp
#: raises inside `_min_admissible_lag_days`.  These are the same instants the
#: gate used; matching its *representation* is what makes the seed-42 control a
#: reproduction rather than a near-miss.
FIT_START = pd.Timestamp("2026-01-14")
GATE_START = pd.Timestamp("2026-07-11")
GATE_END = pd.Timestamp("2026-08-10")
SOURCE = "energy_generation"

#: `GATE_BASIS["abl316-t1b"]`.  BG and CH hold zero solar rows in `forecasts`,
#: so the four-way basis would intersect every cell to n=0; ABL-381 registered
#: the two columns the bar actually names.
GATE_BASIS = ("challenger", "seasonal_naive")

#: The incumbent and TSO columns the gate also reports are omitted: this run
#: merges neither, and the incumbent does not exist for either pair.  Dropping
#: them cannot move a number, because `scored_with_comparators` scores every
#: comparator on its own intersection *with the basis* and the basis is above.
COMPARATORS = (*GATE_BASIS, *MODEL_FREE_COMPARATORS)

#: ABL-381 §3, the two gaps this issue exists to put an error bar on.  The
#: reference is `climatology_oracle`; these are recomputed from the run below
#: and compared against the published values rather than trusted.
PUBLISHED_CELLS = {
    ("BG", "24-36h"): {"challenger": 18.89, "climatology_oracle": 19.15},
    ("BG", "36-48h"): {"challenger": 18.60, "climatology_oracle": 19.15},
    ("BG", "48-64h"): {"challenger": 20.03, "climatology_oracle": 20.38},
    ("CH", "24-36h"): {"challenger": 8.16, "climatology_oracle": 9.02},
    ("CH", "36-48h"): {"challenger": 8.01, "climatology_oracle": 9.02},
    ("CH", "48-64h"): {"challenger": 8.39, "climatology_oracle": 8.70},
}


def _fit_predict(fit: pd.DataFrame, gate_x: pd.DataFrame, seed: int) -> np.ndarray:
    """One fit at one seed.  Everything but `random_seed` is the gate's config."""
    params = dict(config.get_default_params(ALGORITHM))
    params["random_seed"] = seed
    model = CatBoostRegressor(**params)
    model.fit(fit[list(FEATURE_COLUMNS)], fit["actual"])
    return np.asarray(model.predict(gate_x), dtype=float)


def build_country(country: str, replica: str) -> dict:
    """Everything the seed does not touch, built once.

    The gate builds the fit frame, the gate frame, the D-7 baseline and the four
    model-free reference columns before it ever calls `model.predict`, and none
    of them depends on a prediction -- `select_latest_challenger_per_band` sorts
    on `generated_at` alone.  So they are hoisted out of the seed loop, which is
    what makes 20 refits cost a minute instead of an hour.
    """
    started = time.monotonic()
    builder = RenewableFeatureBuilder(country, "solar", FIT_START - pd.Timedelta(days=14),
                                      GATE_END, actuals_source=SOURCE, db_path=replica)

    fit_raw = build_vintage_frame(builder, FIT_START, GATE_START, FEATURE_COLUMNS)
    fit, fit_audit = finite_training_rows(fit_raw, FEATURE_COLUMNS)
    # ABL-381 registered `exclude_impossible_night: False` for this scope, so
    # the fit frame is the unfiltered one.  Stated rather than left implicit:
    # BG's overnight floor (ABL-381 §5) is exactly what that rule would remove,
    # and a spread measured under a different rule would not be this read's.

    gate_raw = build_vintage_frame(builder, GATE_START, GATE_END, FEATURE_COLUMNS)
    gate_finite, gate_audit = finite_training_rows(gate_raw, FEATURE_COLUMNS)
    selected = attach_baselines(select_latest_challenger_per_band(gate_finite),
                                builder._actuals)
    selected, reference_levels = attach_model_free_references(
        selected, builder._actuals, FIT_START, GATE_START, GATE_END)
    selected = selected[selected["horizon_band"].isin(PRIMARY_BANDS)].reset_index(drop=True)
    selected["day_band"] = solar_bands(country, selected["target_ts"]).to_numpy()

    logger.info("%s: %d fit rows, %d scored gate rows, built in %.1f min",
                country, len(fit), len(selected), (time.monotonic() - started) / 60)
    return {"country": country, "fit": fit, "frame": selected,
            "fit_audit": fit_audit, "gate_build_audit": gate_audit,
            "model_free_reference_mw": reference_levels}


def score_one_seed(built: dict, seed: int) -> dict:
    """Fit at `seed`, then score exactly the cells the gate scores."""
    frame = built["frame"].copy()
    began = time.monotonic()
    frame["challenger"] = _fit_predict(built["fit"], frame[list(FEATURE_COLUMNS)], seed)

    cells = {}
    for band in PRIMARY_BANDS:
        group = frame[frame["horizon_band"] == band]
        scores, common, comparator_n = scored_with_comparators(group, GATE_BASIS, COMPARATORS)
        cells[band] = _cell_record(scores, common, comparator_n)
    # The gate's country-level D+2 row: the same three bands pooled, scored once
    # rather than averaged, so it is a WAPE and not a mean of WAPEs.
    scores, common, comparator_n = scored_with_comparators(frame, GATE_BASIS, COMPARATORS)
    pooled = _cell_record(scores, common, comparator_n)

    logger.info("%s seed=%-4d 24-36h WAPE %.4f%%  pooled WAPE %.4f%%  daylight MAE %.2f MW (%.0fs)",
                built["country"], seed, cells["24-36h"]["challenger_wape_pct"],
                pooled["challenger_wape_pct"], pooled["daylight_mae_mw"],
                time.monotonic() - began)
    return {"seed": seed, "bands": cells, "pooled": pooled}


def _cell_record(scores: dict, common: pd.DataFrame, comparator_n: dict) -> dict:
    """One scored cell: the gate's WAPE, and the day-band split beside it.

    The day-band metrics are computed on the **basis intersection** `common`, so
    the daylight MAE and the WAPE describe the same rows.  ABL-385 reads solar on
    daylight MAE and these margins are whole-window WAPE; reporting both from one
    row set is what makes the two CVs comparable rather than merely adjacent.
    """
    record = {
        "n": int(len(common)),
        "challenger_wape_pct": scores["challenger"]["wape_pct"],
        "challenger_mae_mw": scores["challenger"]["mae"],
        "seasonal_naive_wape_pct": scores["seasonal_naive"]["wape_pct"],
        "comparator_n": comparator_n,
    }
    for name in MODEL_FREE_COMPARATORS:
        record[f"{name}_wape_pct"] = comparator_wape(scores, name)

    actual = common["actual"].to_numpy(dtype=float)
    predicted = common["challenger"].to_numpy(dtype=float)
    bands = common["day_band"].to_numpy()
    for band in SOLAR_BANDS:
        mask = bands == band
        record[f"{band}_n"] = int(mask.sum())
        if mask.any():
            # MW only.  A band whose actuals are ~0 has no meaningful relative
            # error -- WAPE there divides by nothing.  Night is exactly that.
            record[f"{band}_mae_mw"] = float(np.abs(predicted[mask] - actual[mask]).mean())
            record[f"{band}_mean_actual_mw"] = float(actual[mask].mean())
            record[f"{band}_mean_pred_mw"] = float(predicted[mask].mean())
    return record


def _spread(values: list[float]) -> dict:
    """Per-fit CV of one statistic across the seeds, with its own error bar.

    `dof = n - 1` and the interval is ABL-385's: a sd from n draws is
    chi-square distributed.  The point estimate alone would repeat, one level
    up, the mistake this line of issues was filed on.
    """
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    cv = sd / abs(mean) if mean else float("nan")
    lo, hi = cv_interval(cv, len(array) - 1)
    return {"n_seeds": int(len(array)), "mean": mean, "sd": sd, "cv_pct": 100.0 * cv,
            "cv_pct_ci95": [100.0 * lo, 100.0 * hi],
            "min": float(array.min()), "max": float(array.max()),
            # The spread a *single-seed* read could have shown, in the units the
            # margins are quoted in.  Not a CV: the raw range, for readers who
            # want the pp figure rather than the ratio.
            "range_pp": float(array.max() - array.min())}


def _margin_reading(cv_fraction: float, challenger: float, reference: float) -> dict:
    """One climatology gap against the margin its own CV implies.

    `c_B = 0`: the reference is deterministic arithmetic on the actuals, not a
    second fit, so the two-fitted-arm margin ABL-385 tabulates shrinks by
    sqrt(2) here.  `delta_min` is ABL-385's function, called with that zero.
    """
    gap_pp = reference - challenger
    gap_relative = 100.0 * gap_pp / challenger if challenger else float("nan")
    readings = {k: 100.0 * delta_min(cv_fraction, 0.0, k) for k in (1, 3, 5, 10, 20)}
    return {"challenger_wape_pct": challenger, "reference_wape_pct": reference,
            "margin_pp": gap_pp, "margin_pct_of_challenger": gap_relative,
            "delta_min_pct": readings,
            "readable_at_k1": bool(abs(gap_relative) >= readings[1]),
            "seeds_needed": _seeds_needed(cv_fraction, gap_relative)}


def _seeds_needed(cv_fraction: float, gap_relative: float) -> int | None:
    """Smallest k with `delta_min(k) <= |gap|`.  None when the gap is zero."""
    if not gap_relative or math.isnan(gap_relative) or math.isnan(cv_fraction):
        return None
    k = (100.0 * 1.96 * cv_fraction / abs(gap_relative)) ** 2
    return max(1, int(math.ceil(k)))


def _git_provenance() -> dict:
    """The seed list is only frozen if it is in a commit that precedes the fit.

    The commit that last touched *this file* is the weaker check -- a typo fix
    would reset it and read as if the seeds had moved.  What must not move is
    the `SEEDS` tuple, so `git log -S` on its literal text is what is recorded:
    the commit that introduced these twenty integers.  Any later edit to them
    would produce a different, later commit here, and that is the thing a
    reviewer can check without taking my word for it.
    """
    root = Path(__file__).parent.parent
    seed_literal = ", ".join(str(s) for s in SEEDS[:6])

    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError) as exc:
            return f"unavailable: {exc}"

    return {
        "head": _git("rev-parse", "HEAD"),
        "seed_tuple_commit": _git("log", "-1", "--format=%H %cI",
                                  f"-S{seed_literal}", "--", "scripts/abl402_bg_ch_seed_cv.py"),
        "file_commit": _git("log", "-1", "--format=%H %cI", "--", "scripts/abl402_bg_ch_seed_cv.py"),
        "seed_line_dirty": seed_literal not in _git("show", "HEAD:scripts/abl402_bg_ch_seed_cv.py"),
        "working_tree_dirty": bool(_git("status", "--porcelain", "--", "scripts/abl402_bg_ch_seed_cv.py")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replica-db", default=None,
                        help="Live replica. Read-only; nothing here writes to it.")
    parser.add_argument("--json-out", default="reports/abl_402_seed_cv.json")
    parser.add_argument("--seeds", type=int, default=len(SEEDS),
                        help="Use only the first N frozen seeds (for a smoke run).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # The measurement is of ABL-381's challenger, and that is a 25-feature fit.
    # Loud and at the top rather than buried in the JSON: if this ever reads 27,
    # every margin below belongs to a different model and the seed-42 control
    # will stop reproducing the published cells.
    if len(FEATURE_COLUMNS) != 25:
        raise SystemExit(f"expected ABL-381's 25-feature challenger, got {len(FEATURE_COLUMNS)}")
    replica = str(args.replica_db or config.DATABASE_PATH)
    seeds = SEEDS[:args.seeds]
    provenance = _git_provenance()
    logger.info("replica=%s  seeds=%s  control=%d", replica, seeds, CONTROL_SEED)
    logger.info("seed tuple committed at: %s (seed line dirty=%s, file dirty=%s)",
                provenance["seed_tuple_commit"], provenance["seed_line_dirty"],
                provenance["working_tree_dirty"])

    results = []
    for country in COUNTRIES:
        built = build_country(country, replica)
        control = score_one_seed(built, CONTROL_SEED)
        runs = [score_one_seed(built, seed) for seed in seeds]
        results.append({"country": country,
                        "fit_rows": int(len(built["fit"])),
                        "fit_audit": built["fit_audit"],
                        "gate_build_audit": built["gate_build_audit"],
                        "model_free_reference_mw": built["model_free_reference_mw"],
                        "control": control, "runs": runs})

    payload = {
        "issue": "ABL-402",
        "parent": "ABL-381",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "replica": replica,
        "interpreter": sys.executable,
        "python": sys.version.split()[0],
        "registration": {
            "source": "experiments/ABL348/config.json, scope abl316-t1b -- read, not re-derived",
            "fit_target_window": [str(FIT_START), str(GATE_START)],
            "gate_target_window": [str(GATE_START), str(GATE_END)],
            "source_table": SOURCE, "primary_bands": list(PRIMARY_BANDS),
            "gate_basis": list(GATE_BASIS), "algorithm": ALGORITHM,
            "exclude_impossible_night": False,
            # ABL-395/ABL-404.  Recorded because it is the one registered
            # property that moved after ABL-381's read, and a CV attaches to the
            # challenger it was measured on.
            "feature_columns": list(FEATURE_COLUMNS),
            "n_features": len(FEATURE_COLUMNS),
            "feature_set": "legacy25 (ABL-381's challenger), pinned via LEGACY_FEATURE_COLUMNS",
        },
        "seeds": list(seeds), "control_seed": CONTROL_SEED,
        "git": provenance,
        "countries": results,
    }
    payload["analysis"] = analyse(payload)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", out)
    return 0


def analyse(payload: dict) -> dict:
    """CV per cell, and every published margin re-read against its own CV."""
    analysis = {"cells": [], "pooled": [], "reproduction": []}
    for country_result in payload["countries"]:
        country = country_result["country"]
        runs, control = country_result["runs"], country_result["control"]

        for band in PRIMARY_BANDS:
            observed = [run["bands"][band] for run in runs]
            wape = _spread([o["challenger_wape_pct"] for o in observed])
            daylight = _spread([o["daylight_mae_mw"] for o in observed])
            reference = _deterministic(observed, "climatology_oracle_wape_pct")
            analysis["cells"].append({
                "country": country, "band": band, "n": observed[0]["n"],
                "wape": wape, "daylight_mae": daylight,
                "climatology_oracle_wape_pct": reference,
                "margin_at_cell_cv": _margin_reading(wape["cv_pct"] / 100.0,
                                                     wape["mean"], reference),
                "margin_at_seed42": _margin_reading(wape["cv_pct"] / 100.0,
                                                    control["bands"][band]["challenger_wape_pct"],
                                                    reference),
            })
            analysis["reproduction"].append({
                "country": country, "band": band,
                "seed42_challenger_wape_pct": control["bands"][band]["challenger_wape_pct"],
                "published_challenger_wape_pct": PUBLISHED_CELLS[(country, band)]["challenger"],
                "seed42_climatology_oracle_wape_pct": control["bands"][band]["climatology_oracle_wape_pct"],
                "published_climatology_oracle_wape_pct": PUBLISHED_CELLS[(country, band)]["climatology_oracle"],
                "seed42_n": control["bands"][band]["n"],
            })

        observed = [run["pooled"] for run in runs]
        wape = _spread([o["challenger_wape_pct"] for o in observed])
        daylight = _spread([o["daylight_mae_mw"] for o in observed])
        reference = _deterministic(observed, "climatology_oracle_wape_pct")
        analysis["pooled"].append({
            "country": country, "n": observed[0]["n"],
            "wape": wape, "daylight_mae": daylight,
            "climatology_oracle_wape_pct": reference,
            "margin_at_cell_cv": _margin_reading(wape["cv_pct"] / 100.0, wape["mean"], reference),
        })
    return analysis


def _deterministic(observed: list[dict], key: str) -> float:
    """A reference column's value, checked to be seed-invariant rather than assumed.

    `c_B = 0` is the whole reason the margin shrinks by sqrt(2), so it is worth
    one assertion: if a reference moved across seeds it would not be arithmetic
    on the actuals and this analysis would be wrong.
    """
    values = {round(o[key], 12) for o in observed}
    if len(values) != 1:
        raise AssertionError(f"{key} moved across seeds: {sorted(values)}")
    return observed[0][key]


if __name__ == "__main__":
    raise SystemExit(main())
