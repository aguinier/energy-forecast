#!/usr/bin/env python3
"""ABL-427: re-read tranche 2c's U(+) pairs IT and HR at k > 1 seeds.

Why this exists
---------------
ABL-419 graded IT, HR and ES `U(+)`.  All three clear ABL-348's registered bar
in all three D+2 bands, and all nine margins sit inside ABL-385's readability
floor at one seed -- 10.6482% on solar, `1.96 * c_A` with `c_B = 0`.  `U(+)` is
not a verdict.  ABL-418 registers it as a *disposition*: re-read at k > 1 seeds.
This is that re-read, and its output is `A` or a plain `U`.

ES is out of scope by CEO direction, not merely optional: its margins need
k = 18+ to clear the nominal floor and it sits behind the ABL-411 / ABL-425
overnight-CSP serving hold regardless, so no seed spend on ES can change a
disposition.  GR and PT are graded `C` -- readable losses, not `U(+)` pairs.

What is measured, and in what order
-----------------------------------
Two questions, and the second one is only reached because the first is answered
by the same fits:

1. **The per-fit seed CV of these pairs**, `c_A`, on the gate's own WAPE.  The
   floor every `U(+)` was called against is ABL-385's **fleet p90**, and ABL-402
   measured that fleet value to be roughly 2x too wide on BG and CH.  The floor
   moves with `c_A`, and the readability call moves with the floor -- so the CV
   is the thing to measure first, and the letter follows from it.

2. **The k-seed read.**  The mean of the k per-seed WAPEs against the same
   deterministic D-7 the gate registered, graded on ABL-418's own ladder.

Protocol
--------
`experiments/ABL427/config.json`, frozen and committed before the first fit.
Everything ABL-348 registered -- windows, bands, metric, baseline, minimum n,
source table -- is inherited unchanged and is not restated here.

One arm, refitted at each of `SEEDS`, on frames built **once** per country and
shared by every fit.  Everything but `random_seed` is the registered gate's own
configuration, and the scoring path is the gate's own functions rather than a
second implementation of them.

**The seed list is ABL-385's, verbatim.**  Read at run time from
`experiments/ABL385/config.json` and cross-checked against the literal below, so
a reviewer can see the twelve integers were committed long before ABL-419 was
fitted.  Its first element is 42, the seed the gate pins, which is what makes
the k = 1 prefix of this read exactly ABL-419's published cell and every k from
1 to 12 an extension of it rather than a different experiment.

**Seed 42 is not excluded from the mean.**  ABL-402 excluded it because a CV
anchored on the arm that produced a headline is not a spread.  Here the k-seed
mean is *supposed* to contain the published draw -- that is what makes this a
re-read of that cell.  The CV excluding 42 is reported beside the CV including
it, so the ABL-402 form is visible too.

The reproduction control
------------------------
Every comparator except the challenger is deterministic arithmetic on the
actuals: D-7, a flat line, an hour-of-day climatology.  Their WAPEs are
recomputed here and compared cell-by-cell against ABL-419's committed record.

That check is not decoration.  The replica has grown since ABL-419 was generated
and the gate window lies inside `energy_generation`'s revision horizon, so a
revised actual would move the *level* of this read without touching its spread.
A deterministic reference that has moved says the actuals moved; a seed-42
challenger that has moved when the references have not would say something far
stranger.  Separating the two before a verdict is taken is the point.

Which floor decides the letter
------------------------------
`delta_min(k) = 1.96 * c_A_upper95 / sqrt(k)`, the **upper end** of the 95%
chi-square interval of the measured per-cell CV.  The upper bound and not the
point estimate: a CV from 12 draws is uncertain by about -29%/+70%, and deciding
a letter on a point estimate would repeat, one level up, the mistake ABL-385 was
filed on.  ABL-385 registers its margin table "per (pair, algorithm) where a
pair-specific value exists", so preferring a measured pair CV is inside that
registration rather than a re-opening of it.

The ladder itself is **called unmodified**.  `grade_cell` derives its floor from
`readability_floor_pct(stream, k)`, so a measured floor enters as the equivalent
k that produces it -- `k_eff = (1.96 * c_fleet / floor)^2` -- and the identity is
asserted rather than trusted.  A second copy of the ladder here, with one
constant changed, is exactly how two implementations drift.

What this is not
----------------
Not a promotion, not a recommendation to serve, and not a re-basing of
`abl316-t2c`.  No registered scope is touched, no row of the six solar-harness
registration tables is edited, no artifact is saved, no row is written to
`forecasts`, and every ABL-419 output file is left byte-unchanged.  The replica
is opened read-only.

(ASCII throughout this docstring on purpose: it is passed as
`description=__doc__`, which ABL-364's sweep reads as help text.)

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl427_tranche2c_seed_reread.py \\
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
from src.evaluation.gate_grading import (  # noqa: E402
    STREAM_FLEET_CV_P90, Z_95, margin_pct_of_own_error, pair_grade,
    readability_floor_pct, skill_pct,
)
from src.evaluation.model_free_reference import (  # noqa: E402
    MODEL_FREE_COMPARATORS, attach_model_free_references, comparator_wape,
)
from src.evaluation.solar_retrain import (  # noqa: E402
    ALGORITHM, PRIMARY_BANDS,
    attach_baselines, build_vintage_frame, finite_training_rows,
    scored_with_comparators, select_latest_challenger_per_band,
)
from src.solar_features import SOLAR_BANDS, solar_bands  # noqa: E402
from src.wind_features import RenewableFeatureBuilder  # noqa: E402

# `delta_min` and `cv_interval` come from ABL-385's reader rather than from a
# second copy here -- `tests/test_abl385_margin.py` pins the chi-square
# approximation against scipy. This is the import form
# `scripts/abl402_bg_ch_seed_cv.py` already uses, and not a second `sys.path`
# entry pointing at `scripts/`: that would make the module reachable under two
# names, the bug class `tests/test_script_imports.py` (ABL-340/ABL-354) forbids.
from scripts.abl385_read_margin import cv_interval, delta_min  # noqa: E402
from scripts.evaluate_solar_retrain import (  # noqa: E402
    CAUSAL_LEVELLING, FEATURE_COLUMNS as LIVE_FEATURE_COLUMNS,
    G23_READABILITY,
)

logger = logging.getLogger("abl427.seed_reread")

REPO_ROOT = Path(__file__).parent.parent
REGISTRATION = REPO_ROOT / "experiments" / "ABL427" / "config.json"
ABL385_REGISTRATION = REPO_ROOT / "experiments" / "ABL385" / "config.json"
ABL419_RECORD = REPO_ROOT / "experiments" / "ABL348" / "results_abl419_tranche2c.json"

#: ABL-385's registered seed list, as a literal, so `git log -S` on these twelve
#: integers finds the commit that froze them. The tuple is *cross-checked*
#: against `experiments/ABL385/config.json` at run time rather than being the
#: source of truth: the registration file is the source, and this is the check
#: that nothing edited it between then and now.
ABL385_SEEDS = (42, 1337, 2718, 7, 13, 101, 271, 314, 577, 863, 1024, 1729)

#: The gate's pinned seed. Here it is the *first member of the read*, not an
#: excluded control -- see the module docstring.
CONTROL_SEED = 42

#: The two `U(+)` pairs the CEO scoped this issue to. ES is excluded by
#: direction and GR/PT are graded `C`; `experiments/ABL427/config.json` carries
#: the reason for each.
COUNTRIES = ("IT", "HR")

#: ABL-419's scope, read for its registration and its published cells. Never
#: written to.
SOURCE_SCOPE = "abl316-t2c"

#: ABL-348's registration, unchanged and deliberately not re-derived here.
#:
#: **tz-naive on purpose.** `experiments/ABL348/config.json` writes these as
#: `...Z`, but the gate reaches the builder through `map(pd.Timestamp, ...)` on
#: bare `YYYY-MM-DD` strings and `RenewableFeatureBuilder` works in naive UTC
#: throughout. These are the same instants the gate used; matching its
#: *representation* is what makes the seed-42 control a reproduction rather than
#: a near-miss.
FIT_START = pd.Timestamp("2026-01-14")
GATE_START = pd.Timestamp("2026-07-11")
GATE_END = pd.Timestamp("2026-08-10")
SOURCE = "energy_generation"

#: `GATE_BASIS["abl316-t2c"]`. IT and HR hold zero solar rows in `forecasts`, so
#: a four-way basis would intersect every cell to n = 0; ABL-419 registered the
#: two columns the bar actually names.
GATE_BASIS = ("challenger", "seasonal_naive")

COMPARATORS = (*GATE_BASIS, *MODEL_FREE_COMPARATORS)

#: `minimum_n` per band, from ABL-419's own gate records. Checked rather than
#: assumed: ABL-434 is open on a ladder that awards a letter to a cell failing
#: this, and a re-read that inherited that defect would launder it.
MINIMUM_N = {"24-36h": 684, "36-48h": 684, "48-64h": 456}

#: How close a recomputed deterministic reference must sit to ABL-419's
#: published value to count as "the actuals did not move". WAPEs are published
#: at full float precision in the record, so this is a float-arithmetic
#: tolerance and not a data tolerance.
REFERENCE_TOLERANCE_PP = 1e-9


def _load_seeds() -> tuple[int, ...]:
    """ABL-385's registered seeds, read from its config and cross-checked."""
    registered = tuple(json.loads(ABL385_REGISTRATION.read_text(encoding="utf-8"))
                       ["scope"]["seeds"])
    if registered != ABL385_SEEDS:
        raise SystemExit(
            f"ABL-385's registered seed list has changed since ABL-427 froze "
            f"against it: config has {registered}, this script pins "
            f"{ABL385_SEEDS}. Refusing to run -- the anti-selection property "
            f"this read depends on is exactly that these did not move.")
    return registered


def _pinned_feature_columns() -> tuple[tuple[str, ...], bool]:
    """The 27 columns ABL-419 fitted, taken from its own record.

    Pinned from the record and not from the live constant, for the reason
    ABL-402 gives at length: `FEATURE_COLUMNS` is a live list that ABL-395 has
    already changed once mid-run, and importing it would have measured the
    spread of a *different challenger* than the one being re-read. The live list
    is compared against it and the agreement is recorded, so a future divergence
    surfaces as a reported fact rather than a silent re-read of something else.
    """
    pinned = tuple(json.loads(ABL419_RECORD.read_text(encoding="utf-8"))
                   ["meta"]["feature_columns"])
    return pinned, tuple(LIVE_FEATURE_COLUMNS) == pinned


def _published_cells() -> dict:
    """ABL-419's committed cells, keyed (country, band). The reproduction target."""
    record = json.loads(ABL419_RECORD.read_text(encoding="utf-8"))
    out = {}
    for cell in record["gate_cells"]:
        if cell["country"] not in COUNTRIES:
            continue
        out[(cell["country"], cell["horizon_band"])] = {
            "scores": cell["scores"], "gate": cell["gate"], "grade": cell["grade"]}
    return out


def _fit_predict(fit: pd.DataFrame, gate_x: pd.DataFrame, seed: int,
                 feature_columns: tuple[str, ...]) -> np.ndarray:
    """One fit at one seed. Everything but `random_seed` is the gate's config."""
    params = dict(config.get_default_params(ALGORITHM))
    params["random_seed"] = seed
    model = CatBoostRegressor(**params)
    model.fit(fit[list(feature_columns)], fit["actual"])
    return np.asarray(model.predict(gate_x), dtype=float)


def build_country(country: str, replica: str, feature_columns: tuple[str, ...]) -> dict:
    """Everything the seed does not touch, built once.

    The gate builds the fit frame, the gate frame, the D-7 baseline and the
    model-free reference columns before it ever calls `model.predict`, and none
    of them depends on a prediction. Hoisting them out of the seed loop is what
    makes twelve refits cost minutes rather than hours.
    """
    started = time.monotonic()
    builder = RenewableFeatureBuilder(country, "solar", FIT_START - pd.Timedelta(days=14),
                                      GATE_END, actuals_source=SOURCE, db_path=replica)

    fit_raw = build_vintage_frame(builder, FIT_START, GATE_START, feature_columns)
    fit, fit_audit = finite_training_rows(fit_raw, feature_columns)
    # ABL-419 registered `exclude_impossible_night: False` for this scope, so the
    # fit frame is the unfiltered one. A spread measured under a different rule
    # would not be this read's.

    gate_raw = build_vintage_frame(builder, GATE_START, GATE_END, feature_columns)
    gate_finite, gate_audit = finite_training_rows(gate_raw, feature_columns)
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


def score_one_seed(built: dict, seed: int, feature_columns: tuple[str, ...]) -> dict:
    """Fit at `seed`, then score exactly the cells the gate scores."""
    frame = built["frame"].copy()
    began = time.monotonic()
    frame["challenger"] = _fit_predict(built["fit"], frame[list(feature_columns)],
                                       seed, feature_columns)

    cells = {}
    for band in PRIMARY_BANDS:
        group = frame[frame["horizon_band"] == band]
        scores, common, comparator_n = scored_with_comparators(group, GATE_BASIS, COMPARATORS)
        cells[band] = _cell_record(scores, common, comparator_n)

    logger.info("%s seed=%-5d WAPE %s (%.0fs)", built["country"], seed,
                "  ".join(f"{b} {cells[b]['scores']['challenger']['wape_pct']:.4f}%"
                          for b in PRIMARY_BANDS),
                time.monotonic() - began)
    return {"seed": seed, "bands": cells}


def _cell_record(scores: dict, common: pd.DataFrame, comparator_n: dict) -> dict:
    """One scored cell: the gate's own `scores` block, and the day-band split.

    `scores` is kept in the exact shape `results.json` writes and `grade_cell`
    reads, so the ladder can be called on it without a translation layer that
    could disagree with the harness.

    The day-band metrics are computed on the **basis intersection** `common`, so
    the daylight MAE and the WAPE describe the same rows. ABL-385's fleet
    percentile is measured on daylight MAE and these margins are whole-window
    WAPE; reporting both from one row set is what makes the pair-specific CV and
    the fleet value comparable rather than merely adjacent.
    """
    record = {"n": int(len(common)), "scores": scores, "comparator_n": comparator_n}

    actual = common["actual"].to_numpy(dtype=float)
    predicted = common["challenger"].to_numpy(dtype=float)
    bands = common["day_band"].to_numpy()
    for band in SOLAR_BANDS:
        mask = bands == band
        record[f"{band}_n"] = int(mask.sum())
        if mask.any():
            # MW only. A band whose actuals are ~0 has no meaningful relative
            # error -- WAPE there divides by nothing. Night is exactly that.
            record[f"{band}_mae_mw"] = float(np.abs(predicted[mask] - actual[mask]).mean())
    return record


def _spread(values: list[float]) -> dict:
    """Per-fit CV of one statistic across the seeds, with its own error bar.

    `dof = n - 1` and the interval is ABL-385's: a sd from n draws is
    chi-square distributed. The point estimate alone would repeat, one level up,
    the mistake this line of issues was filed on.
    """
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    cv = sd / abs(mean) if mean else float("nan")
    # `cv_interval` is ABL-385's Wilson-Hilferty chi-square approximation, which
    # is only defined for dof >= 2 -- at dof = 1 its lower quantile goes negative
    # and it raises. Reachable only from a `--seeds 2` smoke test, never from the
    # registered k = 12, and a smoke test should not be able to crash the rig it
    # is smoke-testing. A NaN interval is the honest answer for one degree of
    # freedom in any case.
    dof = len(array) - 1
    lo, hi = cv_interval(cv, dof) if dof >= 2 else (float("nan"), float("nan"))
    return {"n_seeds": int(len(array)), "mean": mean, "sd": sd,
            "cv_pct": 100.0 * cv, "cv_pct_ci95": [100.0 * lo, 100.0 * hi],
            "cv_fraction": cv, "cv_fraction_upper95": hi,
            "min": float(array.min()), "max": float(array.max()),
            "range_pp": float(array.max() - array.min())}


def _floor_from_cv(cv_fraction: float, k: int) -> float:
    """`delta_min(k)` in percent, with `c_B = 0`, from ABL-385's own function."""
    return 100.0 * delta_min(cv_fraction, 0.0, k)


def _equivalent_k(floor_pct: float) -> float:
    """The `k` at which the ladder's fleet-p90 floor equals `floor_pct`.

    Lets `grade_cell` be called **unmodified** against a measured floor: it
    derives its floor from `readability_floor_pct(stream, k)`, so substituting
    the floor means substituting the k that produces it. The identity is
    asserted by the caller, not trusted.
    """
    return (Z_95 * STREAM_FLEET_CV_P90["solar"] * 100.0 / floor_pct) ** 2


def _seeds_needed(cv_fraction: float, gap_relative: float) -> int | None:
    """Smallest k with `delta_min(k) <= |gap|`. None when the gap is zero."""
    if not gap_relative or math.isnan(gap_relative) or math.isnan(cv_fraction):
        return None
    return max(1, int(math.ceil((100.0 * Z_95 * cv_fraction / abs(gap_relative)) ** 2)))


def _mean_scores(per_seed_cells: list[dict], band: str, k: int) -> dict:
    """The k-seed mean cell, in the `scores` shape the ladder reads.

    The challenger's metrics are averaged over the first `k` seeds -- the
    registered statistic is the mean of the k per-seed WAPEs, not the WAPE of a
    mean prediction, because `delta_min` is derived for the relative gap between
    two k-seed *means of the metric*. Every other comparator is deterministic
    given the actuals and is taken from the first seed's scoring; that it really
    is identical across seeds is asserted separately rather than assumed.
    """
    heads = [cell["bands"][band]["scores"] for cell in per_seed_cells[:k]]
    scores = {name: dict(value) for name, value in heads[0].items()}
    challenger = {}
    for field in ("wape_pct", "mae", "bias_pct", "slope", "correlation"):
        values = [head["challenger"].get(field) for head in heads]
        challenger[field] = (None if any(v is None for v in values)
                             else float(np.mean(values)))
    challenger["n"] = heads[0]["challenger"]["n"]
    scores["challenger"] = challenger
    return scores


def _assert_references_constant(per_seed_cells: list[dict], country: str) -> dict:
    """Every non-challenger comparator must be identical at every seed.

    If one is not, the frames were not built once, and every CV in this record
    is contaminated by whatever else moved. Reported as a checked fact.
    """
    checked = 0
    for band in PRIMARY_BANDS:
        first = per_seed_cells[0]["bands"][band]["scores"]
        for cell in per_seed_cells[1:]:
            other = cell["bands"][band]["scores"]
            for name in first:
                if name == "challenger":
                    continue
                if first[name].get("wape_pct") != other[name].get("wape_pct"):
                    raise SystemExit(
                        f"{country} {band}: comparator {name} moved between "
                        f"seed {per_seed_cells[0]['seed']} and {cell['seed']} "
                        f"-- the frames are not shared and no CV here is valid.")
                checked += 1
    return {"country": country, "comparator_wapes_compared": checked, "all_identical": True}


def _git_provenance() -> dict:
    """The seed list is only frozen if it is in a commit that precedes the fit.

    The commit that last touched *this file* is the weaker check -- a typo fix
    would reset it and read as if the seeds had moved. What must not move is
    ABL-385's registered list, so `git log -S` on its literal text is recorded:
    the commit that introduced those twelve integers. A reviewer can check that
    without taking my word for it.
    """
    seed_literal = ", ".join(str(s) for s in ABL385_SEEDS[:6])

    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError) as exc:
            return f"unavailable: {exc}"

    return {
        "head": _git("rev-parse", "HEAD"),
        "head_subject": _git("log", "-1", "--format=%s"),
        "abl385_seed_list_frozen_by": _git(
            "log", "-1", "--format=%H %ci %s", "-S", seed_literal, "--",
            "experiments/ABL385/config.json"),
        "abl427_registration_frozen_by": _git(
            "log", "-1", "--format=%H %ci %s", "--", "experiments/ABL427/config.json"),
        "abl419_record_blob": _git("rev-parse",
                                   f"HEAD:experiments/ABL348/results_abl419_tranche2c.json"),
        "working_tree_clean": _git("status", "--porcelain") == "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replica-db", default=config.DATABASE_PATH,
                        help="read-only replica")
    parser.add_argument("--seeds", type=int, default=len(ABL385_SEEDS),
                        help="how many of ABL-385's registered seeds to use, "
                             "in registered order. The default is all twelve; a "
                             "smaller value is a smoke test and is recorded as one.")
    parser.add_argument("--countries", nargs="*", default=list(COUNTRIES))
    parser.add_argument("--out", default="reports/abl_427_tranche2c_seed_reread",
                        help="output stem; .json and .md are written")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    seeds = _load_seeds()[:args.seeds]
    k = len(seeds)
    feature_columns, live_matches_pinned = _pinned_feature_columns()
    published = _published_cells()
    replica_bytes = Path(args.replica_db).stat().st_size

    logger.info("ABL-427: %s at k=%d seeds %s", ",".join(args.countries), k, seeds)
    logger.info("replica %s (%d bytes); features pinned=%d live_matches=%s",
                args.replica_db, replica_bytes, len(feature_columns), live_matches_pinned)

    countries, run_started = {}, time.monotonic()
    for country in args.countries:
        built = build_country(country, args.replica_db, feature_columns)
        per_seed = [score_one_seed(built, seed, feature_columns) for seed in seeds]
        countries[country] = {"built": built, "per_seed": per_seed}

    record = _assemble(countries, seeds, k, feature_columns, live_matches_pinned,
                       published, args, replica_bytes, run_started)

    stem = REPO_ROOT / args.out
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(record, indent=1, default=str),
                                         encoding="utf-8")
    stem.with_suffix(".md").write_text(_markdown(record), encoding="utf-8")
    logger.info("wrote %s.json and %s.md in %.1f min", stem, stem,
                (time.monotonic() - run_started) / 60)
    return 0


def _assemble(countries: dict, seeds: tuple, k: int, feature_columns: tuple,
              live_matches_pinned: bool, published: dict, args, replica_bytes: int,
              run_started: float) -> dict:
    """Every number the verdict rests on, derived in one place."""
    from src.evaluation.gate_grading import grade_cell

    levelling = CAUSAL_LEVELLING[SOURCE_SCOPE]
    g23 = G23_READABILITY[SOURCE_SCOPE]
    cells, reproduction, constancy = [], [], []
    #: The real `CellGrade` objects, so `pair_grade` is called on the ladder's
    #: own type rather than on a stand-in that could disagree with it.
    graded_objects: dict = {}

    for country, bundle in countries.items():
        per_seed = bundle["per_seed"]
        constancy.append(_assert_references_constant(per_seed, country))

        for band in PRIMARY_BANDS:
            per_seed_wape = [c["bands"][band]["scores"]["challenger"]["wape_pct"]
                             for c in per_seed]
            daylight = [c["bands"][band].get("daylight_mae_mw") for c in per_seed]
            spread = _spread(per_seed_wape)
            spread_ex42 = (_spread(per_seed_wape[1:]) if k > 2 else None)
            daylight_spread = (_spread(daylight)
                               if all(v is not None for v in daylight) and k > 1 else None)

            mean_scores = _mean_scores(per_seed, band, k)
            d7 = comparator_wape(mean_scores, "seasonal_naive")
            challenger = mean_scores["challenger"]["wape_pct"]

            cv = spread["cv_fraction"]
            cv_upper = spread["cv_fraction_upper95"]
            floor_measured = _floor_from_cv(cv_upper, k)
            floor_point = _floor_from_cv(cv, k)
            floor_fleet = readability_floor_pct("solar", k)

            # The ladder, unmodified, under three floors. `k_eff` is the only
            # lever touched and the identity is asserted, not trusted.
            graded = {}
            for label, floor in (("measured_upper95", floor_measured),
                                 ("measured_point", floor_point),
                                 ("fleet_p90", floor_fleet)):
                # `k_eff < 1` would mean a measured floor *wider* than the
                # ladder's own k = 1 floor, which `readability_floor_pct`
                # refuses. It cannot arise at the registered k = 12 for a CV
                # anywhere near the fleet value, but a rig that crashes instead
                # of reporting on the one case that would matter most is not a
                # rig. `nan` arrives only from a `--seeds 2` smoke test.
                k_eff = _equivalent_k(floor)
                if math.isnan(k_eff) or k_eff < 1.0:
                    graded[label] = {"floor_pct": floor, "equivalent_k": k_eff,
                                     "grade": None, "label": "Not graded",
                                     "reason": "floor is nan or wider than the "
                                               "ladder's k=1 floor"}
                    continue
                assert abs(readability_floor_pct("solar", k_eff) - floor) < 1e-9, label
                grade = grade_cell(mean_scores, "solar", k=k_eff,
                                   levelling=levelling, g23_readability=g23)
                graded_objects[(country, band, label)] = grade
                graded[label] = {"floor_pct": floor, "equivalent_k": k_eff,
                                 **grade.as_dict()}

            gate = published[(country, band)]["gate"]
            cells.append({
                "country": country, "horizon_band": band, "k": k,
                "n": mean_scores["challenger"]["n"],
                "minimum_n": MINIMUM_N[band],
                "meets_minimum_n": mean_scores["challenger"]["n"] >= MINIMUM_N[band],
                "registered_minimum_n_from_abl419": gate["minimum_n"],
                "challenger_wape_pct_k_mean": challenger,
                "challenger_wape_pct_per_seed": dict(zip(map(str, seeds), per_seed_wape)),
                "seasonal_naive_wape_pct": d7,
                "skill_vs_d7_pct": skill_pct(challenger, d7),
                "margin_pct_of_own_error": margin_pct_of_own_error(challenger, d7),
                "seed_cv_wape": spread,
                "seed_cv_wape_excluding_seed42": spread_ex42,
                "seed_cv_daylight_mae": daylight_spread,
                "floor_pct": {"measured_upper95": floor_measured,
                              "measured_point": floor_point,
                              "fleet_p90": floor_fleet},
                "seeds_needed_at_measured_upper95": _seeds_needed(
                    cv_upper, skill_pct(challenger, d7)),
                "grades": graded,
                "abl419_published": {
                    "challenger_wape_pct": published[(country, band)]["scores"]
                                           ["challenger"]["wape_pct"],
                    "skill_vs_d7_pct": published[(country, band)]["grade"]
                                       ["skill_pct"]["seasonal_naive"],
                    "label": published[(country, band)]["grade"]["label"]},
            })

            # Deterministic references: recomputed vs ABL-419's committed cell.
            for name, scored in mean_scores.items():
                if name == "challenger":
                    continue
                was = (published[(country, band)]["scores"].get(name) or {}).get("wape_pct")
                now = scored.get("wape_pct")
                reproduction.append({
                    "country": country, "horizon_band": band, "comparator": name,
                    "abl419_wape_pct": was, "recomputed_wape_pct": now,
                    "delta_pp": (None if was is None or now is None else now - was),
                    "identical": (was is None and now is None) or (
                        was is not None and now is not None
                        and abs(now - was) <= REFERENCE_TOLERANCE_PP)})

    seed42 = [{"country": c["country"], "horizon_band": c["horizon_band"],
               "abl419_wape_pct": c["abl419_published"]["challenger_wape_pct"],
               "seed42_wape_pct": c["challenger_wape_pct_per_seed"][str(CONTROL_SEED)],
               "delta_pp": (c["challenger_wape_pct_per_seed"][str(CONTROL_SEED)]
                            - c["abl419_published"]["challenger_wape_pct"])}
              for c in cells]

    pairs = {}
    for country in countries:
        pairs[country] = {}
        for label in ("measured_upper95", "measured_point", "fleet_p90"):
            bands = [graded_objects[(country, band, label)] for band in PRIMARY_BANDS
                     if (country, band, label) in graded_objects]
            if len(bands) < len(PRIMARY_BANDS):
                pairs[country][label] = {"grade": None, "label": "Not graded",
                                         "detail": f"only {len(bands)}/{len(PRIMARY_BANDS)} "
                                                   f"bands graded", "failed": []}
                continue
            worst = pair_grade(bands)
            pairs[country][label] = {
                "grade": worst.grade, "label": worst.label, "detail": worst.detail,
                "failed": [{"condition": name, "reason": reason}
                           for name, reason in worst.failed]}

    return {
        "meta": {
            "issue": "ABL-427", "registration": str(REGISTRATION.relative_to(REPO_ROOT)),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "elapsed_min": (time.monotonic() - run_started) / 60.0,
            "scope": "abl427-t2c-reread", "re_read_of": SOURCE_SCOPE,
            "countries": list(countries), "k": k, "seeds": list(seeds),
            "seed_source": "experiments/ABL385/config.json -> scope.seeds",
            "is_smoke_test": k < len(ABL385_SEEDS),
            "algorithm": ALGORITHM, "training_source": SOURCE,
            "gate_basis": list(GATE_BASIS), "levelling": levelling,
            "g23_readability": g23,
            "n_features": len(feature_columns),
            "feature_columns_pinned_from": "results_abl419_tranche2c.json meta.feature_columns",
            "live_feature_columns_match_pinned": live_matches_pinned,
            "replica_db": args.replica_db, "replica_bytes": replica_bytes,
            "replica_bytes_at_abl419": 9432453120,
            "fit_window": {"start": str(FIT_START), "end_exclusive": str(GATE_START)},
            "gate_window": {"start": str(GATE_START), "end_exclusive": str(GATE_END)},
            "fleet_cv_p90_solar": STREAM_FLEET_CV_P90["solar"],
            "git": _git_provenance(),
        },
        "cells": cells,
        "pair_grades": pairs,
        "seed_42_reproduction": seed42,
        "deterministic_reference_reproduction": reproduction,
        "reference_constancy_across_seeds": constancy,
        "fit_audit": {c: {"fit_rows": int(len(b["built"]["fit"])),
                          "gate_rows_scored": int(len(b["built"]["frame"])),
                          "fit_audit": b["built"]["fit_audit"],
                          "gate_build_audit": b["built"]["gate_build_audit"],
                          "model_free_reference_mw": b["built"]["model_free_reference_mw"]}
                      for c, b in countries.items()},
    }


def _markdown(record: dict) -> str:
    """The tables a reader checks the verdict against. Prose lives in the report."""
    meta = record["meta"]
    out = [f"# ABL-427 — tranche 2c `U(+)` re-read at k = {meta['k']} seeds",
           "",
           f"Scope `{meta['scope']}`, a re-read of `{meta['re_read_of']}` and not a "
           f"re-basing of it. Registration `{meta['registration']}`, frozen before the "
           f"first fit. Generated {meta['generated_at']} in {meta['elapsed_min']:.1f} min.",
           "",
           f"Seeds (ABL-385's registered list, in registered order): "
           f"`{meta['seeds']}`.",
           ""]
    if meta["is_smoke_test"]:
        out += ["> **SMOKE TEST** — fewer than the twelve registered seeds were run. "
                "Not a verdict.", ""]

    out += ["## The read", "",
            "| pair | band | n | challenger WAPE (k-mean) | D-7 WAPE | skill vs D-7 | "
            "own-error margin | measured floor (95% upper) | fleet floor | grade |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for c in record["cells"]:
        out.append(
            f"| {c['country']} | {c['horizon_band']} | {c['n']} | "
            f"{c['challenger_wape_pct_k_mean']:.4f}% | {c['seasonal_naive_wape_pct']:.4f}% | "
            f"{c['skill_vs_d7_pct']:+.2f}% | {c['margin_pct_of_own_error']:+.2f}% | "
            f"{c['floor_pct']['measured_upper95']:.2f}% | "
            f"{c['floor_pct']['fleet_p90']:.2f}% | "
            f"**{c['grades']['measured_upper95']['label']}** |")

    out += ["", "## The measured per-fit seed CV", "",
            f"ABL-385's fleet p90 for solar is **{100 * meta['fleet_cv_p90_solar']:.4f}%**. "
            "`c_B = 0` throughout: every reference on the ladder is deterministic.", "",
            "| pair | band | c_A (WAPE) | 95% CI | sd (pp) | range (pp) | "
            "c_A excl. seed 42 | vs fleet p90 |",
            "|---|---|---:|---|---:|---:|---:|---:|"]
    for c in record["cells"]:
        s, ex = c["seed_cv_wape"], c["seed_cv_wape_excluding_seed42"]
        ratio = s["cv_fraction"] / meta["fleet_cv_p90_solar"]
        ex_cell = f"{ex['cv_pct']:.4f}%" if ex else "n/a"
        out.append(
            f"| {c['country']} | {c['horizon_band']} | {s['cv_pct']:.4f}% | "
            f"[{s['cv_pct_ci95'][0]:.3f}, {s['cv_pct_ci95'][1]:.3f}]% | {s['sd']:.4f} | "
            f"{s['range_pp']:.4f} | {ex_cell} | {ratio:.2f}x |")

    out += ["", "## Grade under each floor", "",
            "| pair | band | measured (95% upper) | measured (point) | fleet p90 |",
            "|---|---|:---:|:---:|:---:|"]
    for c in record["cells"]:
        g = c["grades"]
        out.append(f"| {c['country']} | {c['horizon_band']} | "
                   f"{g['measured_upper95']['label']} | {g['measured_point']['label']} | "
                   f"{g['fleet_p90']['label']} |")
    out += ["", "**Pair grades** (worst band, ABL-418's `pair_grade`):", ""]
    for country, letters in record["pair_grades"].items():
        out.append(f"- **{country}** — measured 95% upper: "
                   f"**{letters['measured_upper95']['detail']}**; "
                   f"measured point: {letters['measured_point']['label']}; "
                   f"fleet p90: {letters['fleet_p90']['label']}")

    out += ["", "## Reproduction controls", "",
            "### Deterministic references vs ABL-419's committed record", ""]
    moved = [r for r in record["deterministic_reference_reproduction"] if not r["identical"]]
    out.append(f"{len(record['deterministic_reference_reproduction'])} comparator cells "
               f"compared; **{len(moved)} moved**.")
    if moved:
        out += ["", "| pair | band | comparator | ABL-419 | now | Δ (pp) |",
                "|---|---|---|---:|---:|---:|"]
        for r in moved:
            out.append(f"| {r['country']} | {r['horizon_band']} | {r['comparator']} | "
                       f"{r['abl419_wape_pct']} | {r['recomputed_wape_pct']} | "
                       f"{r['delta_pp']:+.6f} |")

    out += ["", "### Seed 42 against ABL-419's published challenger", "",
            "| pair | band | ABL-419 | seed 42 here | Δ (pp) |", "|---|---|---:|---:|---:|"]
    for r in record["seed_42_reproduction"]:
        out.append(f"| {r['country']} | {r['horizon_band']} | {r['abl419_wape_pct']:.4f}% | "
                   f"{r['seed42_wape_pct']:.4f}% | {r['delta_pp']:+.4f} |")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
