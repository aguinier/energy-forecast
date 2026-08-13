"""ABL-376 §5: the seed-spread read, and the invariants that make it a measurement.

Two things can quietly turn this sweep into a number that means nothing, and
neither shows up in the output:

- the fit rule reaching the *scored* frame, so the challenger deletes the rows it
  is supposed to be held to account on;
- the two arms differing by anything other than the rule -- a second seed, a
  second frame -- so the paired difference stops being paired.

Both are pinned here at the call site rather than inferred from a metric.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solar_features import solar_bands  # noqa: E402
from src.solar_geometry import is_night_hour, sun_elevation_deg  # noqa: E402

SWEEP = Path(__file__).parent.parent / "scripts" / "abl376_night_seed_spread.py"

WINTER_DAY = pd.date_range("2026-01-15", periods=24, freq="h")
SUMMER_DAY = pd.date_range("2026-07-29", periods=24, freq="h")


@pytest.fixture(scope="module")
def sweep():
    spec = importlib.util.spec_from_file_location("_abl376_sweep", SWEEP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runs(control, treatment, band="daylight", metric="mae_mw", seeds=None):
    """Synthetic sweep output: one control and one treatment value per seed."""
    seeds = seeds or list(range(len(control)))
    return {"runs": [
        {"arm": arm, "seed": seed, "bands": {band: {metric: value}}}
        for arm, values in (("control", control), ("night_fit", treatment))
        for seed, value in zip(seeds, values)
    ]}


# --------------------------------------------------------------------------
# The band split
# --------------------------------------------------------------------------

def test_the_night_band_is_the_serving_clamps_predicate_not_a_second_copy():
    """If the bands and the clamp disagree about which hours are dark, both are wrong."""
    for country in ("FR", "DE", "BE", "AT"):
        for day in (WINTER_DAY, SUMMER_DAY):
            bands = solar_bands(country, day).to_numpy()
            clamp_view = np.asarray(is_night_hour(country, day), dtype=bool)
            np.testing.assert_array_equal(bands == "night", clamp_view)


def test_the_three_bands_partition_every_hour():
    for country in ("FR", "DE"):
        bands = solar_bands(country, WINTER_DAY).to_numpy()
        assert set(bands) <= {"daylight", "shoulder", "night"}
        assert len(bands) == len(WINTER_DAY)


def test_shoulder_is_below_the_horizon_and_daylight_is_above_it():
    """The shoulder is ABL-337's blind spot: not dark enough to zero, sun still down."""
    for country in ("FR", "DE"):
        bands = solar_bands(country, SUMMER_DAY).to_numpy()
        midpoints = pd.DatetimeIndex(SUMMER_DAY) + pd.Timedelta(minutes=30)
        elevation = np.asarray(sun_elevation_deg(country, midpoints), dtype=float)
        assert (elevation[bands == "shoulder"] <= 0.0).all()
        assert (elevation[bands == "daylight"] > 0.0).all()


def test_empty_input_returns_an_empty_series_rather_than_raising():
    assert len(solar_bands("FR", [])) == 0


# --------------------------------------------------------------------------
# The sweep's protocol
# --------------------------------------------------------------------------

def test_the_registered_seeds_are_distinct_and_disjoint_from_the_gates(sweep):
    """A spread anchored on the seed that produced the headline is not a spread."""
    assert len(set(sweep.SEEDS)) == len(sweep.SEEDS)
    assert 42 not in sweep.SEEDS, "seed 42 is the gate's own read in §4"
    assert len(sweep.SEEDS) >= 4, "too few seeds to quote a null from"


def test_the_two_arms_differ_in_the_rule_and_nothing_else(sweep):
    assert sweep.ARMS == {"control": False, "night_fit": True}


def test_only_the_seed_varies_between_fits(sweep):
    """`_fit_predict` must take the gate's own configuration and change one key.

    An arm that also picked up a different depth or iteration count would still
    produce a difference, and the report would still call it the fit rule.
    """
    tree = ast.parse(SWEEP.read_text(encoding="utf-8"))
    fit = next(node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == "_fit_predict")
    assigned = {ast.unparse(target) for node in ast.walk(fit)
                if isinstance(node, ast.Assign) for target in node.targets}
    # `ast.unparse` normalises string literals to single quotes.
    assert "params['random_seed']" in assigned, "the seed must be set on the params"
    assert not (assigned - {"params", "params['random_seed']", "model"}), (
        f"only the seed may vary between fits; also assigned: {assigned}"
    )


def test_the_rule_is_applied_to_the_fit_frame_and_never_to_the_scored_one(sweep):
    """AST, not text: a filtered scoring frame renders every number identically.

    The night figure would then measure the filter instead of the model, and no
    output in the pack would show it.
    """
    tree = ast.parse(SWEEP.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "exclude_impossible_night_rows"]

    assert len(calls) == 1, "expected exactly one exclusion site in the sweep"
    first_arg = calls[0].args[0]
    assert isinstance(first_arg, ast.Name) and first_arg.id == "fit_all", (
        "the exclusion must build the treatment fit frame from the full fit frame; "
        "applying it to the gate frame would delete the challenger's own scoring rows"
    )


def test_both_arms_are_scored_on_one_shared_frame(sweep):
    """One `gate_x`/`actual`/`bands` per country, built before the arm loop.

    Rebuilding the scored frame inside the loop is how an arm ends up scored on
    its own rows -- the failure this whole design exists to exclude.
    """
    tree = ast.parse(SWEEP.read_text(encoding="utf-8"))
    country = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "sweep_country")
    loops = [node for node in ast.walk(country) if isinstance(node, ast.For)]
    assigned_in_loops = {ast.unparse(target) for loop in loops
                         for node in ast.walk(loop) if isinstance(node, ast.Assign)
                         for target in node.targets}
    assert not assigned_in_loops & {"gate_x", "actual", "bands", "selected", "frames"}, (
        f"the scored frame must be built once per country, not per fit: {assigned_in_loops}"
    )


# --------------------------------------------------------------------------
# The arithmetic the verdict is read off
# --------------------------------------------------------------------------

def test_the_difference_is_taken_within_a_seed(sweep):
    """Pairing is the point: across-seed variance must cancel, not average in."""
    # Control swings wildly across seeds; the rule costs exactly +2 at every one.
    control = [100.0, 300.0, 200.0, 400.0]
    treatment = [value + 2.0 for value in control]
    paired = sweep._paired(_runs(control, treatment), "daylight", "mae_mw")

    assert paired["paired_mean"] == pytest.approx(2.0)
    assert paired["paired_sd"] == pytest.approx(0.0)
    assert paired["seeds_improved"] == 0
    # ...while the unpaired spread of the same data is two orders larger.
    assert paired["null_max"] == pytest.approx(300.0)


def test_the_null_is_every_control_pair_and_is_scaled_by_the_control_mean(sweep):
    control = [90.0, 100.0, 110.0]
    paired = sweep._paired(_runs(control, control), "daylight", "mae_mw")

    assert paired["null_pairs"] == 3
    assert paired["null_max"] == pytest.approx(20.0)
    assert paired["null_max_pct"] == pytest.approx(20.0)
    assert paired["control_mean"] == pytest.approx(100.0)


def test_seeds_improved_counts_the_direction_that_helps(sweep):
    paired = sweep._paired(_runs([10.0, 10.0, 10.0], [9.0, 11.0, 8.0]),
                           "daylight", "mae_mw")
    assert paired["seeds_improved"] == 2
    assert paired["n_seeds"] == 3


def test_a_single_seed_reports_no_null_rather_than_a_zero_one(sweep):
    """A zero null would read as 'no seed sensitivity here', which is the opposite."""
    paired = sweep._paired(_runs([100.0], [98.0]), "daylight", "mae_mw")

    assert paired["null_pairs"] == 0
    assert paired["null_max"] is None
    assert paired["null_max_pct"] is None
    assert paired["control_sd"] is None
    assert paired["paired_mean"] == pytest.approx(-2.0)


def test_a_probe_run_renders_its_missing_null_instead_of_raising(sweep):
    """The degenerate case must still produce a readable report."""
    payload = {
        "meta": {"generated_at": "2026-08-13 00:00 UTC", "seeds": [1],
                 "replica_db": "x", "replica_bytes": 1, "training_source": "energy_renewable",
                 "night_threshold_deg": -8.0, "impossible_night_threshold_mw": 1.0,
                 "fit_window": {"start": "a", "end_exclusive": "b"},
                 "gate_window": {"start": "c", "end_exclusive": "d"}},
        "countries": [{
            "country": "BE",
            "scored_band_n": {"daylight": 90, "shoulder": 17, "night": 23},
            "night_fit_audit": {"night_rows": 280, "excluded_rows": 0,
                                "excluded_targets": 0, "max_excluded_mw": None},
            "runs": [{"arm": arm, "seed": 1,
                      "bands": {band: {"mae_mw": 1.0, "mean_pred_mw": 1.0, "max_pred_mw": 1.0}
                                for band in ("daylight", "shoulder", "night")}}
                     for arm in ("control", "night_fit")],
        }],
    }
    payload["summary"] = sweep.summarise(payload)
    rendered = sweep._render_markdown(payload)

    assert "n/a" in rendered
    assert "no null (single seed)" in rendered


def test_a_clean_country_reports_no_excluded_rows_rather_than_an_empty_cell(sweep):
    payload = {
        "meta": {"generated_at": "2026-08-13 00:00 UTC", "seeds": [1, 2],
                 "replica_db": "x", "replica_bytes": 1, "training_source": "energy_renewable",
                 "night_threshold_deg": -8.0, "impossible_night_threshold_mw": 1.0,
                 "fit_window": {"start": "a", "end_exclusive": "b"},
                 "gate_window": {"start": "c", "end_exclusive": "d"}},
        "countries": [{
            "country": "BE",
            "scored_band_n": {"daylight": 90, "shoulder": 17, "night": 23},
            "night_fit_audit": {"night_rows": 280, "excluded_rows": 0,
                                "excluded_targets": 0, "max_excluded_mw": None},
            "runs": [{"arm": arm, "seed": seed,
                      "bands": {band: {"mae_mw": 1.0, "mean_pred_mw": 1.0, "max_pred_mw": 1.0}
                                for band in ("daylight", "shoulder", "night")}}
                     for arm in ("control", "night_fit") for seed in (1, 2)],
        }],
    }
    payload["summary"] = sweep.summarise(payload)
    rendered = sweep._render_markdown(payload)

    assert "| BE | 280 | 0 | 0 | n/a |" in rendered
