"""ABL-403: the 2x2 statistics, checked on frames no fit is needed to produce.

The probe's contrast machinery is where a 2x2 goes wrong silently -- an
interaction with its sign flipped, or a null built from the wrong arm, reads as
a finished number either way. These check it against cases whose answer is known
by construction, so the 64-fit run does not have to be trusted to also be its
own test.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.abl403_night_rule_interaction_probe import (  # noqa: E402
    ARMS, CONTROL_ARM, FEATURE_COLUMNS_25, SWEEP_SEEDS, _contrast, _effects,
    _interaction, _nulls, _sign_test_p,
)
from src.evaluation.solar_retrain import FEATURE_COLUMNS  # noqa: E402
from src.solar_features import SOLAR_GEOMETRY_FEATURES  # noqa: E402


SEEDS = list(SWEEP_SEEDS)


def _run(arm, seed, night_negative=50.0, night_mae=10.0, wape=20.0):
    """One synthetic run record, shaped exactly as `probe` emits."""
    return {
        "arm": arm, "seed": seed,
        "night": {
            "pct_of_night_rows_negative": night_negative,
            "mean_prediction_at_night_mw": 1.0,
            "night_mae_mw": night_mae,
            "night_bias_mw": 0.5,
            "night_wape_pct": 30.0,
        },
        "cells": [
            {"horizon_band": band, "challenger_wape_pct": wape,
             "daylight_challenger_wape_pct": wape}
            for band in ("24-36h", "36-48h", "48-64h")
        ],
    }


# --------------------------------------------------------------------------
# The arms are the 2x2 the issue registered, and the control carries neither.
# --------------------------------------------------------------------------

def test_arms_are_the_registered_two_by_two():
    assert set(ARMS) == {"f25_off", "f27_off", "f25_on", "f27_on"}
    for name, (columns, exclude) in ARMS.items():
        expected_n = 25 if name.startswith("f25") else 27
        assert len(columns) == expected_n, name
        assert exclude is name.endswith("_on"), name


def test_control_arm_carries_neither_change():
    columns, exclude = ARMS[CONTROL_ARM]
    assert exclude is False
    assert not set(SOLAR_GEOMETRY_FEATURES) & set(columns)


def test_the_25_are_the_live_list_minus_the_geometry_pair():
    """Derived by subtraction, so the control cannot drift on anything else."""
    assert set(FEATURE_COLUMNS) - set(FEATURE_COLUMNS_25) == set(SOLAR_GEOMETRY_FEATURES)
    assert len(FEATURE_COLUMNS_25) == len(FEATURE_COLUMNS) - len(SOLAR_GEOMETRY_FEATURES)


# --------------------------------------------------------------------------
# Sign test.
# --------------------------------------------------------------------------

def test_sign_test_matches_the_values_this_repo_quotes():
    """8/8 -> 0.0078 and 6/8 -> 0.29 are quoted in CLAUDE.md from ABL-395."""
    p, negative, positive, tied = _sign_test_p(np.full(8, -1.0))
    assert (negative, positive, tied) == (8, 0, 0)
    assert p == pytest.approx(0.0078125, abs=1e-6)
    p, _, _, _ = _sign_test_p(np.array([-1.0] * 6 + [1.0] * 2))
    assert p == pytest.approx(0.2890625, abs=1e-6)


def test_sign_test_drops_ties_rather_than_counting_them():
    """A metric an arm could not move must not be scored as evidence for it."""
    p, negative, positive, tied = _sign_test_p(np.array([-1.0, -1.0, 0.0, 0.0]))
    assert (negative, positive, tied) == (2, 0, 2)
    assert p == pytest.approx(0.5, abs=1e-9)  # 2 of 2, not 2 of 4


def test_sign_test_returns_none_when_every_difference_is_zero():
    p, negative, positive, tied = _sign_test_p(np.zeros(8))
    assert p is None and (negative, positive, tied) == (0, 0, 8)


# --------------------------------------------------------------------------
# Contrast and interaction.
# --------------------------------------------------------------------------

def test_contrast_is_treatment_minus_control():
    values = {("f27_off", s): 12.0 for s in SEEDS}
    values.update({("f25_off", s): 10.0 for s in SEEDS})
    out = _contrast(values, "m", SEEDS, "f27_off", "f25_off", np.array([0.5]))
    assert out["paired_mean"] == pytest.approx(2.0)
    assert out["control_mean"] == pytest.approx(10.0)
    assert out["treatment_mean"] == pytest.approx(12.0)
    assert out["seeds_up"] == 8 and out["seeds_down"] == 0
    assert out["outside_the_null"] is True


def test_contrast_reports_an_effect_inside_its_null_as_not_readable():
    values = {("f27_off", s): 10.1 for s in SEEDS}
    values.update({("f25_off", s): 10.0 for s in SEEDS})
    out = _contrast(values, "m", SEEDS, "f27_off", "f25_off", np.array([5.0]))
    assert out["paired_mean"] == pytest.approx(0.1)
    assert out["outside_the_null"] is False


def test_interaction_is_the_difference_of_differences_with_the_stated_sign():
    """The rule buys 5 with geometry and 1 without -> interaction +4."""
    values = {}
    for s in SEEDS:
        values[("f25_off", s)] = 10.0
        values[("f25_on", s)] = 11.0    # rule alone: +1
        values[("f27_off", s)] = 20.0
        values[("f27_on", s)] = 25.0    # rule with geometry: +5
    out = _interaction(values, "m", SEEDS, np.array([1.0]))
    assert out["exclusion_effect_without_geometry_mean"] == pytest.approx(1.0)
    assert out["exclusion_effect_with_geometry_mean"] == pytest.approx(5.0)
    assert out["interaction_mean"] == pytest.approx(4.0)
    assert out["seeds_up"] == 8 and out["sign_test_p"] == pytest.approx(0.0078125, abs=1e-6)


def test_interaction_is_exactly_zero_when_the_rule_does_the_same_thing_in_both():
    """Additive arms have no interaction, whatever the two main effects are."""
    values = {}
    for s in SEEDS:
        values[("f25_off", s)] = 10.0
        values[("f25_on", s)] = 13.0
        values[("f27_off", s)] = 40.0
        values[("f27_on", s)] = 43.0
    out = _interaction(values, "m", SEEDS, np.array([1.0]))
    assert out["interaction_mean"] == pytest.approx(0.0)
    assert out["sign_test_p"] is None and out["seeds_tied"] == 8


# --------------------------------------------------------------------------
# Nulls.
# --------------------------------------------------------------------------

def test_nulls_come_from_the_control_arm_and_size_with_the_seed_count():
    control = np.arange(8, dtype=float)
    pair, quad = _nulls(control)
    assert pair.size == 28            # C(8,2)
    assert quad.size == 1680          # P(8,4)
    assert pair.max() == pytest.approx(7.0)
    # The four seeds are distinct, so the extreme is (7-0) - (1-6), not (7-0) - (0-7).
    assert quad.max() == pytest.approx(12.0)


def test_the_four_fit_null_is_wider_than_the_two_fit_one():
    """An interaction combines four fits, so it must clear a wider bar."""
    pair, quad = _nulls(np.array([0.0, 1.0, 3.0, 7.0, 2.0, 5.0, 4.0, 6.0]))
    assert quad.max() > pair.max()


def test_nulls_are_empty_rather_than_wrong_below_four_seeds():
    pair, quad = _nulls(np.array([1.0, 2.0]))
    assert pair.size == 1 and quad.size == 0
    out = _interaction({("f25_off", 1): 1.0, ("f25_on", 1): 1.0,
                        ("f27_off", 1): 1.0, ("f27_on", 1): 1.0},
                       "m", [1], quad)
    assert out["null_max"] is None and out["outside_the_null"] is None


# --------------------------------------------------------------------------
# End to end over the run records `probe` emits.
# --------------------------------------------------------------------------

def test_effects_covers_every_metric_and_names_the_control_arm():
    runs = [_run(arm, seed) for arm in ARMS for seed in SEEDS]
    effects = _effects(runs)
    assert "pct_of_night_rows_negative" in effects
    assert "night_mae_mw" in effects
    assert "24-36h|challenger_wape_pct" in effects
    assert "48-64h|daylight_challenger_wape_pct" in effects
    for block in effects.values():
        if not block["measured"]:
            continue
        assert block["geometry_rule_off"]["control_arm"] == CONTROL_ARM
        assert block["exclusion_at_f25"]["control_arm"] == CONTROL_ARM
        assert set(block["arm_means"]) == set(ARMS)


def test_effects_marks_a_metric_unmeasurable_rather_than_inventing_a_number():
    """CH's night actuals are 0.00 MW, so night WAPE has no denominator."""
    runs = [_run(arm, seed) for arm in ARMS for seed in SEEDS]
    for run in runs:
        run["night"]["night_wape_pct"] = None
    effects = _effects(runs)
    assert effects["night_wape_pct"]["measured"] is False
    assert effects["night_mae_mw"]["measured"] is True


def test_effects_recovers_a_planted_interaction_end_to_end():
    """Only the both-changes cell moves; the two simple effects must show it."""
    level = {"f25_off": 60.0, "f25_on": 60.0, "f27_off": 60.0, "f27_on": 40.0}
    runs = [_run(arm, seed, night_negative=level[arm] + (seed % 7))
            for arm in ARMS for seed in SEEDS]
    block = _effects(runs)["pct_of_night_rows_negative"]
    assert block["exclusion_at_f25"]["paired_mean"] == pytest.approx(0.0)
    assert block["exclusion_at_f27"]["paired_mean"] == pytest.approx(-20.0)
    assert block["interaction"]["interaction_mean"] == pytest.approx(-20.0)
    assert block["interaction"]["seeds_down"] == 8
    # The seed jitter is common to every arm, so it cancels inside each paired
    # difference and must not reach the effect.
    assert block["geometry_rule_off"]["paired_mean"] == pytest.approx(0.0)
