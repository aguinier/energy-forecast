"""ABL-406: the margin read that qualifies tranche 2b's gate result.

The gate harness answers "did the challenger beat the registered D-7 bar".
`scripts/abl406_margin_read.py` answers the two questions that qualify that
answer -- is the comparison readable, and was the bar what established the pass
-- and neither is a gate criterion. These tests pin the arithmetic and the two
constants, because both are the kind of number that is quietly wrong in a
direction nothing else detects.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "abl406_margin_read",
    Path(__file__).parent.parent / "scripts" / "abl406_margin_read.py")
mr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mr)


def test_cv_constants_match_the_wind_stream_not_the_solar_one():
    """The CV is ABL-385's *wind* fleet p90, and the cross-check is the matched max.

    ABL-381 read its margins against a percentile taken over a different
    stream's fits, which is the specific mistake this constant exists to
    prevent. Solar's p90 is 5.43% and wind's is 3.83% -- a factor of 1.4, so
    borrowing the wrong one moves every readability verdict in the report.
    """
    assert mr.ABL385_WIND_FLEET_P90_CV == pytest.approx(0.038292934379344015)
    # BE/wind_onshore/catboost/control, the largest of the four units matching
    # this challenger's stream and algorithm (AT 1.81, DE 2.03, FR 2.50, BE 3.96).
    assert mr.ABL385_MATCHED_ONSHORE_CATBOOST_CV_MAX == pytest.approx(
        0.039642996663702794)
    # The fleet p90 must not be the solar figure, stated as its own assertion so
    # the failure names the actual confusion rather than an opaque float.
    assert mr.ABL385_WIND_FLEET_P90_CV != pytest.approx(0.05432773918515768)


def test_delta_min_against_a_deterministic_reference_drops_the_sqrt2():
    """`c_B = 0` for D-7, a flat line and a climatology -- none of them is fitted.

    The published two-arm form assumes both arms carry fit noise. Applied
    unchanged against a deterministic reference it is a factor of sqrt(2) too
    wide, which declares real margins unreadable.
    """
    cv = mr.ABL385_WIND_FLEET_P90_CV
    deterministic = mr.delta_min_pct(cv, k=1)
    stochastic = mr.delta_min_pct(cv, k=1, deterministic_reference=False)
    assert deterministic == pytest.approx(100 * 1.96 * cv)
    assert stochastic / deterministic == pytest.approx(math.sqrt(2))
    # The value the tranche is actually read at, pinned: 7.51%.
    assert deterministic == pytest.approx(7.51, abs=0.01)


def test_delta_min_falls_as_the_square_root_of_the_seed_count():
    cv = mr.ABL385_WIND_FLEET_P90_CV
    assert mr.delta_min_pct(cv, k=4) == pytest.approx(mr.delta_min_pct(cv, k=1) / 2)


def test_margin_is_a_percentage_of_the_challengers_own_error():
    """ABL-385's g: the denominator is the challenger, because that is what the
    CV is a CV *of*. Against the reference's error the margin and the threshold
    would be in different units."""
    assert mr.margin_pct(50.0, 60.0) == pytest.approx(20.0)
    assert mr.margin_pct(60.0, 50.0) == pytest.approx(-100 / 6)
    assert mr.margin_pct(50.0, 50.0) == 0.0


def test_it_margins_are_unreadable_at_both_published_cvs():
    """IT's three cells against the bar were -1.05%, -0.76%, +0.57%.

    They must read unreadable at the fleet p90 *and* at the stricter matched
    max, because that robustness is what licenses reporting IT rather than
    dispositioning it.
    """
    for cv in (mr.ABL385_WIND_FLEET_P90_CV,
               mr.ABL385_MATCHED_ONSHORE_CATBOOST_CV_MAX):
        floor = mr.delta_min_pct(cv, k=1)
        for margin in (-1.05, -0.76, 0.57):
            assert abs(margin) < floor


def test_bar_weakness_is_read_against_the_causal_reference_only():
    """An oracle reference knows the gate window, so it could not have set a bar
    in advance. Comparing the registered bar to it would be an anachronism."""
    result = {"gate_cells": [{
        "country": "PL", "horizon_band": "24-36h",
        "scores": {"seasonal_naive": {"wape_pct": 92.76},
                   "constant_causal": {"wape_pct": 61.15},
                   "constant_oracle": {"wape_pct": 51.19},
                   "climatology_causal": {"wape_pct": 59.73}},
    }]}
    row, = mr._bar_weakness(result)
    assert row["bar_weaker_than_constant"] is True
    assert row["bar_weaker_than_climatology"] is True
    assert row["constant_causal_wape"] == 61.15


def test_a_missing_comparator_is_none_not_a_crash_and_not_a_zero():
    """A comparator that never existed must not silently score as a win. The
    incumbent is Not measured in all 24 of this tranche's cells."""
    result = {"gate_cells": [{
        "country": "ES", "horizon_band": "24-36h",
        "scores": {"challenger": {"wape_pct": 54.27},
                   "seasonal_naive": {"wape_pct": 41.04},
                   "constant_causal": {"wape_pct": None}},
        "comparator_n": {"constant_causal": 0},
        "gate": {"n": 720},
    }]}
    rows = mr._cell_rows(result, mr.ABL385_WIND_FLEET_P90_CV, k=1)
    by_ref = {r["reference"]: r for r in rows}
    assert by_ref["constant_causal"]["margin_pct"] is None
    assert by_ref["constant_causal"]["readable"] is None
    # The one reference that does exist is still read, and reads as a loss.
    assert by_ref["seasonal_naive"]["margin_pct"] < 0
    assert by_ref["seasonal_naive"]["readable"] is True
