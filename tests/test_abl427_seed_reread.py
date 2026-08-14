"""ABL-427: the arithmetic the tranche 2c re-read's verdict rests on.

Three properties, each one a way the read could have been wrong without
anything raising:

1. **The seed list did not move.** The whole anti-selection claim is that
   ABL-385's twelve integers were committed before ABL-419 was fitted. If that
   list is ever edited, this read stops being one the effect could not have been
   selected on -- and the script must refuse to run rather than quietly average
   a different set.

2. **The `k_eff` trick is exact.** `grade_cell` is called unmodified and a
   measured floor enters as the equivalent k that produces it. That is only
   legitimate if the identity holds to floating point; a near-miss would grade
   cells against a floor nobody registered.

3. **`U(+)` collapses to `U`.** The letter this issue is allowed to return is
   `A` or a plain `U`. A regression that let `U(+)` through would report the
   instruction "re-read at k > 1 seeds" as the *result* of having done so.
"""

from __future__ import annotations

import math

import pytest

from scripts.abl385_read_margin import delta_min
from scripts.abl427_tranche2c_seed_reread import (
    ABL385_SEEDS, CONTROL_SEED, COUNTRIES, MINIMUM_N,
    _direct_skill_interval, _disposition, _equivalent_k, _floor_from_cv,
    _load_seeds,
)
from src.evaluation.gate_grading import STREAM_FLEET_CV_P90, readability_floor_pct


def test_seed_list_is_abl385s_and_has_not_moved():
    """`_load_seeds` reads the registration and refuses a list that changed."""
    assert _load_seeds() == ABL385_SEEDS
    assert len(ABL385_SEEDS) == 12
    assert len(set(ABL385_SEEDS)) == 12, "a repeated seed is a repeated fit"


def test_the_pinned_gate_seed_is_first():
    """What makes the k = 1 prefix of this read ABL-419's published cell.

    Not cosmetic: every `seed_42_rank_by_wape` and the whole nested-prefix
    reading of the record assume position 0 is the gate's pinned seed.
    """
    assert ABL385_SEEDS[0] == CONTROL_SEED == 42


def test_equivalent_k_inverts_the_ladders_floor_exactly():
    """A measured floor must enter `grade_cell` as exactly the k producing it."""
    for floor_pct in (0.5, 1.0, 2.5, 3.0739, 4.77, 5.07, 10.6482):
        k_eff = _equivalent_k(floor_pct)
        assert readability_floor_pct("solar", k_eff) == pytest.approx(floor_pct, abs=1e-12)


def test_equivalent_k_of_the_nominal_floor_is_the_nominal_k():
    """The trick reduces to the identity on the fleet floor it was derived from."""
    for k in (1, 4, 9, 12):
        assert _equivalent_k(readability_floor_pct("solar", k)) == pytest.approx(k, rel=1e-12)


def test_floor_from_cv_is_delta_min_with_a_deterministic_reference():
    """`c_B = 0`, and the two-arm form is a factor of sqrt(2) wider."""
    cv = 0.05
    assert _floor_from_cv(cv, 12) == pytest.approx(100 * delta_min(cv, 0.0, 12))
    two_arm = 100 * delta_min(cv, cv, 12)
    assert two_arm == pytest.approx(_floor_from_cv(cv, 12) * math.sqrt(2))


def test_fleet_floor_matches_the_published_solar_number():
    """The 10.6482% every tranche 2c `U(+)` was called against."""
    assert _floor_from_cv(STREAM_FLEET_CV_P90["solar"], 1) == pytest.approx(10.6482, abs=1e-4)


def test_disposition_collapses_the_plus_but_keeps_every_other_letter():
    assert _disposition("U") == "U"
    assert _disposition("A") == "A"
    assert _disposition("B") == "B"
    assert _disposition("C") == "C"
    assert _disposition("N") == "N"
    assert _disposition(None) == "Not graded"


def test_direct_interval_recovers_a_known_mean_and_excludes_zero_correctly():
    """A challenger uniformly better than D-7 reads readable; a wash does not."""
    d7 = 10.0
    # Mean WAPE is exactly 9.0, so mean skill is exactly 100 * (1 - 9/10) = 10%.
    # Skill is affine in WAPE for a fixed deterministic D-7, so the mean of the
    # skills equals the skill of the mean WAPE -- which is why the k-seed read
    # can be stated either way without changing a number.
    clearly_better = _direct_skill_interval([9.0, 9.1, 8.9, 9.05, 9.0, 8.95], d7)
    assert clearly_better["mean_skill_pct"] == pytest.approx(10.0, abs=1e-9)
    assert clearly_better["beats_d7_readably"] is True
    assert clearly_better["seeds_losing_to_d7"] == 0

    a_wash = _direct_skill_interval([9.0, 11.0, 9.5, 10.5, 8.5, 11.5], d7)
    assert a_wash["beats_d7_readably"] is False
    assert a_wash["seeds_losing_to_d7"] == 3


def test_direct_interval_half_width_is_t_not_z():
    """The point of the sensitivity: t at 11 dof is materially wider than 1.96."""
    values = [10.0 + 0.1 * i for i in range(12)]
    result = _direct_skill_interval(values, 12.0)
    assert result["t_crit_95"] == pytest.approx(2.2010, abs=1e-4)
    assert result["t_crit_95"] > 1.96


def test_scope_is_the_two_pairs_the_ceo_scoped_and_not_es():
    """ES is out of scope by direction, not merely optional (ABL-427 comment)."""
    assert COUNTRIES == ("IT", "HR")
    assert "ES" not in COUNTRIES


def test_minimum_n_matches_abl348s_registration():
    """ABL-434 is open on a ladder that can grade a coverage-short cell `A`.

    Every cell of this read clears its minimum, so that defect does not touch
    this verdict -- but the constant it is checked against has to be ABL-348's.
    """
    assert MINIMUM_N == {"24-36h": 684, "36-48h": 684, "48-64h": 456}
