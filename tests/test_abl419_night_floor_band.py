"""ABL-419: the night-floor band arithmetic, and ES's serving hold.

Two small things, both of which decide something a reader would otherwise have
to take on trust.

**The band.** ABL-396 section 2 bounds an all-hours WAPE from the daylight-only
one exactly, given `f` = the share of the window's |energy| booked at night.
ABL-419 needs the *inverse* -- the harness measures all-hours and the question
is what a daylight-only read of the same challenger would have been -- and an
inverted bound is exactly the sort of arithmetic that is wrong by a factor of
`(1-f)` without anything looking wrong. It is pinned here against the one case
ABL-396 worked by hand and checked against a real gate read (BG), so the test
fails if the inversion drifts rather than if someone dislikes the number.

**The hold.** ABL-419 originally capped ES at grade B whatever G1-G4 said, with
`ABL-411 hold` named as the failed condition. **That cap is withdrawn.** ABL-411
settled on 2026-08-13 (PR #56): Red Electrica's own `solFot + solTer` split
accounts for 98.55% of the MW the replica books for ES at night over 3,196 night
hours, MAE 5.55 MW against a 263.5 MW mean level, so ES's overnight output is
real generation and the condition the cap named no longer exists.

What replaces it is a *serving* hold -- `ABL-425`, the fleet-wide clamp in
`src/solar_clamp.py` that would zero ES's real 263.5 MW -- carried **beside**
the grade rather than inside it. ABL-418 writes grade A as "promotion-eligible,
subject to any named data hold", so a hold binds without bending a letter, and
the ladder keeps the property it exists for: the letter is a measurement.

These tests pin the *withdrawal*, in both directions. A cap that quietly
survives its own withdrawal would print a grade nobody measured, and a hold that
leaked into `failed_conditions` would reintroduce the same conflation through
the back door. `GRADE_SEVERITY` is still asserted -- `{"A": 0, "U": 1, "B": 2,
"C": 3}`, so `U` is *less severe* than `B` rather than alphabetically after it --
because that ordering is what made the old cap subtle, and it is now what
validates a label rather than what bends one.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.gate_grading import GRADE_SEVERITY  # noqa: E402


def _load(name: str, relative: str):
    """Imported by path: these are scripts, not package modules."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = _load("scripts_abl419_tranche2c_read", "scripts/abl419_tranche2c_read.py")


# ---------------------------------------------------------------------------
# The band
# ---------------------------------------------------------------------------

#: ABL-396 section 2's worked example, and the only one with a real gate read to
#: check against: BG's gate `f` is 4.98% at a daylight-only WAPE of 18.90%, so an
#: all-hours read must sit in [17.96%, 22.94%]. ABL-381 measured 18.89%.
BG_F_PCT = 4.98
BG_DAYLIGHT_PCT = 18.90
BG_MEASURED_ALL_HOURS_PCT = 18.89


def _forward(daylight_pct: float, f_pct: float) -> tuple[float, float]:
    """The published direction: `W(1-f)` to `W(1-f)+f`. Deliberately written out
    here rather than imported, so the inverse is checked against an independent
    statement of the bound instead of against itself."""
    f = f_pct / 100.0
    return daylight_pct * (1.0 - f), daylight_pct * (1.0 - f) + f_pct


def test_the_forward_bound_reproduces_abl396s_worked_bg_example():
    low, high = _forward(BG_DAYLIGHT_PCT, BG_F_PCT)
    assert low == pytest.approx(17.96, abs=0.005)
    assert high == pytest.approx(22.94, abs=0.005)


def test_the_bound_predicts_the_one_gate_read_that_can_check_it():
    """ABL-396's claim that the bound is not decorative. BG's measured all-hours
    read must land inside the band its daylight-only read implies."""
    low, high = _forward(BG_DAYLIGHT_PCT, BG_F_PCT)
    assert low <= BG_MEASURED_ALL_HOURS_PCT <= high


def test_the_inverse_is_the_exact_inverse_of_the_forward_bound():
    """Round-trip, which is what catches a missing `(1-f)`. A challenger sitting
    at the floor-reproducing end of the band must invert to its own daylight
    WAPE, and one at the clamped end must too."""
    low, high = _forward(BG_DAYLIGHT_PCT, BG_F_PCT)
    # All-hours == the low end  ->  the challenger reproduced the floor, so the
    # *high* end of the implied interval is the true daylight WAPE.
    assert reader.implied_daylight(low, BG_F_PCT)[1] == pytest.approx(BG_DAYLIGHT_PCT)
    # All-hours == the high end -> the challenger clamped to zero, so the *low*
    # end of the implied interval is the true daylight WAPE.
    assert reader.implied_daylight(high, BG_F_PCT)[0] == pytest.approx(BG_DAYLIGHT_PCT)


def test_the_band_width_is_f_in_wape_points():
    """The property that makes `f` rankable and quotable at all: the interval's
    width is `f`, not `f` scaled by anything."""
    low, high = _forward(BG_DAYLIGHT_PCT, BG_F_PCT)
    assert high - low == pytest.approx(BG_F_PCT)


def test_a_clean_country_has_a_band_of_zero_width():
    """GR, HR and IT screen at `f` = 0.0000% in the gate window, so the all-hours
    and daylight-only reads of any challenger coincide there. The zeros are
    stated in the pack rather than omitted, and this is why they carry no
    caveat."""
    low, high = _forward(12.34, 0.0)
    assert low == high == pytest.approx(12.34)
    assert reader.implied_daylight(12.34, 0.0) == pytest.approx((12.34, 12.34))


# ---------------------------------------------------------------------------
# The serving hold that replaced the cap
#
# ABL-419 originally capped ES at grade B with `ABL-411 hold` as a failed
# condition. ABL-411 settled (PR #56, 2026-08-13) and the cap was withdrawn: ES
# is graded exactly as G1-G4 read it, and the policy is carried beside the
# letter as a *serving* hold. These tests pin the withdrawal, because a cap that
# quietly survives its own withdrawal is the failure mode here -- it would print
# a grade nobody measured.
# ---------------------------------------------------------------------------

ALL_LABELS = ["A", "B", "U(+)", "U", "C"]


@pytest.mark.parametrize("country", ["ES", "GR", "HR", "IT", "PT"])
@pytest.mark.parametrize("ladder", ALL_LABELS)
def test_no_country_s_grade_is_modified(country, ladder):
    """The cap is gone for every country including ES. This is the assertion
    that would have failed before ABL-411 settled, and it is deliberately the
    broadest one in the file: no country, at no grade, is reported as anything
    other than what the ladder returned."""
    assert reader.reported_grade(country, ladder)[0] == ladder


@pytest.mark.parametrize("ladder", ALL_LABELS)
def test_es_carries_the_hold_at_every_grade_including_a(ladder):
    """The hold is unconditional and orthogonal to the letter. Grade A is the
    case that matters: ABL-418 writes it as "promotion-eligible, **subject to
    any named data hold**", so an A must still print its hold rather than the
    hold being what stops an A from being printed."""
    grade, hold = reader.reported_grade("ES", ladder)
    assert grade == ladder
    assert hold == reader.ES_SERVING_HOLD


def test_the_hold_prints_beside_the_grade_not_inside_it():
    """The rendering ABL-419's correction asked for, pinned literally."""
    assert reader.reported_cell("ES", "A") == "A (serving hold: ABL-425)"
    assert reader.reported_cell("ES", "U(+)") == "U(+) (serving hold: ABL-425)"
    assert reader.reported_cell("GR", "C") == "C"


def test_the_hold_is_never_a_failed_condition():
    """A serving hold and a failed G-condition answer different questions -- what
    policy blocks downstream versus what the read measured -- and merging them
    is exactly what capping did wrong. `reported_grade` returns the hold in its
    own slot; no caller may fold it into `failed_conditions`."""
    _, hold = reader.reported_grade("ES", "U(+)")
    assert hold not in GRADE_SEVERITY
    assert hold == "ABL-425"


def test_the_hold_names_the_issue_that_governs_it():
    """ABL-425, not ABL-411. ABL-411 is *settled*; what still blocks ES from
    serving is the fleet-wide clamp in `src/solar_clamp.py`, which would zero
    ES's real 263.5 MW night level."""
    assert reader.ES_SERVING_HOLD == "ABL-425"
    assert not hasattr(reader, "CAP"), "the grade cap must not come back"
    assert not hasattr(reader, "capped"), "the grade cap must not come back"


@pytest.mark.parametrize("country", ["GR", "HR", "IT", "PT"])
def test_no_other_country_carries_a_hold(country):
    assert reader.serving_hold(country) == ""


def test_a_label_the_ladder_cannot_produce_is_rejected():
    """The severity table is still consulted, for the one thing it is now for:
    a mistyped grade must not reach the table silently."""
    with pytest.raises(ValueError):
        reader.reported_grade("ES", "D")
    assert GRADE_SEVERITY["U"] < GRADE_SEVERITY["B"]
