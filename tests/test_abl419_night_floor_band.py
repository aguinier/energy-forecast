"""ABL-419: the night-floor band arithmetic, and the ES grade cap.

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

**The cap.** ABL-419 caps ES at grade B whatever G1-G4 say, and "higher" here is
ABL-418's *severity* ordering rather than the alphabet:
`GRADE_SEVERITY = {"A": 0, "U": 1, "B": 2, "C": 3}`, so `U` and `U(+)` are
**less severe than `B`** and the cap pulls them down to it. That is not a
hypothetical -- ES's ladder grade on this read is `U(+)`, so a reading that took
`U(+)` for "already below B" would silently not apply the cap on the one read it
was written for. Pinned in both directions, because a cap that could *raise* a
grade would be an upgrade wearing a cap's clothing: a `C` cell reported as `B`
is strictly worse than no cap at all.
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
# The cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ladder,expected", [("A", "B"), ("B", "B"), ("U(+)", "B"),
                                             ("U", "B"), ("C", "C")])
def test_the_es_cap_only_ever_moves_a_grade_down(ladder, expected):
    reported, hold = reader.capped("ES", ladder)
    assert reported == expected
    assert hold == reader.ES_HOLD, "the cap must always name its failed condition"


def test_higher_means_less_severe_and_not_alphabetical():
    """The trap this file exists for, stated as an assertion rather than a
    comment. ABL-418 orders `C > B > U > A` by severity, so `U` and `U(+)` are
    *higher* than the cap and must be pulled down to it. A reading that took
    `U` for "already below B" would leave ES ungraded by the cap on exactly the
    read where it binds -- ES's ladder grade here is `U(+)`."""
    assert GRADE_SEVERITY["U"] < GRADE_SEVERITY["B"]
    assert reader.capped("ES", "U(+)")[0] == "B"


def test_the_cap_never_raises_a_worse_grade():
    """`C` is more severe than the cap, so it survives it. The cap is a ceiling,
    never a floor."""
    assert reader.capped("ES", "C")[0] == "C"
    assert GRADE_SEVERITY["C"] > GRADE_SEVERITY[reader.CAP]


@pytest.mark.parametrize("country", ["GR", "HR", "IT", "PT"])
@pytest.mark.parametrize("ladder", ["A", "B", "U(+)", "U", "C"])
def test_the_cap_touches_no_other_country(country, ladder):
    reported, hold = reader.capped(country, ladder)
    assert reported == ladder
    assert hold == ""


def test_the_hold_is_named_and_is_the_issue_that_governs_it():
    """The cap's whole point is that a reader can tell it from a measurement.
    An unnamed cap is indistinguishable from the ladder having produced a B."""
    assert reader.ES_HOLD == "ABL-411 hold"
