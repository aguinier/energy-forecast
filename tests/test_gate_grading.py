"""ABL-418: the graded gate disposition is a ladder, and it grades what it says.

The registered bar is not re-opened here. Seasonal-naive D-7 stays the gate, and
a cell that clears it still reads PASS -- what the grade adds is what that PASS
entitles the cell to. ABL-406 measured, across eight wind pairs, that the gate
outcome was fully predicted by whether a causal constant clears the bar on its
own (five weak bars, five passes; three strong bars, three failures or ties, no
exceptions), and that NO passed 3/3 while anti-correlated with its own target.
So a PASS is necessary and not sufficient.

Four properties are load-bearing and all four are pinned here.

**The floor is derived, not asserted.** It is ABL-385's ``delta_min`` with
``c_B = 0`` -- correct because every reference on this ladder is deterministic --
and the per-stream CVs it is built from are checked against
``reports/abl_385_decision_margin.json`` itself rather than against a number
retyped from a report. Retyping is how ABL-381 came to quote a different stream's
margins.

**``U`` outranks ``C``.** Both mean "G1 does not hold", but an unreadable margin
and a measured loss are different statements, and reporting the first as the
second invites the feature work ABL-378's ``UNREADABLE`` verdict exists to
prevent. The boundary case -- a *negative* margin sitting inside the floor -- is
exercised directly, because that is where the two rules meet and it is exactly
where IT `wind_onshore` sits.

**The two named retro-grades hold.** ABL-418 asks specifically that NO grades
``B`` and HU grades ``U(+)``. Both are checked against the published results
files rather than against a fixture, so the test fails if either the ladder or
the stored evidence moves. The rest of both tranches is pinned beside them, which
is what makes this a regression test on 48 real cells and not on two.

**There is one implementation.** Both harnesses import the ladder and neither
recomputes it -- the same rule ``model_free_reference.py`` is held to. A second
copy would drift silently, since nothing in a gate report says which code decided
a grade.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from src.evaluation.gate_grading import (
    GRADE_SEVERITY, PUBLISHED_FLOOR_PCT_K1, STREAM_FLEET_CV_P90, attach_grades,
    cell_grade, grade_cell, margin_pct_of_own_error, pair_grade,
    readability_floor_pct, skill_pct,
)

REPO = Path(__file__).resolve().parents[1]
HARNESSES = {"solar": REPO / "scripts" / "evaluate_solar_retrain.py",
             "wind": REPO / "scripts" / "evaluate_wind_retrain.py"}
TRANCHES = {"1b": (REPO / "experiments" / "ABL348" / "results_abl381_tranche1b.json", "solar"),
            "2a": (REPO / "experiments" / "ABL348" / "results_abl405_tranche2a.json", "solar"),
            "2b": (REPO / "experiments" / "ABL348" / "results_abl406_tranche2b.json", "wind")}


def scores(challenger, naive, constant, climatology, slope=0.8, correlation=0.9):
    """One cell's scores in the shape the harnesses write to ``results.json``."""
    # A comparator that scored nothing is written as a record with a null score,
    # not omitted -- that is the shape both harnesses produce.
    def entry(wape):
        return {"wape_pct": wape, "n": 0 if wape is None else 720}
    return {"challenger": {"wape_pct": challenger, "n": 720, "slope": slope,
                           "correlation": correlation},
            "seasonal_naive": entry(naive), "constant_causal": entry(constant),
            "climatology_causal": entry(climatology)}


def pair_label(cell):
    return f"{cell['country']} {cell['forecast_type']}" if "forecast_type" in cell else cell["country"]


def graded_pairs(tranche):
    """Every pair's grade label in one stored tranche, from the published file."""
    path, stream = TRANCHES[tranche]
    cells = json.loads(path.read_text(encoding="utf-8"))["gate_cells"]
    pairs = {}
    for cell in cells:
        pairs.setdefault(pair_label(cell), []).append(grade_cell(cell["scores"], stream))
    return {label: pair_grade(grades).label for label, grades in pairs.items()}


# --------------------------------------------------------------------------
# The floor


@pytest.mark.parametrize("stream", ["solar", "wind"])
def test_stream_cv_matches_the_abl385_machine_record(stream):
    """The CV is ABL-385's, read from its own JSON rather than retyped."""
    registered = json.loads((REPO / "reports" / "abl_385_decision_margin.json")
                            .read_text(encoding="utf-8"))["fleet_margin"][stream]
    assert STREAM_FLEET_CV_P90[stream] == registered["cv_rms_p90"]
    # ABL-385 publishes the two-arm form. The ladder's references are all
    # deterministic, so `c_B = 0` and the floor is that value over sqrt(2).
    assert readability_floor_pct(stream) == pytest.approx(
        registered["delta_min_pct_at_p90"]["1"] / math.sqrt(2), rel=1e-12)


@pytest.mark.parametrize("stream", ["solar", "wind"])
def test_published_floor_is_the_same_number_to_two_places(stream):
    """10.64% and 7.51% are renderings of the derived value, not a second number."""
    assert abs(readability_floor_pct(stream) - PUBLISHED_FLOOR_PCT_K1[stream]) < 0.01


def test_the_two_streams_do_not_share_a_floor():
    """ABL-381 read its margins against another stream's fits. Not repeatable here."""
    assert readability_floor_pct("solar") > readability_floor_pct("wind")


def test_the_floor_shrinks_as_the_square_root_of_the_seed_count():
    assert readability_floor_pct("wind", 4) == pytest.approx(readability_floor_pct("wind") / 2)
    with pytest.raises(ValueError):
        readability_floor_pct("wind", 0)


def test_the_two_denominators_agree_in_sign_and_differ_in_magnitude():
    """Why a denominator can only decide a cell that sits near the floor."""
    for challenger, reference in ((10.0, 12.0), (12.0, 10.0), (50.0, 50.0)):
        skill = skill_pct(challenger, reference)
        own = margin_pct_of_own_error(challenger, reference)
        assert (skill > 0) == (own > 0) and (skill < 0) == (own < 0)


# --------------------------------------------------------------------------
# The ladder


def test_grade_a_needs_all_four_conditions():
    grade = grade_cell(scores(10.0, 20.0, 60.0, 30.0), "wind")
    assert grade.label == "A" and grade.failed == ()
    assert grade.conditions == {"G1": True, "G2": True, "G3": True, "G4": True}


@pytest.mark.parametrize("kwargs,expected", [
    ({"constant": 9.0}, "G2"),
    ({"climatology": 9.0}, "G3"),
    ({"slope": -0.08, "correlation": -0.14}, "G4"),
])
def test_grade_b_holds_g1_and_names_what_failed(kwargs, expected):
    """B is a *measured* failure, so the report must say which one."""
    base = {"challenger": 10.0, "naive": 20.0, "constant": 60.0, "climatology": 30.0}
    grade = grade_cell(scores(**{**base, **kwargs}), "wind")
    assert grade.label == "B"
    assert [name for name, _ in grade.failed] == [expected]
    assert expected in grade.detail


def test_grade_c_is_a_readable_loss_to_the_registered_bar():
    grade = grade_cell(scores(20.0, 10.0, 60.0, 30.0), "wind")
    assert grade.label == "C" and grade.failed[0][0] == "G1"


def test_an_unreadable_margin_is_u_and_never_c():
    """The boundary the two rules meet at -- and where IT wind_onshore sits.

    A margin of -1% is a loss, and calling it one would be reporting an absence
    of measurement as a measured failure.
    """
    floor = readability_floor_pct("wind")
    losing = grade_cell(scores(10.0, 10.0 * (1 - 0.5 * floor / 100), 60.0, 30.0), "wind")
    assert losing.skill["seasonal_naive"] < 0
    assert losing.grade == "U"
    # Just outside it, the same sign is a C.
    clear = grade_cell(scores(10.0, 10.0 * (1 - 2 * floor / 100), 60.0, 30.0), "wind")
    assert clear.grade == "C"


def test_u_plus_needs_g2_and_g3_readable_and_g4():
    """`U(+)` says re-read at k>1 seeds; plain `U` does not."""
    floor = readability_floor_pct("wind")
    inside = 10.0 / (1 - 0.5 * floor / 100)
    assert grade_cell(scores(10.0, inside, 60.0, 30.0), "wind").label == "U(+)"
    # G2 clears, but by less than the floor: not readable, so no plus.
    barely = 10.0 * (1 + 0.5 * floor / 100)
    assert grade_cell(scores(10.0, inside, barely, 30.0), "wind").label == "U"
    # G4 is a sign test, so there is no margin to read -- it enters as-is.
    assert grade_cell(scores(10.0, inside, 60.0, 30.0, slope=-0.1,
                             correlation=-0.2), "wind").label == "U"


def test_a_condition_that_was_not_measured_is_not_satisfied():
    """The net-position gate's INCOMPLETE rule, one level down."""
    grade = grade_cell(scores(10.0, 20.0, None, 30.0), "wind")
    assert grade.label == "B" and grade.conditions["G2"] is None
    assert "not measured" in dict(grade.failed)["G2"]


def test_a_cell_that_scored_nothing_has_no_grade_and_is_not_a_failure():
    """Nothing lost a race here, so it is not a C. ABL-378's rule, one level down."""
    grade = grade_cell(scores(None, None, None, None), "wind")
    assert grade.grade is None and grade.label == "Not measured"
    # A record predating ABL-389 omits the reference keys outright rather than
    # carrying a null score. Both mean "not measured" and neither may raise.
    assert grade_cell({"challenger": {"wape_pct": 10.0, "slope": 0.8, "correlation": 0.9}},
                      "wind").grade is None


# --------------------------------------------------------------------------
# Pairs


def test_a_pair_takes_the_worst_of_its_bands():
    good = grade_cell(scores(10.0, 20.0, 60.0, 30.0), "wind")
    weak = grade_cell(scores(10.0, 20.0, 60.0, 30.0, slope=-0.1, correlation=-0.2), "wind")
    lost = grade_cell(scores(20.0, 10.0, 60.0, 30.0), "wind")
    assert pair_grade([good, good, good]).label == "A"
    assert pair_grade([good, weak, good]).label == "B"
    assert pair_grade([good, weak, lost]).label == "C"
    assert GRADE_SEVERITY["C"] > GRADE_SEVERITY["B"] > GRADE_SEVERITY["U"] > GRADE_SEVERITY["A"]


def test_a_pair_is_u_plus_only_if_every_unreadable_band_is():
    floor = readability_floor_pct("wind")
    inside = 10.0 / (1 - 0.5 * floor / 100)
    good = grade_cell(scores(10.0, 20.0, 60.0, 30.0), "wind")
    plus = grade_cell(scores(10.0, inside, 60.0, 30.0), "wind")
    plain = grade_cell(scores(10.0, inside, 60.0, 30.0, slope=-0.1, correlation=-0.2), "wind")
    assert pair_grade([good, plus, plus]).label == "U(+)"
    assert pair_grade([good, plus, plain]).label == "U"


def test_a_pair_with_nothing_measured_has_no_grade():
    assert pair_grade([grade_cell(scores(None, None, None, None), "wind")]).grade is None


# --------------------------------------------------------------------------
# The retro-grade, against the published evidence


def test_no_grades_b_and_it_is_the_only_2b_disagreement():
    """The grade ABL-418 names: NO clears D-7, the constant *and* the
    climatology, and carries no directional information at all."""
    pairs = graded_pairs("2b")
    assert pairs["NO wind_onshore"] == "B"
    assert pairs == {"ES wind_onshore": "C", "FI wind_onshore": "A",
                     "GR wind_onshore": "A", "IT wind_onshore": "U(+)",
                     "NO wind_onshore": "B", "PL wind_onshore": "A",
                     "PT wind_onshore": "C", "SE wind_onshore": "A"}


def test_hu_grades_u_plus():
    """The other grade ABL-418 names. HU's skill vs D-7 is 4.6/4.6/7.6% against
    a 10.65% floor, so it is unreadable at one seed -- but it clears the causal
    constant and the causal climatology readably, so it is re-read, not
    rejected."""
    pairs = graded_pairs("2a")
    assert pairs["HU"] == "U(+)"
    assert pairs == {"BG": "A", "CH": "A", "CZ": "A", "HU": "U(+)", "PL": "A",
                     "RO": "A", "SI": "A", "SK": "A"}


def test_no_is_anti_correlated_in_every_band_which_is_what_g4_reads():
    path, stream = TRANCHES["2b"]
    cells = [cell for cell in json.loads(path.read_text(encoding="utf-8"))["gate_cells"]
             if cell["country"] == "NO"]
    assert len(cells) == 3
    for cell in cells:
        assert cell["gate"]["pass"] is True          # it clears the registered bar
        assert cell["scores"]["challenger"]["slope"] < 0
        assert cell["scores"]["challenger"]["correlation"] < 0
        assert grade_cell(cell["scores"], stream).conditions["G4"] is False


@pytest.mark.parametrize("tranche", sorted(TRANCHES))
def test_no_cell_changes_grade_on_the_other_denominator(tranche):
    """ABL-418 registers G1 on the printed skill column; ABL-406 quoted margins
    on the challenger's own error. Neither decides anything on these 54 cells
    (48 from ABL-418's two tranches, 6 from 1b's retro-grade on ABL-438), and the
    published 2-dp floors do not either."""
    path, stream = TRANCHES[tranche]
    floor, published = readability_floor_pct(stream), PUBLISHED_FLOOR_PCT_K1[stream]
    for cell in json.loads(path.read_text(encoding="utf-8"))["gate_cells"]:
        challenger = cell["scores"]["challenger"]["wape_pct"]
        bar = cell["scores"]["seasonal_naive"]["wape_pct"]
        skill, own = skill_pct(challenger, bar), margin_pct_of_own_error(challenger, bar)
        def band(value):
            return "U" if abs(value) <= floor else "pass" if value > floor else "C"
        assert band(skill) == band(own), (cell["country"], cell["horizon_band"])
        assert not min(published, floor) < abs(skill) < max(published, floor)


# --------------------------------------------------------------------------
# One implementation, wired into both harnesses, gating nothing


@pytest.mark.parametrize("stream,path", sorted(HARNESSES.items()))
def test_each_harness_grades_against_its_own_stream(stream, path):
    """Handing solar's wider floor to wind would call an unreadable margin
    readable. Read out of the source, so a run is not needed to check it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    literals = [node.value.value for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(getattr(target, "id", "") == "GRADE_STREAM" for target in node.targets)]
    assert literals == [stream]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "id", "") == "attach_grades"]
    assert len(calls) == 1, "exactly one call site, beside where gate_cells is built"
    assert getattr(calls[0].args[1], "id", "") == "GRADE_STREAM"


@pytest.mark.parametrize("path", sorted(HARNESSES.values()))
def test_neither_harness_reimplements_the_ladder(path):
    """The `model_free_reference.py` rule: one definition, imported twice.

    A second copy drifts silently, because nothing in a gate report says which
    code decided a grade.
    """
    source = path.read_text(encoding="utf-8")
    assert "from src.evaluation.gate_grading import" in source
    for forbidden in ("1.96", "10.64", "7.51", "cv_rms_p90"):
        assert forbidden not in source, f"{path.name} appears to recompute the floor"


def test_attaching_grades_moves_no_gate_verdict():
    """A grade is a reading of a cell, never an input to it."""
    path, stream = TRANCHES["2b"]
    cells = json.loads(path.read_text(encoding="utf-8"))["gate_cells"]
    before = [(dict(cell["gate"]), dict(cell["scores"]["challenger"])) for cell in cells]
    attach_grades(cells, stream)
    after = [(dict(cell["gate"]), dict(cell["scores"]["challenger"])) for cell in cells]
    assert before == after
    assert all(cell["grade"]["label"] for cell in cells)


def test_a_recorded_grade_is_read_back_and_not_re_decided():
    """`cell_grade` prefers what the run wrote; recomputing it in the renderer
    would be the second implementation this module exists to prevent."""
    cell = {"scores": scores(10.0, 20.0, 60.0, 30.0)}
    attach_grades([cell], "wind")
    assert cell["grade"]["label"] == "A"
    # Move the stored decision. The renderer must follow the record, not the
    # scores -- otherwise a re-render silently re-decides a published read.
    cell["grade"]["grade"] = "C"
    cell["grade"]["failed"] = [{"condition": "G1", "reason": "recorded by that run"}]
    assert cell_grade(cell, "wind").detail == "C — fails G1"
    # With no record, it is computed -- which is what grades tranche 2a and 2b.
    assert cell_grade({"scores": cell["scores"]}, "wind").label == "A"
