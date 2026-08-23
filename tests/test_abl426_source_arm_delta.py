"""ABL-426: the arm-difference tool, on records whose answer is known by hand.

`scripts/abl426_source_arm_delta.py` produces the numbers ABL-426's evidence pack
quotes and the numbers a shipping decision on CZ and RO would be taken from. It
is arithmetic over two JSON files, which is exactly the kind of code that is
never wrong in a way anyone notices -- a sign flip in `_delta`, or `skill_pct`
called with its arguments the wrong way round, produces a plausible table.

So the fixtures below are synthetic and their answers are computable in the head:
one cell that improves, one that worsens across the gate verdict, one where a
comparator is missing on one arm only, and one where the *reference* moves while
the challenger does not -- which is the replica-vintage signature the pack reads
the D-7 column for.

The one thing not tested here is `_load`'s scope/source assertions against the
real files, because those files are the thing under measurement and pinning them
would make this test a copy of the evidence rather than a check on the tool.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

SCRIPT = REPO / "scripts" / "abl426_source_arm_delta.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("abl426_source_arm_delta", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scores(challenger, **references):
    """A cell's `scores` block. `n` is carried because the tool reports it."""
    out = {"challenger": {"n": 720, "wape_pct": challenger}}
    for name, wape in references.items():
        out[name] = {"n": 720, "wape_pct": wape}
    return out


def _cell(country, band, gate, challenger, n=720, enough_pairs=True, **references):
    """`gate` is ABL-434's coverage block, not a verdict string.

    Built here in full rather than stubbed to `{"pass": ...}`, because the block's
    other fields are exactly what the tool must *not* mistake for a verdict: CZ
    and PL differ in `n` between the two tables while passing on both.
    """
    return {"country": country, "horizon_band": band,
            "gate": {"pass": gate == "PASS", "n": n, "intended_n": 720,
                     "minimum_n": 684, "beats_d7": gate == "PASS",
                     "enough_pairs": enough_pairs},
            "scores": _scores(challenger, **references)}


def _record(cells):
    return {"meta": {"scope": "x", "training_source": "y"}, "gate_cells": cells,
            "verdict": "PASS"}


# A: the published arm. B: the corrected arm.
#   AA 24-36h  challenger improves 20 -> 16 against an unmoved D-7 of 25.
#   BB 24-36h  challenger worsens 20 -> 30 and crosses its D-7 of 25: PASS -> FAIL.
#   CC 24-36h  challenger unmoved at 20; D-7 moves 25 -> 24. Nothing about the
#              model changed, so a tool that attributes this to the table is wrong.
ARM_A = _record([
    _cell("AA", "24-36h", "PASS", 20.0, seasonal_naive=25.0, constant_oracle=80.0,
          climatology_oracle=22.0),
    _cell("BB", "24-36h", "PASS", 20.0, seasonal_naive=25.0, constant_oracle=80.0,
          climatology_oracle=22.0),
    _cell("CC", "24-36h", "PASS", 20.0, seasonal_naive=25.0, constant_oracle=80.0,
          climatology_oracle=22.0),
])
ARM_B = _record([
    _cell("AA", "24-36h", "PASS", 16.0, seasonal_naive=25.0, constant_oracle=80.0,
          climatology_oracle=22.0),
    _cell("BB", "24-36h", "FAIL", 30.0, seasonal_naive=25.0, constant_oracle=80.0,
          climatology_oracle=22.0),
    _cell("CC", "24-36h", "PASS", 20.0, seasonal_naive=24.0, constant_oracle=80.0,
          climatology_oracle=22.0),
])


@pytest.fixture(scope="module")
def comparison(tool):
    return tool.compare(ARM_A, ARM_B)


def _row(comparison, country):
    return next(r for r in comparison["cells"] if r["country"] == country)


def test_the_delta_is_b_minus_a(comparison):
    """Signed so that **negative is the corrected arm doing better**, because WAPE
    is an error. A sign flip here inverts the pack's whole conclusion."""
    assert _row(comparison, "AA")["challenger_wape_pct"]["delta_pp"] == pytest.approx(-4.0)
    assert _row(comparison, "BB")["challenger_wape_pct"]["delta_pp"] == pytest.approx(10.0)
    assert _row(comparison, "CC")["challenger_wape_pct"]["delta_pp"] == pytest.approx(0.0)


def test_skill_is_measured_against_each_arm_s_own_reference(comparison):
    """CC's challenger does not move and its skill does, because its D-7 did.

    This is the case the pack's vintage control exists for: 1 - 20/25 = 20%, and
    1 - 20/24 = 16.67%. A tool that scored arm B's challenger against arm A's
    reference would report no change and hide the confound entirely.
    """
    skill = _row(comparison, "CC")["references"]["seasonal_naive"]["skill_pct"]
    assert skill["renewable"] == pytest.approx(20.0)
    assert skill["generation"] == pytest.approx(100 * (1 - 20.0 / 24.0))
    assert skill["delta_pp"] == pytest.approx(skill["generation"] - skill["renewable"])
    # And the reference's own movement is reported, which is what makes the
    # attribution possible rather than a matter of interpretation.
    assert _row(comparison, "CC")["references"]["seasonal_naive"]["wape_pct"]["delta_pp"] \
        == pytest.approx(-1.0)


def test_a_gate_verdict_change_is_surfaced_not_averaged(tool, comparison):
    summary = tool.summarise(comparison)
    assert summary["n_gate_verdict_changed"] == 1
    assert summary["gate_verdicts_changed"] == ["BB 24-36h: PASS -> FAIL"]
    assert summary["n_cells_compared"] == 3


def test_the_vintage_control_is_reported_separately_from_the_challenger(tool, comparison):
    """The two spreads must not be one number.

    The pack's argument is that a challenger delta is attributable to the source
    table *only* to the extent the D-7 delta is zero. Collapsing them into one
    "how much moved" figure would destroy the argument while looking tidier.
    """
    summary = tool.summarise(comparison)
    assert summary["challenger_wape_delta_pp"]["max_abs"] == pytest.approx(10.0)
    assert summary["seasonal_naive_delta_pp_is_the_vintage_control"]["max_abs"] \
        == pytest.approx(1.0)


def test_a_reference_missing_on_one_arm_yields_none_rather_than_a_number(tool):
    """A comparator that does not exist must not read as a comparator that tied.

    `incumbent` is `n: 0, wape_pct: null` on every tranche cell by construction --
    none of these countries serves a model -- so this is the common case, not an
    edge one.
    """
    a = _record([_cell("AA", "24-36h", "PASS", 20.0, seasonal_naive=25.0)])
    b = _record([{"country": "AA", "horizon_band": "24-36h",
                  "gate": {"pass": True, "n": 720, "intended_n": 720,
                           "minimum_n": 684, "beats_d7": True, "enough_pairs": True},
                  "scores": {"challenger": {"n": 720, "wape_pct": 18.0},
                             "seasonal_naive": {"n": 0, "wape_pct": None}}}])
    row = tool.compare(a, b)["cells"][0]
    ref = row["references"]["seasonal_naive"]
    assert ref["wape_pct"]["delta_pp"] is None
    assert ref["skill_pct"]["generation"] is None
    assert ref["readable"]["generation"] is None
    # The challenger still differences: one missing reference must not blank the row.
    assert row["challenger_wape_pct"]["delta_pp"] == pytest.approx(-2.0)


def test_readability_is_decided_against_the_registered_floor(tool, comparison):
    """`readable` is the floor test the ABL-418 ladder uses, not a sign test.

    AA's climatology_oracle skill is 1 - 16/22 = 27.3% on arm B and 1 - 20/22 =
    9.1% on arm A. The solar k=1 floor is 10.6482%, so this cell crosses from
    unreadable to readable -- exactly the movement the CEO's shipping question
    turns on, and it must not be reported as a mere sign change.
    """
    ref = _row(comparison, "AA")["references"]["climatology_oracle"]["readable"]
    assert ref["renewable"] is False
    assert ref["generation"] is True
    assert comparison["readability_floor_pct"] == pytest.approx(10.6482, abs=1e-3)
    assert "AA 24-36h vs climatology_oracle: False -> True" \
        in tool.summarise(comparison)["oracle_reference_moves"]


def test_a_cell_present_in_only_one_arm_is_named_rather_than_dropped(tool):
    """Silent truncation reads as "covered everything". If the corrected arm
    scores a cell the published one did not -- or fails to -- the comparison has
    to say so instead of differencing the intersection and reporting a count."""
    a = _record([_cell("AA", "24-36h", "PASS", 20.0, seasonal_naive=25.0),
                 _cell("ZZ", "24-36h", "PASS", 20.0, seasonal_naive=25.0)])
    b = _record([_cell("AA", "24-36h", "PASS", 20.0, seasonal_naive=25.0),
                 _cell("YY", "24-36h", "PASS", 20.0, seasonal_naive=25.0)])
    out = tool.compare(a, b)
    assert out["cells_only_in_renewable_arm"] == [("ZZ", "24-36h")]
    assert out["cells_only_in_generation_arm"] == [("YY", "24-36h")]
    assert len(out["cells"]) == 1


def test_the_controls_report_names_the_field_that_moved(tool):
    """A control mismatch must be legible, not a bare False.

    The tool reports controls as data rather than raising, so a reader can see
    *which* registered value differs -- a feature-vector mismatch and a
    gate-window mismatch invalidate the comparison for different reasons.
    """
    a_meta = {"training_source": "energy_renewable", "n_features": 27,
              "gate_basis": ["challenger", "seasonal_naive"]}
    b_meta = {"training_source": "energy_generation", "n_features": 28,
              "gate_basis": ["challenger", "seasonal_naive"]}
    controls = tool._controls(a_meta, b_meta)
    assert controls["all_controls_hold"] is False
    assert controls["must_match"]["n_features"] == {"equal": False, "a": 27, "b": 28}
    assert controls["must_match"]["gate_basis"]["equal"] is True
    # The source table is the variable under test and must not read as a failure.
    assert controls["expected_to_differ"]["training_source"] == \
        ["energy_renewable", "energy_generation"]


def test_a_field_the_published_record_predates_is_not_reported_as_a_control_failure(tool):
    """`all_controls_hold` must mean "the A/B is controlled", not "the two records
    have the same schema".

    ABL-405 ran 2026-08-13, before the harness recorded `causal_levelling`,
    `g23_readability` and `seed_readability` -- they are **absent** from its meta,
    which is also why its cells carry `grade: null`. Arm B records all three. If
    those three sat in `must_match`, this tool would print
    `all_controls_hold: false` on every real run, and a reader could not tell that
    red apart from a feature-vector mismatch, which is the one thing the flag
    exists to surface.
    """
    a_meta = {"training_source": "energy_renewable", "n_features": 27}
    b_meta = {"training_source": "energy_generation", "n_features": 27,
              "causal_levelling": tool.FIT_WINDOW, "g23_readability": tool.SIGN_TEST,
              "seed_readability": tool.DELTA_MIN}
    controls = tool._controls(a_meta, b_meta)

    assert controls["all_controls_hold"] is True
    assert "causal_levelling" not in controls["must_match"]
    added = controls["grading_registrations_added_after_arm_a"]
    assert added["causal_levelling"]["absent_in_arm_a"] is True
    # Absence is not waved through: what replaces equality is that arm B records
    # the same pin `_grade` grades *both* arms under, so the letters are
    # comparable whether or not arm A wrote it down.
    assert controls["grading_registration_reconciled"] is True

    # And it is a real check: an arm B graded under a different levelling from the
    # one the letters are computed with fails it.
    b_off = dict(b_meta, causal_levelling="trailing_28d")
    off = tool._controls(a_meta, b_off)
    assert off["grading_registration_reconciled"] is False
    assert off["all_controls_hold"] is True  # still controlled; a different defect


def test_a_reported_comparator_added_after_arm_a_is_reconciled_not_failed(tool):
    """ABL-437 added two reference columns after ABL-405 ran. Addition is inert.

    A *reported* comparator is scored on its own intersection and is in no gate
    basis, so it cannot move a cell. What must hold is the direction and the
    reach: arm B is a superset, and no added column is in the gate basis or is
    one of the two the grading levelling reads. The third is the one that matters
    here -- the added pair is exactly what `TRAILING_28D` would grade against,
    and both arms are pinned to `FIT_WINDOW`.
    """
    a_meta = {"gate_basis": ["challenger", "seasonal_naive"],
              "reported_comparators": ["challenger", "seasonal_naive",
                                       "constant_causal", "climatology_causal"]}
    b_meta = {"gate_basis": ["challenger", "seasonal_naive"],
              "reported_comparators": ["challenger", "seasonal_naive",
                                       "constant_causal", "climatology_causal",
                                       "constant_causal_28d", "climatology_causal_28d"]}
    rec = tool._controls(a_meta, b_meta)["reported_comparators"]
    assert rec["added_after_arm_a"] == ["constant_causal_28d", "climatology_causal_28d"]
    assert rec["arm_b_is_a_superset"] is True
    assert rec["no_added_column_is_read_by_the_grading_levelling"] is True
    assert rec["reconciled"] is True
    # And it does not silently exempt the two failures that would matter.
    dropped = tool._controls(a_meta, dict(b_meta, reported_comparators=["challenger"]))
    assert dropped["reported_comparators"]["dropped_from_arm_a"] == [
        "seasonal_naive", "constant_causal", "climatology_causal"]
    assert dropped["reported_comparators"]["reconciled"] is False

    into_the_ladder = tool._controls(
        {**a_meta, "reported_comparators": ["challenger", "seasonal_naive"]},
        {**b_meta, "reported_comparators": ["challenger", "seasonal_naive",
                                            "constant_causal"]})
    assert into_the_ladder["reported_comparators"][
        "no_added_column_is_read_by_the_grading_levelling"] is False
    assert into_the_ladder["reported_comparators"]["reconciled"] is False


def test_a_field_present_on_both_arms_is_compared_by_value_not_exempted(tool):
    """The exemption is for the published record's age, not for the field name.

    If a future arm A does record the three grading registrations, they must be
    compared like any other control -- otherwise this carve-out becomes a
    permanent blind spot on exactly the fields that decide the letters.
    """
    a_meta = {"training_source": "energy_renewable",
              "causal_levelling": "trailing_28d"}
    b_meta = {"training_source": "energy_generation",
              "causal_levelling": tool.FIT_WINDOW}
    controls = tool._controls(a_meta, b_meta)
    entry = controls["grading_registrations_added_after_arm_a"]["causal_levelling"]
    assert entry["absent_in_arm_a"] is False
    assert entry["equal"] is False
    assert controls["grading_registration_reconciled"] is False


def test_the_tool_reproduces_abl418_s_published_letters_on_the_arm_it_already_graded(tool):
    """The one place this tool is checkable against published evidence.

    Arm A's cells carry `grade: null` -- ABL-405 predates ABL-418 -- so this tool
    *computes* their letters, which means every grade change it reports depends on
    getting the grading call right. Three things have to be right at once and none
    of them is the default: the levelling (`fit_window`, not ABL-437's
    `TRAILING_28D`), the G2/G3 readability (`sign_test`, not `FLOORED`), and the
    seed readability (`delta_min`, not `STUDENT_T`).

    ABL-418 graded exactly these 24 cells from exactly this record and published
    the letters, so the check is free and the answer is not one this file gets to
    choose. It also pins the `.label` / `.grade` distinction: three HU cells are
    `U(+)`, which `.grade` renders as `U`, so a tool reading the bare field would
    report three grade changes that are only a rendering.
    """
    record = json.loads(
        (REPO / "experiments" / "ABL348" / "results_abl405_tranche2a.json")
        .read_text(encoding="utf-8"))
    published = json.loads(
        (REPO / "reports" / "abl_418_retro_grade.json").read_text(encoding="utf-8"))
    tranche = next(t for t in published["tranches"] if t["tranche"] == "2a")
    # The record ABL-418 graded must be the record we are grading, or the
    # agreement below is about two different reads.
    assert tranche["training_source"] == "energy_renewable"
    assert tranche["floor_pct"] == pytest.approx(10.648236880290906)

    expected = {(c["pair"], c["band"]): c["label"] for c in tranche["cells"]}
    actual = {(c["country"], c["horizon_band"]): tool._grade(c)
              for c in record["gate_cells"]}
    assert len(expected) == 24
    assert actual == expected
    assert sorted(set(expected.values())) == ["A", "U(+)"]
