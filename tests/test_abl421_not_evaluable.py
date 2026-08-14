"""ABL-421: a cell the frozen registration declares NOT-EVALUABLE is not a FAIL.

ABL-348 `not_evaluable` carries a rule the solar harness had no way to obey:

    "A pair listed here is reported NOT-EVALUABLE on the named bands. It is not
     a FAIL and must not be counted as one; a gate read that scores it has
     misread this registration."

`gate_cell` builds a cell for every country-band the run produces rows for and
marks it `pass: False` when `n` falls under the registered minimum. So before
`SCOPE_NOT_EVALUABLE` existed, EE's and FI's four declared cells would each have
arrived as an ordinary failed cell, been counted into `passed/18`, and rendered
the tranche FAIL -- a model-quality verdict on a comparison the registration
forbids, with nothing in the exit status to show it. Every earlier tranche dodged
this by excluding the two pairs; 2d is the tranche they belong to.

The tests below hold three separate things, and they fail for different reasons:

- the declaration **matches ABL-348** and has not been widened by hand (a scope
  that could declare its own cells unscorable is a scope that can drop whatever
  scores badly, which is the entire failure mode pre-registration exists to
  prevent);
- the declared cells are **excluded from the bar** and from the grade ladder;
- every scope that predates the table is **byte-identically unaffected**.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SOLAR_HARNESS = ROOT / "scripts" / "evaluate_solar_retrain.py"
REGISTRATION = ROOT / "experiments" / "ABL348" / "config.json"

SCOPE = "abl316-t2d"
#: The four cells ABL-348 declares, restated so the test is readable. It is not
#: the source of truth -- `test_the_declaration_is_exactly_what_abl348_declares`
#: derives the same set from the frozen config and compares.
DECLARED = {("EE", "24-36h"), ("EE", "36-48h"), ("FI", "24-36h"), ("FI", "36-48h")}


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("solar_harness", SOLAR_HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def registration():
    return json.loads(REGISTRATION.read_text(encoding="utf-8"))


def test_the_tranche_scope_is_the_six_northern_countries(harness):
    assert harness.SCOPES[SCOPE] == ("EE", "FI", "LT", "LV", "NL", "SE")


def test_the_declaration_is_exactly_what_abl348_declares(harness, registration):
    """Derived from the frozen config, not from the harness's own opinion.

    This is the test that makes `SCOPE_NOT_EVALUABLE` a *transcription* of a
    pre-registration rather than a discretionary exclusion list. A cell may be
    declared unscorable only because ABL-348 declared it so, before any fit
    existed; anything else is dropping a cell for how it scored.
    """
    declared = {
        (pair.split("/")[0], band)
        for pair, entry in registration["not_evaluable"]["pairs"].items()
        if pair.endswith("/solar")
        for band in entry["registered_min_n_684_bands"]
    }
    assert declared == DECLARED, "the frozen registration moved; this table must follow it"
    assert harness.not_evaluable_for(SCOPE) == declared, (
        "SCOPE_NOT_EVALUABLE does not match ABL-348's declaration")


def test_the_declaration_covers_only_the_684_bands(harness, registration):
    """48-64h is deliberately absent for both pairs, on ABL-348's instruction.

    `not_evaluable.note_48_64h`: that band selects a 480-510 row subset, so its n
    scales proportionally rather than being hard bounded by `n_d7_scorable`, and
    "a pair declared here may still clear 456 in that band and should be reported
    if it does". Declaring it here would refuse to read a cell the registration
    asks us to read.
    """
    assert "48-64h" not in {band for _, band in harness.not_evaluable_for(SCOPE)}
    assert "note_48_64h" in registration["not_evaluable"], (
        "the instruction this absence rests on is gone from the registration")
    for country in ("EE", "FI"):
        assert (country, "48-64h") not in harness.not_evaluable_for(SCOPE)


def test_the_causes_match_the_frozen_registration(harness, registration):
    """The report's cause strings are a restatement, and restatements drift.

    Only the load-bearing half is asserted verbatim -- the `source_dependent`
    flag, which decides whose problem each shortfall is: EE's survives a revert
    of the source change and FI's does not, so labelling them the same way would
    misdirect the follow-up.
    """
    pairs = registration["not_evaluable"]["pairs"]
    assert set(harness.NOT_EVALUABLE_CAUSES) == {"EE", "FI"}
    assert pairs["EE/solar"]["source_dependent"] is False
    assert "not source-dependent" in harness.NOT_EVALUABLE_CAUSES["EE"]
    assert pairs["FI/solar"]["source_dependent"] is True
    assert "**source-dependent**" in harness.NOT_EVALUABLE_CAUSES["FI"]
    # The n each declaration rests on, so a config edit that moves the shortfall
    # without moving the bands is still caught here.
    assert pairs["EE/solar"]["n_d7_scorable_energy_generation"] == 630
    assert pairs["FI/solar"]["n_d7_scorable_energy_generation"] == 650
    assert "663 of 720" in harness.NOT_EVALUABLE_CAUSES["FI"]


def test_the_bar_is_the_grid_minus_the_declared_cells(harness):
    """6 x 3 = 18, of which 14 are evaluable."""
    grid = len(harness.SCOPES[SCOPE]) * len(harness.PRIMARY_BANDS)
    assert grid == 18
    assert grid - len(harness.not_evaluable_for(SCOPE)) == 14


@pytest.mark.parametrize("scope", ["abl253", "abl316-t1b", "abl316-t2a", "abl316-t2c", "abl376"])
def test_every_scope_predating_the_table_is_unaffected(harness, scope):
    """The table defaults to empty, so the bar is the identity it always was.

    This is what makes ABL-421 landable without re-reading five dispositioned
    scopes: `registered_cells` for each of them is still exactly
    `len(countries) * len(bands)`.
    """
    assert harness.not_evaluable_for(scope) == frozenset()
    grid = len(harness.SCOPES[scope]) * len(harness.PRIMARY_BANDS)
    assert grid - len(harness.not_evaluable_for(scope)) == grid


def test_the_bar_still_derives_from_the_registration_and_holds_no_literal():
    """ABL-379's guard, restated for the subtraction ABL-421 adds.

    `test_solar_bar_is_derived_from_the_scope_not_a_literal` reads the
    `performance_pass` assignment. This reads the `registered_cells` assignment
    that now feeds it, because moving the arithmetic upstream is exactly how a
    hardcoded bar could return without that test noticing.
    """
    tree = ast.parse(SOLAR_HARNESS.read_text(encoding="utf-8"))
    assign = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.Assign)
                  and getattr(node.targets[0], "id", "") == "registered_cells")
    literals = [n.value for n in ast.walk(assign.value)
                if isinstance(n, ast.Constant) and isinstance(n.value, int)]
    assert not literals, (
        f"registered_cells is built from literal(s) {literals}; it must derive from "
        "the registration tables")
    names = {n.id for n in ast.walk(assign.value) if isinstance(n, ast.Name)}
    assert {"registered_countries", "PRIMARY_BANDS", "not_evaluable"} <= names


def test_a_declared_cell_is_kept_out_of_the_graded_list_not_marked_within_it():
    """The exclusion happens where `passed`/`disposition`/`attach_grades` read.

    Pinned by AST rather than by output because a run that appended a declared
    cell to `gate_cells` with a flag on it would render every number, exit 0, and
    still count that cell into the bar -- the flag would be decoration. The
    property is that the cell reaches `not_evaluable_cells` and nothing else.
    """
    source = SOLAR_HARNESS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    branch = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.ops[0], ast.In)
        and getattr(node.test.comparators[0], "id", "") == "not_evaluable")
    appended = {
        getattr(call.func.value, "id", "")
        for body in (branch.body, branch.orelse)
        for stmt in body for call in ast.walk(stmt)
        if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "append"}
    assert appended == {"not_evaluable_cells", "gate_cells"}, (
        f"the declared-cell branch appends to {appended}; a declared cell must reach "
        "not_evaluable_cells and an evaluable one gate_cells, and nothing else")
    # ...and the declared branch is the one that does NOT feed the bar.
    declared_targets = {
        getattr(call.func.value, "id", "")
        for stmt in branch.body for call in ast.walk(stmt)
        if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "append"}
    assert declared_targets == {"not_evaluable_cells"}


def test_the_declared_table_is_not_in_the_import_time_check():
    """`SCOPE_NOT_EVALUABLE` must stay out of `check_registration_tables`.

    An empty declaration is the correct and common case, so requiring an entry
    per scope would raise `KeyError` at import for every scope whose absence is
    right -- taking `--help` and the whole suite with it, which is the tax
    `SCOPE_FEATURES` is kept out for. This pins the asymmetry so a later
    tidying-up does not "complete" the check and break six scopes.
    """
    tree = ast.parse(SOLAR_HARNESS.read_text(encoding="utf-8"))
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "check_registration_tables")
    named = {kw.arg for kw in call.keywords}
    assert named == {"SCOPES", "GATE_BASIS", "SCOPE_OUTPUTS", "FIT_RULES", "SCOPE_TITLES"}
    assert "SCOPE_NOT_EVALUABLE" not in named


@pytest.fixture(scope="module")
def read_script():
    spec = importlib.util.spec_from_file_location(
        "abl421_read", ROOT / "scripts" / "abl421_tranche2d_read.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell(enough: bool, beats: bool = True) -> dict:
    return {"gate": {"enough_pairs": enough, "beats_d7": beats, "pass": enough and beats}}


def test_a_coverage_short_cell_is_not_a_decidable_band(read_script):
    """`beats_d7` and `enough_pairs` are separate, and only the second decides this."""
    cells = [_cell(enough=False, beats=True), _cell(enough=True, beats=True)]
    assert read_script.decidable_bands(cells) == [cells[1]]
    assert read_script.decidable_bands([_cell(enough=False, beats=True)]) == []


def test_a_pair_with_no_decidable_band_is_reported_held_not_graded(read_script):
    """EE and FI: grade A on the margin, reported `—` with the hold named.

    The ladder is handed a cell's `scores` and never sees `n`, so it grades a
    margin. A margin the registration does not consider readable cannot carry a
    promotion, and reporting a bare `A` for such a pair would say it could.
    """
    reported, hold = read_script.held_for_coverage("A", [_cell(enough=False)])
    assert reported == "—"
    assert hold == read_script.COVERAGE_HOLD
    assert "minimum n" in hold


def test_the_hold_does_not_bind_when_any_band_is_decidable(read_script):
    """One readable band is enough for the pair to keep its ladder grade.

    The worst-band rule already accounts for the other bands, so the hold must
    not double-penalise a pair that has something to decide on.
    """
    for label in ("A", "B", "C", "U(+)"):
        reported, hold = read_script.held_for_coverage(
            label, [_cell(enough=False), _cell(enough=True)])
        assert (reported, hold) == (label, "")


def test_the_hold_never_upgrades_a_grade(read_script):
    """It only ever removes eligibility -- it cannot turn a C into anything better."""
    for label in ("A", "B", "C", "U(+)", "U"):
        reported, _ = read_script.held_for_coverage(label, [_cell(enough=False)])
        assert reported == "—", f"{label} was not held"
    # And with no cells at all there is nothing to hold: the label passes through
    # rather than being silently downgraded by an empty list.
    assert read_script.held_for_coverage("A", []) == ("A", "")


def test_the_read_script_does_not_reimplement_the_ladder(read_script):
    """The grades must come from `gate_grading`, not from a second copy here.

    ABL-419 established this and it is the reason the pack and the machine
    record cannot disagree about a grade.
    """
    source = (ROOT / "scripts" / "abl421_tranche2d_read.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.asname or alias.name
                for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and node.module == "src.evaluation.gate_grading"
                for alias in node.names}
    assert {"cell_grade", "pair_grade"} <= imported
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not defined & {"grade_cell", "cell_grade", "pair_grade", "readability_floor_pct"}


def test_the_report_renders_nothing_for_a_scope_that_declares_none(harness):
    """Every already-published report is byte-unchanged by this function."""
    assert harness.not_evaluable_table({"meta": {}, "not_evaluable_cells": []}) == []
    assert harness.not_evaluable_table({"meta": {}}) == []


def test_the_report_table_carries_no_gate_outcome_and_no_grade(harness):
    """Measured and shown, but never dispositioned -- ABL-348's rule, rendered.

    A PASS/FAIL or a grade in this table would be the count the registration
    forbids, in all but name.
    """
    cell = {"country": "EE", "horizon_band": "24-36h",
            "gate": {"n": 630, "minimum_n": 684, "pass": False,
                     "beats_d7": True, "enough_pairs": False},
            "scores": {"challenger": {"wape_pct": 30.0},
                       "seasonal_naive": {"wape_pct": 36.67}}}
    lines = harness.not_evaluable_table({
        "meta": {"registered_cells": 14,
                 "not_evaluable_declared_by": "experiments/ABL348/config.json",
                 "not_evaluable_causes": harness.NOT_EVALUABLE_CAUSES},
        "not_evaluable_cells": [cell]})
    rendered = "\n".join(lines)
    assert "630" in rendered and "684" in rendered
    assert "not source-dependent" in rendered
    # The prose quotes ABL-348's rule and so contains the word FAIL by design.
    # What must carry no outcome is the table itself, so assert on the rows.
    header, *body = [line for line in lines if line.startswith("|")]
    assert "grade" not in header and "gate" not in header
    data_rows = [line for line in body if not set(line) <= set("|-: ")]
    assert data_rows, "the declared cell did not render a row"
    for row in data_rows:
        assert "PASS" not in row and "FAIL" not in row, (
            f"a NOT-EVALUABLE row carries a gate outcome: {row}")
