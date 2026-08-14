"""ABL-438: tranche 1b was never retro-graded, and its record already had what it takes.

ABL-418 retro-graded tranches 2a and 2b. Tranche 1b (BG/CH solar, ABL-381) was
skipped -- not because anything was missing from its record, but because nobody
ran the ladder over it. Its `meta.reported_comparators` already lists all eight
comparators, so this needs **no refit and no re-read**: the scores are on disk
and the grade is arithmetic over them.

Four things are pinned here, and each is a way this could go quietly wrong.

**The grades themselves**, against the committed record rather than a fixture,
so the test fails if either the ladder or the stored evidence moves. ABL-437
landed after this record was committed and added two *provenance* keys to every
cell and pair; the comparisons strip exactly those keys and then assert the
levelling pin separately, so a moved letter still fails here.

**The qualifiers travel with the A.** Grade A reads *promotion-eligible, subject
to any named data hold*, and on these two pairs there are two such qualifiers
that the ladder cannot see. BG carries ABL-396's live night-contamination hold,
which is registered as data (`HOLDS`) rather than left in a comment; and both
pairs beat the *oracle* hour-of-day climatology by less than the readability
floor, which is the ABL-417 lesson -- the floor applies to any margin a reader
ranks on, not only to the one G1 gates on. An `A` published without either line
attached is the failure this file exists to catch.

**Extending the script did not move ABL-418's grades.** `--tranches` was added to
the existing script rather than a second grader being written, so the thing to
prove is that ABL-418's own selection still produces ABL-418's own grades. All 48
cells and 16 pairs are compared against the committed
`reports/abl_418_retro_grade.json`.

**A selection cannot overwrite its predecessor's report.** The `SCOPE_OUTPUTS`
incident in CLAUDE.md is exactly this shape one directory over: a scoped run that
kept a default output path rewrote a dispositioned record under its own heading
and exited 0. Here it is refused, not warned about.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluation.gate_grading import readability_floor_pct
from src.evaluation.model_free_reference import FIT_WINDOW

REPO = Path(__file__).resolve().parents[1]

#: The two keys ABL-437 added to every graded cell and pair, *after* this record
#: was committed.  Both are provenance rather than result: `causal_levelling`
#: names which pair of causal references produced the letter, and
#: `level_inflation_pct` reports how far the one it used is mis-levelled.  A
#: grade that does not name the reference it graded against is the ambiguity
#: ABL-437 exists to remove, so they are kept and this record is a strict
#: superset of the committed one -- which is why the comparisons below strip
#: them and then assert the pin separately, rather than being relaxed to a
#: subset check that would also pass if a letter moved.
ABL437_PROVENANCE_KEYS = frozenset({"causal_levelling", "level_inflation_pct"})


def without_abl437_provenance(value):
    """Strip ABL-437's provenance keys, at any depth, so a record committed
    before ABL-437 can be compared to one produced after it."""
    if isinstance(value, dict):
        return {key: without_abl437_provenance(item) for key, item in value.items()
                if key not in ABL437_PROVENANCE_KEYS}
    if isinstance(value, list):
        return [without_abl437_provenance(item) for item in value]
    return value

_SPEC = importlib.util.spec_from_file_location(
    "abl418_retro_grade", REPO / "scripts" / "abl418_retro_grade.py")
rg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rg)

RECORD = REPO / "reports" / "abl_438_retro_grade.json"
SOURCE = REPO / "experiments" / "ABL348" / "results_abl381_tranche1b.json"


@pytest.fixture(scope="module")
def tranche():
    """Tranche 1b graded live, through the committed path -- not read back."""
    return rg.read_tranche(REPO, rg.TRANCHES["1b"])


@pytest.fixture(scope="module")
def committed():
    return json.loads(RECORD.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The grades


def test_all_six_cells_grade_a(tranche):
    """The reading ABL-438's description proposed, checked rather than adopted.

    Six cells, two pairs, three bands each: BG and CH solar over ABL-348's frozen
    gate window. If this ever disagrees with the description's table, the
    arithmetic is the evidence and the description is the error.
    """
    assert [cell["label"] for cell in tranche["cells"]] == ["A"] * 6
    assert {label: grade["label"] for label, grade in tranche["pair_grades"].items()} == \
        {"BG": "A", "CH": "A"}
    for cell in tranche["cells"]:
        assert cell["conditions"] == {"G1": True, "G2": True, "G3": True, "G4": True}
        assert cell["failed"] == []


def test_the_record_carried_all_eight_comparators_already(tranche):
    """Why this needed no refit: nothing was missing from the stored record.

    ABL-435 had to re-read its tranche because the references were not there.
    1b's were, which is the whole difference between the two issues.
    """
    assert tranche["reported_comparators"] == [
        "challenger", "incumbent", "seasonal_naive", "persistence",
        "constant_causal", "constant_oracle", "climatology_causal", "climatology_oracle"]


def test_every_cell_clears_the_registered_minimum_n(tranche):
    """The column the ladder cannot see.

    A grade is a reading of a *margin*, so a coverage-short cell that beat D-7
    would grade A exactly as a full-coverage one does. `enough_pairs` is the
    check that sees it, and it nests under `gate` where a flat lookup passes
    vacuously -- so it is asserted on the value, not on its presence.
    """
    for cell in tranche["cells"]:
        assert cell["enough_pairs"] is True, (cell["pair"], cell["band"])
        assert cell["n"] >= cell["minimum_n"] > 0, (cell["pair"], cell["band"])


# --------------------------------------------------------------------------
# The two qualifiers that must travel with the A


def test_both_pairs_beat_the_oracle_climatology_only_inside_the_floor(tranche):
    """A win nobody can read is not a win to rank on -- ABL-417's lesson.

    Solar's null model is an hour-of-day climatology, not a flat line (a flat
    line loses by 60pp here and certifies nothing). Against the *oracle* form
    both pairs still win in every band, but by 1.4-11.2% against a 10.65% floor,
    so the worst band of each is inside it. The gate is unaffected -- an oracle
    is not causally available and never gates -- but the margin may not be
    reported as a clean win.
    """
    floor = readability_floor_pct("solar")
    worst = {}
    for cell in tranche["cells"]:
        margin = cell["oracle_skill_pct"]["climatology_oracle"]
        assert margin > 0, (cell["pair"], cell["band"])          # it does win
        worst[cell["pair"]] = min(worst.get(cell["pair"], margin), margin)
    assert worst["BG"] == pytest.approx(1.408, abs=0.005)
    assert worst["CH"] == pytest.approx(3.467, abs=0.005)
    for pair, margin in worst.items():
        assert margin < floor, f"{pair} is readable now -- the qualifier text must be re-derived"
        assert any(cell["oracle_margin_readable"]["climatology_oracle"] is False
                   for cell in tranche["cells"] if cell["pair"] == pair)


def test_the_constant_oracle_is_beaten_readably_and_says_less_than_it_looks(tranche):
    """The other oracle, reported for contrast and for the same reason.

    Both pairs beat a hindsight *constant* by 63-91%, which sounds decisive and
    is not: a flat line cannot represent a diurnal cycle at all, so on solar it
    measures that the sun rises. It is the climatology above that does the work
    a null model is for. Pinned so the two are never read as one result.
    """
    floor = readability_floor_pct("solar")
    for cell in tranche["cells"]:
        assert cell["oracle_skill_pct"]["constant_oracle"] > floor
        assert cell["oracle_margin_readable"]["constant_oracle"] is True


def test_bg_carries_the_abl396_night_hold_as_data_not_prose(tranche):
    """Grade A means *promotion-eligible, subject to any named data hold*.

    A hold that lives only in a comment is a hold the next reader does not get,
    so it is registered against the pair and rendered under every table that pair
    appears in. BG's displacement band is wider than its own +1.41% margin over
    the oracle climatology, which is why it cannot be dropped as a footnote.
    """
    hold = tranche["holds"]["BG"]
    assert hold["issue"] == "ABL-396"
    assert "night" in hold["kind"]
    assert "CH" not in tranche["holds"]
    for cell in tranche["cells"]:
        assert (cell["hold"] == hold) is (cell["pair"] == "BG")


def test_the_rendered_report_states_the_hold_and_the_floor(tranche):
    """The qualifiers are in the *report*, not only in the JSON.

    ABL-438 asks specifically that a grade of A never be reported for BG solar
    without the night-contamination line attached. That is a property of the
    rendered document, so it is asserted on the rendered document.
    """
    report = rg.render([tranche], REPO, "ABL-438")
    assert "**A**" in report
    assert "ABL-396" in report and "night contamination" in report
    assert "1,097 MW" in report
    assert "inside the floor (+1.41%)" in report
    assert "inside the floor (+3.47%)" in report


# --------------------------------------------------------------------------
# The committed record is the grade, and it is re-derivable


def test_the_committed_record_matches_a_live_grade(tranche, committed):
    """The record on disk is what a re-run produces, not a transcription of it."""
    assert committed["issue"] == "ABL-438"
    assert committed["tranche_selection"] == ["1b"]
    stored, = committed["tranches"]
    assert stored["cells"] == without_abl437_provenance(tranche["cells"])
    assert stored["pair_grades"] == without_abl437_provenance(tranche["pair_grades"])
    # ABL-437 pins this record's read to the levelling it was published on, so
    # the letters above are reproduced rather than re-derived on a new reference.
    assert {grade["causal_levelling"] for grade in tranche["pair_grades"].values()} == {FIT_WINDOW}
    assert {cell["causal_levelling"] for cell in tranche["cells"]} == {FIT_WINDOW}


def test_the_record_names_the_bytes_it_graded(committed):
    """The SHA-256 in the record is the source file's, so a later reader can tell
    whether these grades were computed from the bytes that were dispositioned."""
    stored, = committed["tranches"]
    assert stored["results_path"] == "experiments/ABL348/results_abl381_tranche1b.json"
    assert stored["results_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def test_abl381s_published_verdict_is_restated_not_revised(tranche):
    """A grade reads a disposition; it never replaces one. ABL-381 published
    PASS 6/6 and it still says PASS 6/6."""
    results = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert tranche["published_verdict"] == results["verdict"] == "PASS"
    assert all(cell["gate_pass"] is True for cell in tranche["cells"])


# --------------------------------------------------------------------------
# Extending the script moved nothing, and cannot overwrite anything


def test_abl418s_own_selection_still_produces_abl418s_own_grades():
    """`--tranches` was added to the existing script rather than a second grader
    being written. The thing that buys is also the thing to prove."""
    committed = json.loads((REPO / "reports" / "abl_418_retro_grade.json").read_text(encoding="utf-8"))
    assert [tranche["tranche"] for tranche in committed["tranches"]] == list(rg.ABL418_TRANCHES)
    for stored in committed["tranches"]:
        fresh = rg.read_tranche(REPO, rg.TRANCHES[stored["tranche"]])
        assert without_abl437_provenance(fresh["pair_grades"]) == stored["pair_grades"], \
            stored["tranche"]
        assert {grade["causal_levelling"] for grade in fresh["pair_grades"].values()} \
            == {FIT_WINDOW}, stored["tranche"]
        assert [cell["label"] for cell in fresh["cells"]] == \
            [cell["label"] for cell in stored["cells"]], stored["tranche"]
        assert [cell["skill_pct"] for cell in fresh["cells"]] == \
            [cell["skill_pct"] for cell in stored["cells"]], stored["tranche"]


@pytest.mark.parametrize("flag", ["--report-out", "--json-out"])
def test_a_non_default_selection_may_not_write_abl418s_paths(tmp_path, flag):
    """The `SCOPE_OUTPUTS` failure, refused rather than warned about.

    `argparse` resolves a default before any other argument is consulted, so a
    selection that keeps one default path rewrites a dispositioned report under
    a heading that no longer describes it -- and exits 0.
    """
    other = "--json-out" if flag == "--report-out" else "--report-out"
    suffix = "json" if other == "--json-out" else "md"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "abl418_retro_grade.py"),
         "--tranches", "1b", other, str(tmp_path / f"out.{suffix}")],
        capture_output=True, text=True, cwd=REPO)
    assert result.returncode != 0
    assert "may not write ABL-418's" in result.stderr
    assert flag in result.stderr


def test_an_unknown_tranche_is_named_rather_than_silently_dropped(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "abl418_retro_grade.py"),
         "--tranches", "1b,9z", "--report-out", str(tmp_path / "o.md"),
         "--json-out", str(tmp_path / "o.json")],
        capture_output=True, text=True, cwd=REPO)
    assert result.returncode != 0
    assert "unknown tranche(s) 9z" in result.stderr


def test_the_script_does_not_reimplement_the_ladder():
    """The `model_free_reference.py` rule, held here too: one definition,
    imported. A second copy drifts silently, because nothing in a graded report
    says which code decided a grade.

    Read out of the AST rather than by substring: the module docstring and the
    report body both *quote* the formula on purpose, so a text search for `1.96`
    would fail on the prose that documents it and pass on a second copy hidden
    behind a different literal.
    """
    source = (REPO / "scripts" / "abl418_retro_grade.py").read_text(encoding="utf-8")
    assert "from src.evaluation.gate_grading import" in source
    defined = {node.name for node in ast.walk(ast.parse(source))
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for owned in ("grade_cell", "pair_grade", "readability_floor_pct", "skill_pct",
                  "margin_pct_of_own_error", "comparator_wape"):
        assert owned not in defined, f"{owned} belongs to the shared modules, not here"
    # The only numeric literals the renderer may hold are formatting ones. The z
    # and the per-stream CVs come from `gate_grading`.
    assert "Z_95" in source and "STREAM_FLEET_CV_P90" not in source
