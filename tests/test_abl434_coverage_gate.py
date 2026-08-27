"""ABL-434: the grade ladder reads the cell's registered coverage first.

ABL-418's ladder grades a **margin**. ``grade_cell`` is handed a cell's ``scores``
mapping and nothing else, so it never sees ``gate.n``, ``gate.minimum_n`` or
``gate.enough_pairs`` -- and a cell that beats seasonal-naive D-7 by more than the
readability floor while falling short of its registered minimum n graded ``A``,
which means promotion-eligible. Tranche 2d is where the combination first arose:
EE and FI solar 48-64h, ``grade: A`` beside ``pass: false`` in the same cell, FI
missing 456 rows by three.

What is asserted here, in order:

1. the defect reproduces on the committed record, so this is a test of a measured
   thing rather than of a story about one;
2. the gate holds those two cells, naming the numbers;
3. **exactly two cells in the whole programme move**, derived by re-grading every
   committed record rather than listed by hand;
4. the gate is one-way, idempotent, and refuses an unrecorded coverage;
5. ``grade_cell`` is still a function of ``scores`` alone -- which is what lets
   every published margin-only re-read reproduce byte-for-byte -- and every
   caller that grades from ``scores`` while holding a whole cell is on a
   registered list with a reason.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.gate_grading import (  # noqa: E402
    COVERAGE_CONDITION, COVERAGE_SHORT, GRADE_SEVERITY, SIGN_TEST, CellGrade,
    attach_grades, cell_grade, coverage_reason, gated_on_coverage, grade_cell,
    pair_grade,
)
from src.evaluation.model_free_reference import FIT_WINDOW  # noqa: E402

TRANCHE_2D = ROOT / "experiments" / "ABL348" / "results_abl421_tranche2d.json"

#: Every committed gate record, with the stream its cells were graded on. Derived
#: from git rather than globbed, because an untracked local run must not be able
#: to widen or narrow the set this test measures over.
def _committed_records():
    import subprocess
    listed = subprocess.run(["git", "ls-files", "experiments"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.split()
    records = []
    for name in sorted(listed):
        if not name.endswith(".json"):
            continue
        record = json.loads((ROOT / name).read_text(encoding="utf-8"))
        if isinstance(record, dict) and record.get("gate_cells"):
            records.append((name, record))
    return records


COMMITTED = _committed_records()


def _stream(name: str, record: dict) -> str:
    """Solar unless the record's own cells say otherwise.

    The wind harness writes `forecast_type` on every cell and the solar harness
    does not, so the stream is read off the record instead of off the path.
    """
    return "wind" if any("forecast_type" in cell for cell in record["gate_cells"]) else "solar"


def _cell(enough=True, n=720, minimum_n=684, wape=10.0):
    """One cell in the shape both harnesses write, with a clearly-A margin."""
    def entry(value):
        return {"wape_pct": value, "n": 720}
    scores = {"challenger": {"wape_pct": wape, "n": 720, "slope": 0.8, "correlation": 0.9},
              "seasonal_naive": entry(20.0), "constant_causal": entry(60.0),
              "climatology_causal": entry(30.0), "constant_causal_28d": entry(60.0),
              "climatology_causal_28d": entry(30.0)}
    gate = {"n": n, "intended_n": 720, "minimum_n": minimum_n,
            "beats_d7": True, "enough_pairs": enough, "pass": enough}
    return {"country": "XX", "horizon_band": "48-64h", "scores": scores, "gate": gate}


# ---------------------------------------------------------------------------
# 1. The defect, on the committed record


@pytest.fixture(scope="module")
def tranche_2d():
    return json.loads(TRANCHE_2D.read_text(encoding="utf-8"))


def _cells_of(record, country):
    return [cell for cell in record["gate_cells"] if cell["country"] == country]


@pytest.mark.parametrize("country,n", [("EE", 388), ("FI", 453)])
def test_the_margin_instrument_still_grades_the_short_cell_a(tranche_2d, country, n):
    """The defect, reproduced rather than described.

    `grade_cell` reads a margin and must keep reading one: that is what makes it
    re-runnable over a stored record, and what leaves ABL-437's, ABL-443's and
    ABL-444's published re-reads byte-identical. So the A is still there when you
    ask it for a margin -- and this test exists so that a later change which
    quietly gates `grade_cell` too has to come here and say so.
    """
    cell, = _cells_of(tranche_2d, country)
    assert cell["gate"]["n"] == n and cell["gate"]["minimum_n"] == 456
    assert cell["gate"]["enough_pairs"] is False and cell["gate"]["pass"] is False
    assert cell["gate"]["beats_d7"] is True, "a coverage shortfall is not a loss to D-7"
    # Under the forms this read was decided on. 2d predates ABL-437 and ABL-444,
    # so its record carries neither key and rebuilds as fit-window / sign-test --
    # absence dating the read, exactly as those two amendments registered. Taken
    # off the record rather than retyped, so the reproduction cannot drift from
    # what was published.
    recorded = CellGrade.from_dict(cell["grade"])
    assert (recorded.levelling, recorded.g23_readability) == (FIT_WINDOW, SIGN_TEST)
    assert grade_cell(cell["scores"], "solar", levelling=recorded.levelling,
                      g23_readability=recorded.g23_readability).grade == "A"
    # And the record itself carries that A -- the thing the next reader sees.
    assert cell["grade"]["grade"] == "A"


@pytest.mark.parametrize("country,n", [("EE", 388), ("FI", 453)])
def test_a_cell_level_grade_holds_it_and_names_the_numbers(tranche_2d, country, n):
    """`A` means promotion-eligible; a cell the registration does not consider
    readable cannot be. The reason carries n and the minimum, so the hold can be
    checked against ABL-348 by eye rather than taken on trust."""
    cell, = _cells_of(tranche_2d, country)
    grade = cell_grade(cell, "solar")
    assert grade.grade == COVERAGE_SHORT == "X"
    assert grade.enough_pairs is False
    assert grade.conditions[COVERAGE_CONDITION[0]] is False
    assert grade.failed[0][0] == "G0"
    assert f"{n:,}" in grade.failed[0][1] and "456" in grade.failed[0][1]
    # The margin is not erased by the hold -- it is still printed, exactly as
    # ABL-444 requires of its own abstention.
    assert grade.skill["seasonal_naive"] > grade.floor_pct
    assert pair_grade([grade]).grade == "X", "the pair inherits its only band"


# ---------------------------------------------------------------------------
# 2. The affected set, over every committed record


def test_exactly_two_cells_in_the_programme_move():
    """Derived by re-grading every committed record, not listed by hand.

    This is the claim the CEO asked to be reported rather than assumed: fixing
    the ladder moves two letters, both in tranche 2d, and both already reported
    as held by ABL-421's evidence pack. Nothing published is re-graded.
    """
    moved = []
    for name, record in COMMITTED:
        stream = _stream(name, record)
        for cell in record["gate_cells"]:
            before = (cell.get("grade") or {}).get("grade") or \
                grade_cell(cell["scores"], stream).grade
            after = cell_grade(cell, stream).grade
            if before != after:
                moved.append((Path(name).name, cell["country"], cell["horizon_band"],
                              before, after))
    assert moved == [
        ("results_abl421_tranche2d.json", "EE", "48-64h", "A", "X"),
        ("results_abl421_tranche2d.json", "FI", "48-64h", "A", "X"),
    ]


def test_every_committed_cell_records_its_coverage():
    """Why the amendment needs no per-scope pin, and why fail-closed is cheap.

    A record with no `enough_pairs` would grade `X` under this gate -- correctly,
    since coverage that is not recorded is not coverage that holds -- and that
    would re-write published letters wholesale. It does not arise: every cell in
    every committed record carries the column.
    """
    total = 0
    for name, record in COMMITTED:
        for cell in record["gate_cells"]:
            assert isinstance((cell.get("gate") or {}).get("enough_pairs"), bool), \
                (name, cell["country"], cell["horizon_band"])
            total += 1
    # 143 at registration (ABL-434); +24 when ABL-426 committed the tranche
    # 2a-generation record (8 countries x 3 bands). State the delta beside the
    # absolute: the absolute is the tripwire, the delta is what survives the
    # next committed record.
    assert total == 167, "the affected set was measured over this many cells"


# ---------------------------------------------------------------------------
# 3. What the gate may and may not do


@pytest.mark.parametrize("label", ["A", "B", "C", "N", "U"])
def test_the_gate_never_raises_a_letter(label):
    """One-way by construction: it replaces a letter with `X`, never the reverse.

    Asserted on the letters rather than on `GRADE_SEVERITY`, because `X` ranks
    *better* than `B` and `C` -- a `B` cell that is also coverage-short becomes
    `X`, which lowers severity while leaving the cell exactly as non-promotable.
    What must hold is that no cell becomes promotion-eligible.
    """
    held = gated_on_coverage(CellGrade(grade=label), {"enough_pairs": False, "n": 1,
                                                      "minimum_n": 2})
    assert held.grade == "X" and held.grade != "A"
    kept = gated_on_coverage(CellGrade(grade=label), {"enough_pairs": True})
    assert kept.grade == label and kept.enough_pairs is True


def test_a_covered_cell_is_untouched_except_for_the_condition_it_records():
    """The other 141 cells: same letter, same margins, plus one recorded fact."""
    cell = _cell(enough=True)
    margin = grade_cell(cell["scores"], "solar")
    gated = cell_grade(cell, "solar")
    assert (gated.grade, gated.plus, gated.failed) == (margin.grade, margin.plus, margin.failed)
    assert gated.skill == margin.skill and gated.floor_pct == margin.floor_pct
    assert gated.conditions == {**margin.conditions, "G0": True}
    assert gated.enough_pairs is True


def test_unrecorded_coverage_is_not_a_pass():
    """Absence dates a read everywhere else on this ladder; here it also holds it.

    A cell with no `gate` block cannot be shown readable, and the reason says
    exactly that rather than claiming a shortfall nobody measured.
    """
    for gate in (None, {}, {"n": 388}):
        grade = gated_on_coverage(CellGrade(grade="A"), gate)
        assert grade.grade == "X" and grade.enough_pairs is None
        assert "not recorded" in grade.failed[0][1]
    assert "not recorded" in coverage_reason(None)
    assert "not recorded" not in coverage_reason({"enough_pairs": False, "n": 1, "minimum_n": 2})


def test_the_gate_is_idempotent():
    """A gated run writes `X` into its record and that record is read back through
    this same path, so re-gating must re-derive the one condition rather than
    stack it."""
    once = cell_grade(_cell(enough=False, n=453, minimum_n=456), "solar")
    twice = gated_on_coverage(once, {"enough_pairs": False, "n": 453, "minimum_n": 456})
    assert once.as_dict() == twice.as_dict()
    assert [name for name, _ in twice.failed].count("G0") == 1


def test_a_cell_that_measured_nothing_stays_not_measured():
    """`Not measured` is already the weaker statement. Overwriting it with a
    coverage verdict would claim the cell was scored."""
    empty = {"challenger": {"wape_pct": None}, "seasonal_naive": {"wape_pct": None}}
    grade = cell_grade({"scores": empty, "gate": {"enough_pairs": False, "n": 1,
                                                  "minimum_n": 2}}, "solar")
    assert grade.grade is None and grade.label == "Not measured"


def test_a_harness_records_the_held_grade_not_the_margin():
    """The primary fix: what a *future* tranche writes is already held, so no
    later reader has to keep its own books (which is what ABL-421 had to do)."""
    cells = [_cell(enough=True), _cell(enough=False, n=453, minimum_n=456)]
    attach_grades(cells, "solar")
    assert [cell["grade"]["label"] for cell in cells] == ["A", "X"]
    assert cells[1]["grade"]["enough_pairs"] is False
    assert cells[1]["grade"]["conditions"]["G0"] is False
    # And the gate column it disagreed with is untouched: a grade is a reading of
    # a cell, never an input to it.
    assert cells[1]["gate"]["pass"] is False and cells[1]["gate"]["beats_d7"] is True


def test_a_pair_is_held_by_one_short_band():
    """Stricter than ABL-421's reporting-side hold, which held a pair only when
    *no* band was decidable. `A` requires all four conditions in every band, so
    one band short of its registered n is enough."""
    grades = [cell_grade(_cell(enough=True), "solar"),
              cell_grade(_cell(enough=True), "solar"),
              cell_grade(_cell(enough=False, n=453, minimum_n=456), "solar")]
    assert [grade.label for grade in grades] == ["A", "A", "X"]
    assert pair_grade(grades).grade == "X"


def test_severity_places_x_between_u_and_b():
    """Deeper than `U` -- a `U` cell has the rows and cannot resolve the margin,
    an `X` cell does not have the rows. Shallower than `B`/`C`, on ABL-444's rule
    that a definite failure outranks an abstention and `X` would bury it."""
    assert GRADE_SEVERITY["A"] < GRADE_SEVERITY["N"] < GRADE_SEVERITY["U"] \
        < GRADE_SEVERITY["X"] < GRADE_SEVERITY["B"] < GRADE_SEVERITY["C"]
    covered_loss = cell_grade(_cell(enough=True, wape=30.0), "solar")
    assert covered_loss.grade == "C"
    short = cell_grade(_cell(enough=False, n=453, minimum_n=456), "solar")
    assert pair_grade([covered_loss, short]).grade == "C", \
        "a readable loss on a covered band must not be buried under a shortfall"


# ---------------------------------------------------------------------------
# 4. Where the gate lives, held statically


#: Every caller that grades from `scores` while a whole cell is in hand, with the
#: reason it is allowed to. All four read a **published** record and reproduce it
#: byte-for-byte; gating them would re-grade a dispositioned page, which is what
#: the CEO scoped out of ABL-434 and what `reports/abl_434_*` reports instead.
#:
#: Anything not on this list must go through `cell_grade`/`attach_grades`, which
#: read the cell's coverage. A new entry here is a decision to publish an ungated
#: grade and belongs in review.
MARGIN_ONLY_READERS = {
    "scripts/abl418_retro_grade.py":
        "reproduces reports/abl_418_retro_grade.json; its own tranches (1b, 2a, 2b) "
        "are fully covered, and it prints `enough_pairs` in its own column",
    "scripts/abl437_causal_levelling_reread.py":
        "the published levelling re-read; pinned to FIT_WINDOW/SIGN_TEST for the "
        "same reason",
    "scripts/abl443_offshore_trailing_reread.py":
        "the published offshore re-read; both its pairs are fully covered",
    "scripts/abl444_g23_floor_reread.py":
        "the published G2/G3 floor re-read, asserted byte-identical to a live run "
        "by tests/test_abl444_g23_readability_floor.py",
    # Added by ABL-467, and it is a **repair of a red main** rather than a new
    # decision to publish an ungated grade. ABL-427 (PR #80) and ABL-434 (PR #79)
    # merged back to back, each green on the base it was branched from: #80 landed
    # a fifth `grade_cell` caller and #79 landed the registry that has to name it,
    # and neither branch could see the other. This assertion has been failing on
    # `main` since ca3c7f8, on a tree neither author ever ran.
    #
    # The entry is the one ABL-434 would have written had the script existed then
    # -- checked, not assumed: all six of ABL-427's cells clear their registered
    # minimum n (720/684, 720/684, 510/456, and the same three for HR), so routing
    # it through the coverage gate would change no letter.
    "scripts/abl427_tranche2c_seed_reread.py":
        "the published k=12 seed re-read of tranche 2c; all six of its cells meet "
        "their registered minimum n, and it prints `meets_minimum_n` per cell",
}


def _grade_cell_callers():
    found = {}
    for path in sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]):
        if path == ROOT / "src" / "evaluation" / "gate_grading.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and (getattr(node.func, "id", "") == "grade_cell"
                      or getattr(node.func, "attr", "") == "grade_cell")]
        if calls:
            found[path.relative_to(ROOT).as_posix()] = len(calls)
    return found


def test_every_ungated_caller_is_registered_with_a_reason():
    """The `tso_plausibility` sweep pattern: a module may grade a margin only if
    it says why. A new script that copies `grade_cell(cell["scores"], ...)` fails
    here rather than publishing a coverage-blind A."""
    assert set(_grade_cell_callers()) == set(MARGIN_ONLY_READERS), (
        "an unregistered caller grades from scores while holding a cell; route it "
        "through cell_grade/attach_grades or register it above with a reason")
    for reason in MARGIN_ONLY_READERS.values():
        assert len(reason) > 40


def test_the_harnesses_grade_through_the_cell_level_entry_points():
    """Both harnesses must reach the ladder with the whole cell, or the gate is
    decorative: `attach_grades` writes the record and `cell_grade` renders it."""
    for name in ("evaluate_solar_retrain.py", "evaluate_wind_retrain.py"):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {getattr(node.func, "id", "") for node in ast.walk(tree)
                  if isinstance(node, ast.Call)}
        assert {"attach_grades", "cell_grade"} <= called, name
        assert "grade_cell" not in called, f"{name} grades a margin without its cell"


def test_grade_cell_takes_no_coverage_argument():
    """It stays a function of `scores` alone. That is the property which makes a
    stored record re-gradeable and a published re-read reproducible, and it is
    the reason the gate lives one level up instead of here."""
    import inspect
    parameters = set(inspect.signature(grade_cell).parameters)
    assert not parameters & {"cell", "gate", "coverage", "enough_pairs", "n", "minimum_n"}
    # And it emits no coverage key, so a caller splatting `as_dict()` over its own
    # `enough_pairs` column cannot have it silently overwritten with a null.
    assert "enough_pairs" not in grade_cell(_cell()["scores"], "solar").as_dict()
    assert "enough_pairs" in cell_grade(_cell(), "solar").as_dict()
