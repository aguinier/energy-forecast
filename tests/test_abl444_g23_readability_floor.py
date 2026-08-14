"""ABL-444: the readability floor on G2 and G3, and the registration around it.

Five things need holding, and they fail in different directions.

1. **The floor is G1's floor, not a second one.** ABL-444 reuses
   ``readability_floor_pct``; a second constant would drift from it, which is
   the ABL-381 failure of quoting another stream's margins.
2. **``N`` is an abstention, not a failure and not a pass.** It must not promote,
   it must not be reported as ``B``, and a cell with a *readable* failure beside
   an unreadable condition must stay ``B`` -- there is something to report and
   ``N`` would bury it.
3. **The floored form cannot raise a grade.** Over every committed cell, no
   ``B`` becomes ``A`` and no ``C`` or ``U`` moves at all. If it ever could, the
   amendment would be a way of promoting on noise rather than refusing to.
4. **Every published scope is pinned to ``sign_test``.** Derived from
   ``SCOPE_OUTPUTS`` + git rather than typed here, on the ABL-404 precedent, and
   asserted on the *value* rather than on the row's presence.
5. **Every call site that grades a published record names the form.** The
   default is the amendment, so a site that forgets it re-decides a committed
   letter silently -- ABL-404 again, one directory over.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.gate_grading import (  # noqa: E402
    FLOORED, GRADE_SEVERITY, G23_READABILITY_FORMS, SIGN_TEST, CellGrade,
    cell_grade, grade_cell, grading_prose, pair_grade, readability_floor_pct,
)
from src.evaluation.model_free_reference import FIT_WINDOW, TRAILING_28D  # noqa: E402


def _harness(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARNESSES = {"wind": _harness("evaluate_wind_retrain"),
             "solar": _harness("evaluate_solar_retrain")}

#: The re-read record ABL-437 committed, which is every graded cell in the
#: programme with both reference pairs already scored on the same rows. The
#: affected-set assertions below read it rather than refitting anything.
REREAD = ROOT / "reports" / "abl_437_causal_levelling_reread.json"

LADDER_REFS = {FIT_WINDOW: ("constant_causal", "climatology_causal"),
               TRAILING_28D: ("constant_causal_28d", "climatology_causal_28d")}


def _scores(stream, *, d7_skill, g2_skill, g3_skill, slope=1.0, correlation=0.9,
            levelling=TRAILING_28D, challenger=50.0):
    """A cell whose three margins are exactly the percentages asked for.

    ``skill = 100 * (1 - challenger / reference)``, so ``reference =
    challenger / (1 - skill/100)``. Building the references from the margin
    rather than the other way round is what makes a case at 0.99x the floor
    readable in the test as a number, not as a fixture nobody can check.
    """
    def reference(skill):
        return challenger / (1.0 - skill / 100.0)

    level, shape = LADDER_REFS[levelling]
    return {"challenger": {"wape_pct": challenger, "slope": slope,
                           "correlation": correlation},
            "seasonal_naive": {"wape_pct": reference(d7_skill)},
            level: {"wape_pct": reference(g2_skill)},
            shape: {"wape_pct": reference(g3_skill)},
            "constant_oracle": {"wape_pct": reference(g2_skill)}}


# ---------------------------------------------------------------------------
# 1. The floor is G1's floor


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_the_g23_floor_is_the_same_number_g1_uses(stream):
    """One floor, keyed on stream and k, used by every condition that has a
    margin. A second constant is what ABL-444 explicitly declined to add."""
    floor = readability_floor_pct(stream, 1)
    # Just inside on G3, everything else clear: the cell abstains, and the floor
    # it records is the one G1 was tested against.
    inside = grade_cell(_scores(stream, d7_skill=floor * 3, g2_skill=floor * 3,
                                g3_skill=floor * 0.99), stream,
                        g23_readability=FLOORED)
    assert inside.grade == "N"
    assert inside.floor_pct == pytest.approx(floor)
    # Just outside: the same sign, now readable.
    outside = grade_cell(_scores(stream, d7_skill=floor * 3, g2_skill=floor * 3,
                                 g3_skill=floor * 1.01), stream,
                         g23_readability=FLOORED)
    assert outside.grade == "A"


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_the_floor_narrows_with_k_on_g23_exactly_as_on_g1(stream):
    """A margin unreadable at one seed can be readable at four, and the floored
    ladder has to follow `readability_floor_pct` rather than a k=1 constant."""
    floor_k1 = readability_floor_pct(stream, 1)
    margin = floor_k1 * 0.9
    scores = _scores(stream, d7_skill=floor_k1 * 3, g2_skill=floor_k1 * 3,
                     g3_skill=margin)
    assert grade_cell(scores, stream, k=1, g23_readability=FLOORED).grade == "N"
    assert grade_cell(scores, stream, k=4, g23_readability=FLOORED).grade == "A"


def test_an_unknown_readability_form_raises_rather_than_defaulting():
    """A guessed form is a guessed verdict. There is no third value."""
    with pytest.raises(ValueError):
        grade_cell(_scores("wind", d7_skill=30, g2_skill=30, g3_skill=30), "wind",
                   g23_readability="floored_maybe")
    assert set(G23_READABILITY_FORMS) == {SIGN_TEST, FLOORED}


# ---------------------------------------------------------------------------
# 2. N is an abstention


def test_n_is_reached_only_from_a_or_b_and_never_promotes():
    """The claim the caveat rests on: the floored form cannot raise a grade."""
    floor = readability_floor_pct("wind")
    # Would be A under a sign test; abstains under the floor.
    was_a = _scores("wind", d7_skill=floor * 3, g2_skill=floor * 3, g3_skill=floor * 0.5)
    assert grade_cell(was_a, "wind", g23_readability=SIGN_TEST).grade == "A"
    assert grade_cell(was_a, "wind", g23_readability=FLOORED).grade == "N"
    # Would be B under a sign test; abstains under the floor.
    was_b = _scores("wind", d7_skill=floor * 3, g2_skill=floor * 3, g3_skill=-floor * 0.5)
    assert grade_cell(was_b, "wind", g23_readability=SIGN_TEST).grade == "B"
    assert grade_cell(was_b, "wind", g23_readability=FLOORED).grade == "N"
    assert GRADE_SEVERITY["N"] > GRADE_SEVERITY["A"]


def test_a_readable_failure_beside_an_unreadable_condition_stays_b():
    """A definite failure outranks an abstention: there is something to report
    and `N` would bury it. Same argument ABL-418 used for B over U."""
    floor = readability_floor_pct("wind")
    cell = grade_cell(_scores("wind", d7_skill=floor * 3, g2_skill=-floor * 3,
                              g3_skill=floor * 0.5), "wind", g23_readability=FLOORED)
    assert cell.grade == "B"
    assert [name for name, _ in cell.failed] == ["G2"]
    assert [name for name, _ in cell.not_readable] == ["G3"]
    assert cell.detail == "B — fails G2; not readable on G3"


def test_a_failing_g4_beside_an_unreadable_g3_stays_b():
    """G4 is a sign test on the challenger's own slope and correlation. It has
    no margin, so it never routes through the abstention branch."""
    floor = readability_floor_pct("wind")
    cell = grade_cell(_scores("wind", d7_skill=floor * 3, g2_skill=floor * 3,
                              g3_skill=floor * 0.5, slope=-0.08, correlation=-0.14),
                      "wind", g23_readability=FLOORED)
    assert cell.grade == "B"
    assert [name for name, _ in cell.failed] == ["G4"]


def test_an_unmeasured_condition_is_still_a_failure_not_an_abstention():
    """`N` is *not readable*; `Not measured` is a comparator that scored no rows.
    Collapsing them would let a missing reference read as a deferral."""
    floor = readability_floor_pct("solar")
    scores = _scores("solar", d7_skill=floor * 3, g2_skill=floor * 3, g3_skill=floor * 3)
    scores["climatology_causal_28d"] = {"wape_pct": None}
    cell = grade_cell(scores, "solar", g23_readability=FLOORED)
    assert cell.grade == "B"
    assert [name for name, _ in cell.failed] == ["G3"]
    assert "not measured" in dict(cell.failed)["G3"]
    assert cell.not_readable == ()


def test_the_g1_branch_is_untouched_by_the_floored_form():
    """ABL-444 changes the A/B branch. `U`, `U(+)` and `C` are ABL-418's, and the
    plus test is where the floor was *already* applied to G2/G3."""
    floor = readability_floor_pct("solar")
    for readability in G23_READABILITY_FORMS:
        undecided = grade_cell(_scores("solar", d7_skill=floor * 0.5, g2_skill=floor * 3,
                                       g3_skill=floor * 3), "solar",
                               g23_readability=readability)
        assert undecided.label == "U(+)"
        lost = grade_cell(_scores("solar", d7_skill=-floor * 3, g2_skill=floor * 3,
                                  g3_skill=floor * 3), "solar",
                          g23_readability=readability)
        assert lost.grade == "C"


def test_a_pair_takes_its_worst_band_so_one_n_band_carries_the_pair():
    """An `A / A / N` pair is not "mostly promotable"."""
    floor = readability_floor_pct("wind")
    clear = grade_cell(_scores("wind", d7_skill=floor * 3, g2_skill=floor * 3,
                               g3_skill=floor * 3), "wind", g23_readability=FLOORED)
    abstain = grade_cell(_scores("wind", d7_skill=floor * 3, g2_skill=floor * 3,
                                 g3_skill=floor * 0.5), "wind", g23_readability=FLOORED)
    assert pair_grade([clear, clear, abstain]).label == "N"
    # ...and a readable failure in any band still outranks it.
    fails = grade_cell(_scores("wind", d7_skill=floor * 3, g2_skill=-floor * 3,
                               g3_skill=floor * 3), "wind", g23_readability=FLOORED)
    assert pair_grade([abstain, fails]).label == "B"


def test_the_margin_is_recorded_on_an_abstaining_cell():
    """The CEO's binding constraint: the floor decides gradeability, it does not
    replace the number. An `N` cell carries strictly more than the `A` it
    replaces, in both denominators, plus the reason naming the margin."""
    floor = readability_floor_pct("solar")
    cell = grade_cell(_scores("solar", d7_skill=floor * 3, g2_skill=floor * 3,
                              g3_skill=floor * 0.5), "solar", g23_readability=FLOORED)
    record = cell.as_dict()
    assert record["grade"] == "N"
    assert record["skill_pct"]["climatology_causal_28d"] == pytest.approx(floor * 0.5)
    assert record["own_error_margin_pct"]["climatology_causal_28d"] is not None
    assert f"{floor * 0.5:+.2f}%" in record["not_readable"][0]["reason"]
    assert CellGrade.from_dict(record).not_readable == cell.not_readable


def test_a_record_written_before_abl444_rebuilds_as_a_sign_test():
    """Absence dates the read; it is not a default anyone chose after the fact."""
    old = {"grade": "A", "plus": False, "label": "A",
           "conditions": {"G1": True, "G2": True, "G3": True, "G4": True}}
    rebuilt = CellGrade.from_dict(old)
    assert rebuilt.g23_readability == SIGN_TEST
    assert rebuilt.not_readable == ()
    assert cell_grade({"grade": old}, "solar").g23_readability == SIGN_TEST


# ---------------------------------------------------------------------------
# 3. The affected set, over every committed cell


@pytest.fixture(scope="module")
def reread():
    return json.loads(REREAD.read_text(encoding="utf-8"))


def _regrade(reread, levelling, readability):
    """Every committed cell, graded from the ABL-437 re-read's stored WAPE.

    Arithmetic only -- the same inputs both harnesses wrote, no refit and no
    database. `pairs` is keyed by (tranche, pair) because a pair appears in more
    than one tranche.
    """
    pairs = {}
    for tranche in reread["tranches"]:
        stream = tranche["stream"]
        for pair in tranche["pairs"]:
            grades = []
            for cell in pair["cells"]:
                scores = {name: {"wape_pct": value}
                          for name, value in cell["wape"].items()}
                scores["challenger"] = {
                    "wape_pct": cell["wape"]["challenger"],
                    # G4's inputs are not in the re-read record; recover them
                    # from whether the published read named G4 as failed.
                    "slope": -1.0 if _g4_failed(cell) else 1.0,
                    "correlation": -1.0 if _g4_failed(cell) else 0.9}
                grades.append(grade_cell(scores, stream, levelling=levelling,
                                         g23_readability=readability))
            pairs[(tranche["tranche"], pair["pair"])] = (pair_grade(grades), grades)
    return pairs


def _g4_failed(cell):
    """The re-read record stores failed conditions as bare letters."""
    return "G4" in cell["published_failed"] + cell["amended_failed"]


def test_the_floored_form_never_makes_a_pair_promotable(reread):
    """The caveat that travels with this issue, checked over all 113 cells rather
    than argued: it reduces the promotable set or leaves it unchanged.

    Stated on the ``A`` set, not on ``GRADE_SEVERITY``. ``N`` ranks *better* than
    ``B`` on the ladder -- an abstention is a weaker negative than a named
    failure -- so a ``B -> N`` move lowers the severity while leaving the pair
    exactly as non-promotable as it was. Asserting monotone severity would be
    asserting the wrong thing, and it would fail on 2d NL solar.
    """
    for levelling in (FIT_WINDOW, TRAILING_28D):
        signed = _regrade(reread, levelling, SIGN_TEST)
        floored = _regrade(reread, levelling, FLOORED)
        assert set(signed) == set(floored)
        promotable_before = {key for key, (grade, _) in signed.items() if grade.grade == "A"}
        promotable_after = {key for key, (grade, _) in floored.items() if grade.grade == "A"}
        assert promotable_after <= promotable_before, sorted(promotable_after - promotable_before)
        for key, (before, _) in signed.items():
            after, _ = floored[key]
            # `N` is the only new letter, and only ever from A or B.
            if after.grade != before.grade:
                assert after.grade == "N" and before.grade in {"A", "B"}, key
        # ...and the same on every individual cell, not only on the roll-up.
        for key, (_, cells_before) in signed.items():
            for before, after in zip(cells_before, floored[key][1]):
                if after.grade != before.grade:
                    assert after.grade == "N" and before.grade in {"A", "B"}, key
    assert GRADE_SEVERITY["A"] < GRADE_SEVERITY["N"] < GRADE_SEVERITY["B"]


def test_the_published_path_moves_exactly_the_two_pairs_the_pack_names(reread):
    """Section 3.1 of the registration pack, as a number rather than a claim.

    No published `A` becomes `N`, so the promotable set as published is
    unchanged. If this ever moves, the pack is the thing that is wrong.
    """
    signed = _regrade(reread, FIT_WINDOW, SIGN_TEST)
    floored = _regrade(reread, FIT_WINDOW, FLOORED)
    moved = {key: (signed[key][0].label, floored[key][0].label)
             for key in signed if signed[key][0].label != floored[key][0].label}
    assert moved == {("2d", "NL solar"): ("B", "N"),
                     ("2e", "HU wind_onshore"): ("B", "N")}


def test_the_amended_path_moves_ten_tranche_pairs_five_of_them_from_a(reread):
    """Section 3.2 -- the set this issue said had not been enumerated: pairs that
    *pass* G2/G3 on a sub-floor margin. EE and FI solar are the sharpest, each
    grading A on its only gated band at +0.35% and +0.59%.

    Ten here, eleven in the published re-read: this fixture is ABL-437's record,
    which predates ABL-443's offshore scope. DE `wind_offshore` is the eleventh
    and is asserted against the committed re-read below.
    """
    signed = _regrade(reread, TRAILING_28D, SIGN_TEST)
    floored = _regrade(reread, TRAILING_28D, FLOORED)
    moved = {key: (signed[key][0].label, floored[key][0].label)
             for key in signed if signed[key][0].label != floored[key][0].label}
    assert len(moved) == 10
    from_a = sorted(key for key, (before, _) in moved.items() if before == "A")
    assert from_a == [("1b", "BG solar"), ("2a", "BG solar"), ("2d", "EE solar"),
                      ("2d", "FI solar"), ("2e", "HR wind_onshore")]
    assert sorted(key for key, (before, _) in moved.items() if before == "B") == \
        [("2a", "PL solar"), ("2a", "SK solar"), ("2d", "LT solar"),
         ("2d", "NL solar"), ("2d", "SE solar")]


def test_ch_wind_onshore_stays_b_at_pair_level(reread):
    """The correction to this issue's own framing, pinned so it is not re-lost.

    ABL-444 names PL solar (0.36pp) and CH wind_onshore (0.52pp) as the two flips
    inside the floor. PL reproduces at pair level; CH does not -- its 24-36h band
    fails G2 and G3 *readably*, and a pair takes its worst band. The 0.52pp is its
    tightest band, and a tightest-band margin reads like a pair-level one.
    """
    floored = _regrade(reread, TRAILING_28D, FLOORED)
    pair, cells = floored[("2f", "CH wind_onshore")]
    assert pair.label == "B"
    assert [cell.label for cell in cells] == ["B", "N", "N"]
    assert [name for name, _ in cells[0].failed] == ["G2", "G3"]
    assert floored[("2a", "PL solar")][0].label == "N"


def test_the_two_denominators_move_three_observations_and_no_pair_letter(reread):
    """Section 4. The registration is on the printed `skill` column; ABL-385's CV
    is measured in the challenger's own error. Measured rather than assumed --
    and the two do *not* agree everywhere, which is why the number is pinned.

    Three of 452 G2/G3 condition-observations change readability status. One of
    them moves a **cell** letter (1b BG solar 36-48h, `N` on the registered
    column against `A` on ABL-385's). No **pair** letter moves, because that
    pair's other two bands abstain under either denominator.
    """
    differ = []
    for tranche in reread["tranches"]:
        floor = readability_floor_pct(tranche["stream"])
        for pair in tranche["pairs"]:
            for cell in pair["cells"]:
                challenger = cell["wape"]["challenger"]
                for levelling in (FIT_WINDOW, TRAILING_28D):
                    for name in LADDER_REFS[levelling]:
                        reference = cell["wape"].get(name)
                        if reference is None or not challenger or not reference:
                            continue
                        skill = 100.0 * (1.0 - challenger / reference)
                        own = 100.0 * (reference - challenger) / challenger
                        if (abs(skill) <= floor) != (abs(own) <= floor):
                            differ.append((tranche["tranche"], pair["pair"],
                                           cell["band"], name))
    assert differ == [("1b", "BG solar", "36-48h", "climatology_causal_28d"),
                      ("2e", "RO wind_onshore", "36-48h", "constant_causal_28d"),
                      ("2f", "CH wind_onshore", "24-36h", "constant_causal_28d")]


# ---------------------------------------------------------------------------
# 4. Every published scope is pinned


def _tracked(path: Path) -> bool:
    result = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)],
                            cwd=ROOT, capture_output=True, text=True)
    return result.returncode == 0


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_every_published_scope_pins_a_sign_test(stream):
    """Derived from `SCOPE_OUTPUTS` + git, never from a list in this file --
    ABL-404's precedent, where a pin that had to be remembered went missing
    across a merge. Asserted on the value, not on the row's presence."""
    harness = HARNESSES[stream]
    published = {scope for scope, outputs in harness.SCOPE_OUTPUTS.items()
                 if any(_tracked(ROOT / outputs[key])
                        for key in ("json_out", "report_out") if outputs.get(key))}
    assert published, "no published scope found -- the derivation is broken, not the pins"
    missing = published - set(harness.G23_READABILITY)
    assert not missing, f"published scope(s) with no registered readability form: {sorted(missing)}"
    for scope in published:
        assert harness.G23_READABILITY[scope] == SIGN_TEST, (
            f"{scope} is published; its letters were decided by a sign test on G2/G3")


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_an_unregistered_scope_defaults_to_the_amendment(stream):
    """The direction the ABL-444 design sketch guessed the other way.

    A sub-floor margin is not a result that was measured, and the margin prints
    either way -- so inheriting `floored` hides nothing, while inheriting
    `sign_test` awards a letter on noise to the pairs nobody has looked at.
    """
    harness = HARNESSES[stream]
    assert harness.g23_readability_for("a-scope-that-does-not-exist") == FLOORED
    for scope, form in harness.G23_READABILITY.items():
        assert form in G23_READABILITY_FORMS, scope


# ---------------------------------------------------------------------------
# 5. Nothing grades a published record on the default


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_the_run_records_which_form_it_graded_under(stream):
    """A record that does not say cannot be re-read, and absence would otherwise
    resolve to the *amendment* on a read decided by a sign test."""
    source = (ROOT / "scripts" / f"evaluate_{stream}_retrain.py").read_text(encoding="utf-8")
    assert '"g23_readability": g23_readability_for(args.scope),' in source
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "id", "") == "attach_grades"]
    assert len(calls) == 1
    assert "g23_readability" in {kw.arg for kw in calls[0].keywords}


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_the_renderer_follows_the_record_and_not_the_table(stream):
    """Re-rendering a stored read must not re-decide it under a later pin."""
    source = (ROOT / "scripts" / f"evaluate_{stream}_retrain.py").read_text(encoding="utf-8")
    assert 'readability = meta.get("g23_readability", SIGN_TEST)' in source
    assert "g23_readability=readability" in source


@pytest.mark.parametrize("script", ["abl418_retro_grade", "abl437_causal_levelling_reread",
                                    "abl419_tranche2c_read", "abl421_tranche2d_read",
                                    "abl443_offshore_trailing_reread"])
def test_every_published_record_reader_names_the_form(script):
    """Four scripts grade cells from a *committed* record. Two of them read
    records that carry a recorded grade, so the default is unreachable there
    today -- and they name it anyway. A default that only happens to be
    unreachable is the ABL-404 shape."""
    source = (ROOT / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "id", "") in {"grade_cell", "cell_grade"}]
    assert calls, f"{script} no longer grades anything -- update this test, not the pin"
    for call in calls:
        keywords = {kw.arg: kw.value for kw in call.keywords}
        assert "g23_readability" in keywords, f"{script}:{call.lineno} grades on the default"
        assert getattr(keywords["g23_readability"], "id", None) == "SIGN_TEST", \
            f"{script}:{call.lineno} must pin the published form"


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_the_prose_says_which_form_and_what_n_means(stream):
    """A report that prints an `N` without saying what it is is worse than one
    that prints a `B`: the reader knows what a B claims."""
    floored = " ".join(grading_prose(stream, g23_readability=FLOORED))
    assert "ABL-444" in floored and "`floored`" in floored
    assert "not readable" in floored and "abstention" in floored
    assert f"{readability_floor_pct(stream):.2f}%" in floored
    signed = " ".join(grading_prose(stream, g23_readability=SIGN_TEST))
    assert "sign test" in signed and "abl_444_g23_floor_reread.md" in signed


# ---------------------------------------------------------------------------
# 6. The committed re-read


_REREAD_SPEC = importlib.util.spec_from_file_location(
    "abl444_g23_floor_reread", ROOT / "scripts" / "abl444_g23_floor_reread.py")
rr = importlib.util.module_from_spec(_REREAD_SPEC)
_REREAD_SPEC.loader.exec_module(rr)

FLOOR_RECORD = ROOT / "reports" / "abl_444_g23_floor_reread.json"


@pytest.fixture(scope="module")
def committed():
    return json.loads(FLOOR_RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fresh():
    """The re-read computed live, through the committed path -- not read back."""
    return rr.read(ROOT)


def test_the_committed_reread_matches_a_live_run(committed, fresh):
    """The record is arithmetic over files on disk, so it must regenerate exactly.
    `generated_at` is the only field allowed to move."""
    for record in (committed, fresh):
        record.pop("generated_at", None)
    assert fresh == committed


def test_the_reread_names_the_bytes_it_graded(committed):
    """A later reader can tell whether these letters came from the bytes that
    were dispositioned -- the ABL-438 rule, one document over."""
    assert committed["source_reread_sha256"] == \
        hashlib.sha256((ROOT / rr.SOURCE_REREAD).read_bytes()).hexdigest()
    for tranche in committed["tranches"]:
        assert tranche["record_sha256"] == \
            hashlib.sha256((ROOT / tranche["record"]).read_bytes()).hexdigest()


def test_the_sign_test_arms_reproduce_the_published_letters(committed, reread):
    """This document's `floored` column is a comparison, not a restatement, and
    that is only true if its `sign_test` columns reproduce ABL-437's -- through a
    different path, since the challenger side here is read from each tranche's
    own committed record rather than from ABL-437's."""
    published = {(tranche["tranche"], pair["pair"]):
                 (pair["published_pair_grade"], pair["amended_pair_grade"])
                 for tranche in reread["tranches"] for pair in tranche["pairs"]}
    offshore = json.loads((ROOT / rr.OFFSHORE_REREAD).read_text(encoding="utf-8"))
    published.update({("offshore", pair["pair"]):
                      (pair["published_pair_grade"], pair["amended_pair_grade"])
                      for pair in offshore["pairs"]})
    checked = 0
    for tranche in committed["tranches"]:
        for pair in tranche["pairs"]:
            grades = pair["pair_grades"]
            assert (grades[f"{FIT_WINDOW}/{SIGN_TEST}"],
                    grades[f"{TRAILING_28D}/{SIGN_TEST}"]) == \
                published[(tranche["tranche"], pair["pair"])], pair["pair"]
            checked += 1
    assert checked == 41


def test_the_reread_covers_abl437s_cells_and_abl443s(committed, reread):
    """113 from ABL-437's tranches, 6 from ABL-443's offshore scope, which merged
    to main while this issue was open. A floored read of the programme that
    skipped the one pair its own author flagged as unreadable would be worse than
    no read."""
    abl437 = sum(len(pair["cells"])
                 for tranche in reread["tranches"] for pair in tranche["pairs"])
    assert abl437 == 113
    cells = sum(len(pair["cells"])
                for tranche in committed["tranches"] for pair in tranche["pairs"])
    assert cells == 119
    offshore, = [t for t in committed["tranches"] if t["tranche"] == "offshore"]
    assert offshore["scope"] == "abl443-offshore-trailing"
    assert sum(len(pair["cells"]) for pair in offshore["pairs"]) == 6


def test_ee_and_fi_solars_only_gated_band_is_coverage_short_and_sub_floor(committed):
    """Section 3 of the findings, and the reason it is a compounding finding for
    ABL-434 rather than a duplicate of it.

    ABL-421 declares both pairs NOT-EVALUABLE on 24-36h and 36-48h, so 48-64h is
    the only band either carries a letter on -- and that cell failed the gate on
    coverage *and* passes G3 on a margin no seed resolves. Neither guard alone
    makes the letter honest.
    """
    short = [(tranche["tranche"], pair["pair"], cell)
             for tranche in committed["tranches"] for pair in tranche["pairs"]
             for cell in pair["cells"] if not cell["enough_pairs"]]
    assert [(t, p, cell["band"]) for t, p, cell in short] == \
        [("2d", "EE solar", "48-64h"), ("2d", "FI solar", "48-64h")]
    for _, _, cell in short:
        assert cell["gate_pass"] is False
        assert cell["n"] < cell["minimum_n"]
        assert cell["labels"][f"{TRAILING_28D}/{SIGN_TEST}"] == "A"
        assert cell["labels"][f"{TRAILING_28D}/{FLOORED}"] == "N"
        margin = cell["grades"][f"{TRAILING_28D}/{FLOORED}"]["skill_pct"]["climatology_causal_28d"]
        assert 0 < margin < 1.0


def test_bg_solars_hold_travels_with_both_of_its_moves(committed):
    """A grade of N that reads as "just re-run it at k>1" is as wrong as an A
    without the hold: BG's night-contamination displacement is far wider than the
    margin the floor abstains on."""
    held = [(tranche["tranche"], pair["pair"]) for tranche in committed["tranches"]
            for pair in tranche["pairs"] if pair["hold"]]
    assert held == [("1b", "BG solar"), ("2a", "BG solar")]
    for tranche in committed["tranches"]:
        for pair in tranche["pairs"]:
            if pair["hold"]:
                assert "ABL-396" in pair["hold"]
                assert pair["pair_grades"][f"{TRAILING_28D}/{FLOORED}"] == "N"


def test_the_reread_refuses_to_write_where_another_read_writes():
    """The `SCOPE_OUTPUTS` failure one directory over: a run that kept a default
    output path rewrote a dispositioned record and exited 0."""
    for marker in rr.PROTECTED:
        assert marker in " ".join(("abl_437_x", "abl_438_x", "abl_418_x"))
    argv = sys.argv
    try:
        sys.argv = ["abl444_g23_floor_reread.py", "--json-out",
                    "reports/abl_437_causal_levelling_reread.json"]
        with pytest.raises(SystemExit) as caught:
            rr.main()
        assert "refusing to write" in str(caught.value)
    finally:
        sys.argv = argv


def test_de_wind_offshore_is_the_case_the_floor_was_registered_for(committed):
    """ABL-443 did the diagnosis and had no machinery to act on it.

    Its record labels all six DE margins "not readable at one seed" and carries
    `g2_g3_floor_is_a_ladder_condition: false`. Under the floored form all three
    bands abstain -- the two shorter ones were graded `A` on +0.33% to +1.32%,
    and the `B` came from a G3 *failure* of -0.47%. Not one of the six was
    readable.
    """
    offshore, = [t for t in committed["tranches"] if t["tranche"] == "offshore"]
    by_pair = {pair["pair"]: pair for pair in offshore["pairs"]}
    de = by_pair["DE wind_offshore"]
    assert de["pair_grades"][f"{TRAILING_28D}/{SIGN_TEST}"] == "B"
    assert de["pair_grades"][f"{TRAILING_28D}/{FLOORED}"] == "N"
    # ...and its published fit-window letter is untouched: those margins are readable.
    assert de["pair_grades"][f"{FIT_WINDOW}/{SIGN_TEST}"] == "A"
    assert de["pair_grades"][f"{FIT_WINDOW}/{FLOORED}"] == "A"
    floor = readability_floor_pct("wind")
    for cell in de["cells"]:
        assert cell["labels"][f"{TRAILING_28D}/{FLOORED}"] == "N"
        for name in ("constant_causal_28d", "climatology_causal_28d"):
            margin = cell["grades"][f"{TRAILING_28D}/{FLOORED}"]["skill_pct"][name]
            assert abs(margin) <= floor, (cell["band"], name, margin)
    nl = by_pair["NL wind_offshore"]
    assert {nl["pair_grades"][arm] for arm in nl["pair_grades"]} == {"A"}


def test_the_committed_reread_moves_eleven_pairs_on_the_amended_path(committed):
    """The published count, over all 41 pair-records rather than ABL-437's 39."""
    assert sum(len(t["pairs"]) for t in committed["tranches"]) == 41
    assert len(rr._moves(committed, TRAILING_28D)) == 11
    assert len(rr._moves(committed, FIT_WINDOW)) == 2
    assert sum(1 for item in rr._moves(committed, FIT_WINDOW) if item["before"] == "A") == 0
