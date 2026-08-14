"""ABL-443: guards on the offshore trailing-reference read.

Four things have to stay true, and none of them is checkable by reading the
report:

1. **ABL-436's read still reproduces.** `abl322-pilot` is a published scope and
   stays pinned to `fit_window`; this issue adds a *second* scope rather than
   re-levelling the first. The moment that pin moves, ABL-436's committed page of
   `A`s stops being what a re-run produces.
2. **The letters in the record are the ladder's**, not the renderer's. Every
   amended grade in `reports/abl_443_offshore_trailing_reread.json` is re-derived
   here from that cell's own stored scores, and every published grade is checked
   against ABL-436's committed record. A reporting script that grades on the side
   is how two reads of one cell come to disagree.
3. **The refusal list is the registration's**, derived from
   `experiments/ABL443/config.json` rather than retyped. ABL-387's failure was a
   default output path overwriting a dispositioned read and exiting 0.
4. **The readability label is a diagnostic, never a condition.** ABL-437 declined
   to widen G2/G3 to a floor test; a label that could move a letter would be that
   change made silently.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.gate_grading import (  # noqa: E402
    LADDER_REFERENCES, grade_cell, pair_grade, readability_floor_pct,
)
from src.evaluation.model_free_reference import FIT_WINDOW, TRAILING_28D  # noqa: E402

REGISTRATION = json.loads((ROOT / "experiments" / "ABL443" / "config.json")
                          .read_text(encoding="utf-8"))
READ = json.loads((ROOT / "reports" / "abl_443_offshore_trailing_reread.json")
                  .read_text(encoding="utf-8"))
SOURCE = json.loads((ROOT / REGISTRATION["input_record"]["path"]).read_text(encoding="utf-8"))


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


READER = _module("abl443_offshore_trailing_reread")
HARNESSES = {stream: _module(f"evaluate_{stream}_retrain") for stream in ("wind", "solar")}


def _cells():
    for pair in READ["pairs"]:
        for cell in pair["cells"]:
            if cell.get("reconstructed"):
                yield pair, cell


# --------------------------------------------------------------------------
# 1. ABL-436's read still reproduces
# --------------------------------------------------------------------------

def test_the_scope_ABL436_published_under_is_still_pinned_to_the_fit_window():
    """This issue adds a scope; it does not re-level the one already published."""
    harness = HARNESSES["wind"]
    published_scope = READ["source_record_scope"]
    assert published_scope == "abl322-pilot"
    assert harness.CAUSAL_LEVELLING[published_scope] == FIT_WINDOW
    assert harness.causal_levelling_for(published_scope) == FIT_WINDOW


def test_this_read_declares_a_scope_of_its_own_and_registers_no_fit():
    """A re-read is not a run. It must not appear in the tables that drive one.

    `SCOPES` and `SCOPE_OUTPUTS` describe scopes the harness can *fit and write*;
    this scope trains nothing and owns no artifact directory, so appearing there
    would advertise a run that does not exist.
    """
    assert READER.SCOPE == REGISTRATION["scope"] == "abl443-offshore-trailing"
    assert READ["scope"] == READER.SCOPE
    for harness in HARNESSES.values():
        assert READER.SCOPE not in harness.SCOPES
        assert READER.SCOPE not in harness.SCOPE_OUTPUTS
        assert READER.SCOPE not in harness.CAUSAL_LEVELLING


def test_the_source_record_is_the_one_the_registration_pinned():
    """By content, not by filename -- a path can be repointed silently."""
    import hashlib
    path = ROOT / REGISTRATION["input_record"]["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest.startswith(REGISTRATION["input_record"]["sha256"])
    assert READ["source_record_sha256"] == digest
    assert READ["source_record"] == REGISTRATION["input_record"]["path"]


# --------------------------------------------------------------------------
# 2. The letters in the record are the ladder's
# --------------------------------------------------------------------------

def test_every_amended_grade_is_what_the_ladder_computes_from_its_own_scores():
    """Re-derived from the record's own WAPEs, under `trailing_28d`.

    The record carries the four ABL-389 references, the two ABL-437 trailing ones
    and the challenger's slope and correlation, so the whole ladder is checkable
    without the replica. If this ever disagrees, the report is prose and not a
    measurement.
    """
    checked = 0
    for pair, cell in _cells():
        source = next(item for item in SOURCE["gate_cells"]
                      if item["country"] == pair["country"]
                      and item["horizon_band"] == cell["band"])
        scores = dict(source["scores"])
        for name in LADDER_REFERENCES[TRAILING_28D].values():
            scores[name] = {"wape_pct": cell["wape"][name], "n": cell["comparator_n"][name]}
        recomputed = grade_cell(scores, READ["stream"], levelling=TRAILING_28D)
        assert recomputed.label == cell["amended_grade"], (pair["pair"], cell["band"])
        assert [name for name, _ in recomputed.failed] == cell["amended_failed"]
        checked += 1
    assert checked == 6, "all six cells must be re-derivable, not some of them"


def test_every_published_grade_is_ABL436s_committed_letter():
    """The 'before' column is read out of ABL-436's record, never recomputed.

    ABL-436's record predates ABL-437, so its grade blocks carry no
    `causal_levelling` key at all. That absence is not a gap to paper over: it is
    the case `CellGrade.from_dict` resolves to `fit_window`, which is what those
    letters were decided on. Asserted through that read-back rather than off the
    raw key, because the read-back is the path this report actually used.
    """
    from src.evaluation.gate_grading import CellGrade
    for pair, cell in _cells():
        source = next(item for item in SOURCE["gate_cells"]
                      if item["country"] == pair["country"]
                      and item["horizon_band"] == cell["band"])
        assert cell["published_grade"] == source["grade"]["label"], (pair["pair"], cell["band"])
        assert "causal_levelling" not in source["grade"], (
            "ABL-436's record has gained a levelling key -- it was edited")
        assert CellGrade.from_dict(source["grade"]).levelling == FIT_WINDOW


def test_each_pair_grade_is_its_worst_band_under_both_levellings():
    """A pair grades on its worst band; a mean or a majority would hide the B."""
    for pair in READ["pairs"]:
        cells = [cell for cell in pair["cells"] if cell.get("reconstructed")]
        for key, field in (("published_pair_grade", "published_grade"),
                           ("amended_pair_grade", "amended_grade")):
            worst = pair_grade([_grade_of(cell[field]) for cell in cells]).label
            assert pair[key] == worst, (pair["pair"], key)


def _grade_of(label: str):
    """A bare label back into something `pair_grade` can rank."""
    from src.evaluation.gate_grading import CellGrade
    plus = label.endswith("(+)")
    return CellGrade(grade=label[:-3] if plus else label, plus=plus)


def test_the_challenger_and_the_bar_are_ABL436s_numbers_unchanged():
    """No refit. Every column this read did not compute comes from the record."""
    for pair, cell in _cells():
        source = next(item for item in SOURCE["gate_cells"]
                      if item["country"] == pair["country"]
                      and item["horizon_band"] == cell["band"])
        for name in ("challenger", "seasonal_naive", "constant_causal", "constant_oracle",
                     "climatology_causal", "climatology_oracle"):
            assert cell["wape"][name] == source["scores"][name]["wape_pct"], name


# --------------------------------------------------------------------------
# 3. The refusal list is the registration's
# --------------------------------------------------------------------------

def test_the_refusal_list_covers_every_path_the_registration_protects():
    assert set(REGISTRATION["paths_this_read_must_not_write"]) <= set(READER.PROTECTED)


@pytest.mark.parametrize("path", sorted(READER.PROTECTED))
def test_writing_a_protected_path_is_refused_and_the_path_is_tracked(path):
    """Refused, and the thing being refused is a real committed record."""
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path],
                             cwd=ROOT, capture_output=True, text=True)
    if tracked.returncode != 0:
        pytest.skip(f"{path} is not committed in this tree")
    with pytest.raises(SystemExit):
        READER._refuse_protected(ROOT, path)
    READER._refuse_protected(ROOT, "reports/abl_443_offshore_trailing_reread.md")


def test_the_read_writes_only_where_its_registration_says_it_writes():
    registered = REGISTRATION["outputs"]
    assert registered["json_out"] == "reports/abl_443_offshore_trailing_reread.json"
    assert registered["report_out"] == "reports/abl_443_offshore_trailing_reread.md"
    for key in ("json_out", "report_out", "findings"):
        assert (ROOT / registered[key]).exists(), registered[key]
        assert registered[key] not in READER.PROTECTED


# --------------------------------------------------------------------------
# 4. The readability label is a diagnostic, never a condition
# --------------------------------------------------------------------------

def test_the_readability_label_is_derived_from_the_floor_and_nothing_else():
    floor = readability_floor_pct(READ["stream"], 1)
    assert floor == READ["readability_floor_pct_k1"]
    assert READER._readability(None, floor) == "not measured"
    assert READER._readability(floor, floor) == "not readable at one seed"
    assert READER._readability(-floor, floor) == "not readable at one seed"
    assert READER._readability(floor + 0.01, floor) == "readable"
    assert READER._readability(-floor - 0.01, floor) == "readable loss"


def test_a_sub_floor_margin_does_not_move_a_letter():
    """The whole point of calling it a diagnostic.

    Every G2/G3 outcome in the record must equal the plain sign test on its own
    margin, whatever the readability label beside it says.
    """
    assert READ["g2_g3_floor_is_a_ladder_condition"] is False
    for pair, cell in _cells():
        for condition, name in LADDER_REFERENCES[TRAILING_28D].items():
            margin = cell["margins_pct"][name]["skill_pct"]
            assert cell["amended_conditions"][condition] is (margin > 0), (
                pair["pair"], cell["band"], condition)


def test_every_margin_is_reported_including_the_unreadable_ones():
    """The issue asks for the margin in every case. A missing one is a silent cap."""
    required = (set(LADDER_REFERENCES[TRAILING_28D].values())
                | set(LADDER_REFERENCES[FIT_WINDOW].values())
                | {"seasonal_naive", "constant_oracle", "climatology_oracle"})
    for pair, cell in _cells():
        assert required <= set(cell["margins_pct"]), (pair["pair"], cell["band"])
        for name, entry in cell["margins_pct"].items():
            assert entry["skill_pct"] is not None, (pair["pair"], cell["band"], name)
            assert entry["readability"] in {
                "readable", "readable loss", "not readable at one seed", "not measured"}


def test_the_oracle_references_are_reported_and_on_no_ladder():
    """Unchanged from ABL-389 and ABL-437: losing to an oracle bounds a verdict,
    it does not void one. This read must not have quietly promoted one."""
    for levelling in (FIT_WINDOW, TRAILING_28D):
        assert not {"constant_oracle", "climatology_oracle"} & set(
            LADDER_REFERENCES[levelling].values())
    for pair, cell in _cells():
        for name in ("constant_oracle", "climatology_oracle"):
            assert cell["margins_pct"][name]["condition"] == "reported only"


# --------------------------------------------------------------------------
# The read's own integrity
# --------------------------------------------------------------------------

def test_every_cell_reconstructed_or_it_is_named():
    """A cell that did not reproduce its published references is graded by nobody."""
    assert READ["not_reconstructible"] == []
    assert READ["reconstruction_tolerance"] == 1e-09
    assert sum(len(pair["cells"]) for pair in READ["pairs"]) == 6


def test_the_coverage_field_is_read_by_value_and_not_by_presence():
    """ABL-438: `enough_pairs` nests under `gate`, where a flat lookup passes
    vacuously. A coverage-short cell that beat D-7 grades A exactly as a full one."""
    for pair, cell in _cells():
        assert cell["enough_pairs"] is True, (pair["pair"], cell["band"])


def test_the_residual_inflation_is_reported_for_both_causal_references():
    """The trailing window reduces the mis-levelling; it does not remove it, and
    ABL-437 registered that the residual is reported per cell rather than assumed
    away. Both numbers, every cell -- the value is a finding, its presence is a guard."""
    for pair, cell in _cells():
        assert set(cell["level_inflation_pct"]) == {"constant_causal", "constant_causal_28d"}
        for name, value in cell["level_inflation_pct"].items():
            assert value is not None, (pair["pair"], cell["band"], name)


def test_the_registration_was_committed_before_the_read_existed():
    """Order is the whole claim, and it is checkable in git rather than asserted."""
    def first_commit(path):
        out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--", path],
                             cwd=ROOT, capture_output=True, text=True)
        shas = [line for line in out.stdout.split() if line]
        return shas[-1] if shas else None

    registration = first_commit("experiments/ABL443/config.json")
    read_script = first_commit("scripts/abl443_offshore_trailing_reread.py")
    if not registration or not read_script:
        pytest.skip("one of the two paths is not committed yet")
    if registration == read_script:
        pytest.fail("the registration and the read landed in one commit; "
                    "the ordering claim is then unverifiable")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", registration, read_script],
                              cwd=ROOT, capture_output=True)
    assert ancestor.returncode == 0, "the registration must be an ancestor of the read"
