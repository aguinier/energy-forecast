"""ABL-467: the committed re-grade is what a live run of the script produces.

ABL-444's pattern. A committed record and the code that writes it drift silently
otherwise, and this one carries a *letter change* on a published pair -- HR
``U`` -> ``A`` -- so the cost of that drift is a promotion-eligibility claim
nobody can reproduce.

The re-grade reads two records and no database, so a live run here is arithmetic
and costs nothing.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.gate_grading import DELTA_MIN, STUDENT_T  # noqa: E402


def _script():
    spec = importlib.util.spec_from_file_location(
        "abl467_seed_interval_regrade", ROOT / "scripts" / "abl467_seed_interval_regrade.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _script()
COMMITTED = json.loads((ROOT / "reports" / "abl_467_t2c_regrade.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live():
    return SCRIPT.regrade(SCRIPT.load_source())


def test_the_committed_record_is_what_the_script_produces_now(live):
    assert live["cells"] == COMMITTED["cells"]
    assert live["pair_grades"] == COMMITTED["pair_grades"]


def test_the_source_blob_pin_matches_what_git_stores():
    """The pin is the thing standing between this read and a re-graded vintage.

    Computed over LF-normalised bytes so it is invariant to `core.autocrlf`,
    which is what fired on this script's first run and would otherwise make the
    pin a checkout-policy check rather than a content one.
    """
    path = ROOT / SCRIPT.SOURCE
    assert SCRIPT.blob_hash(path) == SCRIPT.SOURCE_BLOB
    assert COMMITTED["meta"]["sources"][SCRIPT.SOURCE]["pinned_blob_matches"] is True


def test_a_different_vintage_of_the_source_is_refused(tmp_path, monkeypatch):
    """Not a hypothetical: that file merged to main while this issue was open."""
    decoy = tmp_path / "abl_427_tranche2c_seed_reread.json"
    decoy.write_text('{"cells": []}', encoding="utf-8")
    monkeypatch.setattr(SCRIPT, "ROOT", tmp_path)
    monkeypatch.setattr(SCRIPT, "SOURCE", decoy.name)
    with pytest.raises(SystemExit, match="not the pinned"):
        SCRIPT.load_source()


def test_a_missing_source_stops_rather_than_degrading(tmp_path, monkeypatch):
    monkeypatch.setattr(SCRIPT, "ROOT", tmp_path)
    monkeypatch.setattr(SCRIPT, "SOURCE", "nothing-here.json")
    with pytest.raises(SystemExit, match="nothing to fall back to"):
        SCRIPT.load_source()


def test_the_outcome_is_the_one_registered_before_it_ran(live):
    """The whole point of §0 of the registration: the prediction was fixed in
    writing first, so the re-grade cannot quietly return something else."""
    assert SCRIPT.check_prediction(live) == [
        line for line in SCRIPT.check_prediction(live) if "MISMATCH" not in line]
    expected = live["prediction_registered_before_this_ran"]["expected"]
    for cell in live["cells"]:
        assert cell["amended_disposition"] == expected[
            f"{cell['country']}_{cell['horizon_band']}"]
    for country, pair in live["pair_grades"].items():
        assert pair["amended_letter"] == expected[f"{country}_pair"]


def test_hr_is_the_only_pair_that_moves(live):
    moved = {country for country, pair in live["pair_grades"].items() if pair["moves"]}
    assert moved == {"HR"}
    assert live["pair_grades"]["HR"]["published_disposition"] == "U"
    assert live["pair_grades"]["HR"]["amended_letter"] == "A"
    assert live["pair_grades"]["IT"]["amended_letter"] == "U"


def test_two_cells_move_and_both_are_the_ones_abl427_named(live):
    moved = {(c["country"], c["horizon_band"]) for c in live["cells"] if c["moves"]}
    assert moved == {("IT", "24-36h"), ("HR", "48-64h")}


def test_the_unamended_ladder_agrees_with_the_amendment_on_every_cell(live):
    """The fact that stops this reading as a rule chosen to pass a pair: this
    module's own floor at k=12, unamended, gives the same six letters. Only
    ABL-427's stricter scope-level choice disagrees."""
    for cell in live["cells"]:
        assert cell["unamended_delta_min_at_k12"]["letter"] == cell["amended_disposition"], \
            f"{cell['country']} {cell['horizon_band']}"


def test_all_three_hr_cells_clear_a_wider_width_than_the_unamended_floor(live):
    floor = live["meta"]["delta_min_floor_pct_at_k12"]
    hr = [c for c in live["cells"] if c["country"] == "HR"]
    assert len(hr) == 3
    for cell in hr:
        assert cell["seed_interval"]["half_width_pp"] > floor
        assert cell["amended_disposition"] == "A"


def test_every_cell_meets_its_registered_minimum_n(live):
    """So ABL-434's coverage gate holds none of them, and the letters here are not
    quietly resting on rows the registration does not have."""
    for cell in live["cells"]:
        assert cell["meets_minimum_n"] is True
        assert cell["n"] >= cell["minimum_n"]


def test_the_read_is_recorded_as_seed_decided_and_carries_its_derivation(live):
    for cell in live["cells"]:
        assert cell["amended"]["readability_test"] == STUDENT_T
        interval = cell["amended"]["seed_interval"]["seasonal_naive"]
        assert interval["n_seeds"] == 12
        # The point estimate is the printed skill column, not a re-derived one.
        assert interval["mean_skill_pct"] == pytest.approx(cell["skill_vs_d7_pct"], abs=1e-9)
        # `draws_losing` is the number ABL-427 §5 says should govern any serving
        # conversation, and an interval does not show it.
        assert 1 <= interval["draws_losing"] <= 3


def test_no_refit_no_replica_and_abl427s_record_is_untouched(live):
    meta = live["meta"]
    assert meta["refit"] is False
    assert meta["replica_opened"] is False
    assert meta["artifact_saved"] is False
    assert meta["scope"] != meta["regrade_of"], "the re-grade must be a new scope"


def test_the_regraded_scope_inherits_the_amendment_and_abl427_stays_pinned():
    spec = importlib.util.spec_from_file_location(
        "evaluate_solar_retrain", ROOT / "scripts" / "evaluate_solar_retrain.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    assert harness.seed_readability_for(COMMITTED["meta"]["scope"]) == STUDENT_T
    assert harness.seed_readability_for(COMMITTED["meta"]["regrade_of"]) == DELTA_MIN


def test_the_plus_is_collapsed_because_this_read_is_the_reread_it_asks_for():
    """`U(+)` means *re-read at k>1 seeds*, and at k=12 that has been done. The
    ladder still cannot see it -- the amendment does not change `plus` -- so the
    collapse is applied here, as ABL-427 applied it, and both forms are recorded."""
    assert SCRIPT.disposition("U(+)") == "U"
    assert SCRIPT.disposition("U") == "U"
    assert SCRIPT.disposition("A") == "A"
    it = next(c for c in COMMITTED["cells"]
              if c["country"] == "IT" and c["horizon_band"] == "36-48h")
    assert it["amended"]["label"] == "U(+)", "the raw ladder label is kept"
    assert it["amended_disposition"] == "U", "and the disposition collapses it"
