"""ABL-379: what a gate read reports when the comparison never happened.

Three behaviours, shared by both retrain harnesses, that decide whether a
new-country tranche read is usable:

  1. `scores_with_comparators` -- a comparator that does not exist for a pair
     costs its own row and nothing else, instead of emptying the cell.
  2. `gate_verdict` -- a cell that scored zero rows reads as no-disposition,
     not as a model-quality FAIL, and the bar is the scope's registered cell
     count rather than a literal 9 or 15.
  3. `render_markdown` -- an unmeasured cell renders as "Not measured", not as
     a number and not as a crash.

Item 2's absence is what made the ABL-322 wind pilot report FAIL on a race it
never ran; items 1 and 3 are why the solar harness could not have reported even
that much for a new country.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from src.evaluation.solar_retrain import PRIMARY_BANDS, gate_verdict, scores_with_comparators

TRANCHE_BASIS = ("challenger", "seasonal_naive")
REPORTED = ("challenger", "incumbent", "seasonal_naive", "persistence")


def _solar_harness():
    spec = importlib.util.spec_from_file_location(
        "_solar_harness", REPO / "scripts" / "evaluate_solar_retrain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(n=8, incumbent=np.nan, persistence=None):
    """A scoreable gate group for a country with no incumbent."""
    rng = np.arange(1.0, n + 1.0)
    return pd.DataFrame({
        "actual": rng * 100.0,
        "challenger": rng * 100.0 + 5.0,
        "seasonal_naive": rng * 100.0 + 25.0,
        "persistence": rng * 100.0 + 15.0 if persistence is None else persistence,
        "incumbent": incumbent,
    })


# --------------------------------------------------------------------------
# 1. A missing comparator reads "Not measured", it does not void the cell.
# --------------------------------------------------------------------------

def test_missing_incumbent_leaves_the_cell_scoreable():
    """The whole defect, in one assertion.

    Every one of ABL-316's 19 remaining solar pairs has zero rows in
    `forecasts`. With `incumbent` inside the basis, `common` is empty, the cell
    scores n=0, and `scores["challenger"]["wape_pct"]` is None.
    """
    frame = _frame(n=8, incumbent=np.nan)
    scores, common, comparator_n = scores_with_comparators(frame, TRANCHE_BASIS, REPORTED)

    assert len(common) == 8, "the gate basis intersection lost rows to a NaN comparator"
    assert scores["challenger"]["wape_pct"] is not None
    assert scores["seasonal_naive"]["wape_pct"] is not None
    # The absent comparator is unmeasured, and says so with its own n.
    assert scores["incumbent"]["wape_pct"] is None
    assert comparator_n["incumbent"] == 0
    assert comparator_n["challenger"] == comparator_n["seasonal_naive"] == 8


def test_the_old_four_way_basis_is_what_emptied_the_cell():
    """Contrast: the same frame under `abl253`'s registered basis scores nothing.

    This is not a regression test on the four-way basis -- ABL-253 keeps it
    deliberately. It pins *why* a tranche scope cannot use it.
    """
    scores, common = _common_scores(_frame(n=8, incumbent=np.nan), REPORTED)
    assert len(common) == 0
    assert scores["challenger"]["wape_pct"] is None


def _common_scores(frame, columns):
    from src.evaluation.solar_retrain import common_scores
    return common_scores(frame, columns)


def test_a_comparator_present_on_some_rows_carries_its_own_n():
    """`persistence` outside the basis is scored on its own intersection with
    it -- it neither empties the cell nor silently borrows the basis's n."""
    frame = _frame(n=8, incumbent=np.nan)
    frame.loc[frame.index[:3], "persistence"] = np.nan
    scores, common, comparator_n = scores_with_comparators(frame, TRANCHE_BASIS, REPORTED)

    assert len(common) == 8
    assert comparator_n["persistence"] == 5
    assert scores["persistence"]["wape_pct"] is not None


def test_comparators_inside_the_basis_report_the_basis_n():
    frame = _frame(n=6, incumbent=np.arange(1.0, 7.0) * 100.0 + 40.0)
    scores, common, comparator_n = scores_with_comparators(frame, REPORTED, REPORTED)
    assert set(comparator_n.values()) == {len(common)} == {6}


# --------------------------------------------------------------------------
# 2. The verdict: derived denominator, and a fourth outcome.
# --------------------------------------------------------------------------

def _cell(passed, n=720):
    return {"gate": {"pass": passed, "n": n}}


def test_a_two_pair_tranche_can_reach_pass():
    """6 registered cells, 6 passes -> PASS.

    Under `len(gate_cells) == 9 and passed == 9` this returned FAIL, and the
    FAIL text rendered it as "only 6/9 primary cells clear the registered bar".
    """
    disposition = gate_verdict([_cell(True)] * 6, registered_cells=6, contaminated=False)
    assert disposition["verdict"] == "PASS"
    assert disposition["passed"] == disposition["registered_cells"] == 6


def test_the_denominator_is_the_scopes_count_not_the_runs():
    """A pair that yields no cells at all must shortfall the bar, not shrink it.

    This is the property the hardcoded 9 existed to protect, and it survives.
    """
    disposition = gate_verdict([_cell(True)] * 3, registered_cells=6, contaminated=False)
    assert disposition["performance_pass"] is False
    assert disposition["verdict"] == "FAIL"
    assert disposition["scored_cells"] == 3
    assert disposition["registered_cells"] == 6


def test_zero_row_cells_read_as_no_disposition_not_fail():
    """The distinction ABL-322 produced and solar did not have.

    A cell that scored no rows did not lose a race -- it never ran one, and the
    correct response to that and to a genuine loss are opposite.
    """
    disposition = gate_verdict([_cell(False, n=0)] * 6, registered_cells=6, contaminated=False)
    assert disposition["verdict"] == "UNREADABLE"
    assert disposition["unreadable"] == 6


def test_one_unreadable_cell_is_enough_to_withhold_a_disposition():
    cells = [_cell(True), _cell(True), _cell(True), _cell(True), _cell(True), _cell(False, n=0)]
    assert gate_verdict(cells, 6, contaminated=False)["verdict"] == "UNREADABLE"


def test_a_genuine_loss_still_reads_fail():
    """No-disposition must not swallow a real model-quality finding."""
    cells = [_cell(True)] * 4 + [_cell(False, n=700)] * 2
    disposition = gate_verdict(cells, 6, contaminated=False)
    assert disposition["verdict"] == "FAIL"
    assert disposition["unreadable"] == 0
    assert disposition["passed"] == 4


def test_contamination_holds_a_full_pass():
    disposition = gate_verdict([_cell(True)] * 6, registered_cells=6, contaminated=True)
    assert disposition["verdict"].startswith("PERFORMANCE PASS")


def test_contamination_does_not_upgrade_a_shortfall():
    assert gate_verdict([_cell(True)] * 5 + [_cell(False)], 6,
                        contaminated=True)["verdict"] == "FAIL"


def test_the_default_scope_still_has_a_bar_of_nine():
    module = _solar_harness()
    assert len(module.SCOPES["abl253"]) * len(PRIMARY_BANDS) == 9
    assert gate_verdict([_cell(True)] * 9, 9, contaminated=False)["verdict"] == "PASS"
    assert gate_verdict([_cell(True)] * 8, 9, contaminated=False)["verdict"] == "FAIL"


# --------------------------------------------------------------------------
# 3. The report renders an unmeasured cell instead of crashing on it.
# --------------------------------------------------------------------------

UNMEASURED = {"n": 0, "wape_pct": None, "mae": None, "bias_pct": None,
              "slope": None, "correlation": None}


def _result(scope, registered_pairs, gate_basis, cells):
    return {
        "meta": {"generated_at": "2026-08-13 00:00 UTC",
                 "replica_db": "replica.db", "replica_bytes": 1,
                 "training_source": "energy_generation",
                 "scope": scope, "registered_pairs": registered_pairs,
                 "registered_cells": len(registered_pairs) * len(PRIMARY_BANDS),
                 "gate_basis": gate_basis,
                 "fit_window": {"start": "2026-01-14", "end_exclusive": "2026-07-11"},
                 "gate_window": {"start": "2026-07-11", "end_exclusive": "2026-08-10"},
                 "screen_window": {"start": "2025-12-31", "end_exclusive": "2026-08-10"}},
        "verdict": "UNREADABLE", "recommendation": "No disposition.",
        "training": [{"country": "BG", "algorithm": "catboost", "constant_runs": [],
                      "artifact_sha256": "deadbeef",
                      "audit": {"retained_rows": 0, "intended_rows": 0, "unique_targets": 0,
                                "excluded_missing_actual_or_feature": 0,
                                "degraded_lag_1d_rows": 0}}],
        "gate_cells": cells,
        "country_d2": [{"country": "BG", "n": 0,
                        "scores": {name: dict(UNMEASURED) for name in REPORTED},
                        "tso": dict(UNMEASURED)}],
    }


def test_an_unmeasured_cell_renders_instead_of_raising():
    """On `origin/main` this was a bare division of two Nones inside the report
    writer, so the no-incumbent case did not even reach its plausible FAIL -- it
    raised TypeError after every pair had already been fitted."""
    cells = [{"country": "BG", "horizon_band": band,
              "scores": {name: dict(UNMEASURED) for name in REPORTED},
              "comparator_n": {name: 0 for name in REPORTED},
              "gate": {"pass": False, "n": 0}} for band in PRIMARY_BANDS]
    report = _solar_harness().render_markdown(
        _result("abl348-level", [["solar", "BG"], ["solar", "CH"]],
                list(TRANCHE_BASIS), cells))

    assert "Not measured" in report
    assert "**Disposition: UNREADABLE**" in report


def test_the_report_names_the_scope_and_its_derived_bar():
    cells = [{"country": "BG", "horizon_band": band,
              "scores": {name: dict(UNMEASURED) for name in REPORTED},
              "comparator_n": {name: 0 for name in REPORTED},
              "gate": {"pass": False, "n": 0}} for band in PRIMARY_BANDS]
    report = _solar_harness().render_markdown(
        _result("abl348-level", [["solar", "BG"], ["solar", "CH"]],
                list(TRANCHE_BASIS), cells))

    assert "`abl348-level`" in report
    assert "all 6 country" in report, "the report still renders a hardcoded 9"
    assert "0/6 cells pass" in report
    assert "BG solar, CH solar" in report
    # ABL-253's measured row counts are not a property of this scope.
    assert "210/570/720/720/510" not in report


def test_the_default_scope_still_carries_its_protocol_count():
    cells = [{"country": country, "horizon_band": band,
              "scores": {name: dict(UNMEASURED) for name in REPORTED},
              "comparator_n": {name: 0 for name in REPORTED},
              "gate": {"pass": False, "n": 0}}
             for country in ("BE", "DE", "FR") for band in PRIMARY_BANDS]
    report = _solar_harness().render_markdown(
        _result("abl253", [["solar", "BE"], ["solar", "DE"], ["solar", "FR"]],
                list(REPORTED), cells))

    assert "210/570/720/720/510" in report
    assert "all 9 country" in report


def test_the_screen_window_is_the_one_actually_screened():
    """It was literal text, so a run with any other window reported an interval
    it did not screen."""
    cells = [{"country": "BG", "horizon_band": band,
              "scores": {name: dict(UNMEASURED) for name in REPORTED},
              "comparator_n": {name: 0 for name in REPORTED},
              "gate": {"pass": False, "n": 0}} for band in PRIMARY_BANDS]
    result = _result("abl348-level", [["solar", "BG"], ["solar", "CH"]],
                     list(TRANCHE_BASIS), cells)
    result["meta"]["screen_window"] = {"start": "2024-01-01", "end_exclusive": "2024-06-01"}
    report = _solar_harness().render_markdown(result)

    assert "2024-01-01 → 2024-06-01 UTC" in report
    assert "2025-12-31 → 2026-08-10 UTC" not in report
