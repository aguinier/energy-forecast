"""ABL-378: an absent incumbent must not read as a solar model-quality FAIL.

`tests/test_gate_scope_registration.py` pins the *shape* of the fix — that a
scope registers its countries and its gate basis in the file. This file pins the
*behaviour*, which is the thing that actually mis-dispositions a tranche.

The mechanism, in three steps:

1. `evaluate_solar_retrain.py` left-merges the incumbent, so `incumbent` is NaN
   on every row for a country with no rows in `forecasts`.
2. `common_scores(frame, columns)` keeps only rows where *every* named column is
   simultaneously finite. Naming `incumbent` therefore empties the intersection
   for such a country: n=0 and every score `None`.
3. `disposition` then has to decide what n=0 means. Before this issue it fell
   through to `FAIL` — a model-quality verdict on a race that never ran — and
   `render_markdown` crashed formatting `None / None` as a skill percentage.

Measured against the live replica on 2026-08-13: of the 32 countries with
non-null `solar_mw` in `energy_generation`, only AT, BE, DE and FR have any rows
in `forecasts`. So 28 of them are in exactly the position step 1 describes, and
they are the countries ABL-316 exists to model.

These tests construct the frames directly rather than fitting: the property
under test is the scoring and disposition logic, and a CatBoost fit over a real
replica would re-derive nothing that is not pinned here.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.wind_retrain import common_scores  # noqa: E402


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scripts_evaluate_solar_retrain_abl378",
        ROOT / "scripts" / "evaluate_solar_retrain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


def _frame(n=48, incumbent=True):
    """A scored frame for one country/band, with or without an incumbent."""
    rng = np.random.default_rng(0)
    actual = rng.uniform(100, 900, n)
    return pd.DataFrame({
        "actual": actual,
        "challenger": actual * 1.02,
        "seasonal_naive": actual * 1.30,
        "persistence": actual * 1.25,
        # A country with zero rows in `forecasts` left-merges to all-NaN.
        "incumbent": actual * 1.10 if incumbent else np.full(n, np.nan),
    })


def _cell(n, passed):
    return {"country": "XX", "horizon_band": "24-36h",
            "gate": {"n": n, "pass": passed}}


# --- step 2: the intersection ------------------------------------------------

def test_incumbent_in_the_basis_empties_the_cell_when_it_is_absent():
    """The defect, reproduced. This is why the basis is a registered property."""
    frame = _frame(incumbent=False)
    scores, common = common_scores(
        frame, ("challenger", "incumbent", "seasonal_naive", "persistence"))
    assert len(common) == 0
    assert scores["challenger"]["wape_pct"] is None
    assert scores["seasonal_naive"]["wape_pct"] is None


def test_a_basis_without_the_incumbent_scores_the_same_rows():
    """The fix: dropping an absent comparator from the basis costs nothing else."""
    frame = _frame(incumbent=False)
    scores, common = common_scores(frame, ("challenger", "seasonal_naive"))
    assert len(common) == len(frame)
    assert scores["challenger"]["wape_pct"] < scores["seasonal_naive"]["wape_pct"]


def test_the_registered_abl253_basis_is_unaffected_where_an_incumbent_exists():
    """ABL-253's four-way basis must keep scoring BE/DE/FR exactly as published."""
    frame = _frame(incumbent=True)
    scores, common = common_scores(
        frame, harness.GATE_BASIS["abl253"])
    assert len(common) == len(frame)
    assert scores["incumbent"]["wape_pct"] is not None


# --- step 3: the disposition -------------------------------------------------

def test_zero_row_cells_are_unreadable_not_fail():
    """The finding. A cell that scored nothing did not lose; it never ran."""
    cells = [_cell(0, False) for _ in range(9)]
    verdict, recommendation = harness.disposition(cells, 9, contaminated=False)
    assert verdict == "UNREADABLE"
    assert "never compared" in recommendation
    assert "9/9" in recommendation


def test_a_genuine_loss_still_fails():
    """UNREADABLE must not become a way for a bad model to escape a FAIL."""
    cells = [_cell(480, False)] + [_cell(480, True) for _ in range(8)]
    verdict, recommendation = harness.disposition(cells, 9, contaminated=False)
    assert verdict == "FAIL"
    assert "8/9" in recommendation


def test_a_partial_read_is_unreadable_not_fail():
    """One empty cell is still an unreadable gate, not a model that lost a cell."""
    cells = [_cell(0, False)] + [_cell(480, True) for _ in range(8)]
    verdict, _ = harness.disposition(cells, 9, contaminated=False)
    assert verdict == "UNREADABLE"


def test_full_pass_is_unchanged():
    cells = [_cell(480, True) for _ in range(9)]
    assert harness.disposition(cells, 9, contaminated=False)[0] == "PASS"


def test_contamination_hold_is_unchanged():
    cells = [_cell(480, True) for _ in range(9)]
    verdict, _ = harness.disposition(cells, 9, contaminated=True)
    assert verdict.startswith("PERFORMANCE PASS")


def test_a_short_scope_cannot_pass_by_producing_fewer_cells():
    """The property the hardcoded bar existed to protect, kept.

    A country that silently yields no cells at all must shortfall the registered
    count rather than quietly shrink the denominator into a vacuous PASS.
    """
    cells = [_cell(480, True) for _ in range(6)]
    assert harness.disposition(cells, 9, contaminated=False)[0] != "PASS"


def test_no_cells_at_all_is_not_a_pass():
    """`all([])` is True; an empty gate must not fall through to PASS."""
    assert harness.disposition([], 9, contaminated=False)[0] != "PASS"


# --- the crash ---------------------------------------------------------------

def test_render_markdown_does_not_crash_on_an_unscored_cell():
    """`100 * (1 - None / None)` is how the wind harness died on the pilot."""
    none_scores = {name: {"wape_pct": None, "mae": None, "bias_pct": None,
                          "slope": None, "correlation": None}
                   for name in ("challenger", "incumbent", "seasonal_naive", "persistence")}
    result = {
        "meta": {"generated_at": "2026-08-13 00:00 UTC", "replica_db": "x.db",
                 "replica_bytes": 1, "training_source": "energy_generation",
                 "scope": "abl253", "registered_countries": ["BE", "DE", "FR"],
                 "registered_cells": 9,
                 "gate_basis": ["challenger", "incumbent", "seasonal_naive", "persistence"],
                 "fit_window": {"start": "2026-01-14", "end_exclusive": "2026-07-11"},
                 "gate_window": {"start": "2026-07-11", "end_exclusive": "2026-08-10"},
                 "registered_intended_n": {}, "schedule_implied_n": {},
                 "vintage_counts": {}, "selection": "test"},
        "verdict": "UNREADABLE", "recommendation": "No disposition.",
        "training": [{"country": "XX", "algorithm": "catboost", "params": {},
                      "audit": {"retained_rows": 0, "intended_rows": 0,
                                "unique_targets": 0,
                                "excluded_missing_actual_or_feature": 0,
                                "degraded_lag_1d_rows": 0},
                      "gate_build_audit": {}, "constant_runs": [],
                      "artifact_path": "x", "artifact_sha256": "0" * 64}],
        "gate_cells": [{"country": "XX", "horizon_band": "24-36h",
                        "scores": none_scores, "comparator_n": {},
                        "gate": {"n": 0, "pass": False}}],
        "country_d2": [{"country": "XX", "n": 0, "scores": none_scores,
                        "comparator_n": {}, "tso": {"wape_pct": None, "n": 0}}],
    }
    text = harness.render_markdown(result)
    assert "Not measured" in text
    assert "UNREADABLE" in text
