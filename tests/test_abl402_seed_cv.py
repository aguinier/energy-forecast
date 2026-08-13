"""ABL-402: the invariants that make the BG/CH seed CV a measurement.

The sweep itself takes minutes and touches a 9.4 GB replica, so what is pinned
here is everything that can silently turn its output into a number that means
nothing:

- the **seed freeze** -- a CV that includes the gate's own pinned seed is a
  spread anchored on the arm that produced the headline, which is not a spread;
- the **`c_B = 0` assumption** -- the whole reason the margin shrinks by
  sqrt(2) against ABL-385's two-fitted-arm table is that the reference is
  deterministic arithmetic on the actuals.  If a reference column ever moved
  across seeds, every margin in the report would be too small and nothing in
  the output would look wrong;
- the **registration** -- this issue is told in terms not to re-derive ABL-348's
  windows, so they are checked against the frozen config rather than retyped
  and hoped over;
- the **boundary** -- no artifact write, no registered scope table touched.
  ABL-381's six cells are dispositioned; this run measures the variance around
  them and must not be able to overwrite them.
"""

import ast
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.abl385_read_margin import delta_min  # noqa: E402

SWEEP = Path(__file__).parent.parent / "scripts" / "abl402_bg_ch_seed_cv.py"
REGISTRATION = Path(__file__).parent.parent / "experiments" / "ABL348" / "config.json"


@pytest.fixture(scope="module")
def sweep():
    spec = importlib.util.spec_from_file_location("_abl402_sweep", SWEEP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source_tree():
    return ast.parse(SWEEP.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The seed freeze
# --------------------------------------------------------------------------

class TestSeedFreeze:
    def test_the_gates_pinned_seed_is_not_in_the_averaged_set(self, sweep):
        """42 is the control, and a control inside the sample is not a control.

        `config.CATBOOST_PARAMS['random_seed']` is 42 and every ABL-381 cell was
        fitted at it. Averaging it in would measure the spread of a set that
        contains the published point, which is the selection this design exists
        to avoid.
        """
        assert sweep.CONTROL_SEED == 42
        assert sweep.CONTROL_SEED not in sweep.SEEDS

    def test_seeds_are_distinct(self, sweep):
        """A repeated seed is a duplicated fit: it shrinks the sd without adding information."""
        assert len(set(sweep.SEEDS)) == len(sweep.SEEDS)

    def test_enough_seeds_to_answer_the_question_that_was_asked(self, sweep):
        """CH's margin is decided at a threshold, and the CI has to clear it.

        The margin is 10.5% of CH's own error; `delta_min(1) = 1.96 c` crosses
        that at c = 5.36%. The point of 20 rather than 12 is that the CV's own
        95% interval is narrow enough to land on one side of that. This pins the
        seed count against the reason for it, so dropping seeds to save a minute
        fails here rather than in a report nobody can reproduce.
        """
        assert len(sweep.SEEDS) >= 20


# --------------------------------------------------------------------------
# c_B = 0 -- the assumption the whole re-read rests on
# --------------------------------------------------------------------------

class TestDeterministicReference:
    def test_a_reference_that_moves_across_seeds_is_an_error_not_a_mean(self, sweep):
        """The failure this catches is silent: it would only make margins look readable."""
        observed = [{"climatology_oracle_wape_pct": 9.02},
                    {"climatology_oracle_wape_pct": 9.03}]
        with pytest.raises(AssertionError):
            sweep._deterministic(observed, "climatology_oracle_wape_pct")

    def test_an_invariant_reference_passes_through(self, sweep):
        observed = [{"climatology_oracle_wape_pct": 9.02}] * 5
        assert sweep._deterministic(observed, "climatology_oracle_wape_pct") == 9.02

    def test_one_fitted_arm_is_the_two_arm_margin_over_root_two(self, sweep):
        """ABL-381 section 3's table, recomputed rather than quoted.

        Its three published k=1 margins for a fitted arm against a deterministic
        reference are 4.55 / 8.76 / 10.64% at the fleet median / p80 / p90. If
        this ever stops holding, the report's headline comparison has silently
        changed convention.
        """
        for cv, published in ((0.0232, 4.55), (0.0447, 8.76), (0.0543, 10.64)):
            one_arm = 100.0 * delta_min(cv, 0.0, 1)
            assert one_arm == pytest.approx(published, abs=0.01)
            two_arms = 100.0 * delta_min(cv, cv, 1)
            assert two_arms / one_arm == pytest.approx(math.sqrt(2), rel=1e-12)


# --------------------------------------------------------------------------
# The statistics
# --------------------------------------------------------------------------

class TestSpread:
    def test_cv_is_the_sample_sd_over_the_mean(self, sweep):
        """ddof=1. ABL-385's convention, and the one `cv_interval`'s dof assumes."""
        spread = sweep._spread([10.0, 12.0, 14.0])
        assert spread["mean"] == pytest.approx(12.0)
        assert spread["sd"] == pytest.approx(2.0)
        assert spread["cv_pct"] == pytest.approx(100.0 * 2.0 / 12.0)
        assert spread["n_seeds"] == 3

    def test_the_interval_brackets_the_point_estimate(self, sweep):
        spread = sweep._spread([10.0, 12.0, 14.0, 11.0, 13.0])
        low, high = spread["cv_pct_ci95"]
        assert low < spread["cv_pct"] < high

    def test_more_draws_narrow_the_interval(self, sweep):
        """The quantitative reason this reads 20 seeds and not 12."""
        few = sweep._spread([10.0, 12.0] * 6)
        many = sweep._spread([10.0, 12.0] * 10)
        few_width = few["cv_pct_ci95"][1] - few["cv_pct_ci95"][0]
        many_width = many["cv_pct_ci95"][1] - many["cv_pct_ci95"][0]
        assert many_width < few_width

    def test_a_dead_flat_arm_has_no_spread(self, sweep):
        spread = sweep._spread([8.16] * 20)
        assert spread["cv_pct"] == pytest.approx(0.0)
        assert spread["range_pp"] == pytest.approx(0.0)

    def test_seeds_needed_inverts_delta_min(self, sweep):
        """k is the smallest integer whose margin the gap clears, not a rounded guess."""
        for cv, gap in ((0.023, 10.5), (0.054, 10.5), (0.023, 1.4), (0.05, 3.7)):
            k = sweep._seeds_needed(cv, gap)
            assert 100.0 * delta_min(cv, 0.0, k) <= abs(gap) + 1e-9
            if k > 1:
                assert 100.0 * delta_min(cv, 0.0, k - 1) > abs(gap)

    def test_a_gap_of_zero_is_never_readable(self, sweep):
        assert sweep._seeds_needed(0.023, 0.0) is None

    def test_readability_is_decided_on_the_relative_gap(self, sweep):
        """A margin is read as a fraction of the challenger's own error, not in pp.

        BG's 0.26pp on an 18.89% challenger and CH's 0.86pp on an 8.16% one are
        1.4% and 10.5% -- a 3.3x ratio in pp becomes 7.5x in the units that
        decide it. Reading the pp figure is the mistake this whole line of
        issues exists to stop.
        """
        bg = sweep._margin_reading(0.023, 18.89, 19.15)
        ch = sweep._margin_reading(0.023, 8.16, 9.02)
        assert bg["margin_pct_of_challenger"] == pytest.approx(1.376, abs=0.01)
        assert ch["margin_pct_of_challenger"] == pytest.approx(10.539, abs=0.01)
        assert not bg["readable_at_k1"]
        assert ch["readable_at_k1"]

    def test_a_challenger_worse_than_the_reference_reads_negative(self, sweep):
        """A losing pair is a finding, not a crash. The sign has to survive."""
        losing = sweep._margin_reading(0.023, 9.50, 9.02)
        assert losing["margin_pp"] < 0
        assert losing["margin_pct_of_challenger"] < 0


# --------------------------------------------------------------------------
# The registration is read, not re-derived
# --------------------------------------------------------------------------

class TestRegistration:
    def test_windows_and_source_match_the_frozen_config(self, sweep):
        """ABL-402 is told not to re-derive ABL-348's registration.

        The constants in the sweep are a transcription, and a transcription is
        checkable. `config.json` writes the instants with a `Z`; the sweep holds
        them tz-naive because that is the representation the gate itself passes
        to the builder, so the comparison is on the date part.
        """
        registered = json.loads(REGISTRATION.read_text(encoding="utf-8"))
        assert str(sweep.FIT_START.date()) == registered["fit_target_window"][0][:10]
        assert str(sweep.GATE_START.date()) == registered["fit_target_window"][1][:10]
        assert str(sweep.GATE_START.date()) == registered["gate_target_window"][0][:10]
        assert str(sweep.GATE_END.date()) == registered["gate_target_window"][1][:10]
        assert sweep.SOURCE == registered["training_source"]["table"]
        assert list(sweep.PRIMARY_BANDS) == registered["primary_bands"]
        assert registered["metric"] == "WAPE"

    def test_the_windows_are_ordered(self, sweep):
        assert sweep.FIT_START < sweep.GATE_START < sweep.GATE_END

    def test_the_pairs_are_the_registered_scope(self, sweep):
        """`SCOPES['abl316-t1b']`, and nothing crept in beside it."""
        from scripts.evaluate_solar_retrain import SCOPES

        assert tuple(sweep.COUNTRIES) == SCOPES["abl316-t1b"]

    def test_the_basis_is_the_registered_two_way_one(self, sweep):
        """BG and CH hold zero solar rows in `forecasts`; the four-way basis empties every cell."""
        from scripts.evaluate_solar_retrain import GATE_BASIS

        assert tuple(sweep.GATE_BASIS) == GATE_BASIS["abl316-t1b"]

    def test_the_fit_rule_matches_what_the_scope_registered(self, sweep):
        """ABL-376's night rule is off for this scope, and the spread must be measured under the read's own rule."""
        from scripts.evaluate_solar_retrain import fit_rules_for

        assert fit_rules_for("abl316-t1b")["exclude_impossible_night"] is False

    def test_published_cells_cover_the_whole_registered_grid(self, sweep):
        """All six cells, not just the two ABL-381 quoted in prose.

        CH's 48-64h margin is 0.31pp on an 8.39% challenger -- 3.7% of its own
        error, a third of the headline cell's 10.5%. A re-read that only checked
        the quoted cell would miss the weakest one.
        """
        expected = {(c, b) for c in sweep.COUNTRIES for b in sweep.PRIMARY_BANDS}
        assert set(sweep.PUBLISHED_CELLS) == expected


# --------------------------------------------------------------------------
# The boundary
# --------------------------------------------------------------------------

class TestBoundary:
    def test_no_artifact_is_written(self, source_tree):
        """`save_gate_artifact` re-fits into `experiments/ABL316/artifacts` in place.

        ABL-402's scope says in terms: no refit of the dispositioned artifacts.
        The gate calls that writer; this sweep must never reach it, and a static
        check is the one that holds without running a fit.
        """
        called = {node.func.attr if isinstance(node.func, ast.Attribute)
                  else getattr(node.func, "id", "")
                  for node in ast.walk(source_tree) if isinstance(node, ast.Call)}
        assert "save_gate_artifact" not in called

    def test_the_registration_tables_are_read_and_never_assigned(self, source_tree):
        """Five tables since ABL-376, and `abl316-t1b` has a row in each.

        Reading them is the point; rebinding or mutating one from a probe script
        would move a dispositioned scope from outside the harness that owns it.
        """
        tables = {"SCOPES", "GATE_BASIS", "SCOPE_OUTPUTS", "FIT_RULES", "SCOPE_TITLES"}
        for node in ast.walk(source_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    # The sweep's own module-level GATE_BASIS constant is its
                    # transcription of the registered tuple, pinned equal to the
                    # harness's above; what must not happen is an assignment
                    # *into* one, which is a Subscript target.
                    if isinstance(target, ast.Subscript):
                        name = getattr(target.value, "id", "")
                        assert name not in tables, f"assigns into {name}"
            if isinstance(node, ast.Call):
                method = node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if method in {"update", "setdefault", "pop", "clear"}:
                    owner = getattr(node.func.value, "id", "")
                    assert owner not in tables, f"mutates {owner}"

    def test_the_replica_is_never_opened_for_writing(self, source_tree):
        """No `sqlite3.connect` at all here: every read goes through the builder."""
        for node in ast.walk(source_tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                assert name != "connect"

    def test_help_text_is_ascii(self):
        """The docstring is passed as `description=__doc__` (ABL-364).

        Covered by the repo-wide sweep too; kept here because this script's
        comments deliberately *do* carry non-ASCII typography, so the line
        between the two is worth pinning at the file it applies to.
        """
        docstring = ast.get_docstring(ast.parse(SWEEP.read_text(encoding="utf-8")), clean=False)
        assert docstring.isascii()
