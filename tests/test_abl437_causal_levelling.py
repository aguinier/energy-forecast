"""ABL-437: the trailing-window causal references, and the registration around them.

Four things need holding, and they are not the same kind of thing.

1. **The reference is what it says it is.** A trailing window ending at
   ``generated_at``, inclusive and hour-floored, over the same series the
   challenger's own rolling features are built from. If that drifts, the
   reference stops being causal and the whole amendment is worse than what it
   replaced.
2. **The ladder logic did not change.** ABL-437 re-levels the *reference* G2 and
   G3 read. Given two reference pairs carrying identical numbers, every cell must
   grade identically under either levelling -- otherwise the amendment smuggled
   in a rule change alongside the re-levelling.
3. **Every published scope is pinned.** ``CAUSAL_LEVELLING`` defaults *toward*
   the amendment, which is the right default for a scope nobody has read yet and
   the wrong one for a scope whose letters are committed. The published set is
   derived from ``SCOPE_OUTPUTS`` and git rather than typed here, on the ABL-404
   precedent: a pin that has to be remembered is a pin that goes missing across a
   merge.
4. **The window is in the name.** ``TRAILING_WINDOW_DAYS`` and the two column
   names must move together, or two reads levelled on different windows end up
   wearing one name.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.gate_grading import (  # noqa: E402
    LADDER_REFERENCES, cell_grade, conditions_for, grade_cell, grading_prose,
    scored_conditions,
)
from src.evaluation.model_free_reference import (  # noqa: E402
    CAUSAL_LEVELLINGS, FIT_WINDOW, MODEL_FREE_COMPARATORS, TRAILING_28D,
    TRAILING_COMPARATORS, TRAILING_WINDOW_DAYS, attach_model_free_references,
    attach_trailing_references, level_inflation, trailing_reference_levels,
)

import importlib.util  # noqa: E402


def _harness(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wind = _harness("evaluate_wind_retrain")
solar = _harness("evaluate_solar_retrain")
HARNESSES = {"wind": wind, "solar": solar}

FIT_START = pd.Timestamp("2026-01-14")
GATE_START = pd.Timestamp("2026-07-11")
GATE_END = pd.Timestamp("2026-08-10")


def _seasonal(fit_level=800.0, gate_level=200.0):
    """A series whose level drops between the fit window and the gate window.

    This is not a contrived shape. ABL-348's fit window runs 2026-01-14 to
    2026-07-11 and its gate window is high summer, and wind is seasonal, so the
    real NL ``wind_onshore`` read has exactly this structure -- which is how a
    flat line at the fit-window mean came to score 225.54% there against an
    oracle constant at 73.85%.
    """
    hours = pd.date_range(FIT_START - pd.Timedelta(days=14), GATE_END, freq="h",
                          inclusive="left")
    return pd.Series(np.where(hours < GATE_START, fit_level, gate_level),
                     index=hours, dtype=float)


def _gate_frame(n=72, actual=200.0):
    targets = pd.date_range(GATE_START, periods=n, freq="h")
    return pd.DataFrame({"target_ts": targets,
                         "generated_at": targets - pd.Timedelta(hours=48),
                         "actual": np.full(n, actual)})


# ---------------------------------------------------------------------------
# 1. The reference is what it says it is.
# ---------------------------------------------------------------------------

def test_the_trailing_level_is_the_mean_of_the_window_ending_at_the_issue_instant():
    """Computed by hand from the series, not read back out of the module."""
    hours = pd.date_range("2026-06-01", periods=24 * 60, freq="h")
    actuals = pd.Series(np.arange(len(hours), dtype=float), index=hours)
    as_of = pd.Timestamp("2026-07-09 07:00")
    levels = trailing_reference_levels(actuals, [as_of])

    window = actuals[(actuals.index >= as_of - pd.Timedelta(hours=TRAILING_WINDOW_DAYS * 24 - 1))
                     & (actuals.index <= as_of)]
    assert len(window) == TRAILING_WINDOW_DAYS * 24
    assert levels[as_of]["constant"] == pytest.approx(window.mean(), rel=1e-12)
    assert len(levels[as_of]["climatology"]) == 24
    assert levels[as_of]["climatology"][7] == pytest.approx(
        window[window.index.hour == 7].mean(), rel=1e-12)


def test_the_trailing_level_never_reads_past_the_issue_instant():
    """The causality claim, tested by making the future impossible to miss.

    Everything at or before the issue instant is 100; everything after it is
    10,000. A level that leaked one future hour could not come back 100.
    """
    hours = pd.date_range("2026-06-01", periods=24 * 60, freq="h")
    as_of = pd.Timestamp("2026-07-09 07:00")
    actuals = pd.Series(np.where(hours <= as_of, 100.0, 10_000.0), index=hours)
    levels = trailing_reference_levels(actuals, [as_of])
    assert levels[as_of]["constant"] == pytest.approx(100.0)
    assert set(levels[as_of]["climatology"].values()) == {100.0}


def test_the_anchor_matches_the_builders_own_rolling_window_bound():
    """The whole causality argument is that this is not a new rule.

    `wind_features._rolling_features` anchors at `observation_as_of.floor("h")`,
    inclusive, and spans `window_hours - 1` back. Read out of that module's
    source rather than restated, so the two cannot drift apart silently.
    """
    from src import wind_features

    source = (ROOT / "src" / "wind_features.py").read_text(encoding="utf-8")
    assert 'anchor = req.observation_as_of.floor("h")' in source
    assert "window_start = anchor - pd.Timedelta(hours=window_hours - 1)" in source
    assert "bounded = actuals[actuals.index <= anchor]" in source
    # And 168h -- a 7-day trailing mean -- is one of the challenger's own
    # features, which is what makes a 28-day one information it already had.
    assert 168 in wind_features.ROLLING_WINDOWS_HOURS


def test_a_level_the_window_cannot_measure_is_not_a_number():
    """An unmeasurable reference reads Not measured; it never becomes a zero."""
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    as_of = pd.Timestamp("2026-07-09 07:00")
    levels = trailing_reference_levels(empty, [as_of])
    assert levels[as_of] == {"constant": None, "climatology": {}}

    frame, summary = attach_trailing_references(_gate_frame(), empty)
    assert frame[list(TRAILING_COMPARATORS)].isna().all().all()
    assert summary["constant_mean_mw"] is None


def test_a_frame_with_no_issue_instant_is_unmeasurable_and_not_fit_window():
    """The safe direction, stated as a test because the unsafe one is silent.

    Falling back to the fit-window level here would restore the exact defect the
    reference exists to remove, and nothing in a report would say so.
    """
    frame = _gate_frame().drop(columns=["generated_at"])
    attached, summary = attach_trailing_references(frame, _seasonal())
    assert attached[list(TRAILING_COMPARATORS)].isna().all().all()
    assert summary["as_of_count"] == 0


def test_the_trailing_reference_tracks_a_level_the_fit_window_mean_misses():
    """The measurement this amendment turns on, on a series built to have it.

    And the limitation that comes with it, asserted rather than left implicit:
    a trailing window **converges**, it does not teleport. On a step change at
    the gate boundary the reference still carries the old level on day 1 and
    reaches the new one only after the window has rolled through -- so on
    ABL-348's 30-day gate window it spends most of the window catching up, and
    it is a *closer* level rather than the right one. That residual is what the
    per-cell `level inflation` column on the trailing reference exists to show;
    it is not a number this amendment can drive to zero, and a read that quoted
    the corrected reference as exact would be overclaiming.
    """
    gate_hours = int((GATE_END - GATE_START).total_seconds() // 3600)
    frame, levels = attach_model_free_references(
        _gate_frame(n=gate_hours), _seasonal(fit_level=800.0, gate_level=200.0),
        FIT_START, GATE_START, GATE_END)
    # The fit-window mean is a winter-and-spring average scored against summer.
    assert levels["constant_causal"] == pytest.approx(800.0)
    assert levels["constant_oracle"] == pytest.approx(200.0)

    trailing = frame["constant_causal_28d"]
    # Day 1 still carries the pre-gate level: the window has not rolled yet.
    assert trailing.iloc[0] == pytest.approx(800.0)
    # By the end of the window it has converged onto the gate level.
    assert trailing.iloc[-1] == pytest.approx(200.0, rel=0.05)
    # And what the ladder actually reads is the reference's WAPE, which on this
    # deliberately worst case -- a step change exactly at the gate boundary,
    # where a trailing window has the least warning it could have -- is halved.
    # Real seasonality is not a step, and the trailing window's *starting*
    # position here is the pathological one: on ABL-348's windows it starts as
    # the last 28 days of the fit window, which are already in the gate season.
    def wape(predicted):
        return float(np.mean(np.abs(frame["actual"] - predicted)) / np.mean(frame["actual"]))

    assert wape(levels["constant_causal"]) == pytest.approx(3.0)
    assert wape(trailing) < wape(levels["constant_causal"]) / 1.8


# ---------------------------------------------------------------------------
# 2. The ladder logic did not change -- only the reference it reads.
# ---------------------------------------------------------------------------

def _scores(challenger, naive, constant, climatology, slope=0.8, correlation=0.9):
    def entry(wape):
        return {"wape_pct": wape, "n": 0 if wape is None else 720}
    return {"challenger": {"wape_pct": challenger, "n": 720, "slope": slope,
                           "correlation": correlation},
            "seasonal_naive": entry(naive),
            "constant_causal": entry(constant), "climatology_causal": entry(climatology),
            "constant_causal_28d": entry(constant), "climatology_causal_28d": entry(climatology)}


@pytest.mark.parametrize("case", [
    (10.0, 20.0, 60.0, 30.0),      # A
    (10.0, 20.0, 9.0, 30.0),       # B, fails G2
    (10.0, 20.0, 60.0, 9.0),       # B, fails G3
    (20.0, 10.0, 60.0, 30.0),      # C
    (10.0, 10.2, 60.0, 30.0),      # U
    (None, None, None, None),      # not measured
])
@pytest.mark.parametrize("stream", ["wind", "solar"])
def test_the_two_levellings_grade_identically_on_identical_numbers(case, stream):
    """The amendment re-levels a reference. It does not touch a rule.

    Both reference pairs carry the same numbers here, so any difference in the
    letter would be a rule change riding along with the re-levelling -- which is
    the thing a pre-registration is supposed to make impossible to do quietly.
    """
    scores = _scores(*case)
    fit = grade_cell(scores, stream, levelling=FIT_WINDOW)
    trailing = grade_cell(scores, stream, levelling=TRAILING_28D)
    assert fit.label == trailing.label
    assert fit.conditions == trailing.conditions
    assert [name for name, _ in fit.failed] == [name for name, _ in trailing.failed]


def test_the_ladder_reads_the_registered_reference_and_no_other():
    """A cell carrying only ABL-437's pair grades on it, and vice versa."""
    both = _scores(10.0, 20.0, 60.0, 30.0)
    only_fit = {key: value for key, value in both.items()
                if key not in TRAILING_COMPARATORS}
    only_trailing = {key: value for key, value in both.items()
                     if key not in ("constant_causal", "climatology_causal")}

    assert grade_cell(only_fit, "wind", levelling=FIT_WINDOW).label == "A"
    assert grade_cell(only_trailing, "wind", levelling=TRAILING_28D).label == "A"
    # And a cell graded on references it does not carry is B with both named,
    # never A -- ABL-418's "a condition that cannot be evaluated is not
    # satisfied", which is what stops the amendment passing a cell for free.
    missing = grade_cell(only_fit, "wind", levelling=TRAILING_28D)
    assert missing.label == "B"
    assert [name for name, _ in missing.failed] == ["G2", "G3"]
    assert all("not measured" in reason for _, reason in missing.failed)


def test_no_oracle_is_on_either_ladder():
    """ABL-389's hard rule, which this amendment explicitly does not relax."""
    for levelling in CAUSAL_LEVELLINGS:
        named = set(LADDER_REFERENCES[levelling].values())
        assert not named & {"constant_oracle", "climatology_oracle"}
        assert not {name for _, name in scored_conditions(levelling)} & {
            "constant_oracle", "climatology_oracle"}


def test_g1_is_the_registered_bar_under_every_levelling():
    """Re-levelling a reference is an amendment; re-levelling the bar is not."""
    for levelling in CAUSAL_LEVELLINGS:
        assert scored_conditions(levelling)[0] == ("G1", "seasonal_naive")
        assert conditions_for(levelling)[0][0] == "G1"
        assert "seasonal_naive" in conditions_for(levelling)[0][2]


def test_a_record_written_before_this_amendment_reads_back_as_fit_window():
    """Absence dates the read. It is not a default anyone chose afterwards."""
    cell = {"scores": _scores(10.0, 20.0, 60.0, 30.0),
            "grade": {"grade": "A", "conditions": {}, "skill_pct": {}}}
    assert cell_grade(cell, "wind").levelling == FIT_WINDOW


def test_the_level_inflation_diagnostic_is_the_number_the_issue_was_opened_on():
    """CH `wind_onshore`, from ABL-380's own published pair of numbers."""
    scores = {"challenger": {"wape_pct": 47.42},
              "constant_causal": {"wape_pct": 79.07},
              "constant_oracle": {"wape_pct": 40.29},
              "constant_causal_28d": {"wape_pct": 45.0}}
    assert level_inflation(scores) == pytest.approx(96.25, abs=0.01)
    assert level_inflation(scores, "constant_causal_28d") == pytest.approx(11.69, abs=0.01)
    assert level_inflation({"constant_causal": {"wape_pct": 10.0}}) is None


# ---------------------------------------------------------------------------
# 3. Every published scope is pinned.
# ---------------------------------------------------------------------------

def _tracked(path: Path) -> bool:
    result = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)],
                            cwd=ROOT, capture_output=True, text=True)
    return result.returncode == 0


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_every_published_scope_pins_its_levelling(stream):
    """Derived from `SCOPE_OUTPUTS` + git, never from a list in this file.

    *Published* means a record committed to the repository, on ABL-404's
    reading: a scope whose `json_out` or `report_out` is tracked has letters a
    reader can cite, and a re-run that graded them under a later registration
    would disagree with its own evidence and exit 0. A local run of an open
    scope cannot promote itself into this set.
    """
    harness = HARNESSES[stream]
    published = {scope for scope, outputs in harness.SCOPE_OUTPUTS.items()
                 if any(_tracked(ROOT / outputs[key])
                        for key in ("json_out", "report_out") if outputs.get(key))}
    assert published, "no published scope found -- the derivation is broken, not the pins"
    missing = published - set(harness.CAUSAL_LEVELLING)
    assert not missing, f"published scope(s) with no registered levelling: {sorted(missing)}"
    for scope in published:
        assert harness.CAUSAL_LEVELLING[scope] == FIT_WINDOW, (
            f"{scope} is published; its letters were decided on the fit-window references")


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_an_unregistered_scope_defaults_to_the_amendment(stream):
    """The default direction, which is the opposite of `SCOPE_FEATURES`'.

    Inheriting the *old* reference silently is the defect ABL-437 removes, and
    it would land on the pairs nobody has looked at yet.
    """
    harness = HARNESSES[stream]
    assert harness.causal_levelling_for("a-scope-that-does-not-exist") == TRAILING_28D
    for scope, levelling in harness.CAUSAL_LEVELLING.items():
        assert levelling in CAUSAL_LEVELLINGS, scope


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_the_run_records_which_levelling_it_graded_under(stream):
    """A record that does not say cannot be re-read, and this is the one field
    that distinguishes an inflated `A` from a corrected one."""
    source = (ROOT / "scripts" / f"evaluate_{stream}_retrain.py").read_text(encoding="utf-8")
    assert '"causal_levelling": causal_levelling_for(args.scope),' in source
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "id", "") == "attach_grades"]
    assert len(calls) == 1
    keywords = {kw.arg for kw in calls[0].keywords}
    assert "levelling" in keywords, "the one grading call must name its levelling"


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_the_renderer_follows_the_record_and_not_the_table(stream):
    """Re-rendering a stored read must not re-decide it under a later pin."""
    source = (ROOT / "scripts" / f"evaluate_{stream}_retrain.py").read_text(encoding="utf-8")
    assert 'levelling = meta.get("causal_levelling", FIT_WINDOW)' in source


def test_the_abl418_retro_grade_is_pinned_to_the_levelling_it_published():
    """Its two tranches carry no trailing column, so the amended default would
    turn a published page of A's into a page of B's on re-run."""
    source = (ROOT / "scripts" / "abl418_retro_grade.py").read_text(encoding="utf-8")
    assert "levelling=FIT_WINDOW" in source


# ---------------------------------------------------------------------------
# 4. The window is in the name, and both harnesses report both pairs.
# ---------------------------------------------------------------------------

def test_the_window_and_the_column_names_move_together():
    """Two reads levelled on different windows must not wear one name."""
    for name in TRAILING_COMPARATORS:
        assert name.endswith(f"_{TRAILING_WINDOW_DAYS}d")


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_both_harnesses_report_both_causal_pairs_and_gate_on_neither(stream):
    harness = HARNESSES[stream]
    assert set(MODEL_FREE_COMPARATORS) <= set(harness.REPORTED_COMPARATORS)
    for basis in harness.GATE_BASIS.values():
        assert not set(TRAILING_COMPARATORS) & set(basis)


@pytest.mark.parametrize("stream", sorted(HARNESSES))
@pytest.mark.parametrize("levelling", CAUSAL_LEVELLINGS)
def test_the_prose_names_the_references_the_ladder_actually_read(stream, levelling):
    """A report whose words and columns disagree is worse than one with neither."""
    text = " ".join(grading_prose(stream, levelling=levelling))
    references = LADDER_REFERENCES[levelling]
    assert f"`{references['G2']}`" in text and f"`{references['G3']}`" in text
    assert f"ABL-437): `{levelling}`" in text
    assert "registered bar is **not** re-opened" in text or "G1 is unchanged" in text


def test_the_pre_registration_record_states_the_form_before_it_grades_anything():
    """The registration is a committed artefact, not a paragraph in a report."""
    config = json.loads((ROOT / "experiments" / "ABL437" / "config.json")
                        .read_text(encoding="utf-8"))
    assert config["issue"] == "ABL-437"
    assert config["adopted_form"] == TRAILING_28D
    assert config["window_days"] == TRAILING_WINDOW_DAYS
    assert config["ladder_references"] == LADDER_REFERENCES[TRAILING_28D]
    assert config["oracles_stay_off_the_ladder"] is True
    assert config["registered_bar_unchanged"] == "seasonal_naive D-7"
