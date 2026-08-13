"""ABL-389: the model-free references are *reported*, and gate nothing.

ABL-380 passed 6/6 and reported, against its own passing result, that CH
wind_onshore cleared all three cells at 47.42% WAPE while a flat line at the
gate-window median scored 40.29% — the fitted model was 7.1pp worse than a
constant — and that BG's registered D-7 bar of 93.75% is cleared outright by a
causal constant at the fit-window mean (82.77%), with no model at all. 33 more
pairs are queued behind that tranche, all with zero rows in `forecasts`, all
otherwise dispositioned against the same weak floor. So both harnesses now print
what a PASS is worth beside the PASS.

Four references, not two. The flat line alone was measured across ABL-381's six
solar cells and scores 63–95% WAPE on every one: it cannot represent a diurnal
cycle, and on solar the diurnal cycle is the signal, so it is a comparator the
challenger cannot lose to — a formality where a bar was wanted, which is the
ABL-380 defect one level up. The hour-of-day climatology is the same idea (no
model, causal form available at forecast time) with the one structure a constant
cannot carry, and it is the tighter reference on **both** technologies: CH wind's
oracle climatology is 38.20% against the constant's 40.29%, widening the finding
that motivated this issue from 7.1pp to 9.2pp. Both are kept, because the gap
between them is how much of a series is forced diurnal structure.

Three properties are load-bearing and all three are pinned here.

**It is a reference, not a criterion.** The comparators are in
`REPORTED_COMPARATORS` and in no `GATE_BASIS` entry, so no cell verdict, no
band, no bar, no window and no minimum n can move because of them. Moving a bar
after seeing a result is exactly what the pre-registration apparatus exists to
prevent, and the fact that this particular move would tighten rather than loosen
does not exempt it.

**Already-dispositioned reads do not move.** `abl195`, `abl253` and
`abl322-pilot` are read and closed. Adding a reported column to their output is
expected; changing a score or a verdict is a bug. The mechanism by which that
could happen is exactly one function — the scorer that turns a frame plus a
registered basis into scores, an intersection and a verdict — so that function
is exercised on each of the three registered bases with and without the new
columns, and every basis score, the intersection size and the gate verdict must
be identical. What this cannot prove is that a full replica re-read reproduces
its published table; that is a live run, and ABL-389 records one on the issue.

**A partially measurable reference stays partial.** A climatology is 24 levels,
not one, so unlike every comparator that came before it a row can fail to have a
level while its neighbours have one. Those rows must drop out of that column's
own intersection and lower only that column's n — never be filled from an
adjacent hour, and never be scored against an invented level. That is the
`NULL`-is-not-`0` rule applied to a comparator instead of to a source column.

The frames are constructed rather than fitted: the property under test is the
scoring path, and a CatBoost fit over a 9.4 GB replica would re-derive nothing
that is not pinned here.
"""
import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.model_free_reference import (  # noqa: E402
    CLIMATOLOGY_COMPARATORS, CONSTANT_COMPARATORS, MODEL_FREE_COMPARATORS,
    attach_model_free_references, climatology_reference_levels, comparator_wape,
    constant_reference_levels, levels_table, lost_to_a_model_free_reference,
)
from src.evaluation.wind_retrain import (  # noqa: E402
    common_scores, gate_cell, scored_with_comparators,
)

WIND_HARNESS = ROOT / "scripts" / "evaluate_wind_retrain.py"
SOLAR_HARNESS = ROOT / "scripts" / "evaluate_solar_retrain.py"

FIT_START = pd.Timestamp("2026-01-14")
GATE_START = pd.Timestamp("2026-07-11")
GATE_END = pd.Timestamp("2026-08-10")

#: The three scopes that have been read and dispositioned, with the basis each
#: was published under. Written out here, not read from the harness: a fixture
#: that follows the file it is guarding cannot catch the file changing.
DISPOSITIONED_BASES = {
    "abl195": ("challenger", "incumbent", "seasonal_naive", "persistence"),
    "abl253": ("challenger", "incumbent", "seasonal_naive", "persistence"),
    "abl322-pilot": ("challenger", "seasonal_naive"),
}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wind = _load(WIND_HARNESS, "scripts_evaluate_wind_retrain_abl389")
solar = _load(SOLAR_HARNESS, "scripts_evaluate_solar_retrain_abl389")


def _gate_basis_literal(path):
    """`GATE_BASIS` as written in the file, not as imported.

    The registration is the *literal in the source*, committed before the run.
    Reading it through the module would still pass if some future edit computed
    it, which is the thing a pre-registration must not be.
    """
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "GATE_BASIS":
            return {name: tuple(cols) for name, cols in ast.literal_eval(node.value).items()}
    raise AssertionError(f"GATE_BASIS not found in {path.name}")


def _series(values, start, freq="h"):
    return pd.Series(np.asarray(values, dtype=float),
                     index=pd.date_range(start, periods=len(values), freq=freq))


def _spanning(fit_level=250.0, gate_level=250.0):
    """A target series covering *both* registered windows, so both levels exist.

    A short series anchored at the fit start does not reach the gate window five
    months later, and its oracle level is then correctly `None` — which is the
    right answer and the wrong fixture for a test about a live column. That
    distinction is the whole subject of this module, so it is made explicit here
    rather than left to whether a row count happened to be large enough.
    """
    hours = pd.date_range(FIT_START, GATE_END, freq="h", inclusive="left")
    return pd.Series(np.where(hours < GATE_START, fit_level, gate_level),
                     index=hours, dtype=float)


def _diurnal(peak=100.0, night=0.0, gate_peak=None):
    """A target series with a real daily cycle, over both registered windows.

    Solar in one line: nothing at night, a smooth hump around midday. This is
    the shape a constant cannot represent at any level — which is why the flat
    reference reads 63-95% on every solar cell measured — so it is the fixture
    that distinguishes the two references rather than one where they agree.
    """
    hours = pd.date_range(FIT_START, GATE_END, freq="h", inclusive="left")
    shape = np.maximum(0.0, np.sin((hours.hour - 6) / 12 * np.pi))
    amplitude = np.where(hours < GATE_START, peak, peak if gate_peak is None else gate_peak)
    return pd.Series(night + shape * (amplitude - night), index=hours, dtype=float)


def _frame(n=60, seed=0, level=400.0):
    """One country/band of scored gate rows, in the shape `main()` builds.

    `target_ts` is hourly from the gate start, because the climatology columns
    are keyed on hour of day — a frame without it would be scoring a reference
    against timestamps it does not have.
    """
    rng = np.random.default_rng(seed)
    actual = rng.uniform(0.5 * level, 1.5 * level, n)
    return pd.DataFrame({
        "target_ts": pd.date_range(GATE_START, periods=n, freq="h"),
        "actual": actual,
        "challenger": actual * 1.05 + rng.normal(0, 0.05 * level, n),
        "seasonal_naive": np.roll(actual, 7),
        "persistence": np.roll(actual, 1),
        "incumbent": actual * 0.95 + rng.normal(0, 0.08 * level, n),
    })


# ---------------------------------------------------------------------------
# 1. A reference, not a criterion.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [WIND_HARNESS, SOLAR_HARNESS], ids=["wind", "solar"])
def test_the_references_are_in_no_registered_gate_basis(path):
    """The whole hard constraint of ABL-389, in one assertion per harness.

    If any of the four names reaches a `GATE_BASIS` value it stops being a
    reference: it joins the finite-intersection conjunction, and a pair whose fit
    window holds no finite observation would then score n=0 across every cell and
    render a model-quality verdict on a comparison that never ran — the ABL-322
    defect, reintroduced by the fix for a different one. The climatology raises
    the stake, because it can be partially missing where a constant is all or
    nothing: in a basis it would silently *shrink* an intersection rather than
    empty it, which is the harder failure to notice.
    """
    for scope, basis in _gate_basis_literal(path).items():
        leaked = set(MODEL_FREE_COMPARATORS) & set(basis)
        assert not leaked, (
            f"{path.name}: scope {scope!r} gates on {sorted(leaked)}. The model-free "
            "predictors are reported references; they must never be gate criteria.")


@pytest.mark.parametrize("harness", [wind, solar], ids=["wind", "solar"])
def test_both_harnesses_report_all_four_references(harness):
    """Reported by both, from one module, so the two gates cannot drift."""
    assert set(MODEL_FREE_COMPARATORS) <= set(harness.REPORTED_COMPARATORS)
    assert CONSTANT_COMPARATORS == ("constant_causal", "constant_oracle")
    assert CLIMATOLOGY_COMPARATORS == ("climatology_causal", "climatology_oracle")
    assert MODEL_FREE_COMPARATORS == CONSTANT_COMPARATORS + CLIMATOLOGY_COMPARATORS


@pytest.mark.parametrize("harness", [wind, solar], ids=["wind", "solar"])
def test_the_registered_bar_still_names_only_d7(harness):
    """The PASS rule is `challenger WAPE < seasonal_naive WAPE`, unchanged.

    `gate_cell` takes exactly two scores and a count. A constant cannot reach it
    without a signature change, and this is what would fail if one were made.
    """
    passes = gate_cell(challenger_wape=10.0, naive_wape=20.0, n=720, intended_n=720)
    loses = gate_cell(challenger_wape=30.0, naive_wape=20.0, n=720, intended_n=720)
    assert passes["pass"] and passes["beats_d7"]
    assert not loses["pass"]
    # A cell that clears D-7 while losing to both constants is still a PASS.
    assert "constant" not in str(sorted(passes)), (
        "gate_cell has grown a constant-predictor term; ABL-389 is reporting only")


# ---------------------------------------------------------------------------
# 2. Already-dispositioned reads do not move.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scope,basis", sorted(DISPOSITIONED_BASES.items()))
def test_dispositioned_scopes_keep_their_scores_and_verdicts(scope, basis):
    """Adding the columns changes no basis score, no n, and no verdict.

    Run against each dispositioned scope's own registered basis, because the
    four-way `abl195`/`abl253` basis and the two-way `abl322-pilot` basis fail
    differently: only the four-way one has a comparator that can drop rows.
    """
    before_frame = _frame()
    after_frame, levels = attach_model_free_references(
        before_frame, _spanning(fit_level=310.0, gate_level=280.0),
        FIT_START, GATE_START, GATE_END)

    before, before_common, before_n = scored_with_comparators(
        before_frame, basis, ("challenger", "incumbent", "seasonal_naive", "persistence"))
    after, after_common, after_n = scored_with_comparators(
        after_frame, basis, ("challenger", "incumbent", "seasonal_naive", "persistence",
                             *MODEL_FREE_COMPARATORS))

    assert all(levels[name] is not None for name in CONSTANT_COMPARATORS)
    assert all(len(levels[name]) == 24 for name in CLIMATOLOGY_COMPARATORS), (
        "all four columns must be live here — and the climatologies fully covered — "
        "or this proves nothing about whichever one is not")
    for name in ("challenger", "incumbent", "seasonal_naive", "persistence"):
        assert after[name] == before[name], f"{scope}: {name} moved"
        assert after_n[name] == before_n[name], f"{scope}: {name} n moved"
    assert len(after_common) == len(before_common), f"{scope}: the gate intersection moved"

    verdict_before = gate_cell(before["challenger"]["wape_pct"],
                               before["seasonal_naive"]["wape_pct"], len(before_common), 60)
    verdict_after = gate_cell(after["challenger"]["wape_pct"],
                              after["seasonal_naive"]["wape_pct"], len(after_common), 60)
    assert verdict_after == verdict_before, f"{scope}: the cell verdict moved"


@pytest.mark.parametrize("scope,basis", sorted(DISPOSITIONED_BASES.items()))
def test_the_extracted_scorer_still_computes_what_the_closure_computed(scope, basis):
    """ABL-389 lifted a byte-identical closure out of both harnesses.

    The closure is inlined here so the extraction is checked against what it
    replaced rather than against itself.
    """
    frame, _ = attach_model_free_references(
        _frame(seed=3), _series([10.0] * 500, FIT_START), FIT_START, GATE_START, GATE_END)
    reported = ("challenger", "incumbent", "seasonal_naive", "persistence",
                *MODEL_FREE_COMPARATORS)

    scores, common = common_scores(frame, basis)
    comparator_n = {name: len(common) for name in basis}
    for name in reported:
        if name in scores:
            continue
        sub_scores, sub_common = common_scores(frame, (*basis, name))
        scores[name], comparator_n[name] = sub_scores[name], len(sub_common)

    assert scored_with_comparators(frame, basis, reported) [0] == scores
    assert scored_with_comparators(frame, basis, reported)[2] == comparator_n


def test_an_absent_comparator_still_costs_only_its_own_row():
    """The property ABL-322/ABL-378 bought must survive this change.

    A country with zero rows in `forecasts` has an all-NaN `incumbent`. Under a
    basis that excludes it, it reads `Not measured` with n=0 while every other
    comparator — including the four new ones — keeps the full intersection.
    """
    frame = _frame(n=48)
    frame["incumbent"] = np.nan
    frame, _ = attach_model_free_references(
        frame, _spanning(), FIT_START, GATE_START, GATE_END)
    scores, common, comparator_n = scored_with_comparators(
        frame, ("challenger", "seasonal_naive"),
        ("challenger", "incumbent", "seasonal_naive", "persistence", *MODEL_FREE_COMPARATORS))

    assert len(common) == 48
    assert scores["incumbent"]["wape_pct"] is None and comparator_n["incumbent"] == 0
    for name in MODEL_FREE_COMPARATORS:
        assert comparator_n[name] == 48
        assert scores[name]["wape_pct"] is not None


# ---------------------------------------------------------------------------
# 3. The definitions themselves.
# ---------------------------------------------------------------------------

def test_causal_is_the_fit_window_mean_and_oracle_the_gate_window_median():
    """Mean for the causal level, median for the oracle level — not either for
    both.

    The series is deliberately right-skewed, because a symmetric one has mean ==
    median and would pass this test under any of the four possible mix-ups. Wind
    and solar output *are* right-skewed — mostly low with a thin high tail — so
    the skew is the realistic case rather than an adversarial one. Measured on
    the replica: CH's fit-window mean is 21.97 MW against a gate-window median
    of 10.68 MW, and choosing the wrong statistic there moves the reported WAPE
    by tens of percentage points.
    """
    hours = pd.date_range(FIT_START, GATE_END, freq="h", inclusive="left")
    ramp = np.arange(len(hours), dtype=float)
    series = pd.Series(ramp ** 2, index=hours)
    levels = constant_reference_levels(series, FIT_START, GATE_START, GATE_END)

    fit = series[(series.index >= FIT_START) & (series.index < GATE_START)]
    gate = series[(series.index >= GATE_START) & (series.index < GATE_END)]
    assert levels["constant_causal"] == pytest.approx(float(fit.mean()))
    assert levels["constant_oracle"] == pytest.approx(float(gate.median()))
    # The skew is real, so each level is distinguishable from the other statistic
    # of its own window — the assertion above cannot be satisfied by an accident.
    assert float(fit.mean()) > 1.3 * float(fit.median())
    assert levels["constant_causal"] != pytest.approx(float(fit.median()))
    assert levels["constant_oracle"] != pytest.approx(float(gate.mean()))


def test_the_lookback_and_the_post_gate_tail_are_excluded():
    """The builder loads 14 days before the fit window for lags, and its loader
    asks for `gate_end + 1 day`. Neither may reach the levels."""
    hours = pd.date_range(FIT_START - pd.Timedelta(days=14),
                          GATE_END + pd.Timedelta(days=1), freq="h", inclusive="left")
    series = pd.Series(np.where((hours >= FIT_START) & (hours < GATE_END), 100.0, 1e6),
                       index=hours)
    levels = constant_reference_levels(series, FIT_START, GATE_START, GATE_END)
    assert levels["constant_causal"] == pytest.approx(100.0)
    assert levels["constant_oracle"] == pytest.approx(100.0)


def test_the_causal_level_cannot_see_the_gate_window():
    """The point of reporting two levels: the causal one is fixed before the
    gate window opens, so a level shift inside it moves only the oracle."""
    hours = pd.date_range(FIT_START, GATE_END, freq="h", inclusive="left")
    flat = pd.Series(np.where(hours < GATE_START, 200.0, 200.0), index=hours)
    shifted = pd.Series(np.where(hours < GATE_START, 200.0, 20.0), index=hours)
    a = constant_reference_levels(flat, FIT_START, GATE_START, GATE_END)
    b = constant_reference_levels(shifted, FIT_START, GATE_START, GATE_END)
    assert a["constant_causal"] == b["constant_causal"] == pytest.approx(200.0)
    assert a["constant_oracle"] == pytest.approx(200.0)
    assert b["constant_oracle"] == pytest.approx(20.0)


def test_an_unmeasurable_level_is_none_and_reads_not_measured():
    """`NULL` is not `0`. A window with no finite observation has no level, the
    column is all-NaN, its own intersection is empty, and it reports as
    unmeasured — it never stands in as a zero-error flat line."""
    gate_only = _series([5.0] * 720, GATE_START)
    levels = constant_reference_levels(gate_only, FIT_START, GATE_START, GATE_END)
    assert levels["constant_causal"] is None
    assert levels["constant_oracle"] == pytest.approx(5.0)

    frame, attached = attach_model_free_references(
        _frame(n=30), gate_only, FIT_START, GATE_START, GATE_END)
    assert {name: attached[name] for name in CONSTANT_COMPARATORS} == levels
    assert frame["constant_causal"].isna().all()
    scores, common, comparator_n = scored_with_comparators(
        frame, ("challenger", "seasonal_naive"),
        ("challenger", "seasonal_naive", *MODEL_FREE_COMPARATORS))
    assert comparator_n["constant_causal"] == 0
    assert scores["constant_causal"]["wape_pct"] is None
    assert comparator_n["constant_oracle"] == len(common) == 30


def test_an_all_nan_series_yields_no_level_at_all():
    series = _series([np.nan] * 500, FIT_START)
    assert constant_reference_levels(series, FIT_START, GATE_START, GATE_END) == {
        "constant_causal": None, "constant_oracle": None}
    assert constant_reference_levels(pd.Series(dtype=float), FIT_START, GATE_START,
                                     GATE_END) == {"constant_causal": None,
                                                   "constant_oracle": None}


def test_missing_hours_do_not_drag_a_level_toward_zero():
    """A gap is a gap. Skipping NaN is what keeps the fit-window mean the mean
    of what was measured, not of what was measured plus a run of implied zeros."""
    values = [100.0] * 200 + [np.nan] * 200
    levels = constant_reference_levels(_series(values, FIT_START), FIT_START,
                                       GATE_START, GATE_END)
    assert levels["constant_causal"] == pytest.approx(100.0)


def test_the_attached_column_is_one_constant_per_pair():
    frame, levels = attach_model_free_references(
        _frame(n=40), _series([7.5] * 900, FIT_START), FIT_START, GATE_START, GATE_END)
    assert frame["constant_causal"].nunique() == 1
    assert frame["constant_causal"].iloc[0] == pytest.approx(levels["constant_causal"])
    assert len(frame) == 40


def test_a_flat_line_scores_as_a_flat_line():
    """End to end on the scoring path: a constant's WAPE is
    `sum|c - actual| / sum|actual|`, which is what ABL-380 computed by hand."""
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    frame = pd.DataFrame({"actual": actual, "challenger": actual,
                          "seasonal_naive": actual, "constant_oracle": 25.0})
    scores, _, _ = scored_with_comparators(
        frame, ("challenger", "seasonal_naive"), ("challenger", "constant_oracle"))
    expected = 100.0 * np.sum(np.abs(25.0 - actual)) / np.sum(np.abs(actual))
    assert scores["constant_oracle"]["wape_pct"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 3b. The climatology, and why the constant alone was not enough.
# ---------------------------------------------------------------------------

def test_climatology_is_the_hourly_mean_causally_and_the_hourly_median_with_hindsight():
    """One level per hour, from the same two windows as the constants.

    Mean for the causal form because that is what a forecaster would have
    fitted; median for the oracle form because it minimises `sum|a - c|` inside
    each bucket, which is what makes the oracle a true upper bound on any
    hour-of-day predictor rather than merely a good one.
    """
    series = _diurnal(peak=120.0)
    levels = climatology_reference_levels(series, FIT_START, GATE_START, GATE_END)

    fit = series[(series.index >= FIT_START) & (series.index < GATE_START)]
    gate = series[(series.index >= GATE_START) & (series.index < GATE_END)]
    for hour in range(24):
        assert levels["climatology_causal"][hour] == pytest.approx(
            float(fit[fit.index.hour == hour].mean()))
        assert levels["climatology_oracle"][hour] == pytest.approx(
            float(gate[gate.index.hour == hour].median()))
    # The shape is real: midday and midnight are different levels, which is the
    # thing a constant cannot say and this reference exists to say.
    assert levels["climatology_causal"][12] > 100.0
    assert levels["climatology_causal"][0] == pytest.approx(0.0)


def test_on_a_diurnal_series_the_oracle_constant_is_useless_and_the_climatology_is_not():
    """The measured finding that put the climatology in this module.

    On solar every constant scores 63–95% WAPE, because the WAPE-optimal flat
    line on a series that is zero over half the day sits near zero. The oracle
    constant is the *best* any flat line can do, so this is not a tuning problem
    — it is the ceiling. A comparator a challenger cannot lose to certifies
    nothing, which is the ABL-380 defect one level up.
    """
    series = _diurnal(peak=100.0)
    gate = series[(series.index >= GATE_START) & (series.index < GATE_END)]
    frame = pd.DataFrame({"target_ts": gate.index, "actual": gate.to_numpy(),
                          "challenger": gate.to_numpy() * 1.1, "seasonal_naive": gate.to_numpy()})
    frame, levels = attach_model_free_references(
        frame, series, FIT_START, GATE_START, GATE_END)
    scores, _, _ = scored_with_comparators(
        frame, ("challenger", "seasonal_naive"), ("challenger", *MODEL_FREE_COMPARATORS))

    # More than half the hours are night, so the WAPE-minimising constant is 0 MW
    # — and it is still the best flat line there is.
    assert levels["constant_oracle"] == pytest.approx(0.0)
    assert scores["constant_oracle"]["wape_pct"] == pytest.approx(100.0)
    assert scores["constant_causal"]["wape_pct"] > 60.0
    # The climatology reconstructs the shape exactly here, so it is worth
    # something on the same rows where the flat line is worth nothing.
    assert scores["climatology_oracle"]["wape_pct"] == pytest.approx(0.0, abs=1e-9)
    assert scores["climatology_oracle"]["wape_pct"] < scores["constant_oracle"]["wape_pct"]


def test_a_climatology_of_a_shapeless_series_is_just_the_constant():
    """The constant is the one-bucket degenerate case, so on a series with no
    daily cycle the two references must agree — that is what makes the *gap*
    between them readable as diurnal structure rather than as noise."""
    frame, levels = attach_model_free_references(
        _frame(n=48), _spanning(fit_level=310.0, gate_level=280.0),
        FIT_START, GATE_START, GATE_END)
    assert sorted(levels["climatology_causal"]) == list(range(24))
    assert list(levels["climatology_causal"].values()) == pytest.approx([310.0] * 24)
    assert list(levels["climatology_oracle"].values()) == pytest.approx([280.0] * 24)
    for constant, climatology in zip(CONSTANT_COMPARATORS, CLIMATOLOGY_COMPARATORS):
        assert frame[climatology].to_numpy() == pytest.approx(frame[constant].to_numpy())


def test_an_hour_never_observed_drops_its_rows_instead_of_being_filled():
    """`NULL` is not `0`, and it is not the neighbouring hour either.

    A country that reports nothing at 03:00 has no 03:00 climatology. Those rows
    leave *this column's* intersection and lower *this column's* n; every other
    comparator keeps the full basis intersection. Interpolating across the gap
    would be filling a hole to make a table look complete, which is the one thing
    this codebase does not do.
    """
    series = _diurnal(peak=100.0, night=10.0)
    gapped = series[pd.DatetimeIndex(series.index).hour != 3]
    frame, levels = attach_model_free_references(
        _frame(n=72), gapped, FIT_START, GATE_START, GATE_END)

    assert 3 not in levels["climatology_causal"] and 3 not in levels["climatology_oracle"]
    assert len(levels["climatology_causal"]) == 23
    missing = frame["target_ts"].dt.hour == 3
    assert missing.sum() == 3
    assert frame.loc[missing, "climatology_causal"].isna().all()
    assert frame.loc[~missing, "climatology_causal"].notna().all()

    scores, common, comparator_n = scored_with_comparators(
        frame, ("challenger", "seasonal_naive"),
        ("challenger", "seasonal_naive", *MODEL_FREE_COMPARATORS))
    assert len(common) == 72
    # Only the two climatology columns lose the three rows, and they still carry
    # a number — a partially measured reference is not an unmeasured one.
    for name in ("challenger", "seasonal_naive", *CONSTANT_COMPARATORS):
        assert comparator_n[name] == 72
    for name in CLIMATOLOGY_COMPARATORS:
        assert comparator_n[name] == 69, "a missing hour must cost its own rows and no others"
        assert scores[name]["wape_pct"] is not None


def test_a_climatology_that_can_never_be_measured_is_empty_not_zero():
    """Same contract as an unmeasurable constant: no level, an all-NaN column,
    n=0, and `Not measured` — never a flat line at zero."""
    gate_only = _series([5.0] * 720, GATE_START)
    levels = climatology_reference_levels(gate_only, FIT_START, GATE_START, GATE_END)
    assert levels["climatology_causal"] == {}
    assert len(levels["climatology_oracle"]) == 24

    assert climatology_reference_levels(_series([np.nan] * 500, FIT_START), FIT_START,
                                        GATE_START, GATE_END) == {
        "climatology_causal": {}, "climatology_oracle": {}}
    assert climatology_reference_levels(pd.Series(dtype=float), FIT_START, GATE_START,
                                        GATE_END) == {"climatology_causal": {},
                                                      "climatology_oracle": {}}

    frame, _ = attach_model_free_references(_frame(n=30), gate_only,
                                            FIT_START, GATE_START, GATE_END)
    assert frame["climatology_causal"].isna().all()
    scores, _, comparator_n = scored_with_comparators(
        frame, ("challenger", "seasonal_naive"),
        ("challenger", "seasonal_naive", *MODEL_FREE_COMPARATORS))
    assert comparator_n["climatology_causal"] == 0
    assert scores["climatology_causal"]["wape_pct"] is None


def test_the_causal_climatology_cannot_see_the_gate_window():
    """The causal/oracle split has to hold per hour, not just in aggregate."""
    flat_gate = _diurnal(peak=100.0)
    shifted = _diurnal(peak=100.0, gate_peak=10.0)
    a = climatology_reference_levels(flat_gate, FIT_START, GATE_START, GATE_END)
    b = climatology_reference_levels(shifted, FIT_START, GATE_START, GATE_END)
    assert a["climatology_causal"] == pytest.approx(b["climatology_causal"])
    assert a["climatology_oracle"][12] == pytest.approx(100.0)
    assert b["climatology_oracle"][12] == pytest.approx(10.0)


def test_the_lookback_and_the_post_gate_tail_are_excluded_from_the_climatology_too():
    """The 14-day lag lookback and the loader's `gate_end + 1 day` tail are
    excluded per hour by the same window helper the constants use."""
    hours = pd.date_range(FIT_START - pd.Timedelta(days=14),
                          GATE_END + pd.Timedelta(days=1), freq="h", inclusive="left")
    series = pd.Series(np.where((hours >= FIT_START) & (hours < GATE_END), 100.0, 1e6),
                       index=hours)
    levels = climatology_reference_levels(series, FIT_START, GATE_START, GATE_END)
    for name in CLIMATOLOGY_COMPARATORS:
        assert sorted(levels[name]) == list(range(24))
        assert list(levels[name].values()) == pytest.approx([100.0] * 24)


# ---------------------------------------------------------------------------
# 4. Reporting.
# ---------------------------------------------------------------------------

def test_a_pre_abl389_record_renders_without_the_columns_rather_than_raising():
    """Re-rendering an older `results.json` must print `Not measured`, not crash.
    An absent measurement and an unmeasurable one read the same; neither is a
    number."""
    assert comparator_wape({"challenger": {"wape_pct": 12.0}}, "constant_oracle") is None
    assert comparator_wape({"constant_oracle": None}, "constant_oracle") is None
    assert levels_table([{"country": "BE", "audit": {}}]) == []


def test_a_challenger_beaten_by_an_oracle_reference_is_named():
    """ABL-380's CH: PASS in the gate column, worse than a flat line. Reported,
    never gating — the note names cells, and returns nothing when there are
    none."""
    cells = [
        {"country": "CH", "horizon_band": "24-36h",
         "scores": {"challenger": {"wape_pct": 47.42},
                    "constant_oracle": {"wape_pct": 40.29},
                    "climatology_oracle": {"wape_pct": 38.20}}},
        {"country": "BG", "horizon_band": "24-36h",
         "scores": {"challenger": {"wape_pct": 56.86},
                    "constant_oracle": {"wape_pct": 63.78},
                    "climatology_oracle": {"wape_pct": 62.50}}},
    ]
    note = "\n".join(lost_to_a_model_free_reference(
        cells, lambda row: f"{row['country']} {row['horizon_band']}"))
    assert "CH 24-36h" in note and "47.42" in note and "40.29" in note and "+7.13pp" in note
    # The two oracles are reported separately, because losing to the level and
    # losing to the average day are different statements about the model.
    assert "38.20" in note and "+9.22pp" in note
    assert "loses to a constant chosen with hindsight in 1 cell(s)" in note
    assert "loses to a climatology chosen with hindsight in 1 cell(s)" in note
    assert "BG" not in note
    assert lost_to_a_model_free_reference(cells[1:], lambda row: row["country"]) == []
    # An unmeasured comparator is not a loss.
    assert lost_to_a_model_free_reference(
        [{"country": "XX", "scores": {"challenger": {"wape_pct": 50.0}}}],
        lambda row: row["country"]) == []


def test_a_cell_can_beat_the_constant_and_still_lose_to_the_climatology():
    """The case that makes reporting the two oracles separately worth a column.

    Constructed, but no longer hypothetical. The nearest real case is ABL-381's
    BG solar, where the challenger clears every flat line by 55pp and then, at
    the pinned seed, edged the hindsight hour-of-day median by 0.26pp. ABL-402
    refit it at 20 pre-committed seeds: at the seed mean that 24-36h margin is
    +0.009pp — a tie — and on **48-64h it is negative (-0.058pp)**, so BG's
    central estimate does land in this quadrant. It is the one that matters most
    on solar, where a constant is 63-95% on every cell: collapsed into one
    "beaten by a model-free reference" bullet, an easy win over the flat line
    would mask a loss to the average day and the report would say nothing at all.
    """
    note = "\n".join(lost_to_a_model_free_reference(
        [{"country": "XX", "scores": {"challenger": {"wape_pct": 40.00},
                                      "constant_oracle": {"wape_pct": 73.49},
                                      "climatology_oracle": {"wape_pct": 38.50}}}],
        lambda row: row["country"]))
    assert "loses to a constant" not in note
    assert "loses to a climatology chosen with hindsight in 1 cell(s)" in note
    assert "challenger 40.00% vs oracle climatology 38.50% (+1.50pp)" in note


def _score(wape):
    """One `score_predictions` record.

    `None` renders the unmeasured shape that function actually returns for an
    empty intersection — every key present, every value `None`. A fixture that
    dropped the key instead would let a `KeyError` in the renderer pass as a
    test failure about something else, and one that used `0.0` would assert the
    exact confusion this issue exists to prevent.
    """
    if wape is None:
        return {"n": 0, "wape_pct": None, "mae": None, "bias_pct": None,
                "slope": None, "correlation": None}
    return {"n": 720, "wape_pct": wape, "mae": 7.71, "bias_pct": 61.53,
            "slope": 0.094, "correlation": 0.176}


#: ABL-380 tranche 1a as published (merged `69f8cd5`, PASS 6/6), beside the four
#: references this issue adds. The challenger WAPEs are that read's; every
#: reference WAPE and level is measured — `_load_actuals_series` on
#: `energy_generation` over ABL-348's frozen windows, against the
#: 9,432,453,120-byte replica on 2026-08-13:
#:
#: * BG: constants 141.54 MW / 74.69 MW giving 82.77% / 63.78%; climatology
#:   81.03% / 62.50%.
#: * CH: constants 21.97 MW / 10.68 MW giving 79.07% / 40.29%; climatology
#:   77.82% / 38.20%.
#:
#: The `incumbent` column is `None` because BG and CH hold zero rows in
#: `forecasts`, which is the real condition of all 37 ABL-316 pairs.
#:
#: The D-7 column carries ABL-348's registered whole-window bar (BG 93.75%, CH
#: 59.26%) rather than the harness's per-band D-7, and MAE/bias/slope/corr are
#: placeholders. Nothing rendered under test reads them; only the challenger and
#: the four references are compared, and those are the measured ones.
ABL380_TRANCHE_1A = (
    ("BG", 93.75, (56.86, 56.82, 57.76), 82.77, 63.78, 141.54, 74.69, 81.03, 62.50),
    ("CH", 59.26, (47.42, 44.99, 44.31), 79.07, 40.29, 21.97, 10.68, 77.82, 38.20),
)
BANDS = ("24-36h", "36-48h", "48-64h")


def _report_fixture(harness, key):
    """The smallest result record `render_markdown` accepts, per harness.

    `key` is `"forecast_type"` for the wind harness and `None` for solar, which
    is the one shape difference between the two records: wind gates
    (type, country) pairs and solar gates countries. Built from
    `ABL380_TRANCHE_1A` for both, so the solar rendering is exercised on the
    same numbers rather than on a second set nobody has measured — the solar
    record is a rendering fixture and is not a solar measurement.
    """
    scope = "abl380-tranche1a" if key else "abl253"
    pair = ({"forecast_type": "wind_onshore"} if key else {})
    cells, country_d2, training = [], [], []
    for (country, naive, challengers, causal, oracle, causal_mw, oracle_mw,
         clim_causal, clim_oracle) in ABL380_TRANCHE_1A:
        for band, challenger in zip(BANDS, challengers):
            cells.append({**pair, "country": country, "horizon_band": band,
                          "gate": gate_cell(challenger, naive, 720, 720),
                          "scores": {"challenger": _score(challenger),
                                     "seasonal_naive": _score(naive),
                                     "persistence": _score(None),
                                     "incumbent": _score(None),
                                     "constant_causal": _score(causal),
                                     "constant_oracle": _score(oracle),
                                     "climatology_causal": _score(clim_causal),
                                     "climatology_oracle": _score(clim_oracle)}})
        country_d2.append({**pair, "country": country, "n": 1_920,
                           "scores": cells[-1]["scores"],
                           "tso": {"wape_pct": None, "n": 0}})
        # The hourly levels are recorded in full in `results.json`; the markdown
        # summarises them, so the fixture carries a real 24-hour set rather than
        # a shape the renderer would never receive.
        hourly = {hour: causal_mw * (0.5 + hour / 24) for hour in range(24)}
        training.append({**pair, "country": country, "algorithm": "catboost",
                         "artifact_sha256": "0" * 64, "constant_runs": [],
                         "model_free_reference_mw": {"constant_causal": causal_mw,
                                                     "constant_oracle": oracle_mw,
                                                     "climatology_causal": hourly,
                                                     "climatology_oracle": hourly},
                         "audit": {"retained_rows": 4_272, "intended_rows": 4_272,
                                   "unique_targets": 4_272, "degraded_lag_1d_rows": 0,
                                   "excluded_missing_actual_or_feature": 0}})
    registered = ({"registered_pairs": [("wind_onshore", c) for c, *_ in ABL380_TRANCHE_1A]}
                  if key else {"registered_countries": [c for c, *_ in ABL380_TRANCHE_1A]})
    return {
        "verdict": "PASS", "recommendation": "Evidence only. No promotion.",
        "gate_cells": cells, "country_d2": country_d2, "training": training,
        "meta": {"scope": scope, "generated_at": "2026-08-13T00:00:00Z",
                 "registered_cells": len(cells), "replica_bytes": 9_432_453_120,
                 "gate_basis": list(harness.GATE_BASIS[scope]),
                 "reported_comparators": list(harness.REPORTED_COMPARATORS),
                 "training_source": "energy_generation",
                 "fit_window": {"start": str(FIT_START), "end_exclusive": str(GATE_START)},
                 "gate_window": {"start": str(GATE_START), "end_exclusive": str(GATE_END)},
                 "databases": {"replica": "/replica.db", "features": "/replica.db",
                               "sidecar": None, "features_match_replica": True,
                               "ambient_matches_replica": True,
                               "ambient_energy_db_path": "/replica.db"},
                 **registered},
    }


@pytest.mark.parametrize("harness,key", [(wind, "forecast_type"), (solar, None)],
                         ids=["wind", "solar"])
def test_the_report_states_that_the_references_gate_nothing(harness, key):
    """A reader must not be able to mistake a reported column for a criterion.

    Both tables carry the columns, and the paragraph above them says in words
    what `test_the_references_are_in_no_registered_gate_basis` pins in code. A
    report that grew the columns without the sentence would be the ABL-380
    reading failure with more numbers in it.
    """
    text = harness.render_markdown(_report_fixture(harness, key))
    assert text.count("| constant causal WAPE | constant oracle WAPE | "
                      "climatology causal WAPE | climatology oracle WAPE |") == 2, (
        "both the per-cell table and the per-country summary must carry all four")
    assert "reported references and not gate criteria" in text
    assert "still reads PASS" in text
    # A climatology can be scored on fewer rows than the cell, so the report has
    # to say to check its n — a WAPE on other rows is not the same measurement.
    assert "own n before comparing" in text
    # The verdict column is untouched: all six cells clear their D-7 bar, and
    # the three that lose to a reference are still PASS.
    assert text.count("| PASS |") == 6 and "| FAIL |" not in text


@pytest.mark.parametrize("harness,key", [(wind, "forecast_type"), (solar, None)],
                         ids=["wind", "solar"])
def test_the_report_reproduces_the_abl380_finding_that_motivated_this_issue(harness, key):
    """The harness now prints, unprompted, what a human had to go looking for.

    ABL-380 reported 6/6 PASS on a pair whose fitted model was 7.1pp worse than
    a flat line, and said so only because someone computed it by hand. These are
    the measured numbers from that pair, rendered through the harness.
    """
    text = harness.render_markdown(_report_fixture(harness, key))
    # CH's cells: challenger 47.4% / D-7 59.3% / skill / the four references.
    # It clears the registered bar by 20pp and still loses to a flat line by 7pp
    # and to an hour-of-day median by 9pp.
    assert "| 47.4% | 59.3% | +20.0% | 79.1% | 40.3% | 77.8% | 38.2% |" in text
    assert "| 56.9% | 93.8% | +39.3% | 82.8% | 63.8% | 81.0% | 62.5% |" in text, (
        "BG clears a 93.75% D-7 bar that its own causal constant clears at 82.8%")
    assert "21.97 MW | 10.68 MW" in text and "141.54 MW | 74.69 MW" in text
    assert "The challenger loses to a constant chosen with hindsight in 3 cell(s)" in text
    assert "challenger 47.42% vs oracle constant 40.29% (+7.13pp)" in text
    # And the climatology, which is the tighter of the two and was in no ABL-380
    # evidence pack at all — a human computed the flat line and stopped there.
    assert "The challenger loses to a climatology chosen with hindsight in 3 cell(s)" in text
    assert "challenger 47.42% vs oracle climatology 38.20% (+9.22pp)" in text
    # BG beats both its oracles (56.86% vs 63.78% and 62.50%) and is named in no
    # bullet under either.
    assert "  - BG" not in text
