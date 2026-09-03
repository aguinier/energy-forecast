"""The intercept-only correction cannot move slope, and cannot fire on noise.

Two properties carry this module, and both are measured here rather than
asserted in prose:

* **`test_slope_corr_and_sd_ratio_are_invariant`** -- the property the Board
  relied on when it authorised an intercept and not a general correction layer.
  V016's affine term moved slope away from the gate band in 15 of 19 zones; an
  added constant provably cannot, and "provably" is worth nothing if the code
  quietly grows a scale term later. The bound is 1e-12, and the measured value
  on live data is 1.1e-16.
* **the refusals** -- a zone whose bias flips sign between halves, lives in one
  half only, is smaller than the gate's own bar, or is inside its own noise gets
  an identity *with the number that produced it*. The DE and RO shapes are
  reproduced here as named cases, because both were measured on live post-fix
  data and both would have been corrected by a test that only looked at the
  pooled mean.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.net_position import point_metrics
from src.evaluation.static_bias import (BREAK_EVEN_ABS_T, DEGENERATE_MAX_ABS_MW,
                                        MIN_ABS_T, MIN_BIAS_FRAC, MIN_PAIRS,
                                        MIN_TARGET_DAYS, Thresholds,
                                        apply_static_bias, apply_to_frame,
                                        fit_static_bias, level_drift_diagnostic,
                                        measure, qualify, split_halves)

RNG = np.random.default_rng(20260903)


def _zone(n_days=27, bias=0.0, bias_second_half=None, level=0.0,
          level_second_half=None, shrink=1.0, noise=300.0, day_noise=0.0,
          amplitude=1000.0, start="2026-08-07", country="XX"):
    """A zone shaped like the champion: daily vintages, 24 target hours each.

    `shrink` is the amplitude the forecast reproduces, so a zone built with
    `shrink < 1` carries the ABL-24 signature and its window bias moves with
    `level` -- which is exactly the DE shape the magnitude test has to refuse.

    `day_noise` is a per-day common shock on the error. It is the difference
    between a toy and the real thing: live day-bias standard deviations run
    600-1,300 MW per zone, so a generator with i.i.d. hourly noise only would
    make 648 hourly errors look like 648 independent observations -- the exact
    overstatement the day-level unit exists to avoid.
    """
    rows = []
    for d in range(n_days):
        gen = pd.Timestamp(start) + pd.Timedelta(days=d - 2, hours=6)
        targets = pd.date_range(pd.Timestamp(start) + pd.Timedelta(days=d),
                                periods=24, freq="h")
        second = d >= n_days // 2
        lv = level_second_half if (second and level_second_half is not None) else level
        b = bias_second_half if (second and bias_second_half is not None) else bias
        shock = RNG.normal(0, day_noise) if day_noise else 0.0
        actual = (lv + amplitude * np.sin(np.arange(24) * 2 * np.pi / 24)
                  + RNG.normal(0, 200, 24))
        forecast = shrink * actual + b + shock + RNG.normal(0, noise, 24)
        rows.append(pd.DataFrame({
            "country_code": country, "generated_at": gen, "target_ts": targets,
            "actual": actual, "forecast_value": forecast}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# The invariance the authorisation rests on
# ---------------------------------------------------------------------------

def test_slope_corr_and_sd_ratio_are_invariant():
    pairs = _zone(bias=400.0, shrink=0.5)
    a = pairs["actual"].to_numpy()
    f = pairs["forecast_value"].to_numpy()
    for intercept in (-5000.0, -1.0, 0.0, 0.5, 1234.5, 1e6):
        dec = qualify("XX", pairs)
        dec = type(dec)(country="XX", applied=True, intercept_mw=intercept,
                        reason="test")
        after = point_metrics(a, apply_static_bias(f, dec))
        before = point_metrics(a, f)
        assert abs(after["slope"] - before["slope"]) < 1e-12
        assert abs(after["corr"] - before["corr"]) < 1e-12
        assert abs(after["sd_ratio"] - before["sd_ratio"]) < 1e-12
        # And the thing it *is* allowed to move, moves by exactly the intercept.
        assert after["bias_mw"] == pytest.approx(before["bias_mw"] - intercept,
                                                 abs=1e-9)


def test_correction_has_no_scale_term_to_set():
    """A regression guard: `apply_static_bias` must stay a pure translation.

    If a slope coefficient is ever added, two forecasts differing by a constant
    stop differing by that same constant after correction. That is the whole
    failure mode this issue was scoped away from.
    """
    pairs = _zone(bias=250.0)
    dec = qualify("XX", pairs)
    assert dec.applied
    f = pairs["forecast_value"].to_numpy()
    offset = 777.0
    assert np.allclose(apply_static_bias(f + offset, dec),
                       apply_static_bias(f, dec) + offset, atol=1e-12)


def test_apply_never_reads_the_actual():
    """The correction is fitted offline and applied blind; poisoning `actual`
    at apply time must change nothing, bit for bit."""
    pairs = _zone(bias=300.0)
    decisions = fit_static_bias(pairs)
    clean = apply_to_frame(pairs, decisions)["corrected"].to_numpy()
    poisoned = pairs.assign(actual=np.nan)
    dirty = apply_to_frame(poisoned, decisions)["corrected"].to_numpy()
    assert np.array_equal(clean, dirty)


def test_unqualified_zone_passes_through_bitwise_unchanged():
    pairs = _zone(bias=0.0)
    decisions = fit_static_bias(pairs)
    assert not decisions["XX"].applied
    out = apply_to_frame(pairs, decisions)
    assert np.array_equal(out["corrected"].to_numpy(),
                          out["forecast_value"].to_numpy())


def test_a_zone_with_no_decision_is_passed_through_not_dropped():
    pairs = _zone(bias=300.0, country="YY")
    out = apply_to_frame(pairs, {})
    assert len(out) == len(pairs)
    assert np.array_equal(out["corrected"].to_numpy(),
                          out["forecast_value"].to_numpy())


# ---------------------------------------------------------------------------
# Conventions must match the gate's
# ---------------------------------------------------------------------------

def test_bias_and_slope_conventions_match_point_metrics():
    pairs = _zone(bias=210.0, shrink=0.6)
    stats = measure(pairs)
    ref = point_metrics(pairs["actual"].to_numpy(),
                        pairs["forecast_value"].to_numpy())
    assert stats.bias_mw == pytest.approx(ref["bias_mw"], rel=1e-12)
    assert stats.slope == pytest.approx(ref["slope"], rel=1e-12)
    assert stats.mae_mw == pytest.approx(ref["mae_mw"], rel=1e-12)
    assert stats.mean_abs_actual_mw == pytest.approx(ref["mean_abs_actual_mw"],
                                                     rel=1e-12)


# ---------------------------------------------------------------------------
# Qualification: what fires, and what does not
# ---------------------------------------------------------------------------

def test_a_real_offset_is_qualified_and_removed():
    pairs = _zone(bias=350.0, noise=200.0)
    dec = qualify("XX", pairs)
    assert dec.applied, dec.reason
    assert dec.intercept_mw == pytest.approx(measure(pairs).bias_mw)
    after = point_metrics(pairs["actual"].to_numpy(),
                          apply_static_bias(pairs["forecast_value"].to_numpy(), dec))
    assert abs(after["bias_mw"]) < 1e-9


def test_sign_flip_across_halves_is_refused():
    """The PL shape: -354 MW in the first half, +135 MW in the second."""
    pairs = _zone(bias=-900.0, bias_second_half=900.0, noise=200.0)
    dec = qualify("XX", pairs)
    assert not dec.applied
    assert not dec.tests["sign_agrees_across_halves"]["pass"]
    assert dec.intercept_mw == 0.0


def test_a_bias_living_in_one_half_only_is_refused():
    """The DE shape: quiet first half, extreme second half, large pooled mean."""
    pairs = _zone(bias=0.0, bias_second_half=2400.0, noise=200.0)
    dec = qualify("XX", pairs)
    assert not dec.applied
    assert not dec.tests["magnitude_agrees_across_halves"]["pass"]


def test_an_immaterial_bias_is_refused_even_when_perfectly_stable():
    """Tiny, utterly reproducible, and still not the defect this issue names."""
    pairs = _zone(bias=1.0, noise=5.0, amplitude=5000.0)
    dec = qualify("XX", pairs)
    assert not dec.applied
    assert not dec.tests["material"]["pass"]


def test_a_bias_inside_its_own_noise_is_refused():
    """Material by size, and still not separable from zero once the day-level
    shock is what it is on live data -- the FI shape (18.3% of mean |net
    position|, t = +1.34)."""
    pairs = _zone(bias=120.0, noise=300.0, day_noise=900.0, amplitude=400.0)
    dec = qualify("XX", pairs)
    assert not dec.applied
    assert not dec.tests["separated_from_zero"]["pass"]


def test_thin_window_is_refused_with_the_counts_that_refused_it():
    pairs = _zone(n_days=MIN_TARGET_DAYS - 1, bias=800.0, noise=100.0)
    dec = qualify("XX", pairs)
    assert not dec.applied
    assert "insufficient fitting data" in dec.reason
    assert str(MIN_TARGET_DAYS) in dec.reason


def test_degenerate_series_is_refused():
    """ABL-31: never dress a zero-filled context in a fitted level."""
    pairs = _zone(bias=0.0)
    pairs["forecast_value"] = 0.5 * DEGENERATE_MAX_ABS_MW
    dec = qualify("XX", pairs)
    assert not dec.applied
    assert "degenerate" in dec.reason


def test_every_refusal_carries_a_number():
    """'Left alone' and 'measured and left alone' are different claims."""
    for pairs in (_zone(bias=-900.0, bias_second_half=900.0, noise=200.0),
                  _zone(bias=0.0, bias_second_half=2400.0, noise=200.0),
                  _zone(bias=1.0, noise=5.0, amplitude=5000.0)):
        dec = qualify("XX", pairs)
        assert not dec.applied
        failed = [k for k, v in dec.tests.items() if not v["pass"]]
        assert failed
        for k in failed:
            assert any(ch.isdigit() for ch in dec.tests[k]["detail"]), k
            assert dec.tests[k]["detail"] in dec.reason


def test_thresholds_are_overridable_and_are_what_decides():
    """A small, utterly stable offset: refused only because it is immaterial,
    and admitted the moment materiality is the thing that is relaxed."""
    pairs = _zone(bias=50.0, noise=100.0, amplitude=5000.0)
    strict = qualify("XX", pairs)
    assert not strict.applied
    assert not strict.tests["material"]["pass"]
    assert strict.tests["separated_from_zero"]["pass"]
    assert qualify("XX", pairs, Thresholds(min_bias_frac=0.0)).applied


def test_the_bar_sits_above_the_break_even():
    """The bar is conservative *of a stated break-even*, not of nothing."""
    assert MIN_ABS_T > BREAK_EVEN_ABS_T
    assert BREAK_EVEN_ABS_T == 1.0


# ---------------------------------------------------------------------------
# Machinery
# ---------------------------------------------------------------------------

def test_split_halves_never_shares_a_target_day():
    pairs = _zone(n_days=27)
    h1, h2 = split_halves(pairs)
    d1 = set(h1["target_ts"].dt.normalize())
    d2 = set(h2["target_ts"].dt.normalize())
    assert d1 and d2
    assert not (d1 & d2)
    assert len(d1) + len(d2) == pairs["target_ts"].dt.normalize().nunique()


def test_day_is_the_independence_unit_not_the_hour():
    """An hourly t would be ~2.8x larger here and would fire on a day shock.

    Pinned as an inequality with a margin rather than a ratio, because the
    factor depends on how much of the variance is day-level -- the point is the
    direction and that it is large, not a particular multiple.
    """
    pairs = _zone(bias=150.0, noise=1200.0, day_noise=800.0, amplitude=800.0)
    stats = measure(pairs)
    err = pairs["forecast_value"].to_numpy() - pairs["actual"].to_numpy()
    hourly_se = err.std(ddof=1) / np.sqrt(len(err))
    assert stats.day_se_mw > 2.0 * hourly_se


def test_level_drift_diagnostic_recovers_a_pure_shrinkage_bias():
    """A zone with *no* offset at all, only shrinkage and a level move, must
    read as fully explained -- this is the DE discriminator."""
    pairs = _zone(bias=0.0, shrink=0.5, level=-4000.0, level_second_half=4000.0,
                  noise=1.0, amplitude=100.0)
    h1, h2 = split_halves(pairs)
    diag = level_drift_diagnostic(measure(h1), measure(h2), measure(pairs))
    assert diag["measurable"]
    assert diag["explained_frac"] == pytest.approx(1.0, abs=0.05)


def test_newey_west_se_tracks_the_plain_se_on_independent_days():
    pairs = _zone(bias=200.0, noise=400.0)
    stats = measure(pairs)
    assert stats.nw_se_mw is not None
    assert stats.nw_se_mw == pytest.approx(stats.day_se_mw, rel=0.6)


def test_fit_static_bias_covers_every_requested_zone():
    pairs = pd.concat([_zone(bias=350.0, country="AA"),
                       _zone(bias=0.0, country="BB")], ignore_index=True)
    out = fit_static_bias(pairs, ["AA", "BB", "CC"])
    assert set(out) == {"AA", "BB", "CC"}
    assert out["AA"].applied
    assert not out["BB"].applied
    assert not out["CC"].applied
    assert "no scored pairs" in out["CC"].reason


def test_min_pairs_floor_matches_the_day_floor():
    """The two coverage bounds must not be able to drift apart: 20 complete
    target days is 480 hourly pairs, so a 400-pair floor is the slack, not a
    second, looser rule."""
    assert MIN_PAIRS <= MIN_TARGET_DAYS * 24
    assert MIN_BIAS_FRAC == 0.05      # the gate's own bar (GATE_BIAS_FRAC)
