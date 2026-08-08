"""V016's correction layer refuses to correct what it cannot fit (ABL-68).

The layer's whole risk is that it turns a visibly-shrunk-but-honest forecast
into a confidently wrong one, by fitting three per-country parameters on a
sample too thin to support them. These tests pin both halves: it recovers real
coefficients when the data supports them, and it falls back to the champion
unchanged — with a stated reason — when it does not.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.challengers.correction import (
    MIN_SERVE_LEAD_H,
    CountryCorrection,
    apply_correction,
    fit_country_correction,
    latest_residual,
)

RNG = np.random.default_rng(20260807)


def _pairs(n_days=60, shrink=0.4, noise=100.0, level=3000.0, phi=0.0):
    """Champion-like pairs: a real signal the forecast reproduces shrunk.

    `shrink` is the slope of forecast-on-actual — the ABL-24 defect. A perfect
    affine correction should recover slope 1/shrink.
    """
    idx = pd.date_range("2026-01-01", periods=24 * n_days, freq="h")
    actual = level * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) \
        + RNG.normal(0, level / 5, len(idx))
    err = RNG.normal(0, noise, len(idx))
    if phi:
        for i in range(1, len(err)):
            err[i] = phi * err[i - 1] + err[i] * np.sqrt(1 - phi ** 2)
    forecast = shrink * actual + err
    return pd.DataFrame({"target_ts": idx, "forecast_value": forecast,
                         "actual": actual})


def test_fit_recovers_the_amplification_that_undoes_shrinkage():
    df = _pairs(shrink=0.4, noise=50.0)
    fit = fit_country_correction(df, "FR")
    assert fit.applied, fit.reason
    # OLS of actual on forecast inverts the shrinkage up to noise attenuation.
    assert fit.slope == pytest.approx(1 / 0.4, rel=0.15)
    assert fit.slope_forecast_on_actual == pytest.approx(0.4, rel=0.15)


def test_correction_removes_the_shrinkage_it_was_fitted_for():
    """The property that matters: after correction, forecast-on-actual slope
    sits near 1 instead of near `shrink`."""
    df = _pairs(shrink=0.4, noise=50.0)
    fit = fit_country_correction(df, "FR")
    corrected = apply_correction(df["forecast_value"].to_numpy(),
                                 pd.DatetimeIndex(df["target_ts"]), fit)
    a = df["actual"].to_numpy()
    before = np.cov(a, df["forecast_value"].to_numpy(), bias=True)[0, 1] / np.var(a)
    after = np.cov(a, corrected, bias=True)[0, 1] / np.var(a)
    assert before == pytest.approx(0.4, rel=0.15)
    assert abs(after - 1.0) < abs(before - 1.0)
    assert np.mean(np.abs(corrected - a)) < np.mean(np.abs(df["forecast_value"] - a))


def test_thin_sample_passes_through_uncorrected():
    """22 pairs from one target day is what the live window actually held on
    2026-08-07. It must produce V010, not a fitted-on-noise V016."""
    idx = pd.date_range("2026-08-06", periods=22, freq="h")
    df = pd.DataFrame({"target_ts": idx,
                       "forecast_value": RNG.normal(1000, 300, 22),
                       "actual": RNG.normal(2500, 800, 22)})
    fit = fit_country_correction(df, "BE")
    assert not fit.applied
    assert "insufficient fitting data" in fit.reason
    assert fit.n_pairs == 22 and fit.n_target_days <= 2
    original = df["forecast_value"].to_numpy()
    assert np.array_equal(apply_correction(original, idx, fit), original)


def test_many_pairs_over_too_few_days_still_refuses():
    """Sample size alone is not evidence: 480 pairs drawn from 5 target days
    carries roughly 5 days of information."""
    idx = pd.date_range("2026-03-01", periods=24 * 5, freq="h")
    df = pd.concat([pd.DataFrame({"target_ts": idx,
                                  "forecast_value": RNG.normal(1000, 300, len(idx)),
                                  "actual": RNG.normal(2500, 800, len(idx))})] * 4)
    fit = fit_country_correction(df, "AT")
    assert not fit.applied and "target days" in fit.reason


def test_signal_free_country_passes_through():
    idx = pd.date_range("2026-01-01", periods=24 * 60, freq="h")
    df = pd.DataFrame({"target_ts": idx,
                       "forecast_value": RNG.normal(0, 500, len(idx)),
                       "actual": RNG.normal(0, 500, len(idx))})
    fit = fit_country_correction(df, "LT")
    assert not fit.applied and "too little signal" in fit.reason


def test_unverified_serve_parity_passes_through():
    """LT/RO/BG reconstruct materially differently from what production served,
    so their fits are not trustworthy however clean they look."""
    df = _pairs(shrink=0.4)
    fit = fit_country_correction(df, "LT", serve_parity_verified=False)
    assert not fit.applied and "serve-parity unverified" in fit.reason


def test_implausible_slope_passes_through():
    df = _pairs(shrink=0.02, noise=20.0)  # needs ~50x amplification
    fit = fit_country_correction(df, "BE")
    assert not fit.applied and "implausible recalibration slope" in fit.reason


def test_ar1_decays_over_the_real_serve_lead():
    """The honest ceiling: the nearest hour V016 corrects is 27h after the last
    residual it can see, so even phi=0.96 carries only a third of it."""
    fit = CountryCorrection(country="FR", n_pairs=1000, n_target_days=40,
                            intercept_mw=0.0, slope=1.0, ar1_phi=0.96,
                            applied=True, reason="fitted")
    last_ts = pd.Timestamp("2026-08-06 21:00")
    targets = pd.date_range("2026-08-08", periods=24, freq="h")
    base = np.zeros(24)
    out = apply_correction(base, targets, fit, last_residual=1000.0,
                           last_residual_ts=last_ts)
    leads = ((targets - last_ts) / pd.Timedelta(hours=1)).to_numpy()
    assert leads[0] == MIN_SERVE_LEAD_H and leads[-1] == 50
    assert out[0] == pytest.approx(1000.0 * 0.96 ** 27)
    assert out[-1] == pytest.approx(1000.0 * 0.96 ** 50)
    assert out[0] < 350.0            # a third of the residual, not all of it
    assert np.all(np.diff(out) < 0)  # decays monotonically with lead


def test_ar1_is_not_carried_backwards_in_time():
    """A residual observed after a target hour is information the run did not
    have; it must contribute nothing."""
    fit = CountryCorrection(country="FR", n_pairs=1000, n_target_days=40,
                            intercept_mw=0.0, slope=1.0, ar1_phi=0.9,
                            applied=True, reason="fitted")
    targets = pd.date_range("2026-08-01", periods=3, freq="h")
    out = apply_correction(np.zeros(3), targets, fit, last_residual=500.0,
                           last_residual_ts=pd.Timestamp("2026-08-05 00:00"))
    assert np.array_equal(out, np.zeros(3))


def test_negative_lag1_is_clipped_not_oscillated():
    """A negative phi carried 27-51 hours would flip sign with the lead. Clip
    to no-AR-term rather than invent an alternating correction."""
    idx = pd.date_range("2026-01-01", periods=24 * 60, freq="h")
    actual = RNG.normal(0, 2000, len(idx))
    alternating = np.where(np.arange(len(idx)) % 2 == 0, 400.0, -400.0)
    df = pd.DataFrame({"target_ts": idx, "actual": actual,
                       "forecast_value": 0.5 * actual + alternating})
    fit = fit_country_correction(df, "SK")
    assert fit.ar1_phi >= 0.0


def test_latest_residual_respects_as_of_and_uses_recalibrated_forecast():
    fit = CountryCorrection(country="FR", n_pairs=1000, n_target_days=40,
                            intercept_mw=100.0, slope=2.0, applied=True,
                            ar1_phi=0.9, reason="fitted")
    history = pd.DataFrame({
        "target_ts": pd.to_datetime(["2026-08-06 20:00", "2026-08-06 21:00",
                                     "2026-08-06 23:00"]),
        "forecast_value": [500.0, 600.0, 700.0],
        "actual": [1000.0, 1400.0, 1600.0]})
    resid, ts = latest_residual(history, pd.Timestamp("2026-08-06 22:00"), fit)
    assert ts == pd.Timestamp("2026-08-06 21:00")   # 23:00 is past as_of
    assert resid == pytest.approx(1400.0 - (100.0 + 2.0 * 600.0))


def test_latest_residual_is_none_for_an_identity_correction():
    fit = fit_country_correction(pd.DataFrame(
        {"target_ts": pd.to_datetime([]), "forecast_value": [], "actual": []}), "XX")
    assert latest_residual(pd.DataFrame(), pd.Timestamp("2026-08-06"), fit) == (None, None)


def test_all_nan_actuals_do_not_become_a_fit():
    idx = pd.date_range("2026-01-01", periods=24 * 60, freq="h")
    df = pd.DataFrame({"target_ts": idx,
                       "forecast_value": RNG.normal(1000, 300, len(idx)),
                       "actual": np.nan})
    fit = fit_country_correction(df, "GR")
    assert not fit.applied and fit.n_pairs == 0
