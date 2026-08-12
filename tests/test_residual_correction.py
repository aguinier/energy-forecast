"""The correction study cannot read what the run could not see (ABL-65).

The measured answer to ABL-65 is that no correction shape helps. That answer is
only worth anything if the harness could not have cheated, so the load-bearing
test here is not "does the offset recover a constant" — it is **`test_no_lookahead`**:
appending rows dated after the vintage's publication cutoff must change nothing,
bit for bit. A correction that peeks is worthless (ABL-65's constraints), so the
peek is made structurally impossible and pinned here rather than audited by eye.

The rest pin the two refusals — thin history and an ABL-31 degenerate vintage
are identities with a reason, never small corrections.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.net_position import as_of_for_vintage
from src.evaluation.residual_correction import (
    DEGENERATE_MAX_ABS_MW, MIN_HISTORY_DAYS, SERVE_LEADS_H, CorrectionSpec,
    backtest_corrections, default_specs, estimate_error, score_corrections,
)

RNG = np.random.default_rng(20260812)


def _vintage_pairs(n_days=40, bias=0.0, hour_profile=None, noise=100.0,
                   start="2026-03-01", country="XX"):
    """Daily 06:00Z vintages, each covering D+2 00:00-23:00 — the real geometry.

    A synthetic cohort shaped exactly like the champion's so a shape that works
    here is not relying on a geometry the product does not have.
    """
    rows = []
    for d in range(n_days):
        gen = pd.Timestamp(start) + pd.Timedelta(days=d, hours=6)
        targets = pd.date_range(gen.normalize() + pd.Timedelta(days=2),
                                periods=24, freq="h")
        actual = 1000 * np.sin(np.arange(24) * 2 * np.pi / 24) + RNG.normal(0, 300, 24)
        err = bias + RNG.normal(0, noise, 24)
        if hour_profile is not None:
            err = err + np.asarray(hour_profile)[targets.hour]
        rows.append(pd.DataFrame({
            "country_code": country, "generated_at": gen, "target_ts": targets,
            "actual": actual, "forecast_value": actual + err}))
    return pd.concat(rows, ignore_index=True)


def test_serve_leads_match_the_real_vintage_geometry():
    """27-50h is the whole reason the AR idea fails; it must not drift silently."""
    gen = pd.Timestamp("2026-08-10 06:00:00")
    as_of = as_of_for_vintage(gen)
    assert as_of == pd.Timestamp("2026-08-10 22:00:00")
    last_observable = as_of - pd.Timedelta(hours=1)      # D 21:00
    targets = pd.date_range(gen.normalize() + pd.Timedelta(days=2), periods=24, freq="h")
    leads = ((targets - last_observable) / pd.Timedelta(hours=1)).to_numpy()
    assert (int(leads.min()), int(leads.max())) == SERVE_LEADS_H


def test_no_lookahead():
    """Rows dated after the cutoff must not move a single corrected value.

    This is the test the whole result rests on. It appends a violently biased
    future to every country and asserts the output is bit-identical.
    """
    pairs = _vintage_pairs(n_days=30, bias=250.0)
    specs = default_specs()
    before = backtest_corrections(pairs, specs)

    future = pairs.copy()
    future["generated_at"] = future["generated_at"] + pd.Timedelta(days=60)
    future["target_ts"] = future["target_ts"] + pd.Timedelta(days=60)
    future["forecast_value"] = future["actual"] + 50_000.0    # unmissable if leaked
    after = backtest_corrections(pd.concat([pairs, future], ignore_index=True), specs)
    after = after[after["generated_at"].isin(before["generated_at"].unique())]

    key = ["country_code", "generated_at", "target_ts", "spec"]
    a = before.sort_values(key).reset_index(drop=True)
    b = after.sort_values(key).reset_index(drop=True)
    assert len(a) == len(b)
    np.testing.assert_array_equal(a["corrected"].to_numpy(), b["corrected"].to_numpy())
    np.testing.assert_array_equal(a["estimate_mw"].to_numpy(), b["estimate_mw"].to_numpy())


def test_history_excludes_the_vintage_being_corrected():
    """A vintage may not learn from its own targets, even though they are past."""
    pairs = _vintage_pairs(n_days=20, bias=400.0)
    out = backtest_corrections(pairs, [CorrectionSpec("off", "offset", window_days=7)])
    first = out[out["generated_at"] == out["generated_at"].min()]
    assert bool(first["applied"].iloc[0]) is False
    # The earliest vintage has no prior vintage at all, so the refusal is the
    # absence of history rather than its thinness — either way it must not have
    # reached for the 400 MW bias sitting in its own target hours.
    assert "history" in first["reason"].iloc[0]
    assert np.all(first["estimate_mw"].to_numpy() == 0.0)


def test_thin_history_is_an_identity_with_a_reason():
    hist = pd.DataFrame({
        "target_ts": pd.date_range("2026-03-01", periods=24, freq="h"),
        "err": np.full(24, 500.0)})
    targets = pd.date_range("2026-03-04", periods=24, freq="h")
    res = estimate_error(hist, targets, pd.Timestamp("2026-03-02 22:00"),
                         CorrectionSpec("off", "offset", window_days=7))
    assert res.is_identity
    assert np.all(res.estimate_mw == 0.0)
    assert f"need {MIN_HISTORY_DAYS}" in res.reason


def test_degenerate_vintage_is_never_corrected():
    """ABL-31: a zero-filled context must not be dressed in a fitted level."""
    hist = pd.DataFrame({
        "target_ts": pd.date_range("2026-03-01", periods=24 * 10, freq="h"),
        "err": np.full(24 * 10, 800.0)})
    targets = pd.date_range("2026-03-13", periods=24, freq="h")
    spec = CorrectionSpec("off", "offset", window_days=7)

    live = estimate_error(hist, targets, pd.Timestamp("2026-03-11 22:00"), spec,
                          forecast=np.full(24, 900.0))
    assert live.applied and np.allclose(live.estimate_mw, 800.0)

    degenerate = estimate_error(hist, targets, pd.Timestamp("2026-03-11 22:00"), spec,
                                forecast=np.full(24, DEGENERATE_MAX_ABS_MW / 2))
    assert degenerate.is_identity
    assert np.all(degenerate.estimate_mw == 0.0)
    assert "ABL-31" in degenerate.reason


def test_offset_recovers_a_real_constant_bias():
    """The shape works when a persistent bias exists — so a null result is a
    statement about the data, not about a broken estimator."""
    pairs = _vintage_pairs(n_days=40, bias=600.0, noise=50.0)
    out = backtest_corrections(pairs, [CorrectionSpec("off", "offset", window_days=14)])
    late = out[(out["generated_at"] > pairs["generated_at"].min() + pd.Timedelta(days=20))
               & out["applied"]]
    assert len(late) > 0
    assert abs(late["estimate_mw"].mean() - 600.0) < 40.0
    scored = score_corrections(late)
    assert scored["mae_delta_pct"].iloc[0] > 50.0


def test_diurnal_recovers_an_hour_of_day_profile():
    profile = 400 * np.sin(np.arange(24) * 2 * np.pi / 24)
    pairs = _vintage_pairs(n_days=40, hour_profile=profile, noise=40.0)
    out = backtest_corrections(pairs, [CorrectionSpec("diu", "diurnal", window_days=14)])
    late = out[(out["generated_at"] > pairs["generated_at"].min() + pd.Timedelta(days=20))
               & out["applied"]].copy()
    late["hour"] = late["target_ts"].dt.hour
    got = late.groupby("hour")["estimate_mw"].mean().to_numpy()
    assert np.corrcoef(got, profile)[0, 1] > 0.95


def test_diurnal_shrink_zero_is_the_grand_mean():
    profile = 400 * np.sin(np.arange(24) * 2 * np.pi / 24)
    hist = pd.DataFrame({"target_ts": pd.date_range("2026-03-01", periods=24 * 14, freq="h")})
    hist["err"] = 100.0 + profile[hist["target_ts"].dt.hour]
    targets = pd.date_range("2026-03-17", periods=24, freq="h")
    as_of = pd.Timestamp("2026-03-15 22:00")
    flat = estimate_error(hist, targets, as_of,
                          CorrectionSpec("d", "diurnal", window_days=28, shrink=0.0))
    assert np.allclose(flat.estimate_mw, flat.estimate_mw[0])
    shaped = estimate_error(hist, targets, as_of,
                            CorrectionSpec("d", "diurnal", window_days=28, shrink=1.0))
    assert shaped.estimate_mw.std() > 100.0


def test_lead_ar_refuses_a_residual_dated_after_the_target():
    """Carrying a residual backwards would be information the run did not have."""
    hist = pd.DataFrame({
        "target_ts": pd.date_range("2026-03-01", periods=24 * 14, freq="h"),
        "err": RNG.normal(0, 100, 24 * 14)})
    # Targets that predate the whole history: every lead is negative.
    targets = pd.date_range("2026-02-20", periods=24, freq="h")
    res = estimate_error(hist, targets, pd.Timestamp("2026-03-15 22:00"),
                         CorrectionSpec("ar", "lead_ar", window_days=14))
    assert np.all(res.estimate_mw == 0.0)


def test_score_corrections_only_claims_skill_where_a_baseline_exists():
    pairs = _vintage_pairs(n_days=20, bias=300.0)
    out = backtest_corrections(pairs, [CorrectionSpec("id", "identity")])
    base = out[["country_code", "generated_at", "target_ts"]].copy()
    base["baseline_ensemble"] = np.nan
    base.loc[base.index[:50], "baseline_ensemble"] = 0.0
    scored = score_corrections(out, baselines=base)
    assert scored["n"].iloc[0] == len(out)
    assert scored["n_vs_ensemble"].iloc[0] == 50


def test_unknown_kind_is_an_error_not_a_silent_identity():
    with pytest.raises(ValueError):
        estimate_error(pd.DataFrame({"target_ts": [], "err": []}),
                       pd.DatetimeIndex([]), pd.Timestamp("2026-03-01"),
                       CorrectionSpec("x", "not_a_shape"))
