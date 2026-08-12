"""The ABL-280 per-country re-read must measure what it claims (ABL-280).

Four properties, each of which a refactor breaks without any test noticing:

- the zero baseline is *identically* WAPE's denominator, so "loses to zero" and
  "WAPE > 100%" are one fact and can never disagree;
- demeaning within the vintage day separates level error from shape error, and
  recovers an injected day-level bias exactly;
- the minimum-vintage precondition counts vintages that carry a scored pair,
  not vintages that merely exist — the two differ by the two newest vintages on
  every real run, because the rail generates at D for D+2;
- a baseline that is NaN on part of the window is scored against the model on
  the same subset, so a partly-missing baseline cannot move the model's number.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.country_reread import (  # noqa: E402
    baseline_table,
    country_reread,
    evidence_vintages,
    fleet_summary,
    level_vs_shape,
    per_vintage_day,
    render_fleet_markdown,
    render_markdown,
    zero_baseline_mae,
)
from src.evaluation.net_position import FIX_DEPLOYED_UTC, point_metrics  # noqa: E402

SPLIT = FIX_DEPLOYED_UTC


def make_paired(days, country="RO", hours=24, seed=0, day_bias=None,
                unscored_days=0, noise=50.0):
    """A left-merged frame shaped like `net_position.evaluate`'s `paired`.

    `day_bias` injects a known per-vintage-day level error into the forecast.
    `unscored_days` appends vintages whose actuals are NaN — the shape the real
    rail always has for its two newest runs.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(days):
        gen = pd.Timestamp("2026-08-05 06:00:00") + pd.Timedelta(days=i)
        bias = 0.0 if day_bias is None else float(day_bias[i])
        for h in range(hours):
            target = gen.normalize() + pd.Timedelta(days=2, hours=h)
            actual = 800.0 * np.sin(2 * np.pi * h / 24) + rng.normal(0, noise)
            rows.append({
                "country_code": country, "target_ts": target, "generated_at": gen,
                "horizon_hours": 24 + h, "forecast_value": actual + bias,
                "actual": actual,
                "persistence": actual + rng.normal(0, 300),
                "climatology": actual + rng.normal(0, 200),
            })
    for j in range(unscored_days):
        gen = pd.Timestamp("2026-08-05 06:00:00") + pd.Timedelta(days=days + j)
        for h in range(hours):
            rows.append({
                "country_code": country,
                "target_ts": gen.normalize() + pd.Timedelta(days=2, hours=h),
                "generated_at": gen, "horizon_hours": 24 + h,
                "forecast_value": 100.0, "actual": np.nan,
                "persistence": np.nan, "climatology": np.nan,
            })
    df = pd.DataFrame(rows)
    df["baseline_ensemble"] = df[["persistence", "climatology"]].mean(axis=1)
    return df


# --------------------------------------------------------------------------
# The zero baseline is WAPE's denominator, not an independent measurement
# --------------------------------------------------------------------------

def test_zero_baseline_mae_is_mean_abs_actual():
    actual = np.array([-1200.0, 300.0, 0.0, 900.0])
    assert zero_baseline_mae(actual) == pytest.approx(np.mean(np.abs(actual)))


@pytest.mark.parametrize("bias", [0.0, 400.0, 1500.0, -2000.0])
def test_loses_to_zero_iff_wape_over_100(bias):
    """The identity that keeps 'WAPE 102.6%' and 'loses to the zero forecast'
    from ever being reported as two independent findings."""
    paired = make_paired(4, day_bias=[bias] * 4)
    scored = paired.dropna(subset=["actual"])
    rows = {r["baseline"]: r for r in baseline_table(scored)}
    wape = point_metrics(scored["actual"].to_numpy(),
                         scored["forecast_value"].to_numpy())["wape_pct"]
    assert (rows["zero"]["skill_pct"] < 0) == (wape > 100.0)
    # and quantitatively: skill vs zero == 100 - WAPE
    assert rows["zero"]["skill_pct"] == pytest.approx(100.0 - wape)


# --------------------------------------------------------------------------
# Level vs shape
# --------------------------------------------------------------------------

def test_within_day_demeaning_recovers_injected_level_error():
    """A forecast with a perfect profile and a large, swinging day offset must
    read as near-perfect within-day and much worse pooled — the RO signature."""
    biases = [259.3, -1067.0, -691.2, 195.9, 519.4, -1095.1]
    paired = make_paired(6, day_bias=biases, noise=0.0)
    lvs = level_vs_shape(paired.dropna(subset=["actual"]))
    assert lvs["within_day"]["corr"] == pytest.approx(1.0, abs=1e-9)
    assert lvs["within_day"]["slope"] == pytest.approx(1.0, abs=1e-9)
    assert lvs["pooled"]["corr"] < 0.9          # the level destroys the pooled fit
    assert lvs["bias_sd_mw"] == pytest.approx(float(np.std(biases, ddof=1)), rel=1e-9)
    # day_bias_mw is keyed by date, so compare in vintage-day order
    for got, want in zip(lvs["day_bias_mw"].values(), biases):
        assert got == pytest.approx(want, abs=1e-6)


def test_bias_sd_uses_ddof_1_and_is_stated():
    """ddof=0 and ddof=1 differ by ~10% on six days; the module commits to the
    sample sd so a published figure is reproducible."""
    biases = [259.3, -1067.0, -691.2, 195.9, 519.4, -1095.1]
    lvs = level_vs_shape(make_paired(6, day_bias=biases, noise=0.0)
                         .dropna(subset=["actual"]))
    assert lvs["bias_sd_mw"] == pytest.approx(np.std(biases, ddof=1), rel=1e-9)
    assert lvs["bias_sd_mw"] != pytest.approx(np.std(biases, ddof=0), rel=1e-3)


def test_shape_error_does_not_improve_on_demeaning():
    """The control for the test above: a genuinely wrong profile stays wrong."""
    paired = make_paired(4, noise=0.0)
    scored = paired.dropna(subset=["actual"]).copy()
    rng = np.random.default_rng(7)
    scored["forecast_value"] = rng.normal(0, 800, len(scored))  # no shape at all
    lvs = level_vs_shape(scored)
    assert abs(lvs["within_day"]["corr"]) < 0.4


# --------------------------------------------------------------------------
# Vintages that carry evidence vs vintages that merely exist
# --------------------------------------------------------------------------

def test_unscored_vintages_are_counted_separately():
    paired = make_paired(7, unscored_days=2)
    ev = evidence_vintages(paired, SPLIT)
    assert ev["counted"] == 9
    assert ev["scored"] == 7
    assert len(ev["unscored_vintages"]) == 2


def test_min_vintage_precondition_reads_scored_not_counted():
    """9 vintages present, 7 with actuals: a minimum of 8 is NOT met.

    This is the whole reason the module keeps its own counter instead of
    reusing the gate's — `build_gate_scope` counts off the left-merged frame,
    where an unscored vintage still counts.
    """
    paired = make_paired(7, unscored_days=2)
    read = country_reread(paired, "RO", cohort_split=SPLIT, min_scored_vintages=8)
    assert read["vintages"]["counted"] == 9
    assert read["vintages"]["scored"] == 7
    assert read["meets_min_vintages"] is False
    assert read["read_kind"] == "interim"
    assert "not** the confirmatory read" in render_markdown(read, "now")


def test_confirmatory_once_scored_vintages_reach_the_minimum():
    read = country_reread(make_paired(8, unscored_days=2), "RO",
                          cohort_split=SPLIT, min_scored_vintages=8)
    assert read["meets_min_vintages"] is True
    assert read["read_kind"] == "confirmatory"
    assert "not** the confirmatory read" not in render_markdown(read, "now")


def test_pre_fix_vintages_are_excluded_from_the_cohort():
    paired = make_paired(3)
    old = paired.copy()
    old["generated_at"] = pd.Timestamp("2026-07-01 06:00:00")
    read = country_reread(pd.concat([paired, old], ignore_index=True), "RO",
                          cohort_split=SPLIT, min_scored_vintages=3)
    assert read["vintages"]["scored"] == 3
    assert read["n_pairs"] == 3 * 24


# --------------------------------------------------------------------------
# Baseline bookkeeping
# --------------------------------------------------------------------------

def test_partly_missing_baseline_is_scored_on_its_own_subset():
    """A baseline with no history for the first day must not shift the model's
    MAE in the comparison against it."""
    paired = make_paired(4, day_bias=[500.0] * 4)
    scored = paired.dropna(subset=["actual"]).copy()
    first = scored["generated_at"] == scored["generated_at"].min()
    scored.loc[first, "climatology"] = np.nan
    rows = {r["baseline"]: r for r in baseline_table(scored)}
    assert rows["climatology"]["n"] == len(scored) - int(first.sum())
    sub = scored[~first]
    expect = 100.0 * (1.0 - np.mean(np.abs(sub["forecast_value"] - sub["actual"]))
                      / np.mean(np.abs(sub["climatology"] - sub["actual"])))
    assert rows["climatology"]["skill_pct"] == pytest.approx(expect)
    # the model row itself still reports the full window
    assert rows["model"]["n"] == len(scored)


def test_country_with_no_paired_actuals_is_not_a_flawless_zero():
    """The GR shape: vintages exist, nothing scores. Must never render metrics."""
    paired = make_paired(0, unscored_days=3)
    read = country_reread(paired, "RO", cohort_split=SPLIT)
    assert read["coverage"] == "no_paired_actuals"
    assert read["n_pairs"] == 0
    assert "metrics" not in read
    assert "No paired actuals" in render_markdown(read, "now")


def test_excluded_countries_carry_their_reason():
    read = country_reread(make_paired(3, country="GR"), "GR",
                          cohort_split=SPLIT, min_scored_vintages=3)
    assert read["excluded_from_gate"] is not None
    assert "Excluded from the promotion gate by name" in render_markdown(read, "now")


def test_per_vintage_day_groups_same_day_reruns_together():
    """2026-08-06 really does carry both a 06:00 and a 10:52 run; the level
    analysis is per day, so those must land in one row of 48 pairs."""
    paired = make_paired(3)
    rerun = paired[paired["generated_at"] == paired["generated_at"].min()].copy()
    rerun["generated_at"] = rerun["generated_at"] + pd.Timedelta(hours=5)
    both = pd.concat([paired, rerun], ignore_index=True)
    rows = per_vintage_day(both.dropna(subset=["actual"]))
    assert len(rows) == 3
    assert rows[0]["n"] == 48
    assert evidence_vintages(both, SPLIT) == {
        **evidence_vintages(both, SPLIT), "counted": 4, "scored": 4,
        "counted_days": 3, "scored_days": 3}


# --------------------------------------------------------------------------
# Fleet sweep — the context that stops a zone-specific reading of a fleet fact
# --------------------------------------------------------------------------

def test_fleet_summary_flags_the_level_dominated_zones():
    """A zone with a big day-level swing must show a large `level_gap`; a
    well-behaved zone must not."""
    swing = make_paired(5, country="RO", day_bias=[259, -1067, -691, 196, 519],
                        noise=120.0, seed=1)
    calm = make_paired(5, country="DE", day_bias=[0.0] * 5, noise=120.0, seed=2)
    rows = {r["country"]: r
            for r in fleet_summary(pd.concat([swing, calm], ignore_index=True),
                                   ["RO", "DE"], SPLIT, min_scored_vintages=5)}
    assert rows["RO"]["level_gap"] > 0.15
    assert rows["DE"]["level_gap"] < 0.05
    assert rows["RO"]["bias_sd_frac"] > rows["DE"]["bias_sd_frac"]


def test_fleet_summary_reports_a_country_with_no_pairs_rather_than_dropping_it():
    rows = {r["country"]: r for r in
            fleet_summary(make_paired(3, country="RO"), ["RO", "GR"], SPLIT,
                          min_scored_vintages=3)}
    assert rows["GR"]["coverage"] == "no_paired_actuals"
    assert rows["GR"]["n"] == 0


def test_fleet_markdown_counts_losses_against_each_baseline():
    bad = make_paired(4, country="RO", day_bias=[3000.0] * 4, noise=10.0)
    good = make_paired(4, country="DE", day_bias=[0.0] * 4, noise=10.0, seed=3)
    rows = fleet_summary(pd.concat([bad, good], ignore_index=True),
                         ["RO", "DE"], SPLIT, min_scored_vintages=4)
    md = render_fleet_markdown(rows, "now", min_scored_vintages=4)
    assert "Loses to **zero**: RO (1/2)" in md
    assert "CONFIRMATORY" in md


def test_fleet_markdown_labels_an_underpowered_sweep_interim():
    rows = fleet_summary(make_paired(3, country="RO"), ["RO"], SPLIT,
                         min_scored_vintages=14)
    md = render_fleet_markdown(rows, "now", min_scored_vintages=14)
    assert "INTERIM" in md and "Flags, not findings" in md


def test_loses_to_lists_only_negative_skill_baselines():
    paired = make_paired(4, day_bias=[2500.0] * 4)  # far worse than zero
    read = country_reread(paired, "RO", cohort_split=SPLIT, min_scored_vintages=4)
    assert "zero" in read["loses_to"]
    assert read["metrics"]["wape_pct"] > 100.0
