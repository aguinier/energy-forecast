"""The net-position eval must measure what it claims to measure (ABL-30).

The properties pinned here are the ones a refactor breaks silently:

- slope/sd-ratio recover a known affine relationship (the ABL-24 shrinkage
  signature must be *measurable*, not just printed);
- baselines are serve-faithful — they read nothing at or after the vintage's
  publication cutoff, and the cutoff itself follows the day-ahead rule;
- a country with vintages but no actuals scores as `no_paired_actuals`,
  never as a flawless zero (the GR shape);
- the sidecar wins on overlap with the prod-pushed copy, and the overlap
  diff is reported, not fixed;
- the error decomposition's fractions sum to 1;
- the promotion gate passes and fails on exactly the pre-registered C3 rules.

And, since ABL-72, the four properties that make the gate's answer mean what it
says: it scores its own vintage window rather than every stored vintage, it
emits all eight pre-registered criteria and cannot report PASS while one is
absent or un-evaluable, it excludes LU/GR by name rather than by symptom, and a
multi-model comparison measures every column over one identical window.
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.net_position import (
    GATE_EXCLUDED_COUNTRIES, GATE_MIN_LIVE_VINTAGES, GATE_SLOPE_RANGE,
    PRE_REGISTERED_CHECKS,
    EvalConfig, as_of_for_vintage, baseline_predictions, compare_models,
    decompose_error, evaluate, point_metrics, promotion_gate,
    render_comparison_markdown, render_markdown,
)

COUNTRY = "AA"
NO_ACTUALS_COUNTRY = "BB"   # forecasts exist, actuals never published


ACTUALS_START = "2026-07-01"
ACTUALS_END = "2026-09-10 23:00"   # wide enough for post-cohort-split vintages


def _make_dbs(tmp_path, forecast_fn, quantile_fn=None, actual_fn=None,
              vintage_days=("2026-07-28", "2026-07-29"),
              vintage_forecast_fn=None, countries=(COUNTRY, NO_ACTUALS_COUNTRY),
              actual_countries=(COUNTRY,), model_name="chronos-2-V010"):
    """Build replica + sidecar with the production column layout.

    forecast_fn(actual) -> forecast value; actual_fn(ts) -> actual value;
    quantile_fn(forecast, q, ts) -> stored quantile value (None = no quantiles).
    vintage_forecast_fn(actual, generated_at) overrides forecast_fn when the
    forecast must differ *by vintage* — which is how the ABL-72 G1 contamination
    is reproduced: bad pre-fix vintages beside good post-fix ones.
    """
    replica = tmp_path / "replica.db"
    sidecar = tmp_path / "sidecar.db"
    if actual_fn is None:
        actual_fn = lambda ts: 300.0 + 200.0 * np.sin(2 * np.pi * ts.hour / 24)

    rcon = sqlite3.connect(replica)
    rcon.execute("""CREATE TABLE net_position (
        id INTEGER PRIMARY KEY, country_code TEXT, timestamp_utc TEXT,
        net_position_mw REAL, data_quality TEXT,
        publication_timestamp_utc TEXT, fetched_at TEXT)""")
    hours = pd.date_range(ACTUALS_START, ACTUALS_END, freq="h")
    rcon.executemany(
        "INSERT INTO net_position (country_code, timestamp_utc, net_position_mw) VALUES (?,?,?)",
        [(cc, str(ts), actual_fn(ts)) for cc in actual_countries for ts in hours])

    for path in (replica, sidecar):
        con = sqlite3.connect(path) if path != replica else rcon
        con.execute("""CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY, country_code TEXT, forecast_type TEXT,
            target_timestamp_utc TEXT, generated_at TEXT, horizon_hours INTEGER,
            forecast_value REAL, model_name TEXT, model_version TEXT)""")
        con.execute("""CREATE TABLE IF NOT EXISTS forecast_quantiles (
            id INTEGER PRIMARY KEY, country_code TEXT, forecast_type TEXT,
            target_timestamp_utc TEXT, generated_at TEXT, quantile REAL,
            forecast_value REAL, model_name TEXT)""")
        if path != replica:
            con.commit()

    scon = sqlite3.connect(sidecar)
    for day in vintage_days:
        gen = pd.Timestamp(f"{day} 06:00:00")
        targets = pd.date_range(pd.Timestamp(day) + pd.Timedelta(days=2),
                                periods=24, freq="h")
        for cc in countries:
            for ts in targets:
                a = actual_fn(ts)
                f = (forecast_fn(a) if vintage_forecast_fn is None
                     else vintage_forecast_fn(a, gen))
                horizon = int((ts - gen).total_seconds() // 3600)
                row = (cc, "net_position", str(ts), str(gen), horizon, f,
                       model_name, "test")
                scon.execute("""INSERT INTO forecasts (country_code, forecast_type,
                    target_timestamp_utc, generated_at, horizon_hours,
                    forecast_value, model_name, model_version)
                    VALUES (?,?,?,?,?,?,?,?)""", row)
                if quantile_fn:
                    for q in (0.1, 0.5, 0.9):
                        scon.execute("""INSERT INTO forecast_quantiles (country_code,
                            forecast_type, target_timestamp_utc, generated_at,
                            quantile, forecast_value, model_name)
                            VALUES (?,?,?,?,?,?,?)""",
                            (cc, "net_position", str(ts), str(gen), q,
                             quantile_fn(f, q, ts), model_name))
    scon.commit(); scon.close()
    rcon.commit(); rcon.close()
    return EvalConfig(replica_db=str(replica), sidecar_db=str(sidecar),
                      model_name=model_name, serve_faithful_attestation=None)


# ---------------------------------------------------------------------------
# Amplitude metrics — the ABL-24 signature must be measurable
# ---------------------------------------------------------------------------

def test_point_metrics_recover_known_shrinkage():
    rng = np.random.default_rng(7)
    actual = rng.normal(0, 1000, 500)
    forecast = 0.5 * actual + 100.0
    m = point_metrics(actual, forecast)
    assert m["slope"] == pytest.approx(0.5, abs=1e-9)
    assert m["sd_ratio"] == pytest.approx(0.5, abs=1e-9)
    assert m["corr"] == pytest.approx(1.0, abs=1e-9)
    assert m["bias_mw"] == pytest.approx(100 - 0.5 * actual.mean(), abs=1e-6)
    assert m["nmae"] == pytest.approx(m["mae_mw"] / np.mean(np.abs(actual)))


def test_evaluate_end_to_end_measures_shrinkage(tmp_path):
    cfg = _make_dbs(tmp_path, forecast_fn=lambda a: 0.5 * a)
    res = evaluate(cfg)
    m = res["per_country"][COUNTRY]
    assert m["n"] == 48
    assert m["slope"] == pytest.approx(0.5, abs=1e-6)
    assert m["sd_ratio"] == pytest.approx(0.5, abs=1e-6)
    # markdown renders without crashing and names the country
    md = render_markdown(res, "test")
    assert COUNTRY in md and "Promotion gate" in md


def test_no_actuals_country_reads_as_no_coverage_not_zero_error(tmp_path):
    cfg = _make_dbs(tmp_path, forecast_fn=lambda a: a)
    res = evaluate(cfg)
    m = res["per_country"][NO_ACTUALS_COUNTRY]
    assert m["coverage"] == "no_paired_actuals"
    assert m["n"] == 0 and "mae_mw" not in m


# ---------------------------------------------------------------------------
# Serve-faithful cutoff and baselines
# ---------------------------------------------------------------------------

def test_as_of_follows_day_ahead_publication_rule():
    # 06:00Z scheduled run: sees through run-day 21:00 (ABL-28's as_of = D 22:00)
    assert as_of_for_vintage(pd.Timestamp("2026-08-04 06:00:44")) == \
        pd.Timestamp("2026-08-04 22:00:00")
    # ad-hoc 16:31Z run: next day's publication is already out
    assert as_of_for_vintage(pd.Timestamp("2026-07-24 16:31:19")) == \
        pd.Timestamp("2026-07-25 22:00:00")


def test_baselines_read_nothing_at_or_after_as_of():
    idx = pd.date_range("2026-07-01", "2026-07-10 23:00", freq="h")
    actuals = pd.Series(100.0, index=idx)
    # everything from the 8th onward is poisoned — a leak would be visible
    actuals[actuals.index >= "2026-07-08"] = 1e9
    targets = pd.date_range("2026-07-09", periods=24, freq="h")
    preds = baseline_predictions(actuals, pd.Timestamp("2026-07-08 00:00"), targets)
    assert (preds["persistence"] == 100.0).all()
    assert (preds["climatology"] == 100.0).all()


def test_persistence_is_last_available_day_same_hour():
    idx = pd.date_range("2026-07-01", "2026-07-07 23:00", freq="h")
    actuals = pd.Series([float(ts.day * 100 + ts.hour) for ts in idx], index=idx)
    targets = pd.date_range("2026-07-09", periods=24, freq="h")
    # as_of mid-day on the 7th: hours 0-9 come from the 7th, hours 10+ from the 6th
    preds = baseline_predictions(actuals, pd.Timestamp("2026-07-07 10:00"), targets)
    assert preds["persistence"].iloc[3] == 703.0
    assert preds["persistence"].iloc[15] == 615.0


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def test_decomposition_fractions_sum_to_one_and_find_planted_structure():
    rng = np.random.default_rng(11)
    n = 24 * 30
    hours = np.tile(np.arange(24), n // 24)
    actual = rng.normal(0, 800, n)
    forecast = 0.6 * actual + 300 + 50 * np.sin(2 * np.pi * hours / 24) \
        + rng.normal(0, 40, n)
    d = decompose_error(actual, forecast, hours)
    total = d["frac_static_bias"] + d["frac_affine"] + d["frac_diurnal"] + d["frac_residual"]
    assert total == pytest.approx(1.0, abs=1e-9)
    assert d["frac_static_bias"] > 0.3      # the planted +300
    assert d["frac_affine"] > 0.3           # the planted 0.6 slope
    assert d["frac_residual"] < 0.1
    assert d["affine_alpha"] == pytest.approx(1 / 0.6, rel=0.05)


def test_decomposition_refuses_tiny_samples():
    assert "note" in decompose_error(np.ones(10), np.ones(10), np.zeros(10))


# ---------------------------------------------------------------------------
# Promotion gate — pre-registered C3 rules
# ---------------------------------------------------------------------------

# The gate scores vintages at or after the cohort split, and requires at least
# GATE_MIN_LIVE_VINTAGES of them, so every gate fixture must sit *after*
# FIX_DEPLOYED_UTC (2026-08-04 14:29) and supply enough days. A fixture dated
# before the split now scores zero vintages — which is the point of ABL-72 G1,
# and is asserted directly in test_gate_ignores_pre_fix_vintages.
GATE_VINTAGE_DAYS = tuple(str(d.date()) for d in
                          pd.date_range("2026-08-05", periods=GATE_MIN_LIVE_VINTAGES))
# Dominated by pre-fix vintages, as the champion's stored set really is: 31 of
# 45 here (69%) against 6,312 of 6,730 (94%) measured on the replica 2026-08-07.
PRE_FIX_VINTAGE_DAYS = tuple(str(d.date()) for d in
                             pd.date_range("2026-07-02", "2026-08-01"))


def _noisy_actual_fn(seed=3):
    rng = np.random.default_rng(seed)
    noise = {  # reproducible per-timestamp noise so persistence is imperfect
        ts: float(rng.normal(0, 150))
        for ts in pd.date_range(ACTUALS_START, ACTUALS_END, freq="h")}
    return lambda ts: 300.0 + 200.0 * np.sin(2 * np.pi * ts.hour / 24) + noise[ts]


def _gate_fixture(tmp_path, forecast_fn, quantile_fn, noisy=True,
                  vintage_days=GATE_VINTAGE_DAYS, **kw):
    return _make_dbs(tmp_path, forecast_fn=forecast_fn, quantile_fn=quantile_fn,
                     actual_fn=_noisy_actual_fn() if noisy else None,
                     vintage_days=vintage_days, **kw)


def _calibrated_quantile_fn(f, q, ts):
    # 10-90 band drawn so coverage lands at 75-85%: hours 0-4 fall outside
    # the band, 19 of 24 inside -> 79.2%
    if q == 0.5:
        return f
    wide = ts.hour >= 5
    if q == 0.1:
        return f - (500 if wide else -1)
    return f + (500 if wide else -0.5)


def _passing_gate_cfg(tmp_path, **kw):
    """A fixture that clears all eight criteria — the baseline the negative
    gate tests perturb one property at a time from."""
    cfg = _gate_fixture(tmp_path, forecast_fn=lambda a: a,
                        quantile_fn=_calibrated_quantile_fn, **kw)
    ref = {"V010": {COUNTRY: {"net_position": {"W01": {"mae": 500.0}}}}}
    for name in ("ref.json", "cand.json"):
        (tmp_path / name).write_text(json.dumps(ref))
    cfg.reference_backtest = str(tmp_path / "ref.json")
    cfg.candidate_backtest = str(tmp_path / "cand.json")
    cfg.serve_faithful_verified = True
    return cfg


def test_gate_passes_a_calibrated_forecast(tmp_path):
    res = evaluate(_passing_gate_cfg(tmp_path))
    gate = res["gate"]
    failing = {k: v for k, v in gate["checks"].items() if v["pass"] is False}
    assert not failing, failing
    assert gate["verdict"] == "PASS"


def test_gate_fails_shrinkage_and_unattested_serve_parity(tmp_path):
    cfg = _gate_fixture(tmp_path, forecast_fn=lambda a: 0.5 * a,
                        quantile_fn=lambda f, q, ts: f + (q - 0.5) * 20)
    res = evaluate(cfg)
    checks = res["gate"]["checks"]
    assert res["gate"]["verdict"] == "FAIL"
    assert COUNTRY in checks["slope_in_range_per_country"]["countries_failing"]
    assert checks["serve_faithful_inputs_verified"]["pass"] is False
    # narrow band around a shrunk forecast cannot cover 75-85%
    assert COUNTRY in checks["coverage_10_90_in_band_per_country"]["countries_failing"]
    # and the missing candidate backtest is not evaluable, never a silent pass
    assert checks["no_regression_W01_W12"]["pass"] is None


def test_gate_reads_model_keyed_serve_attestation(tmp_path):
    cfg = _gate_fixture(tmp_path, forecast_fn=lambda a: a,
                        quantile_fn=_calibrated_quantile_fn)
    artifact = tmp_path / "attestation.json"
    artifact.write_text(json.dumps({"models": {cfg.model_name: {
        "model_name": cfg.model_name,
        "verified": True,
        "vintage": {
            "generated_at_utc": "2026-08-11 06:00:55",
            "publication_cutoff_exclusive_utc": "2026-08-11 22:00:00",
        },
        "max_abs_delta_mw": 0.0,
        "rows_compared": 24,
        "per_country": {COUNTRY: {"max_abs_delta_mw": 0.0}},
    }}}), encoding="utf-8")
    cfg.serve_faithful_attestation = str(artifact)
    check = evaluate(cfg)["gate"]["checks"]["serve_faithful_inputs_verified"]
    assert check["pass"] is True
    assert "max |delta| 0 MW over 24 rows / 1 countries" in check["detail"]


def test_gate_rejects_attestation_for_another_model(tmp_path):
    cfg = _gate_fixture(tmp_path, forecast_fn=lambda a: a,
                        quantile_fn=_calibrated_quantile_fn)
    artifact = tmp_path / "attestation.json"
    artifact.write_text(json.dumps({"models": {"other": {}}}), encoding="utf-8")
    cfg.serve_faithful_attestation = str(artifact)
    check = evaluate(cfg)["gate"]["checks"]["serve_faithful_inputs_verified"]
    assert check["pass"] is False
    assert "invalid serve-faithful attestation" in check["detail"]


def test_gate_flags_backtest_regression(tmp_path):
    cfg = _gate_fixture(tmp_path, forecast_fn=lambda a: a,
                        quantile_fn=lambda f, q, ts: f + (q - 0.5) * 2000)
    ref = {"V010": {COUNTRY: {"net_position": {"W01": {"mae": 500.0}}}}}
    cand = {"V013": {COUNTRY: {"net_position": {"W01": {"mae": 600.0}}}}}
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    cfg.reference_backtest = str(tmp_path / "ref.json")
    cfg.candidate_backtest = str(tmp_path / "cand.json")
    res = evaluate(cfg)
    check = res["gate"]["checks"]["no_regression_W01_W12"]
    assert check["pass"] is False and COUNTRY in check["detail"]
    # the credibility table carries the live/backtest ratio for gate countries
    row = next(r for r in res["backtest_vs_live"] if r["country"] == COUNTRY)
    assert row["backtest_mae_mw"] == pytest.approx(500.0)
    assert row["live_over_backtest"] == pytest.approx(row["live_mae_mw"] / 500.0)


def test_gate_reports_limited_backtest_country_coverage(tmp_path):
    cfg = _passing_gate_cfg(tmp_path)
    check = evaluate(cfg)["gate"]["checks"]["no_regression_W01_W12"]
    assert check["pass"] is True
    assert check["countries_compared"] == [COUNTRY]
    assert check["coverage_complete"] is False
    assert check["coverage"].endswith("/19 gated countries")
    assert "does not establish no-regression" in check["detail"]


def test_gate_fails_when_candidate_omits_a_reference_backtest_country(tmp_path):
    cfg = _passing_gate_cfg(tmp_path)
    ref = {"V010": {
        COUNTRY: {"net_position": {"W01": {"mae": 500.0}}},
        "FR": {"net_position": {"W01": {"mae": 900.0}}},
    }}
    cand = {"V012": {
        COUNTRY: {"net_position": {"W01": {"mae": 400.0}}},
    }}
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    check = evaluate(cfg)["gate"]["checks"]["no_regression_W01_W12"]
    assert check["pass"] is False
    assert check["countries_missing_from_candidate"] == ["FR"]
    assert "missing candidate backtest" in check["detail"]


# ---------------------------------------------------------------------------
# ABL-72 — the gate must score the right data, and all eight criteria
# ---------------------------------------------------------------------------

def test_gate_ignores_pre_fix_vintages(tmp_path):
    """G1. The gate read `per_country`, built from every stored vintage. For the
    champion that is 94% pre-context-fix data: MAE 1,439 MW / slope 0.26 against
    a real post-fix 553 MW / 0.90, so a challenger cleared a bar 2.60x easier
    than the one it should face. Here the pre-fix vintages are shrunk 0.3x and
    the post-fix ones are exact; the gate must see only the exact ones."""
    split = pd.Timestamp("2026-08-04 14:29:00")
    cfg = _gate_fixture(
        tmp_path, forecast_fn=None, quantile_fn=None,
        vintage_days=PRE_FIX_VINTAGE_DAYS + GATE_VINTAGE_DAYS,
        vintage_forecast_fn=lambda a, gen: (0.95 * a) if gen >= split else 0.3 * a)
    res = evaluate(cfg)

    # The report still covers every vintage — only the gate is restricted.
    assert res["meta"]["vintages"] == len(PRE_FIX_VINTAGE_DAYS) + len(GATE_VINTAGE_DAYS)
    assert res["gate_scope"]["vintages"] == len(GATE_VINTAGE_DAYS)

    report, gated = res["per_country"][COUNTRY], res["gate_scope"]["per_country"][COUNTRY]
    assert gated["slope"] == pytest.approx(0.95, abs=1e-6)   # what the model serves
    assert report["slope"] < 0.6                             # what the old gate read
    assert report["mae_mw"] > 5 * gated["mae_mw"]            # the handicap, ~2.60x live

    # The consequence, stated directly: the honest window passes the slope
    # criterion and the contaminated one would have failed it, so the two
    # readings do not merely differ in precision — they disagree on the verdict.
    assert res["gate"]["checks"]["slope_in_range_per_country"]["pass"] is True
    lo, hi = GATE_SLOPE_RANGE
    assert not (lo <= report["slope"] <= hi)


def test_gate_window_and_vintage_count_are_reported(tmp_path):
    """A restriction nobody can see is not a restriction."""
    md = render_markdown(evaluate(_passing_gate_cfg(tmp_path)), "test")
    assert "Gate vintage window" in md
    assert f"**{GATE_MIN_LIVE_VINTAGES} vintages**" in md
    assert "2026-08-05" in md


def test_gate_fails_closed_below_min_live_vintages(tmp_path):
    """G2 (plan Rev 3:54). `meta.vintages` was reported and never gated on."""
    few = GATE_VINTAGE_DAYS[:GATE_MIN_LIVE_VINTAGES - 1]
    res = evaluate(_passing_gate_cfg(tmp_path, vintage_days=few))
    check = res["gate"]["checks"]["min_live_shadow_vintages"]
    assert check["pass"] is False
    assert f"{len(few)} live shadow vintages" in check["detail"]
    assert res["gate"]["verdict"] == "FAIL"


def test_same_day_reruns_count_as_vintages_but_run_days_are_reported(tmp_path):
    """A same-day re-run makes two vintages out of one day of evidence — live on
    the replica 2026-08-07, where 4 post-fix vintages come from 3 run-days
    (08-06 has both a 06:00 and a 10:52 run). The criterion is pre-registered in
    vintages and is still scored in vintages; the run-day count is surfaced
    beside it so '14 vintages' cannot quietly mean five days of re-runs."""
    days = list(GATE_VINTAGE_DAYS)
    cfg = _passing_gate_cfg(tmp_path, vintage_days=tuple(days))
    _add_model_rows(cfg, cfg.model_name, (days[-1],), lambda a: a,
                    _noisy_actual_fn(), gen_hour=10)   # a second run, same day
    scope = evaluate(cfg)["gate_scope"]
    assert scope["vintages"] == len(days) + 1
    assert scope["vintage_days"] == len(days)
    check = evaluate(cfg)["gate"]["checks"]["min_live_shadow_vintages"]
    assert check["pass"] is True                       # scored in vintages, as written
    assert f"from {len(days)} distinct run-days" in check["detail"]


def test_gate_emits_exactly_the_eight_pre_registered_criteria(tmp_path):
    res = evaluate(_passing_gate_cfg(tmp_path))
    assert len(PRE_REGISTERED_CHECKS) == 8
    assert set(res["gate"]["checks"]) == set(PRE_REGISTERED_CHECKS)
    assert res["gate"]["criteria_missing"] == []


def test_an_unevaluable_criterion_cannot_read_as_pass(tmp_path):
    """G2's other half. The verdict spanned 'only evaluable checks', so a
    criterion that was never implemented could not fail — it was simply absent,
    and PASS was reported over the checks that happened to exist."""
    cfg = _passing_gate_cfg(tmp_path)
    cfg.candidate_backtest = None       # leaves no_regression_W01_W12 at pass=None
    res = evaluate(cfg)
    assert res["gate"]["checks"]["no_regression_W01_W12"]["pass"] is None
    assert res["gate"]["verdict"] == "INCOMPLETE"
    assert "no_regression_W01_W12" in res["gate"]["criteria_unevaluable"]


def test_a_missing_criterion_is_named_in_the_report(tmp_path):
    """The report iterates the pre-registered tuple, not whatever the gate
    happened to emit, so a criterion that is absent is printed as absent — the
    failure mode that hid `min_live_shadow_vintages` for the whole of C2."""
    res = evaluate(_passing_gate_cfg(tmp_path))
    res["gate"]["checks"].pop("slope_in_range_per_country")
    md = render_markdown(res, "test")
    assert "NOT IMPLEMENTED" in md
    assert "slope_in_range_per_country" in md


def test_gate_with_no_scope_fails_closed():
    """Called on a results dict with no gate scope — e.g. by a caller that
    skipped `build_gate_scope` — the gate must not report a clean sheet."""
    gate = promotion_gate({}, EvalConfig(replica_db=":memory:"))
    assert gate["verdict"] != "PASS"
    assert gate["checks"]["min_live_shadow_vintages"]["pass"] is False
    assert set(gate["checks"]) == set(PRE_REGISTERED_CHECKS)


def test_excluded_zone_is_excluded_by_name_even_when_it_has_data(tmp_path):
    """G3 (plan Rev 3:55). GR is excluded today only as a side-effect of having
    zero paired actuals. Give it actuals — as a partial upstream resume would —
    and it silently re-enters the gate and fails it on thin data."""
    cfg = _gate_fixture(tmp_path, forecast_fn=lambda a: 0.2 * a, quantile_fn=None,
                        countries=(COUNTRY, "GR"), actual_countries=(COUNTRY, "GR"))
    res = evaluate(cfg)
    assert res["gate_scope"]["per_country"]["GR"]["n"] > 0   # GR really has pairs

    check = res["gate"]["checks"]["excluded_zones_LU_GR"]
    assert check["pass"] is True
    assert check["excluded_with_data"] == ["GR"]
    assert set(check["excluded"]) == set(GATE_EXCLUDED_COUNTRIES) == {"LU", "GR"}
    assert "ABL-35" in GATE_EXCLUDED_COUNTRIES["GR"]
    # GR's 0.2x shrinkage must not appear in any other criterion's failures
    for name, c in res["gate"]["checks"].items():
        assert "GR" not in (c.get("countries_failing") or []), name


def _add_model_rows(cfg, model_name, vintage_days, forecast_fn, actual_fn,
                    countries=(COUNTRY,), gen_hour=6):
    con = sqlite3.connect(cfg.sidecar_db)
    for day in vintage_days:
        gen = pd.Timestamp(f"{day} {gen_hour:02d}:00:00")
        for ts in pd.date_range(pd.Timestamp(day) + pd.Timedelta(days=2),
                                periods=24, freq="h"):
            for cc in countries:
                con.execute("""INSERT INTO forecasts (country_code, forecast_type,
                    target_timestamp_utc, generated_at, horizon_hours,
                    forecast_value, model_name, model_version)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (cc, "net_position", str(ts), str(gen),
                     int((ts - gen).total_seconds() // 3600),
                     forecast_fn(actual_fn(ts)), model_name, "test"))
    con.commit(); con.close()


def test_compare_models_scores_every_model_over_one_window(tmp_path):
    """G4. The C2c deliverable is one table with a column per candidate. The
    load-bearing property is that the columns share a window — comparing a
    challenger's recent vintages against a champion's longer, partly pre-fix
    history is G1's defect moved from inside one model to between two."""
    afn = _noisy_actual_fn()
    cfg = _make_dbs(tmp_path, forecast_fn=lambda a: a, actual_fn=afn,
                    vintage_days=GATE_VINTAGE_DAYS, countries=(COUNTRY,),
                    model_name="chronos-2-V010")
    late = GATE_VINTAGE_DAYS[7:]        # the challenger started shadowing later
    _add_model_rows(cfg, "V016", late, lambda a: 0.5 * a, afn)

    cmp = compare_models(cfg, ["chronos-2-V010", "V016"])
    assert cmp["window"]["vintage_start"][:10] == late[0]   # the overlap, not 08-05
    assert cmp["vintages_per_model"] == {"chronos-2-V010": len(late), "V016": len(late)}
    # the champion is not credited for the seven vintages the challenger lacks
    assert cmp["pairs_per_model"]["chronos-2-V010"] == cmp["pairs_per_model"]["V016"]
    v010 = cmp["per_model"]["chronos-2-V010"]["gate_scope"]["per_country"][COUNTRY]
    v016 = cmp["per_model"]["V016"]["gate_scope"]["per_country"][COUNTRY]
    assert v010["mae_mw"] == pytest.approx(0.0, abs=1e-9)
    assert v016["slope"] == pytest.approx(0.5, abs=1e-6)

    md = render_comparison_markdown(cmp, "test")
    assert "Identical vintage window" in md
    assert "MAE (MW) by country" in md and "V016" in md


def test_compare_models_reports_a_model_with_no_vintages(tmp_path):
    """A model that stored nothing must read as absent, never as a clean sweep."""
    cfg = _make_dbs(tmp_path, forecast_fn=lambda a: a, actual_fn=_noisy_actual_fn(),
                    vintage_days=GATE_VINTAGE_DAYS, countries=(COUNTRY,),
                    model_name="chronos-2-V010")
    cmp = compare_models(cfg, ["chronos-2-V010", "V999-never-ran"])
    assert cmp["window"]["models_with_no_vintages"] == ["V999-never-ran"]
    assert "V999-never-ran" in cmp["errors"]
    assert "V999-never-ran" not in cmp["vintages_per_model"]
    md = render_comparison_markdown(cmp, "test")
    assert "Not scored" in md and "V999-never-ran" in md


# ---------------------------------------------------------------------------
# Sidecar/replica overlap
# ---------------------------------------------------------------------------

def test_sidecar_wins_overlap_and_divergence_is_reported(tmp_path):
    cfg = _make_dbs(tmp_path, forecast_fn=lambda a: a)
    # push a diverged copy of every sidecar row into the replica
    scon = sqlite3.connect(cfg.sidecar_db)
    rows = scon.execute("""SELECT country_code, forecast_type, target_timestamp_utc,
        generated_at, horizon_hours, forecast_value, model_name, model_version
        FROM forecasts""").fetchall()
    scon.close()
    rcon = sqlite3.connect(cfg.replica_db)
    rcon.executemany("""INSERT INTO forecasts (country_code, forecast_type,
        target_timestamp_utc, generated_at, horizon_hours, forecast_value,
        model_name, model_version) VALUES (?,?,?,?,?,?,?,?)""",
        [(r[0], r[1], r[2], r[3], r[4], r[5] + 7.0, r[6], r[7]) for r in rows])
    rcon.commit(); rcon.close()

    res = evaluate(cfg)
    meta = res["meta"]
    assert meta["sidecar_vs_pushed_max_abs_diff_mw"] == pytest.approx(7.0)
    # sidecar value won: a forecast==actual sidecar row scores MAE 0, not 7
    assert res["per_country"][COUNTRY]["mae_mw"] == pytest.approx(0.0, abs=1e-9)
