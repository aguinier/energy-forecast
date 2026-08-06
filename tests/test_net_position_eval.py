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
    EvalConfig, as_of_for_vintage, baseline_predictions, decompose_error,
    evaluate, point_metrics, promotion_gate, render_markdown,
)

COUNTRY = "AA"
NO_ACTUALS_COUNTRY = "BB"   # forecasts exist, actuals never published


def _make_dbs(tmp_path, forecast_fn, quantile_fn=None, actual_fn=None,
              vintage_days=("2026-07-28", "2026-07-29")):
    """Build replica + sidecar with the production column layout.

    forecast_fn(actual) -> forecast value; actual_fn(ts) -> actual value;
    quantile_fn(forecast, q) -> stored quantile value (None = no quantiles).
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
    hours = pd.date_range("2026-07-01", "2026-07-31 23:00", freq="h")
    rcon.executemany(
        "INSERT INTO net_position (country_code, timestamp_utc, net_position_mw) VALUES (?,?,?)",
        [(COUNTRY, str(ts), actual_fn(ts)) for ts in hours])

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
        for cc in (COUNTRY, NO_ACTUALS_COUNTRY):
            for ts in targets:
                a = actual_fn(ts)
                f = forecast_fn(a)
                horizon = int((ts - gen).total_seconds() // 3600)
                row = (cc, "net_position", str(ts), str(gen), horizon, f,
                       "chronos-2-V010", "test")
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
                             quantile_fn(f, q, ts), "chronos-2-V010"))
    scon.commit(); scon.close()
    rcon.commit(); rcon.close()
    return EvalConfig(replica_db=str(replica), sidecar_db=str(sidecar))


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

def _gate_fixture(tmp_path, forecast_fn, quantile_fn, noisy=True):
    rng = np.random.default_rng(3)
    noise = {  # reproducible per-timestamp noise so persistence is imperfect
        ts: float(rng.normal(0, 150))
        for ts in pd.date_range("2026-07-01", "2026-07-31 23:00", freq="h")}
    actual_fn = (lambda ts: 300.0 + 200.0 * np.sin(2 * np.pi * ts.hour / 24)
                 + noise[ts]) if noisy else None
    return _make_dbs(tmp_path, forecast_fn=forecast_fn, quantile_fn=quantile_fn,
                     actual_fn=actual_fn)


def test_gate_passes_a_calibrated_forecast(tmp_path):
    # forecast == actual; 10-90 band drawn so coverage lands at 75-85%:
    # hours 0-4 fall outside the band, 19 of 24 inside -> 79.2%
    def quantile_fn(f, q, ts):
        if q == 0.5:
            return f
        wide = ts.hour >= 5
        if q == 0.1:
            return f - (500 if wide else -1)
        return f + (500 if wide else -0.5)
    cfg = _gate_fixture(tmp_path, forecast_fn=lambda a: a, quantile_fn=quantile_fn)
    ref = {"V010": {COUNTRY: {"net_position": {"W01": {"mae": 500.0}}}}}
    for name in ("ref.json", "cand.json"):
        (tmp_path / name).write_text(json.dumps(ref))
    cfg.reference_backtest = str(tmp_path / "ref.json")
    cfg.candidate_backtest = str(tmp_path / "cand.json")
    cfg.serve_faithful_verified = True
    res = evaluate(cfg)
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
