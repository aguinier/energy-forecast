"""The shadow rail refuses rather than improvises (ABL-68).

A challenger that quietly produces *something* when its inputs are missing is
worse than one that produces nothing: the row lands in the sidecar, the eval
scores it, and the report carries a number nobody can trace. These tests pin the
refusals, and pin that V012 never writes a zero it did not measure.
"""
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from src.challengers.registry import (CHALLENGERS, CHAMPION_MODEL_NAME,
                                      model_name_for, spec_for)

TARGET_DATE = "2026-08-09"
GENERATED_AT = pd.Timestamp("2026-08-07 06:00:00")


# --- registry ---------------------------------------------------------------

def test_unregistered_experiment_is_refused_with_a_reason():
    with pytest.raises(KeyError) as exc:
        spec_for("V999")
    assert "not a registered challenger" in str(exc.value)


def test_challenger_model_names_are_distinct_and_not_the_champion():
    names = [s.model_name for s in CHALLENGERS.values()]
    assert len(names) == len(set(names))
    assert CHAMPION_MODEL_NAME not in names, \
        "a challenger sharing the champion's model_name would be pushed to prod"


def test_champion_model_name_resolves_from_its_experiment_config():
    assert model_name_for("V010") == CHAMPION_MODEL_NAME


def test_v012_is_not_a_promotion_candidate():
    """It is the floor. Promoting the floor would be promoting the yardstick."""
    assert spec_for("V012").promotion_candidate is False
    assert spec_for("V016").promotion_candidate is True


# --- serving guards ---------------------------------------------------------

@pytest.fixture
def rail(monkeypatch, tmp_path):
    monkeypatch.setenv("ENERGY_DB_PATH", str(tmp_path / "replica.db"))
    monkeypatch.setenv("FORECAST_OUTPUT_DB", str(tmp_path / "sidecar.db"))
    import config
    importlib.reload(config)
    mod = importlib.import_module("forecast_challengers")
    return importlib.reload(mod)


def _sidecar_with_champion(path, values=None):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE forecasts (id INTEGER PRIMARY KEY AUTOINCREMENT,
        country_code TEXT, forecast_type TEXT, target_timestamp_utc TIMESTAMP,
        generated_at TIMESTAMP, horizon_hours INTEGER, forecast_value REAL,
        model_name TEXT, model_version TEXT)""")
    con.execute("""CREATE TABLE forecast_quantiles (id INTEGER PRIMARY KEY AUTOINCREMENT,
        country_code TEXT, forecast_type TEXT, target_timestamp_utc TIMESTAMP,
        generated_at TIMESTAMP, quantile REAL, forecast_value REAL, model_name TEXT)""")
    if values is not None:
        idx = pd.date_range(TARGET_DATE, periods=24, freq="h")
        con.executemany("""INSERT INTO forecasts (country_code, forecast_type,
            target_timestamp_utc, generated_at, horizon_hours, forecast_value,
            model_name, model_version) VALUES (?,?,?,?,?,?,?,?)""",
            [("FR", "net_position", str(ts), str(GENERATED_AT), 42, float(v),
              CHAMPION_MODEL_NAME, "v") for ts, v in zip(idx, values)])
        con.executemany("""INSERT INTO forecast_quantiles (country_code, forecast_type,
            target_timestamp_utc, generated_at, quantile, forecast_value, model_name)
            VALUES (?,?,?,?,?,?,?)""",
            [("FR", "net_position", str(ts), str(GENERATED_AT), q, float(v) + off,
              CHAMPION_MODEL_NAME)
             for q, off in ((0.1, -500.0), (0.5, 0.0), (0.9, 500.0))
             for ts, v in zip(idx, values)])
    con.commit(); con.close()


def test_v016_refuses_to_serve_without_a_fit(rail, tmp_path, monkeypatch):
    """Without coefficients V016 would just be V010 republished under a second
    name — two model rows carrying one model's numbers."""
    _sidecar_with_champion(tmp_path / "sidecar.db", values=np.full(24, 1000.0))
    monkeypatch.setattr(rail.config, "EXPERIMENTS_DIR", tmp_path / "no_experiments")
    rows, qrows = rail.run_v016(spec_for("V016"), ["FR"], TARGET_DATE, GENERATED_AT,
                                {}, sidecar_db=str(tmp_path / "sidecar.db"))
    assert rows == [] and qrows == []


def test_v016_refuses_when_the_champion_has_not_run(rail, tmp_path, monkeypatch):
    _sidecar_with_champion(tmp_path / "sidecar.db", values=None)
    exp = tmp_path / "experiments" / "V016"
    exp.mkdir(parents=True)
    (exp / "correction.json").write_text(json.dumps({"corrections": {}}))
    monkeypatch.setattr(rail.config, "EXPERIMENTS_DIR", tmp_path / "experiments")
    rows, _ = rail.run_v016(spec_for("V016"), ["FR"], TARGET_DATE, GENERATED_AT,
                            {}, sidecar_db=str(tmp_path / "sidecar.db"))
    assert rows == []


def test_v016_applies_the_fit_and_keeps_quantiles_ordered(rail, tmp_path, monkeypatch):
    champion = np.linspace(500.0, 1500.0, 24)
    _sidecar_with_champion(tmp_path / "sidecar.db", values=champion)
    exp = tmp_path / "experiments" / "V016"
    exp.mkdir(parents=True)
    (exp / "correction.json").write_text(json.dumps({"corrections": {"FR": {
        "country": "FR", "n_pairs": 5000, "n_target_days": 120,
        "intercept_mw": 50.0, "slope": 2.0, "ar1_phi": 0.0, "applied": True,
        "reason": "fitted"}}}))
    monkeypatch.setattr(rail.config, "EXPERIMENTS_DIR", tmp_path / "experiments")

    rows, qrows = rail.run_v016(spec_for("V016"), ["FR"], TARGET_DATE, GENERATED_AT,
                                {}, sidecar_db=str(tmp_path / "sidecar.db"))
    assert len(rows) == 24
    assert rows[0]["forecast_value"] == pytest.approx(50.0 + 2.0 * champion[0])
    assert rows[0]["model_name"] == "chronos-2-V016"

    q = pd.DataFrame(qrows).pivot_table(index="target_ts", columns="quantile",
                                        values="forecast_value")
    assert (q[0.1] < q[0.5]).all() and (q[0.5] < q[0.9]).all()
    # An affine map with slope 2 widens the band by exactly 2.
    assert (q[0.9] - q[0.1]).iloc[0] == pytest.approx(2.0 * 1000.0)


def test_v016_passthrough_country_reproduces_the_champion(rail, tmp_path, monkeypatch):
    champion = np.linspace(500.0, 1500.0, 24)
    _sidecar_with_champion(tmp_path / "sidecar.db", values=champion)
    exp = tmp_path / "experiments" / "V016"
    exp.mkdir(parents=True)
    (exp / "correction.json").write_text(json.dumps({"corrections": {"FR": {
        "country": "FR", "n_pairs": 22, "n_target_days": 1, "intercept_mw": 0.0,
        "slope": 1.0, "ar1_phi": 0.0, "applied": False,
        "reason": "insufficient fitting data"}}}))
    monkeypatch.setattr(rail.config, "EXPERIMENTS_DIR", tmp_path / "experiments")
    rows, _ = rail.run_v016(spec_for("V016"), ["FR"], TARGET_DATE, GENERATED_AT,
                            {}, sidecar_db=str(tmp_path / "sidecar.db"))
    assert np.allclose([r["forecast_value"] for r in rows], champion)


def test_v012_skips_a_country_with_no_actuals_instead_of_writing_zero(rail):
    rows, _ = rail.run_v012(spec_for("V012"), ["XX"], TARGET_DATE, GENERATED_AT, {})
    assert rows == []


def test_v012_writes_measured_values_for_a_country_with_history(rail):
    idx = pd.date_range(GENERATED_AT - pd.Timedelta(days=30), GENERATED_AT, freq="h")
    actuals = {"FR": pd.Series(2000.0, index=idx)}
    rows, qrows = rail.run_v012(spec_for("V012"), ["FR"], TARGET_DATE,
                                GENERATED_AT, actuals)
    assert len(rows) == 24
    assert all(r["forecast_value"] == pytest.approx(2000.0) for r in rows)
    assert all(r["model_name"] == "baseline-V012" for r in rows)
    # No quantiles: a point ensemble has no calibrated band, and V012 is not a
    # promotion candidate so the gate's coverage check does not apply to it.
    assert qrows == []
