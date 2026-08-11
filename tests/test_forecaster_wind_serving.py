"""ABL-183: `Forecaster.predict_d2` wired to the shared wind feature builder,
extended to solar by ABL-191, and the xgboost intercept witness extended to
the legacy `Forecaster.save`/`load` artifacts (ABL-69's guard, applied to a
second caller).

The predict_d2 tests check the *feature vector actually handed to the
model*, not just that a forecast DataFrame comes back — the proxy-row bug
this replaces produced a plausible-shaped, wrong vector, so shape alone
proves nothing.
"""
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.forecaster import Forecaster
from src import xgboost_artifact_guard
from src.wind_features import RenewableFeatureBuilder, to_vector

COUNTRY = "XX"
OBS = pd.Timestamp("2026-08-11 08:00:00")

FEATURE_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "month_sin", "month_cos",
    "target_value_lag_1d", "target_value_lag_7d", "target_value_lag_14d",
    "target_value_roll_24h_mean", "target_value_roll_24h_std",
    "target_value_roll_24h_min", "target_value_roll_24h_max",
    "target_value_roll_168h_mean", "target_value_roll_168h_std",
    "target_value_roll_168h_min", "target_value_roll_168h_max",
    "wind_speed_100m_ms", "wind_speed_10m_ms", "temperature_c",
]

SOLAR_FEATURE_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "month_sin", "month_cos",
    "target_value_lag_1d", "target_value_lag_7d", "target_value_lag_14d",
    "target_value_roll_24h_mean", "target_value_roll_24h_std",
    "target_value_roll_24h_min", "target_value_roll_24h_max",
    "target_value_roll_168h_mean", "target_value_roll_168h_std",
    "target_value_roll_168h_min", "target_value_roll_168h_max",
    "shortwave_radiation_wm2", "direct_radiation_wm2", "diffuse_radiation_wm2",
    "temperature_c",
]


def _epoch_hours(ts: pd.Timestamp) -> float:
    return (ts - pd.Timestamp("2000-01-01")).total_seconds() / 3600.0


@pytest.fixture
def replica(tmp_path, monkeypatch):
    """Deliberately has no `energy_generation`/legacy training table — if
    predict_d2 ever fell back to the old proxy-row path for a wind type, it
    would raise on a missing table instead of silently passing."""
    path = tmp_path / "replica.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE energy_renewable (country_code TEXT, timestamp_utc TIMESTAMP,
            wind_offshore_mw REAL, wind_onshore_mw REAL, solar_mw REAL);
        CREATE TABLE weather_data (country_code TEXT, timestamp_utc TIMESTAMP,
            forecast_run_time TIMESTAMP, data_quality TEXT,
            temperature_2m_k REAL, wind_speed_10m_ms REAL, wind_speed_100m_ms REAL,
            shortwave_radiation_wm2 REAL, direct_radiation_wm2 REAL, diffuse_radiation_wm2 REAL);
        """
    )
    for ts in pd.date_range(OBS - pd.Timedelta(days=40), OBS, freq="h"):
        value = _epoch_hours(ts)
        con.execute(
            "INSERT INTO energy_renewable VALUES (?, ?, ?, ?, ?)",
            (COUNTRY, str(ts), value, value, value),
        )
    run = OBS - pd.Timedelta(hours=6)
    for ts in pd.date_range(OBS - pd.Timedelta(days=1), OBS + pd.Timedelta(days=3), freq="h"):
        con.execute(
            "INSERT INTO weather_data VALUES (?, ?, ?, 'forecast', ?, ?, ?, ?, ?, ?)",
            (
                COUNTRY, str(ts), str(run),
                280.0 + ts.hour, 5.0 + ts.hour * 0.1, 8.0 + ts.hour * 0.1,
                100.0 + ts.hour * 2.0, 50.0 + ts.hour, 20.0 + ts.hour * 0.5,
            ),
        )
    con.commit()
    con.close()

    monkeypatch.setenv("ENERGY_DB_PATH", str(path))
    import importlib

    importlib.reload(config)
    return path


class _CapturingModel:
    """Stands in for a fitted booster: records the exact frame it was asked
    to predict on, so the test can compare it against the builder's own
    output rather than trusting predict_d2 not to have mangled it."""

    def __init__(self):
        self.calls = []

    def predict(self, X):
        self.calls.append(X.copy())
        return np.zeros(len(X))


def _forecaster(forecast_type):
    f = Forecaster(COUNTRY, forecast_type, algorithm="xgboost")
    f.model = _CapturingModel()
    f.feature_columns = list(FEATURE_COLUMNS)
    f.model_version = "test"
    return f


def test_predict_d2_wind_feeds_the_model_the_builders_own_vector(replica):
    forecaster = _forecaster("wind_offshore")
    forecaster.predict_d2(
        reference_date=OBS.date(), horizon_days=1, hours=[5],
        observation_as_of=OBS, weather_publication_as_of=OBS,
    )
    [captured] = forecaster.model.calls
    assert list(captured.columns) == FEATURE_COLUMNS

    target = pd.Timestamp("2026-08-12 05:00:00")
    builder = RenewableFeatureBuilder(COUNTRY, "wind_offshore", OBS - pd.Timedelta(days=45), OBS + pd.Timedelta(days=3))
    expected = to_vector(builder.row(target, OBS, OBS), FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        assert captured.iloc[0][col] == pytest.approx(expected[col]), col


def test_predict_d2_wind_horizon_hours_is_hours_from_observation_as_of(replica):
    forecaster = _forecaster("wind_offshore")
    df = forecaster.predict_d2(
        reference_date=OBS.date(), horizon_days=2, hours=[10],
        observation_as_of=OBS, weather_publication_as_of=OBS,
    )
    target = pd.Timestamp("2026-08-13 10:00:00")
    assert df.iloc[0]["horizon_hours"] == int((target - OBS).total_seconds() / 3600)
    assert df.iloc[0]["target_timestamp_utc"] == target.to_pydatetime()


def test_predict_d2_wind_d1_and_d2_share_one_rolling_anchor(replica):
    """Two separate predict_d2 calls (D+1, D+2) from the same generation
    instant must see identical rolling stats — same invariant as the
    single-call multi-hour case in test_wind_features.py, exercised here
    through the actual serving entrypoint."""
    forecaster = _forecaster("wind_offshore")
    forecaster.predict_d2(reference_date=OBS.date(), horizon_days=1, hours=[3],
                          observation_as_of=OBS, weather_publication_as_of=OBS)
    forecaster.predict_d2(reference_date=OBS.date(), horizon_days=2, hours=[3],
                          observation_as_of=OBS, weather_publication_as_of=OBS)
    d1_call, d2_call = forecaster.model.calls
    for col in FEATURE_COLUMNS:
        if "roll" in col:
            assert d1_call.iloc[0][col] == pytest.approx(d2_call.iloc[0][col]), col


def test_predict_d2_wind_onshore_is_also_wired(replica):
    forecaster = _forecaster("wind_onshore")
    df = forecaster.predict_d2(reference_date=OBS.date(), horizon_days=1, hours=[0],
                               observation_as_of=OBS, weather_publication_as_of=OBS)
    assert len(df) == 1
    assert not df["forecast_value"].isna().any()


def test_predict_d2_solar_feeds_the_model_the_builders_own_vector(replica):
    """ABL-191: solar goes through the same serve-faithful entrypoint as
    wind, with no solar-specific branch in Forecaster — this exercises the
    real `predict_d2` call, not just the builder directly."""
    forecaster = _forecaster("solar")
    forecaster.feature_columns = list(SOLAR_FEATURE_COLUMNS)
    forecaster.predict_d2(
        reference_date=OBS.date(), horizon_days=1, hours=[5],
        observation_as_of=OBS, weather_publication_as_of=OBS,
    )
    [captured] = forecaster.model.calls
    assert list(captured.columns) == SOLAR_FEATURE_COLUMNS

    target = pd.Timestamp("2026-08-12 05:00:00")
    builder = RenewableFeatureBuilder(COUNTRY, "solar", OBS - pd.Timedelta(days=45), OBS + pd.Timedelta(days=3))
    expected = to_vector(builder.row(target, OBS, OBS), SOLAR_FEATURE_COLUMNS)
    for col in SOLAR_FEATURE_COLUMNS:
        assert captured.iloc[0][col] == pytest.approx(expected[col]), col


def test_predict_d2_solar_is_also_wired(replica):
    forecaster = _forecaster("solar")
    forecaster.feature_columns = list(SOLAR_FEATURE_COLUMNS)
    df = forecaster.predict_d2(reference_date=OBS.date(), horizon_days=1, hours=[0],
                               observation_as_of=OBS, weather_publication_as_of=OBS)
    assert len(df) == 1
    assert not df["forecast_value"].isna().any()


# --- xgboost intercept witness, extended to Forecaster.save/load -----------


def _fit_tiny_xgboost():
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = 500.0 + np.arange(20, dtype=float)
    model = xgboost.XGBRegressor(n_estimators=3, max_depth=2)
    model.fit(X, y)
    return model


def test_save_writes_a_real_intercept_witness_for_xgboost(tmp_path):
    forecaster = Forecaster(COUNTRY, "wind_offshore", algorithm="xgboost")
    forecaster.model = _fit_tiny_xgboost()
    forecaster.feature_columns = ["x"]
    forecaster.model_version = "v1"
    path = forecaster.save(str(tmp_path / "model.joblib"))

    blob = joblib.load(path)
    assert blob["xgboost_version"] == xgboost.__version__
    assert blob["base_score"] == pytest.approx(xgboost_artifact_guard.base_score(forecaster.model))
    assert blob["base_score"] is not None


def test_load_refuses_a_model_whose_intercept_did_not_survive(tmp_path):
    forecaster = Forecaster(COUNTRY, "wind_offshore", algorithm="xgboost")
    forecaster.model = _fit_tiny_xgboost()
    forecaster.feature_columns = ["x"]
    forecaster.model_version = "v1"
    path = forecaster.save(str(tmp_path / "model.joblib"))

    blob = joblib.load(path)
    blob["base_score"] = blob["base_score"] + 5000.0  # corruption ABL-69 measured: intercept resets far off
    joblib.dump(blob, path)

    with pytest.raises(xgboost_artifact_guard.ModelArtifactError) as exc:
        Forecaster.load(COUNTRY, "wind_offshore", path=path)
    assert "did not survive loading" in str(exc.value)


def test_load_tolerates_a_legacy_artifact_carrying_no_witness(tmp_path):
    """Mirrors the real BE/FR wind_offshore artifacts as measured 2026-08-11:
    saved before this guard existed, with neither base_score nor
    xgboost_version keys at all. Absent witness means 'cannot check', not
    'corrupt' — this must not force a retrain to keep serving them."""
    path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": _fit_tiny_xgboost(),
            "feature_columns": ["x"],
            "country_code": COUNTRY,
            "forecast_type": "wind_offshore",
            "model_version": "legacy",
            "training_metrics": {},
            "saved_at": "2025-01-01T00:00:00",
        },
        path,
    )
    restored = Forecaster.load(COUNTRY, "wind_offshore", path=path)
    assert restored.feature_columns == ["x"]


def test_save_skips_the_witness_for_a_non_xgboost_algorithm(tmp_path):
    from catboost import CatBoostRegressor

    forecaster = Forecaster(COUNTRY, "wind_onshore", algorithm="catboost")
    model = CatBoostRegressor(iterations=2, depth=2, verbose=False)
    model.fit(np.arange(20).reshape(-1, 1).astype(float), 500.0 + np.arange(20, dtype=float))
    forecaster.model = model
    forecaster.feature_columns = ["x"]
    forecaster.model_version = "v1"
    path = forecaster.save(str(tmp_path / "model.joblib"))

    blob = joblib.load(path)
    assert "base_score" not in blob
    assert "xgboost_version" not in blob


def test_an_intact_xgboost_artifact_round_trips(tmp_path):
    forecaster = Forecaster(COUNTRY, "wind_offshore", algorithm="xgboost")
    forecaster.model = _fit_tiny_xgboost()
    forecaster.feature_columns = ["x"]
    forecaster.model_version = "v1"
    path = forecaster.save(str(tmp_path / "model.joblib"))

    restored = Forecaster.load(COUNTRY, "wind_offshore", path=path)
    assert restored.feature_columns == ["x"]
    assert restored.algorithm == "xgboost"
