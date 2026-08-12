"""Golden D+1/D+2 feature-vector tests for solar via the shared renewable
feature builder (ABL-191, extending ABL-183's `src/wind_features.py` to
solar per ABL-185's diagnosis).

These pin *meaning*, not shape, same standard as `test_wind_features.py`:
which `as_of` each lag and rolling stat resolves against, that weather never
reaches past its publication cutoff, and that solar's actuals inherit the
ABL-188 constant-run exclusion with no solar-specific code. The lag/rolling/
calendar logic under test is the same code path wind's tests already cover;
this file exists to prove it holds for a second `forecast_type` rather than
assuming it, and to pin solar's own weather columns (radiation, not wind
speed) and its frozen artifacts' exact feature order. Every actuals row
after `OBS` here is poisoned (`POISON`) so a leak shows up as a wrong value,
not just a wrong index.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.wind_features import (
    ROLLING_WINDOWS_HOURS,
    RenewableFeatureBuilder,
    SUPPORTED_FORECAST_TYPES,
    to_vector,
)

COUNTRY = "XX"
FORECAST_TYPE = "solar"
OBS = pd.Timestamp("2026-08-11 08:00:00")  # the generation instant
POISON = -99999.0                            # must never surface in a feature


def _epoch_hours(ts: pd.Timestamp) -> float:
    """Deterministic, strictly-increasing-by-hour value so lag/rolling
    lookups can be checked against a formula instead of a hand-built table."""
    return (ts - pd.Timestamp("2000-01-01")).total_seconds() / 3600.0


@pytest.fixture
def replica(tmp_path, monkeypatch):
    path = tmp_path / "replica.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE energy_renewable (country_code TEXT, timestamp_utc TIMESTAMP,
            solar_mw REAL);
        CREATE TABLE weather_data (country_code TEXT, timestamp_utc TIMESTAMP,
            forecast_run_time TIMESTAMP, data_quality TEXT,
            temperature_2m_k REAL, wind_speed_10m_ms REAL, wind_speed_100m_ms REAL,
            shortwave_radiation_wm2 REAL, direct_radiation_wm2 REAL, diffuse_radiation_wm2 REAL);
        """
    )

    # Actuals: hourly from well before the lookback window through three days
    # *past* OBS. Rows after OBS are poisoned -- the builder must never read
    # them (the dashboard-wide "never extrapolate" invariant, served here as
    # "never look past observation_as_of").
    for ts in pd.date_range(OBS - pd.Timedelta(days=40), OBS + pd.Timedelta(days=3), freq="h"):
        value = POISON if ts > OBS else _epoch_hours(ts)
        con.execute(
            "INSERT INTO energy_renewable VALUES (?, ?, ?)",
            (COUNTRY, str(ts), value),
        )

    # Weather: two forecast runs per target hour. `early_run` is before every
    # publication cutoff used below and must be selected; `late_run` is after
    # OBS, poisoned, and must never be selected when weather_publication_as_of
    # <= OBS. Wind-speed columns are irrelevant to solar but must be present
    # -- _WEATHER_RAW_COLUMNS fetches them for every forecast_type.
    early_run = OBS - pd.Timedelta(hours=6)
    late_run = OBS + pd.Timedelta(hours=6)
    for ts in pd.date_range(OBS - pd.Timedelta(days=1), OBS + pd.Timedelta(days=3), freq="h"):
        temp_k = 290.0 + ts.hour
        shortwave = 100.0 + ts.hour * 2.0
        direct = 50.0 + ts.hour
        diffuse = 20.0 + ts.hour * 0.5
        con.execute(
            "INSERT INTO weather_data VALUES (?, ?, ?, 'forecast', ?, 0, 0, ?, ?, ?)",
            (COUNTRY, str(ts), str(early_run), temp_k, shortwave, direct, diffuse),
        )
        con.execute(
            "INSERT INTO weather_data VALUES (?, ?, ?, 'forecast', ?, ?, ?, ?, ?, ?)",
            (COUNTRY, str(ts), str(late_run), POISON, POISON, POISON, POISON, POISON, POISON),
        )
    con.commit()
    con.close()

    monkeypatch.setenv("ENERGY_DB_PATH", str(path))
    import importlib

    importlib.reload(config)
    return path


def _builder():
    span_start = OBS - pd.Timedelta(days=45)
    span_end = OBS + pd.Timedelta(days=3)
    return RenewableFeatureBuilder(COUNTRY, FORECAST_TYPE, span_start, span_end)


# --- point lags: which as_of each lag resolves against ----------------------


def test_d1_early_hour_lag_1d_is_the_true_one_day_lag(replica):
    """D+1, hour 5 (horizon 21h): target-1d is already in the past relative
    to OBS, so lag_1d must be the true D-1 value, not a fallback."""
    target = pd.Timestamp("2026-08-12 05:00:00")
    row = _builder().row(target, OBS, OBS)
    lag1 = row["target_value_lag_1d"]
    assert lag1.source_timestamp == target - pd.Timedelta(days=1)
    assert lag1.value == pytest.approx(_epoch_hours(target - pd.Timedelta(days=1)))
    assert lag1.degraded is False


def test_d1_late_hour_lag_1d_degrades_to_the_nearest_reachable_same_hour(replica):
    """D+1, hour 15 (horizon 31h): target-1d is 2026-08-11 15:00, seven hours
    *after* OBS (08:00) -- not yet observed. The builder must fall back to
    target-2d rather than serve an unobserved value, and must say so."""
    target = pd.Timestamp("2026-08-12 15:00:00")
    assert target - pd.Timedelta(days=1) > OBS  # the condition this test exists for
    row = _builder().row(target, OBS, OBS)
    lag1 = row["target_value_lag_1d"]
    assert lag1.source_timestamp == target - pd.Timedelta(days=2)
    assert lag1.value == pytest.approx(_epoch_hours(target - pd.Timedelta(days=2)))
    assert lag1.degraded is True


def test_d2_lag_1d_is_always_degraded(replica):
    """D+2 horizons (40-63h at this OBS) always exceed 24h, so lag_1d can
    never be a true one-day lag at D+2 -- the diagnosis's central claim,
    confirmed here for solar rather than assumed from wind."""
    for hour in (0, 12, 23):
        target = pd.Timestamp(f"2026-08-13 {hour:02d}:00:00")
        row = _builder().row(target, OBS, OBS)
        lag1 = row["target_value_lag_1d"]
        assert lag1.degraded is True
        assert lag1.source_timestamp < target - pd.Timedelta(days=1)
        assert lag1.value == pytest.approx(_epoch_hours(lag1.source_timestamp))


def test_lag_7d_and_lag_14d_are_true_lags_at_both_horizons_never_degraded(replica):
    """7 and 14 days exceed any stored forecast horizon (max ~64h), so these
    never hit the D+2 problem lag_1d does."""
    for target in (pd.Timestamp("2026-08-12 09:00:00"), pd.Timestamp("2026-08-13 20:00:00")):
        row = _builder().row(target, OBS, OBS)
        for days in (7, 14):
            feat = row[f"target_value_lag_{days}d"]
            assert feat.degraded is False
            assert feat.source_timestamp == target - pd.Timedelta(days=days)
            assert feat.value == pytest.approx(_epoch_hours(target - pd.Timedelta(days=days)))


# --- rolling windows: anchored at observation_as_of, not the target --------


def test_rolling_windows_are_anchored_at_observation_as_of(replica):
    """Both windows end at OBS (inclusive), not at the target hour -- the
    fix for the same "always in the future at D+1/D+2" problem lag_1d has."""
    target = pd.Timestamp("2026-08-12 05:00:00")
    row = _builder().row(target, OBS, OBS)
    anchor = OBS.floor("h")
    for hours in ROLLING_WINDOWS_HOURS:
        mean_feat = row[f"target_value_roll_{hours}h_mean"]
        assert mean_feat.source_timestamp == anchor
        window = [_epoch_hours(anchor - pd.Timedelta(hours=h)) for h in range(hours)]
        assert mean_feat.value == pytest.approx(np.mean(window))
        assert row[f"target_value_roll_{hours}h_min"].value == pytest.approx(min(window))
        assert row[f"target_value_roll_{hours}h_max"].value == pytest.approx(max(window))


def test_rolling_windows_are_identical_across_every_target_hour_of_one_run(replica):
    """A single generation run anchors every target hour's rolling stats to
    the same instant -- D+1 hour 3, D+1 hour 22, and D+2 hour 10 must all
    report the *same* rolling values, because the run happened once."""
    builder = _builder()
    targets = [
        pd.Timestamp("2026-08-12 03:00:00"),
        pd.Timestamp("2026-08-12 22:00:00"),
        pd.Timestamp("2026-08-13 10:00:00"),
    ]
    rows = [builder.row(t, OBS, OBS) for t in targets]
    for hours in ROLLING_WINDOWS_HOURS:
        for stat in ("mean", "std", "min", "max"):
            key = f"target_value_roll_{hours}h_{stat}"
            values = {r[key].value for r in rows}
            assert len(values) == 1, f"{key} differed across target hours of one run: {values}"


def test_no_actuals_after_observation_as_of_ever_reach_a_feature(replica):
    """Every FeatureValue's source_timestamp must be <= OBS, and the poison
    marker must never appear -- the leakage check the poisoned future rows
    exist for."""
    for target in (pd.Timestamp("2026-08-12 05:00:00"), pd.Timestamp("2026-08-13 20:00:00")):
        row = _builder().row(target, OBS, OBS)
        for name, feat in row.items():
            assert feat.value != POISON, f"{name} leaked a post-observation actual"
            if ("roll" in name or "lag" in name) and feat.source_timestamp is not None:
                assert feat.source_timestamp <= OBS, f"{name} source is after observation_as_of"


# --- weather: resolved at the target hour, bounded by publication cutoff ---


def test_weather_resolves_the_latest_radiation_run_at_or_before_the_publication_cutoff(replica):
    """Solar's weather allow-list is the radiation trio, not wind speed --
    confirm the same publication-cutoff selection ABL-183 built applies to
    it: the newest forecast run at or before the cutoff wins."""
    target = pd.Timestamp("2026-08-12 05:00:00")
    early_run = OBS - pd.Timedelta(hours=6)
    row = _builder().row(target, OBS, weather_publication_as_of=OBS)
    for col, formula in (
        ("shortwave_radiation_wm2", lambda h: 100.0 + h * 2.0),
        ("direct_radiation_wm2", lambda h: 50.0 + h),
        ("diffuse_radiation_wm2", lambda h: 20.0 + h * 0.5),
    ):
        feat = row[col]
        assert feat.published_at == early_run
        assert feat.value == pytest.approx(formula(target.hour))
        assert feat.value != POISON


def test_a_tighter_weather_cutoff_than_observation_as_of_is_honoured(replica):
    """weather_publication_as_of is a distinct parameter from observation_as_of
    -- a scheduler that generated late but still wants the weather run that
    was live at the nominal schedule time must be able to pin it separately."""
    target = pd.Timestamp("2026-08-12 05:00:00")
    tight_cutoff = OBS - pd.Timedelta(hours=12)  # earlier than the only run before OBS
    row = _builder().row(target, OBS, weather_publication_as_of=tight_cutoff)
    shortwave = row["shortwave_radiation_wm2"]
    assert shortwave.published_at is None
    assert np.isnan(shortwave.value)


def test_temperature_c_is_always_populated_from_the_same_weather_row(replica):
    """The issue's open question: does solar's generic temperature_c
    inference still hold under the shared builder? `_weather_features`
    resolves temperature_c unconditionally for every forecast_type, not
    gated by config.WEATHER_FEATURES (which has no temperature_2m_k entry
    for solar, same as wind) -- confirmed here rather than assumed."""
    target = pd.Timestamp("2026-08-12 05:00:00")
    row = _builder().row(target, OBS, OBS)
    early_run = OBS - pd.Timedelta(hours=6)
    temp = row["temperature_c"]
    assert temp.published_at == early_run
    assert temp.value == pytest.approx((290.0 + target.hour) - 273.15)


def test_weather_publication_as_of_defaults_to_observation_as_of(replica):
    """FeatureRequest.build defaults weather_publication_as_of to
    observation_as_of when not given explicitly."""
    target = pd.Timestamp("2026-08-12 05:00:00")
    explicit = _builder().row(target, OBS, OBS)
    defaulted = _builder().row(target, OBS)
    assert defaulted["shortwave_radiation_wm2"].value == pytest.approx(explicit["shortwave_radiation_wm2"].value)
    assert defaulted["shortwave_radiation_wm2"].published_at == explicit["shortwave_radiation_wm2"].published_at


# --- ABL-188: the constant-run invariant is inherited, not reimplemented ---


@pytest.fixture
def constant_run_replica(tmp_path, monkeypatch):
    """A dedicated, minimal fixture isolating one suspect constant run inside
    the lookback window a lag_7d lookup will hit -- proves the DE-solar-zero
    invariant (`exclude_suspect_constant_runs`, ABL-188) reaches solar
    through this builder with no solar-specific code, since `_load_actuals_series`
    calls the same `load_renewable_type_data` wind already goes through."""
    path = tmp_path / "replica.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE energy_renewable (country_code TEXT, timestamp_utc TIMESTAMP, solar_mw REAL);"
        "CREATE TABLE weather_data (country_code TEXT, timestamp_utc TIMESTAMP, "
        "forecast_run_time TIMESTAMP, data_quality TEXT, temperature_2m_k REAL, "
        "wind_speed_10m_ms REAL, wind_speed_100m_ms REAL, shortwave_radiation_wm2 REAL, "
        "direct_radiation_wm2 REAL, diffuse_radiation_wm2 REAL);"
    )

    target = pd.Timestamp("2026-08-12 05:00:00")
    lag7_source = target - pd.Timedelta(days=7)
    run_start = lag7_source - pd.Timedelta(hours=15)
    run_end = lag7_source + pd.Timedelta(hours=15)  # 30h span, over the 24h min

    for ts in pd.date_range(OBS - pd.Timedelta(days=40), OBS, freq="h"):
        value = 0.0 if run_start <= ts <= run_end else _epoch_hours(ts)
        con.execute("INSERT INTO energy_renewable VALUES (?, ?, ?)", (COUNTRY, str(ts), value))
    con.commit()
    con.close()

    monkeypatch.setenv("ENERGY_DB_PATH", str(path))
    import importlib

    importlib.reload(config)
    return {"target": target, "lag7_source": lag7_source}


def test_a_suspect_constant_actuals_run_is_excluded_from_lags_and_rolling(constant_run_replica):
    target = constant_run_replica["target"]
    lag7_source = constant_run_replica["lag7_source"]
    row = _builder().row(target, OBS, OBS)
    lag7 = row["target_value_lag_7d"]
    assert lag7.source_timestamp == lag7_source
    # Not 0.0 -- ABL-188: a bit-identical 30h run is unadjudicated-missing,
    # not a measured value, so the invariant nulls it before it reaches a lag.
    assert np.isnan(lag7.value)


# --- contract: to_vector / SUPPORTED_FORECAST_TYPES / artifact shape -------


#: The exact 24 names, in the exact order, read from the real frozen
#: artifacts (models/AT|BE|DE|FR/solar/model.joblib), 2026-08-11. Same shape
#: as wind's 24 (10 calendar + 3 lags + 8 rolling + temperature_c) with the
#: two wind-speed columns swapped for solar's three radiation columns.
REAL_ARTIFACT_FEATURE_COLUMNS = [
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


def test_to_vector_produces_exactly_the_real_artifacts_24_columns_in_order(replica):
    row = _builder().row(pd.Timestamp("2026-08-12 05:00:00"), OBS, OBS)
    vector = to_vector(row, REAL_ARTIFACT_FEATURE_COLUMNS)
    assert list(vector.keys()) == REAL_ARTIFACT_FEATURE_COLUMNS
    assert all(isinstance(v, float) for v in vector.values())


def test_solar_is_in_supported_forecast_types():
    assert "solar" in SUPPORTED_FORECAST_TYPES
