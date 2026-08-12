"""Golden D+1/D+2 feature-vector tests for the shared wind feature builder
(ABL-183, `src/wind_features.py`). Solar's equivalent golden tests
(ABL-191) are in `tests/test_solar_features.py`, since this module now
serves both — the lag/rolling/calendar logic under test here is identical
for either forecast_type.

These pin *meaning*, not shape: which `as_of` each lag and rolling stat
resolves against, and that weather never reaches past its publication
cutoff. Feature order was already 24/24 correct against the frozen wind
artifacts while the underlying semantics were wrong (ABL-179) — a
shape-only test would have passed throughout that bug. Every actuals row
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
    POINT_LAG_DAYS,
    ROLLING_WINDOWS_HOURS,
    RenewableFeatureBuilder,
    ServeFaithfulnessError,
    SUPPORTED_FORECAST_TYPES,
    to_vector,
)

COUNTRY = "XX"
FORECAST_TYPE = "wind_offshore"
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
        CREATE TABLE energy_generation (country_code TEXT, timestamp_utc TIMESTAMP,
            wind_offshore_mw REAL, wind_onshore_mw REAL,
            data_quality TEXT DEFAULT 'actual');
        CREATE TABLE energy_renewable (country_code TEXT, timestamp_utc TIMESTAMP,
            wind_offshore_mw REAL DEFAULT 0, wind_onshore_mw REAL DEFAULT 0,
            data_quality TEXT DEFAULT 'actual');
        CREATE TABLE weather_data (country_code TEXT, timestamp_utc TIMESTAMP,
            forecast_run_time TIMESTAMP, data_quality TEXT,
            temperature_2m_k REAL, wind_speed_10m_ms REAL, wind_speed_100m_ms REAL,
            shortwave_radiation_wm2 REAL, direct_radiation_wm2 REAL, diffuse_radiation_wm2 REAL);
        """
    )

    # Actuals: hourly from well before the lookback window through three days
    # *past* OBS. Rows after OBS are poisoned — the builder must never read
    # them (the dashboard-wide "never extrapolate" invariant, served here as
    # "never look past observation_as_of").
    for ts in pd.date_range(OBS - pd.Timedelta(days=40), OBS + pd.Timedelta(days=3), freq="h"):
        value = POISON if ts > OBS else _epoch_hours(ts)
        con.execute(
            "INSERT INTO energy_generation VALUES (?, ?, ?, ?, 'actual')",
            (COUNTRY, str(ts), value, value),
        )
        con.execute(
            "INSERT INTO energy_renewable VALUES (?, ?, ?, ?, 'actual')",
            (COUNTRY, str(ts), value, value),
        )

    # Weather: two forecast runs per target hour. `early_run` is before every
    # publication cutoff used below and must be selected; `late_run` is after
    # OBS, poisoned, and must never be selected when weather_publication_as_of
    # <= OBS. Radiation columns are irrelevant to wind but must be present --
    # _WEATHER_RAW_COLUMNS fetches them for every forecast_type since ABL-191.
    early_run = OBS - pd.Timedelta(hours=6)
    late_run = OBS + pd.Timedelta(hours=6)
    for ts in pd.date_range(OBS - pd.Timedelta(days=1), OBS + pd.Timedelta(days=3), freq="h"):
        temp_k = 280.0 + ts.hour
        wind10 = 5.0 + ts.hour * 0.1
        wind100 = 8.0 + ts.hour * 0.1
        con.execute(
            "INSERT INTO weather_data VALUES (?, ?, ?, 'forecast', ?, ?, ?, 0, 0, 0)",
            (COUNTRY, str(ts), str(early_run), temp_k, wind10, wind100),
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


def _builder(forecast_type=FORECAST_TYPE):
    span_start = OBS - pd.Timedelta(days=45)
    span_end = OBS + pd.Timedelta(days=3)
    return RenewableFeatureBuilder(COUNTRY, forecast_type, span_start, span_end)


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
    *after* OBS (08:00) — not yet observed. The builder must fall back to
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
    never be a true one-day lag at D+2 — the diagnosis's central claim."""
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


def test_a_lag_that_would_reach_the_future_raises_rather_than_degrading(replica):
    """lag_7d/lag_14d have no fallback (see POINT_LAG_DAYS vs STRICT_LAG_DAYS
    in wind_features.py) — reaching the future for them is a bug, not an
    expected D+1/D+2 condition, so it must raise, not silently serve a wrong
    value."""
    target = OBS + pd.Timedelta(days=8)  # target-7d is one day *after* OBS
    with pytest.raises(ServeFaithfulnessError):
        _builder().row(target, OBS, OBS)


# --- rolling windows: anchored at observation_as_of, not the target --------


def test_rolling_windows_are_anchored_at_observation_as_of(replica):
    """Both windows end at OBS (inclusive), not at the target hour — the
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
    the same instant — D+1 hour 3, D+1 hour 22, and D+2 hour 10 must all
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
    marker must never appear — the leakage check the poisoned future rows
    exist for."""
    for target in (pd.Timestamp("2026-08-12 05:00:00"), pd.Timestamp("2026-08-13 20:00:00")):
        row = _builder().row(target, OBS, OBS)
        for name, feat in row.items():
            assert feat.value != POISON, f"{name} leaked a post-observation actual"
            if ("roll" in name or "lag" in name) and feat.source_timestamp is not None:
                assert feat.source_timestamp <= OBS, f"{name} source is after observation_as_of"


# --- weather: resolved at the target hour, bounded by publication cutoff ---


def test_weather_resolves_the_latest_run_at_or_before_the_publication_cutoff(replica):
    target = pd.Timestamp("2026-08-12 05:00:00")
    early_run = OBS - pd.Timedelta(hours=6)
    row = _builder().row(target, OBS, weather_publication_as_of=OBS)
    wind100 = row["wind_speed_100m_ms"]
    assert wind100.published_at == early_run
    assert wind100.value == pytest.approx(8.0 + target.hour * 0.1)
    assert wind100.value != POISON


def test_a_tighter_weather_cutoff_than_observation_as_of_is_honoured(replica):
    """weather_publication_as_of is a distinct parameter from observation_as_of
    — a scheduler that generated late but still wants the weather run that
    was live at the nominal schedule time must be able to pin it separately."""
    target = pd.Timestamp("2026-08-12 05:00:00")
    tight_cutoff = OBS - pd.Timedelta(hours=12)  # earlier than the only run before OBS
    row = _builder().row(target, OBS, weather_publication_as_of=tight_cutoff)
    wind100 = row["wind_speed_100m_ms"]
    assert wind100.published_at is None
    assert np.isnan(wind100.value)


def test_temperature_c_is_always_populated_from_the_same_weather_row(replica):
    """Diagnosis finding #3: wind's weather allow-list has no temperature
    column, so serving never overrode it. This builder resolves temperature_c
    unconditionally, from the same row as the allow-listed wind columns."""
    target = pd.Timestamp("2026-08-12 05:00:00")
    row = _builder().row(target, OBS, OBS)
    early_run = OBS - pd.Timedelta(hours=6)
    temp = row["temperature_c"]
    assert temp.published_at == early_run
    assert temp.value == pytest.approx((280.0 + target.hour) - 273.15)


def test_weather_publication_as_of_defaults_to_observation_as_of(replica):
    """FeatureRequest.build defaults weather_publication_as_of to
    observation_as_of when not given explicitly."""
    target = pd.Timestamp("2026-08-12 05:00:00")
    explicit = _builder().row(target, OBS, OBS)
    defaulted = _builder().row(target, OBS)
    assert defaulted["wind_speed_100m_ms"].value == pytest.approx(explicit["wind_speed_100m_ms"].value)
    assert defaulted["wind_speed_100m_ms"].published_at == explicit["wind_speed_100m_ms"].published_at


# --- calendar features: pure functions of the target hour ------------------


def test_calendar_features_describe_the_target_hour_not_the_generation_hour(replica):
    target = pd.Timestamp("2026-08-13 15:00:00")  # a Thursday
    row = _builder().row(target, OBS, OBS)
    assert row["hour"].value == 15
    assert row["day_of_week"].value == target.dayofweek
    assert row["is_weekend"].value == 0
    assert row["hour"].source_timestamp == target


# --- contract: to_vector / SUPPORTED_FORECAST_TYPES / artifact shape -------


#: The exact 24 names, in the exact order, read from the real frozen
#: artifacts (models/BE/wind_offshore, models/FR/wind_offshore,
#: models/BE|DE|FR/wind_onshore, models/AT/wind_onshore), 2026-08-11.
REAL_ARTIFACT_FEATURE_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "month_sin", "month_cos",
    "target_value_lag_1d", "target_value_lag_7d", "target_value_lag_14d",
    "target_value_roll_24h_mean", "target_value_roll_24h_std",
    "target_value_roll_24h_min", "target_value_roll_24h_max",
    "target_value_roll_168h_mean", "target_value_roll_168h_std",
    "target_value_roll_168h_min", "target_value_roll_168h_max",
    "wind_speed_100m_ms", "wind_speed_10m_ms", "temperature_c",
]


def test_to_vector_produces_exactly_the_real_artifacts_24_columns_in_order(replica):
    row = _builder().row(pd.Timestamp("2026-08-12 05:00:00"), OBS, OBS)
    vector = to_vector(row, REAL_ARTIFACT_FEATURE_COLUMNS)
    assert list(vector.keys()) == REAL_ARTIFACT_FEATURE_COLUMNS
    assert all(isinstance(v, float) for v in vector.values())


def test_to_vector_raises_for_a_column_the_builder_cannot_build(replica):
    row = _builder().row(pd.Timestamp("2026-08-12 05:00:00"), OBS, OBS)
    with pytest.raises(KeyError):
        to_vector(row, ["is_holiday"])


def test_supported_forecast_types_is_wind_and_solar():
    """ABL-183 scope was the two wind types ABL-179 diagnosed; ABL-191 adds
    solar, per ABL-185's diagnosis. See wind_features.py's
    SUPPORTED_FORECAST_TYPES docstring for why hydro_total/biomass/renewable
    are not included yet."""
    assert set(SUPPORTED_FORECAST_TYPES) == {"wind_onshore", "wind_offshore", "solar"}


def test_builder_refuses_an_unsupported_forecast_type(replica):
    with pytest.raises(ValueError):
        RenewableFeatureBuilder(COUNTRY, "hydro_total", OBS - pd.Timedelta(days=1), OBS)
