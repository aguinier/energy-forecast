"""ABL-342: the gate harnesses must not be a second artifact writer.

`Forecaster.save` is the only writer that derives `training_source` (ABL-331)
and the xgboost intercept witness (ABL-183). Both gate harnesses
(`scripts/evaluate_wind_retrain.py`, `scripts/evaluate_solar_retrain.py`)
instead `joblib.dump`ed seven keys of their own, and `src/model_registry.py`
takes a caller's dict verbatim.

That is not a missing-metadata complaint. `Forecaster.load` reads every key with
`.get(..., default)` and resolves an **absent** `training_source` to
`LEGACY_RENEWABLE_TRAINING_SOURCE` ('energy_renewable') for any
`config.RENEWABLE_TYPES` artifact. So a pair fitted on `energy_generation` and
written in the bare shape loads with no error, no warning and no shape change,
and serves every lag and rolling feature from the table it was never fitted on.
The harnesses are what the remaining ABL-316 pairs are fitted by, and
`--artifact-dir` is a command-line argument.

As in `test_per_artifact_training_source.py`, the replica gives the two tables
**deliberately different values** for the same country and instant and the model
returns its own `lag_1d`, so a forecast value names the table it was read from.
A test that only compared the recorded string would pass just as happily if the
string never reached the serving features.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src import db
from src.evaluation.gate_artifacts import save_gate_artifact
from src.forecaster import Forecaster
from src.model_registry import ModelRegistry
from src.wind_features import RenewableFeatureBuilder
from src.xgboost_artifact_guard import ModelArtifactError

COUNTRY = "XX"
OBS = pd.Timestamp("2026-08-11 08:00:00")

#: `energy_renewable` carries the same series offset by this much, so any
#: feature derived from the target series differs between the two tables by a
#: margin no rounding can explain.
RENEWABLE_OFFSET = 1_000_000.0

_SHARED_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "month_sin", "month_cos",
    "target_value_lag_1d", "target_value_lag_7d", "target_value_lag_14d",
    "target_value_roll_24h_mean", "target_value_roll_24h_std",
    "target_value_roll_24h_min", "target_value_roll_24h_max",
    "target_value_roll_168h_mean", "target_value_roll_168h_std",
    "target_value_roll_168h_min", "target_value_roll_168h_max",
]
WIND_COLUMNS = _SHARED_COLUMNS + [
    "wind_speed_100m_ms", "wind_speed_10m_ms", "temperature_c",
]


def _epoch_hours(ts) -> float:
    return (ts - pd.Timestamp("2000-01-01")).total_seconds() / 3600.0


@pytest.fixture
def replica(tmp_path, monkeypatch):
    path = tmp_path / "replica.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE energy_generation (country_code TEXT, timestamp_utc TIMESTAMP,
            wind_offshore_mw REAL, wind_onshore_mw REAL, solar_mw REAL,
            hydro_run_mw REAL, hydro_reservoir_mw REAL, biomass_mw REAL,
            data_quality TEXT DEFAULT 'actual');
        CREATE TABLE energy_renewable (country_code TEXT, timestamp_utc TIMESTAMP,
            wind_offshore_mw REAL DEFAULT 0, wind_onshore_mw REAL DEFAULT 0,
            solar_mw REAL DEFAULT 0, hydro_run_mw REAL DEFAULT 0,
            hydro_reservoir_mw REAL DEFAULT 0, biomass_mw REAL DEFAULT 0,
            data_quality TEXT DEFAULT 'actual');
        CREATE TABLE weather_data (country_code TEXT, timestamp_utc TIMESTAMP,
            forecast_run_time TIMESTAMP, data_quality TEXT,
            temperature_2m_k REAL, wind_speed_10m_ms REAL, wind_speed_100m_ms REAL,
            shortwave_radiation_wm2 REAL, direct_radiation_wm2 REAL,
            diffuse_radiation_wm2 REAL);
        """
    )
    for ts in pd.date_range(OBS - pd.Timedelta(days=40), OBS, freq="h"):
        gen = _epoch_hours(ts)
        ren = gen + RENEWABLE_OFFSET
        con.execute(
            "INSERT INTO energy_generation VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'actual')",
            (COUNTRY, str(ts), gen, gen, gen, gen, gen, gen),
        )
        con.execute(
            "INSERT INTO energy_renewable VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'actual')",
            (COUNTRY, str(ts), ren, ren, ren, ren, ren, ren),
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
    importlib.reload(config)
    importlib.reload(db)
    yield path
    monkeypatch.undo()
    importlib.reload(config)
    importlib.reload(db)


class EchoLagModel:
    """Returns its own `target_value_lag_1d`, so the forecast value is a direct
    readout of which table fed the target series. Module-level (not a closure)
    because these get pickled into artifacts."""

    def predict(self, X):
        return X["target_value_lag_1d"].to_numpy(dtype=float)


def _builder(source=None, forecast_type="wind_onshore"):
    """The builder a harness constructs, over the span its fit covers."""
    return RenewableFeatureBuilder(
        COUNTRY, forecast_type, OBS - pd.Timedelta(days=30),
        OBS + pd.Timedelta(days=3), actuals_source=source,
    )


def _params():
    return {"n_estimators": 7, "depth": 3}


def _write(tmp_path, source=None, forecast_type="wind_onshore", model=None,
           algorithm="catboost", name="gate"):
    return save_gate_artifact(
        tmp_path / name / COUNTRY / forecast_type / "model.joblib",
        model=EchoLagModel() if model is None else model,
        builder=_builder(source, forecast_type),
        algorithm=algorithm,
        params=_params(),
        feature_columns=WIND_COLUMNS,
        fit_window=(pd.Timestamp("2026-01-14"), pd.Timestamp("2026-07-11")),
    )


def _served_value(path, forecast_type="wind_onshore") -> float:
    forecaster = Forecaster.load(COUNTRY, forecast_type, path=str(path))
    df = forecaster.predict_d2(
        reference_date=OBS.date(), horizon_days=1, hours=[5],
        observation_as_of=OBS, weather_publication_as_of=OBS,
    )
    return float(df["forecast_value"].iloc[0])


# ---------------------------------------------------------------------------
# The regression the issue asks for: fitted source in, fitted source served.
# ---------------------------------------------------------------------------


def test_a_gate_artifact_fitted_on_energy_generation_is_served_from_it(replica, tmp_path):
    """The headline. A harness pair fitted on `energy_generation` must not come
    back resolved to the legacy default, and the resolution has to reach the
    features — so this asserts a *value*, not the recorded string."""
    served = _served_value(_write(tmp_path, source="energy_generation"))

    assert db.LEGACY_RENEWABLE_TRAINING_SOURCE == "energy_renewable", "precondition"
    assert served < RENEWABLE_OFFSET, (
        f"a gate artifact fitted on `energy_generation` served {served}, an "
        "`energy_renewable` value. This is the ABL-342 skew: the artifact loads "
        "clean and serves lags from a table it was never fitted on."
    )


def test_the_bare_dump_the_harnesses_used_to_write_does_serve_the_wrong_table(replica, tmp_path):
    """Guards the test above: reconstruct the exact pre-ABL-342 shape from the
    same builder and show it lands on the other table. Without this, the test
    above would pass just as well if the two tables happened to agree here."""
    path = tmp_path / "bare" / "model.joblib"
    path.parent.mkdir(parents=True)
    joblib.dump({"model": EchoLagModel(), "feature_columns": list(WIND_COLUMNS),
                 "country_code": COUNTRY, "forecast_type": "wind_onshore",
                 "algorithm": "catboost", "params": _params(),
                 "fit_window": ["2026-01-14", "2026-07-11"]}, path)

    assert _served_value(path) > RENEWABLE_OFFSET, (
        "the pre-ABL-342 shape no longer reproduces the defect, so the "
        "regression above is not measuring anything."
    )


def test_the_recorded_source_survives_the_round_trip(replica, tmp_path):
    path = _write(tmp_path, source="energy_generation")
    assert joblib.load(path)["training_source"] == "energy_generation"
    assert Forecaster.load(
        COUNTRY, "wind_onshore", path=str(path)
    ).training_source == "energy_generation"


def test_a_harness_naming_no_source_records_the_table_it_read(replica, tmp_path):
    """`None` means 'db's default'. The artifact has to record which table that
    actually was — an absent key is what `load` then has to guess at, and the
    guess is a literal that a later flip of the training default will not move."""
    data = joblib.load(_write(tmp_path, source=None))
    assert data["training_source"] == db.RENEWABLE_TYPE_SOURCE_TABLE
    assert data["training_source"] is not None


@pytest.mark.parametrize("forecast_type", ["wind_onshore", "wind_offshore", "solar"])
def test_every_type_both_harnesses_fit_carries_a_source(replica, tmp_path, forecast_type):
    data = joblib.load(_write(tmp_path, source="energy_generation",
                              forecast_type=forecast_type))
    assert data["training_source"] == "energy_generation"


# ---------------------------------------------------------------------------
# The ABL-183 intercept witness comes along too.
# ---------------------------------------------------------------------------


def _fitted_xgb():
    from xgboost import XGBRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, len(WIND_COLUMNS)))
    model = XGBRegressor(n_estimators=4, max_depth=2)
    model.fit(X, rng.normal(loc=4000.0, scale=10.0, size=64))
    return model


def test_an_xgboost_gate_artifact_records_the_intercept_witness(replica, tmp_path):
    """The wind gate fits `wind_offshore` with xgboost. The bare dump omitted
    `base_score`/`xgboost_version`, and `assert_survived_load` treats an absent
    witness as 'cannot check' — so the guard was off for exactly the artifacts
    the harness produces."""
    data = joblib.load(_write(tmp_path, source="energy_generation",
                              forecast_type="wind_offshore",
                              model=_fitted_xgb(), algorithm="xgboost"))
    assert data["base_score"] is not None
    assert data["xgboost_version"]


def test_the_recorded_witness_is_live_not_decorative(replica, tmp_path):
    """Tamper with the stored intercept: `Forecaster.load` must refuse. Proves
    the witness written above is compared rather than merely present."""
    path = _write(tmp_path, source="energy_generation", forecast_type="wind_offshore",
                  model=_fitted_xgb(), algorithm="xgboost")
    data = joblib.load(path)
    data["base_score"] = data["base_score"] + 5_000.0
    joblib.dump(data, path)

    with pytest.raises(ModelArtifactError):
        Forecaster.load(COUNTRY, "wind_offshore", path=str(path))


# ---------------------------------------------------------------------------
# The harnesses' own keys still have to be there.
# ---------------------------------------------------------------------------


def test_the_harness_keys_downstream_readers_use_are_preserved(replica, tmp_path):
    data = joblib.load(_write(tmp_path, source="energy_generation"))
    assert data["params"] == _params()
    assert data["fit_window"] == ["2026-01-14 00:00:00", "2026-07-11 00:00:00"]
    assert data["feature_columns"] == list(WIND_COLUMNS)
    assert data["country_code"] == COUNTRY
    assert data["forecast_type"] == "wind_onshore"
    assert data["algorithm"] == "catboost"


def test_the_recorded_params_are_the_fitted_ones_not_the_algorithm_defaults(replica, tmp_path):
    """`Forecaster.__init__` merges hyperparams *over* the defaults. The wind
    gate pops `early_stopping_rounds` because its final fit has no validation
    set; if the merge restated it, the artifact would claim a setting the model
    was not fitted with."""
    data = joblib.load(_write(tmp_path, source="energy_generation"))
    assert data["hyperparams"] == _params()
    assert "early_stopping_rounds" not in data["hyperparams"]


def test_extra_metadata_cannot_overwrite_a_derived_key(replica, tmp_path):
    """The whole point of one writer is that provenance is derived, not passed
    in. A caller that could set `training_source` through the escape hatch would
    be the bypass this change removes."""
    forecaster = Forecaster(COUNTRY, "wind_onshore", algorithm="catboost",
                            training_source="energy_generation")
    forecaster.model = EchoLagModel()
    forecaster.feature_columns = list(WIND_COLUMNS)
    path = tmp_path / "clash" / "model.joblib"
    path.parent.mkdir(parents=True)

    with pytest.raises(ValueError, match="training_source"):
        forecaster.save(str(path), extra_metadata={"training_source": "energy_renewable"})


# ---------------------------------------------------------------------------
# The registry: the third writer, closed by refusal.
# ---------------------------------------------------------------------------


def test_the_registry_refuses_a_renewable_payload_with_no_source(replica, tmp_path):
    registry = ModelRegistry(models_dir=tmp_path / "models")
    with pytest.raises(ValueError, match="training_source"):
        registry.save_model(
            {"model": EchoLagModel(), "feature_columns": list(WIND_COLUMNS)},
            COUNTRY, "solar", "candidate",
        )


def test_the_registry_refuses_a_source_that_is_present_but_empty(replica, tmp_path):
    registry = ModelRegistry(models_dir=tmp_path / "models")
    with pytest.raises(ValueError, match="training_source"):
        registry.save_model(
            {"model": EchoLagModel(), "training_source": None},
            COUNTRY, "wind_onshore", "candidate",
        )


def test_the_registry_accepts_what_the_normal_pipeline_builds(replica, tmp_path):
    """`scripts/train.py` saves `Forecaster._get_model_data()` through the
    registry. The guard must not break that path."""
    forecaster = Forecaster(COUNTRY, "solar", algorithm="catboost",
                            training_source="energy_generation")
    forecaster.model = EchoLagModel()
    forecaster.feature_columns = list(WIND_COLUMNS)
    registry = ModelRegistry(models_dir=tmp_path / "models")

    path = registry.save_model(forecaster._get_model_data(), COUNTRY, "solar", "candidate")
    assert joblib.load(path)["training_source"] == "energy_generation"


@pytest.mark.parametrize("forecast_type", ["load", "price", "renewable", "net_position"])
def test_the_registry_still_accepts_an_aggregate_type_without_a_source(replica, tmp_path,
                                                                      forecast_type):
    """The aggregate types read one fixed table each, so they carry no renewable
    source and must not be caught by the guard."""
    registry = ModelRegistry(models_dir=tmp_path / "models")
    path = registry.save_model({"model": EchoLagModel()}, COUNTRY, forecast_type, "candidate")
    assert path.exists()
