"""ABL-331: the renewable source is a property of the artifact, not a global.

Before this change one constant (`db.RENEWABLE_TYPE_SOURCE_TABLE`) decided the
training source *and* the serve-time feature source for all 49 country/stream
pairs at once — `Forecaster` built its `RenewableFeatureBuilder` with no source
argument, so every lag, rolling and anchor feature of the ten already-serving
models was read from whatever that constant said at inference time. ABL-321
measured that flipping it makes three of those ten materially worse, while the
other 39 pairs cannot use `energy_renewable` at all. One constant cannot serve
both groups, and flipping it alone would have produced a fourth state nobody
measured: artifacts fitted on one table being served features from the other.

So the source moves onto the artifact. The properties pinned here:

1. **Every artifact on disk today serves bit-identically.** None of them carries
   a `training_source` key — measured across all 88 renewable artifacts in the
   primary checkout on 2026-08-12 — so the absent-key default is what the whole
   live fleet runs on. It must reproduce the pre-change construction exactly,
   not approximately.
2. An artifact that names `energy_generation` is served from `energy_generation`
   even though the global constant says otherwise.
3. The absent-key default is `energy_renewable` *as a literal*, so a later flip
   of the training default cannot retroactively move a legacy artifact onto a
   table it was never fitted on.

The replica gives the two tables **deliberately different values** for the same
country and instant, and the stand-in model returns its own `lag_1d` feature.
A forecast value therefore names the table it was read from: a test that only
checked shapes would pass no matter which table won.
"""
import importlib
import sqlite3
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src import db
from src.forecaster import Forecaster
from src.wind_features import RenewableFeatureBuilder, to_vector

COUNTRY = "XX"
OBS = pd.Timestamp("2026-08-11 08:00:00")

#: `energy_renewable` carries the same series offset by this much. Any feature
#: derived from the target series therefore differs between the two tables by a
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

FEATURE_COLUMNS = _SHARED_COLUMNS + [
    "wind_speed_100m_ms", "wind_speed_10m_ms", "temperature_c",
]

SOLAR_FEATURE_COLUMNS = _SHARED_COLUMNS + [
    "shortwave_radiation_wm2", "direct_radiation_wm2", "diffuse_radiation_wm2",
    "temperature_c",
]

#: The builder emits a different weather block per type (ABL-183/ABL-191), and
#: `to_vector` raises on a column it cannot build — so the artifact's columns
#: have to match the type under test.
def _columns_for(forecast_type: str) -> list:
    return list(SOLAR_FEATURE_COLUMNS if forecast_type == "solar" else FEATURE_COLUMNS)


def _epoch_hours(ts: pd.Timestamp) -> float:
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
    """Returns its own `target_value_lag_1d` column, so the forecast value is a
    direct readout of which table fed the target series, and records every frame
    it was asked to predict on so a test can compare the *vector* rather than
    trust the scalar. Module-level (not a closure) because these get pickled
    into artifacts."""

    def __init__(self):
        self.calls = []

    def predict(self, X):
        self.calls.append(X.copy())
        return X["target_value_lag_1d"].to_numpy(dtype=float)


def _write_artifact(path: Path, forecast_type: str, extra: dict = None) -> Path:
    """Write a model artifact by hand.

    Criterion 3 says *construct* a key-less artifact rather than go looking for
    one — a test that hunts for a legacy file on disk stops testing anything the
    day the last one is retrained. `algorithm` is catboost so the xgboost
    intercept witness (ABL-183) stays out of the way of what is being measured.
    """
    payload = {
        "model": EchoLagModel(),
        "algorithm": "catboost",
        "feature_columns": _columns_for(forecast_type),
        "country_code": COUNTRY,
        "forecast_type": forecast_type,
        "model_version": "abl331-test",
    }
    payload.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path


def _forecast(path: Path, forecast_type: str, hours=(5,)):
    """Returns (forecast frame, the loaded Forecaster) — the caller usually
    wants the model's recorded input frames as well as the output."""
    forecaster = Forecaster.load(COUNTRY, forecast_type, path=str(path))
    df = forecaster.predict_d2(
        reference_date=OBS.date(), horizon_days=1, hours=list(hours),
        observation_as_of=OBS, weather_publication_as_of=OBS,
    )
    return df, forecaster


def _served_value(path: Path, forecast_type: str) -> float:
    df, _ = _forecast(path, forecast_type)
    return float(df["forecast_value"].iloc[0])


def _pre_change_vectors(forecast_type: str, hours) -> list:
    """The feature vectors the **pre-ABL-331** serving path produced.

    `_predict_d2_serve_faithful` used to construct its builder with no source
    argument at all; this reproduces that call verbatim, including the span
    arithmetic, so the comparison is against the old code's actual behaviour
    rather than against a restatement of the new code's.
    """
    target_hours = [
        pd.Timestamp(OBS.year, OBS.month, OBS.day, h) + pd.Timedelta(days=1)
        for h in hours
    ]
    lookback_days = max(config.LAG_DAYS) + 7
    span_start = min(min(target_hours), OBS) - pd.Timedelta(days=lookback_days)
    span_end = max(max(target_hours), OBS)
    builder = RenewableFeatureBuilder(COUNTRY, forecast_type, span_start, span_end)
    columns = _columns_for(forecast_type)
    return [to_vector(builder.row(t, OBS, OBS), columns) for t in target_hours]


# ---------------------------------------------------------------------------
# Criterion 1 — every artifact on disk today serves bit-identically.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forecast_type", ["wind_onshore", "wind_offshore", "solar"])
def test_a_keyless_artifact_serves_the_pre_change_vector_bit_for_bit(replica, tmp_path, forecast_type):
    """Acceptance criterion 1, and the one that matters most.

    No artifact in the fleet carries `training_source`, so every live forecast
    runs through the absent-key default. If that default resolves to anything
    other than what the old no-argument builder resolved to, ten models start
    predicting from a series they were never fitted on — silently, with no
    error, no gap and no shape change. Bit-for-bit or it did not hold.
    """
    hours = [0, 5, 13, 23]
    columns = _columns_for(forecast_type)
    path = _write_artifact(tmp_path / "legacy" / "model.joblib", forecast_type)
    served, forecaster = _forecast(path, forecast_type, hours)
    expected = _pre_change_vectors(forecast_type, hours)

    complaint = (
        "a pre-ABL-331 artifact no longer serves what it served before. "
        "ABL-331's entire premise is that making the source per-artifact moves "
        "no live forecast; if this fails, it moved one."
    )

    # The whole feature vector, not just the output — a wrong source that
    # happened to agree on `lag_1d` would still have moved the rollings.
    assert len(forecaster.model.calls) == len(hours)
    for frame, exp in zip(forecaster.model.calls, expected):
        assert list(frame.columns) == columns, complaint
        np.testing.assert_array_equal(
            frame.iloc[0].to_numpy(dtype=float),
            np.array([exp[c] for c in columns], dtype=float),
            err_msg=complaint,
        )

    np.testing.assert_array_equal(
        served["forecast_value"].to_numpy(dtype=float),
        np.array([exp["target_value_lag_1d"] for exp in expected], dtype=float),
        err_msg=complaint,
    )


def test_a_keyless_artifact_reports_the_source_it_is_actually_reading(replica, tmp_path):
    path = _write_artifact(tmp_path / "legacy" / "model.joblib", "wind_onshore")
    forecaster = Forecaster.load(COUNTRY, "wind_onshore", path=str(path))
    assert forecaster.training_source == "energy_renewable"


def test_a_keyless_artifact_reads_energy_renewable_not_energy_generation(replica, tmp_path):
    """Criterion 3 stated as a value, not as a table name: the two tables differ
    by `RENEWABLE_OFFSET` here, so the forecast itself says which one was read."""
    path = _write_artifact(tmp_path / "legacy" / "model.joblib", "wind_onshore")
    served = _served_value(path, "wind_onshore")

    assert served > RENEWABLE_OFFSET, (
        f"a key-less artifact served {served}, which is an `energy_generation` "
        "value. Artifacts predating ABL-331 were fitted on `energy_renewable` "
        "and must keep being served from it."
    )


# ---------------------------------------------------------------------------
# Criterion 2 — a named source wins over the global constant.
# ---------------------------------------------------------------------------


def test_an_artifact_naming_energy_generation_is_served_from_it(replica, tmp_path):
    """Acceptance criterion 2.

    `db.RENEWABLE_TYPE_SOURCE_TABLE` is `energy_renewable` and the two tables
    differ by `RENEWABLE_OFFSET`, so if the global constant were still consulted
    this returns a value above the offset and the assertion fails."""
    path = _write_artifact(
        tmp_path / "gen" / "model.joblib", "wind_onshore",
        {"training_source": "energy_generation"},
    )
    served = _served_value(path, "wind_onshore")

    assert db.RENEWABLE_TYPE_SOURCE_TABLE == "energy_renewable", "precondition"
    assert served < RENEWABLE_OFFSET, (
        f"an artifact declaring training_source='energy_generation' served "
        f"{served}, an `energy_renewable` value. The global constant is still "
        "being consulted at inference, which is the ABL-331 defect."
    )


def test_the_two_sources_actually_disagree_here(replica, tmp_path):
    """Guards the two tests above: if the fixture ever made the tables agree,
    both would pass while proving nothing."""
    legacy = _write_artifact(tmp_path / "a" / "model.joblib", "wind_onshore")
    named = _write_artifact(
        tmp_path / "b" / "model.joblib", "wind_onshore",
        {"training_source": "energy_generation"},
    )
    assert _served_value(legacy, "wind_onshore") != pytest.approx(
        _served_value(named, "wind_onshore")
    )


def test_a_flipped_global_default_does_not_move_a_legacy_artifact(replica, tmp_path, monkeypatch):
    """The absent-key default is a literal, not an alias of the training default.

    If someone later flips `RENEWABLE_TYPE_SOURCE_TABLE` to `energy_generation`
    for new training runs, every artifact already on disk was still fitted on
    `energy_renewable`. Serving it from the new default is exactly the unmeasured
    train/serve skew the ABL-321 verdict rejected — so the legacy default cannot
    be allowed to follow the constant.
    """
    monkeypatch.setattr(db, "RENEWABLE_TYPE_SOURCE_TABLE", "energy_generation")
    path = _write_artifact(tmp_path / "legacy" / "model.joblib", "wind_onshore")
    served = _served_value(path, "wind_onshore")

    assert served > RENEWABLE_OFFSET, (
        "flipping the training default moved a legacy artifact's serving source. "
        "`LEGACY_RENEWABLE_TRAINING_SOURCE` must stay a literal."
    )


# ---------------------------------------------------------------------------
# Save / load round trip — the recorded value must be the value that was used.
# ---------------------------------------------------------------------------


def test_save_records_the_resolved_source_not_the_none_it_was_given(replica, tmp_path):
    forecaster = Forecaster(COUNTRY, "wind_onshore", algorithm="catboost")
    forecaster.model = EchoLagModel()
    forecaster.feature_columns = list(FEATURE_COLUMNS)
    path = tmp_path / "saved" / "model.joblib"
    path.parent.mkdir(parents=True)
    forecaster.save(str(path))

    assert joblib.load(path)["training_source"] == "energy_renewable", (
        "`None` means 'db's default'; the artifact has to record which table "
        "that actually was, or it records an intention instead of a fact."
    )


def test_save_records_an_explicitly_named_source(replica, tmp_path):
    forecaster = Forecaster(
        COUNTRY, "wind_onshore", algorithm="catboost", training_source="energy_generation"
    )
    forecaster.model = EchoLagModel()
    forecaster.feature_columns = list(FEATURE_COLUMNS)
    path = tmp_path / "saved" / "model.joblib"
    path.parent.mkdir(parents=True)
    forecaster.save(str(path))

    assert joblib.load(path)["training_source"] == "energy_generation"


def test_the_registry_payload_carries_it_too(replica):
    """`scripts/train.py` saves through `_get_model_data` + the registry, not
    through `Forecaster.save`. A key added to only one of the two would leave
    every model trained by the normal pipeline without a source."""
    forecaster = Forecaster(
        COUNTRY, "solar", algorithm="catboost", training_source="energy_generation"
    )
    forecaster.model = EchoLagModel()
    forecaster.feature_columns = list(FEATURE_COLUMNS)
    assert forecaster._get_model_data()["training_source"] == "energy_generation"


@pytest.mark.parametrize("forecast_type", ["load", "price", "renewable"])
def test_an_aggregate_type_records_no_renewable_source(replica, forecast_type):
    """`load`, `price` and `renewable` read one fixed table each. Stamping a
    renewable source table on them would be a field that reads as true and means
    nothing — the species of metadata this codebase treats as a defect."""
    forecaster = Forecaster(COUNTRY, forecast_type, algorithm="catboost")
    forecaster.model = EchoLagModel()
    forecaster.feature_columns = list(FEATURE_COLUMNS)
    assert forecaster._get_model_data()["training_source"] is None


def test_a_round_trip_preserves_a_named_source(replica, tmp_path):
    path = _write_artifact(
        tmp_path / "gen" / "model.joblib", "solar",
        {"training_source": "energy_generation"},
    )
    assert Forecaster.load(COUNTRY, "solar", path=str(path)).training_source == "energy_generation"


# ---------------------------------------------------------------------------
# The recorded source has to be the one training actually read.
# ---------------------------------------------------------------------------


def test_training_reads_the_source_the_artifact_will_claim(replica, monkeypatch):
    """If `train()` ignored `training_source` while `save()` recorded it, the
    artifact would assert a provenance it does not have — and would then be
    served features from a table it was never fitted on, for the rest of its
    life. Capture what the loader was actually handed."""
    seen = {}

    def _capture(country_code, forecast_type, start_date, end_date, source=None):
        seen["source"] = source
        return pd.DataFrame()

    monkeypatch.setattr("src.forecaster.load_training_data", _capture)
    forecaster = Forecaster(
        COUNTRY, "wind_onshore", algorithm="catboost", training_source="energy_generation"
    )
    with pytest.raises(ValueError):  # empty frame -> "No training data available"
        forecaster.train(start_date="2026-01-01", end_date="2026-02-01")

    assert seen["source"] == "energy_generation"


# ---------------------------------------------------------------------------
# ABL-331 follow-up — the training *window* closes on the artifact's table too.
#
# `save`, `load` and the feature builder are threaded above, but both `train`
# entry points resolve an open-ended window (`end_date is None`) by asking
# `get_latest_data_timestamp` how fresh the data is, and that read the global
# constant regardless of the table the run was about to train on. A run naming
# `energy_generation` therefore closed its window on `energy_renewable`'s last
# instant: silently truncated where that table lags, and left to `datetime.now()`
# where the pair has no rows in it at all — the normal case for the 39 unmodelled
# pairs ABL-316 exists to cover. Same species as the bug above, on a path nobody
# had looked at: a source-blind global read.
# ---------------------------------------------------------------------------

#: A last instant present in one table and absent from the other, so a resolved
#: window end names the table it was read from.
FRESHER = OBS + pd.Timedelta(hours=12)


def _extend(path: Path, table: str) -> None:
    """Give one table a later last instant than the other."""
    con = sqlite3.connect(path)
    value = _epoch_hours(FRESHER)
    con.execute(
        f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'actual')",
        (COUNTRY, str(FRESHER), value, value, value, value, value, value),
    )
    con.commit()
    con.close()


def test_the_freshness_probe_reads_the_source_it_is_given(replica):
    _extend(replica, "energy_generation")
    assert db.get_latest_data_timestamp(
        COUNTRY, "wind_onshore", source="energy_generation"
    ) == FRESHER
    assert db.get_latest_data_timestamp(
        COUNTRY, "wind_onshore", source="energy_renewable"
    ) == OBS


def test_the_freshness_probe_default_is_the_pre_change_table(replica):
    """Criterion 1's spirit on this path: a caller passing no source gets
    exactly the table the pre-change code read."""
    _extend(replica, "energy_generation")
    assert db.get_latest_data_timestamp(COUNTRY, "wind_onshore") == OBS


def test_an_aggregate_type_ignores_a_renewable_source(replica):
    """`load`, `price` and `renewable` read one fixed table each — a renewable
    source table is meaningless for them and must not redirect the read."""
    _extend(replica, "energy_generation")
    assert db.get_latest_data_timestamp(
        COUNTRY, "renewable", source="energy_generation"
    ) == OBS


def test_the_freshness_probe_rejects_a_table_it_does_not_know(replica):
    """Mirrors `load_renewable_type_data`: the table name is interpolated into
    the query, so an unknown one has to fail before it reaches SQL."""
    with pytest.raises(ValueError, match="Unknown renewable source table"):
        db.get_latest_data_timestamp(COUNTRY, "wind_onshore", source="forecasts")


@pytest.mark.parametrize(
    "training_source,expected",
    [("energy_generation", "energy_generation"), (None, "energy_renewable")],
)
def test_an_open_window_asks_the_freshness_of_the_table_it_will_train_on(
    replica, monkeypatch, training_source, expected
):
    """The `None` case is the back-compatibility half: a legacy artifact still
    closes its window on `energy_renewable`, exactly as before."""
    asked = {}

    def _probe(country_code, data_type, source=None):
        asked["source"] = source
        return None  # end_date then falls back to now(); the empty load stops us

    monkeypatch.setattr("src.forecaster.get_latest_data_timestamp", _probe)
    monkeypatch.setattr("src.forecaster.load_training_data", lambda *a, **k: pd.DataFrame())
    forecaster = Forecaster(
        COUNTRY, "wind_onshore", algorithm="catboost", training_source=training_source
    )
    with pytest.raises(ValueError):  # empty frame -> "No training data available"
        forecaster.train(start_date="2026-01-01", end_date=None)

    assert asked["source"] == expected


def test_an_aggregate_model_asks_for_no_renewable_source(replica, monkeypatch):
    """A `load` model must not send a renewable source table into the probe."""
    asked = {}

    def _probe(country_code, data_type, source=None):
        asked["source"] = source
        return None

    monkeypatch.setattr("src.forecaster.get_latest_data_timestamp", _probe)
    monkeypatch.setattr("src.forecaster.load_training_data", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError):
        Forecaster(COUNTRY, "load", algorithm="catboost").train(
            start_date="2026-01-01", end_date=None
        )

    assert asked["source"] is None


def test_the_walk_forward_window_asks_the_same_question(replica, monkeypatch):
    """`train_with_walk_forward` resolves its own window and had the same
    source-blind read. `validation` is stubbed because that method imports it
    flat (`from validation import ...`) before reaching the window code."""
    asked = {}

    def _probe(country_code, data_type, source=None):
        asked["source"] = source
        return None

    monkeypatch.setitem(
        sys.modules, "validation", types.SimpleNamespace(WalkForwardValidator=object)
    )
    monkeypatch.setattr("src.forecaster.get_latest_data_timestamp", _probe)
    monkeypatch.setattr("src.forecaster.load_training_data", lambda *a, **k: pd.DataFrame())
    forecaster = Forecaster(
        COUNTRY, "wind_onshore", algorithm="catboost", training_source="energy_generation"
    )
    with pytest.raises(ValueError):
        forecaster.train_with_walk_forward(start_date="2026-01-01", end_date=None)

    assert asked["source"] == "energy_generation"
