"""ABL-321: the individual-renewable training source and its NULL-vs-0 contract.

The whole point of moving `load_renewable_type_data` off `energy_renewable`
is that `energy_renewable` **cannot say "not reported"** -- its columns are
`REAL DEFAULT 0` and its per-column mapper initialises each to 0.0 before
checking whether ENTSO-E returned the production type (ABL-188). Measured on
the replica: the 15 countries whose `wind_offshore_mw` is NULL in every
`energy_generation` row read as 100.0% exactly 0.0 in `energy_renewable`.

So the property these tests pin is not "we read a different table" -- it is
that a country/stream the TSO does not report reaches a trainer as an **empty
frame**, and that a country/stream it *does* report is unaffected. A test that
only asserted the table name would still pass if the NULL rows came back as a
zero series.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src import db
from src.wind_features import RenewableFeatureBuilder

COUNTRY = "XX"
START, END = "2026-01-01", "2026-01-04"

#: A country with an onshore fleet and no offshore fleet -- the exact shape
#: the 15 real countries in the ABL-318 audit have.
_HOURS = pd.date_range("2026-01-01", "2026-01-03 23:00", freq="h")

#: A 12-hour zero-filled solar block -- half the 24-hour minimum
#: `exclude_suspect_constant_runs` needs to fire. The real instance is LV
#: solar: 374 zero-filled rows and *zero* runs reaching 24 h, so every one of
#: them passes the ABL-188 guard and enters training as a measured zero.
ZERO_FILL_HOURS = range(24, 36)


@pytest.fixture
def replica(tmp_path, monkeypatch):
    path = tmp_path / "replica.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE energy_generation (country_code TEXT, timestamp_utc TIMESTAMP,
            solar_mw REAL, wind_onshore_mw REAL, wind_offshore_mw REAL,
            hydro_run_mw REAL, hydro_reservoir_mw REAL, biomass_mw REAL,
            data_quality TEXT DEFAULT 'actual');
        CREATE TABLE energy_renewable (country_code TEXT, timestamp_utc TIMESTAMP,
            solar_mw REAL DEFAULT 0, wind_onshore_mw REAL DEFAULT 0,
            wind_offshore_mw REAL DEFAULT 0, hydro_run_mw REAL DEFAULT 0,
            hydro_reservoir_mw REAL DEFAULT 0, biomass_mw REAL DEFAULT 0,
            data_quality TEXT DEFAULT 'actual');
        CREATE TABLE weather_data (country_code TEXT, timestamp_utc TIMESTAMP,
            forecast_run_time TIMESTAMP, data_quality TEXT,
            temperature_2m_k REAL, wind_speed_10m_ms REAL, wind_speed_100m_ms REAL,
            shortwave_radiation_wm2 REAL, direct_radiation_wm2 REAL,
            diffuse_radiation_wm2 REAL);
        """
    )
    for i, ts in enumerate(_HOURS):
        onshore = 1000.0 + i
        solar = 500.0 + i
        # energy_generation: offshore and biomass NULL (not reported by this
        # TSO); hydro_run reported, hydro_reservoir never reported.
        con.execute(
            "INSERT INTO energy_generation VALUES (?, ?, ?, ?, NULL, ?, NULL, NULL, 'actual')",
            (COUNTRY, str(ts), solar, onshore, 42.0 + i),
        )
        # energy_renewable: the same instants, but the unreported types arrive
        # as a measured 0.0, and solar is zero-filled across ZERO_FILL_HOURS --
        # a run deliberately shorter than `exclude_suspect_constant_runs`'
        # 24-hour minimum, which is the case the guard cannot catch.
        con.execute(
            "INSERT INTO energy_renewable VALUES (?, ?, ?, ?, 0.0, ?, 0.0, 0.0, 'actual')",
            (COUNTRY, str(ts), 0.0 if i in ZERO_FILL_HOURS else solar, onshore, 42.0 + i),
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


def test_default_source_is_energy_generation():
    """The switch itself. Pinned as a constant so a silent revert is a test
    failure rather than a quiet retraining against the wrong table."""
    assert db.RENEWABLE_TYPE_SOURCE_TABLE == "energy_generation"


def test_an_unreported_stream_yields_an_empty_frame_not_a_zero_series(replica):
    """Acceptance criterion 3. This is the property the whole issue turns on."""
    frame = db.load_renewable_type_data(COUNTRY, "wind_offshore", START, END)
    assert frame.empty, (
        "NOT REPORTED IS NOT ZERO. A country/stream the TSO does not report must "
        f"reach the trainer as an empty frame; got {len(frame)} rows.\n"
        "\n"
        "If you are reading this because you changed the loader: the rows you are "
        "about to train on are almost certainly fabricated. `energy_renewable`'s "
        "columns are REAL DEFAULT 0 and its mapper initialises each to 0.0 before "
        "checking whether ENTSO-E returned the production type (ABL-188), so the 15 "
        "countries with no offshore fleet read as 100.0% exactly 0.0 -- a complete, "
        "non-null, perfectly valid series for a fleet that does not exist. Nothing "
        "downstream can tell that apart from a measurement: no NaN, no gap, no "
        "warning, and a model fitted on it scores beautifully against the same "
        "zeros. ABL-321 exists because we were one issue away from training 15 such "
        "models and reporting them as working.\n"
        "\n"
        "A NULL run and an absent row both mean 'not reported', and neither is a "
        "zero. Make the loader drop them; do not make this test tolerate them."
    )


def test_the_old_source_returns_that_same_stream_as_a_full_length_frame(replica):
    """The counterfactual, so the test above cannot pass vacuously. This is
    what a trainer received before ABL-321: a complete, non-null, perfectly
    'valid' series for a fleet that does not exist.

    Here the ABL-188 guard does null the values, because a 72-hour synthetic
    run clears its 24-hour minimum -- the row count is the tell. The case the
    guard genuinely cannot catch is the sub-24h one below."""
    frame = db.load_renewable_type_data(
        COUNTRY, "wind_offshore", START, END, source="energy_renewable"
    )
    assert len(frame) == len(_HOURS)


def test_a_sub_24h_zero_fill_survives_the_guard_on_the_old_source(replica):
    """The load-bearing argument for the switch. `exclude_suspect_constant_runs`
    only rejects bit-identical runs of 24 hours or more, so a shorter zero-fill
    reaches the trainer as a measured zero -- 6,711 such rows across 35 of 72
    country/stream pairs in the ABL-318 census, including LV solar's 374 with
    no qualifying run at all."""
    old = db.load_renewable_type_data(
        COUNTRY, "solar", START, END, source="energy_renewable"
    )
    zero_filled = old["target_value"].iloc[list(ZERO_FILL_HOURS)]
    assert (zero_filled == 0.0).all(), (
        "expected the guard to miss a sub-24h run; if this fails the guard's "
        "minimum changed and the argument in this test needs restating"
    )


def test_the_new_source_carries_the_real_values_the_zero_fill_masked(replica):
    """...and `energy_generation` has the real, non-zero generation at exactly
    those instants. This is ABL-188's own adjudication test, in miniature."""
    new = db.load_renewable_type_data(COUNTRY, "solar", START, END)
    recovered = new["target_value"].iloc[list(ZERO_FILL_HOURS)]
    assert (recovered > 0).all()
    assert recovered.iloc[0] == pytest.approx(500.0 + ZERO_FILL_HOURS[0])


def test_a_reported_stream_is_unchanged_by_the_switch(replica):
    """The switch must not cost coverage where both tables agree."""
    new = db.load_renewable_type_data(COUNTRY, "wind_onshore", START, END)
    old = db.load_renewable_type_data(
        COUNTRY, "wind_onshore", START, END, source="energy_renewable"
    )
    assert len(new) == len(old) == len(_HOURS)
    pd.testing.assert_series_equal(
        new["target_value"], old["target_value"], check_names=False
    )


def test_hydro_total_sums_the_reported_component_when_the_other_is_never_reported(replica):
    """SQL's `+` propagates NULL, so a plain `hydro_run_mw + hydro_reservoir_mw`
    would return NULL for every row here and erase the country. Measured on the
    replica 2026-08-12, that is not hypothetical: for 9 of the 24 supported
    countries exactly one hydro component is 100% NULL in `energy_generation`
    (BE/EE/FI/LT/LV/NL/SI report run-of-river only, GR/SE reservoir only)."""
    frame = db.load_renewable_type_data(COUNTRY, "hydro_total", START, END)
    assert len(frame) == len(_HOURS)
    assert frame["target_value"].iloc[0] == pytest.approx(42.0)
    assert frame["target_value"].iloc[-1] == pytest.approx(42.0 + len(_HOURS) - 1)


def test_hydro_total_is_empty_when_neither_component_is_reported(replica, tmp_path, monkeypatch):
    """...but "not reported at all" still has to survive as an empty frame,
    rather than being coalesced into a fabricated 0."""
    path = tmp_path / "nohydro.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE energy_generation (country_code TEXT, timestamp_utc TIMESTAMP,
            hydro_run_mw REAL, hydro_reservoir_mw REAL,
            data_quality TEXT DEFAULT 'actual');
        """
    )
    for ts in _HOURS:
        con.execute(
            "INSERT INTO energy_generation VALUES (?, ?, NULL, NULL, 'actual')",
            (COUNTRY, str(ts)),
        )
    con.commit()
    con.close()
    monkeypatch.setenv("ENERGY_DB_PATH", str(path))
    importlib.reload(config)
    importlib.reload(db)
    assert db.load_renewable_type_data(COUNTRY, "hydro_total", START, END).empty


def test_an_unknown_source_is_rejected_rather_than_interpolated(replica):
    with pytest.raises(ValueError, match="Unknown renewable source table"):
        db.load_renewable_type_data(COUNTRY, "solar", START, END, source="forecasts")


def test_the_feature_builder_carries_the_source_through_to_its_actuals(replica):
    """`RenewableFeatureBuilder` is where the switch actually reaches every lag
    and rolling feature, not just the training target -- so the A/B harness can
    vary exactly one thing, and so a serving path cannot end up reading a
    different table than the one its model was fitted on."""
    span = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03 23:00"))
    new = RenewableFeatureBuilder(COUNTRY, "wind_offshore", *span)
    old = RenewableFeatureBuilder(
        COUNTRY, "wind_offshore", *span, actuals_source="energy_renewable"
    )
    assert new._actuals.empty
    assert len(old._actuals) == len(_HOURS)
