"""The context must end where the observations actually stop.

A D+2 run fires at ~06:00 on day D, but the schedule's nominal cutoff is
D+1 23:00 — some 42h later. Building the context out to that cutoff meant
`_align_to_index` forward-filled 6h and wrote 0.0 into the remaining ~36,
so the most recent thing the model saw was a block of zeros. Net position is
signed and centred near zero, which made those zeros look like plausible
observations: measured FR forecasts came out at 6% of actual and DE was
sign-flipped.

These tests pin the two properties that prevent it: no zero-filled tail, and
a horizon long enough to reach the target day from wherever the data stops.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# The run would fire here; the target day is two days later.
AS_OF = "2026-06-10 06:00:00"
TARGET_DATE = "2026-06-12"
# Observations stop at the last completed hour before the run.
LAST_OBS = pd.Timestamp("2026-06-10 05:00:00")
# Nominal cutoff the schedule implies: target 00:00 - 1h.
NOMINAL_CUTOFF = pd.Timestamp("2026-06-11 23:00:00")
GAP_HOURS = int((NOMINAL_CUTOFF - LAST_OBS) / pd.Timedelta(hours=1))  # 42

LEVEL = 8000.0  # a country sitting well away from zero, like FR


def _seed_db(path, last_obs=LAST_OBS):
    """Minimal DB: 60 days of hourly net position at a steady non-zero level."""
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE net_position "
        "(country_code TEXT, timestamp_utc TEXT, net_position_mw REAL)"
    )
    # Tables the covariate mapper reaches for; empty is fine, the loaders
    # fall back to zeros and none of these assertions depend on them.
    con.execute("CREATE TABLE weather_data (country_code TEXT, timestamp_utc TEXT, "
                "temperature_2m_k REAL, wind_speed_100m_ms REAL, "
                "shortwave_radiation_wm2 REAL, data_quality TEXT, forecast_run_time TEXT)")
    con.execute("CREATE TABLE energy_price (country_code TEXT, timestamp_utc TEXT, "
                "price_eur_mwh REAL, data_quality TEXT)")
    con.execute("CREATE TABLE energy_load (country_code TEXT, timestamp_utc TEXT, "
                "load_mw REAL, data_quality TEXT)")
    con.execute("CREATE TABLE energy_load_forecast (country_code TEXT, "
                "target_timestamp_utc TEXT, forecast_value_mw REAL)")
    con.execute("CREATE TABLE crossborder_flows (country_from TEXT, country_to TEXT, "
                "timestamp_utc TEXT, flow_mw REAL)")

    start = last_obs - pd.Timedelta(days=60)
    idx = pd.date_range(start, last_obs, freq="h")
    con.executemany(
        "INSERT INTO net_position VALUES (?,?,?)",
        [("FR", ts.strftime("%Y-%m-%d %H:%M:%S"), LEVEL) for ts in idx],
    )
    con.commit()
    con.close()


def _builder(monkeypatch, db_path):
    monkeypatch.setenv("ENERGY_DB_PATH", str(db_path))
    import config
    importlib.reload(config)
    from src.chronos2 import input_builder
    importlib.reload(input_builder)
    return input_builder


@pytest.fixture
def build(monkeypatch, tmp_path):
    db = tmp_path / "t.db"
    _seed_db(db)
    ib = _builder(monkeypatch, db)
    return ib.InputBuilder()


def test_context_has_no_zero_filled_tail(build):
    """The regression itself: nothing in the context may be invented zeros."""
    inp = build.build_for_country("FR", "net_position", TARGET_DATE, as_of=AS_OF)
    target = np.asarray(inp["target"], dtype=float)

    assert not np.any(target == 0.0), (
        f"{int((target == 0.0).sum())} zero-valued hours in the context; "
        "the unobserved tail was padded instead of excluded"
    )
    assert target[-1] == pytest.approx(LEVEL)


def test_horizon_spans_the_gap_plus_the_target_day(build):
    inp = build.build_for_country("FR", "net_position", TARGET_DATE, as_of=AS_OF)
    assert inp["prediction_length"] == GAP_HOURS + 24


def test_horizon_tail_is_exactly_the_target_day(build):
    """Callers publish the last 24 points, so those must be the target day."""
    inp = build.build_for_country("FR", "net_position", TARGET_DATE, as_of=AS_OF)
    day = inp["future_index"][-24:]

    expected = pd.date_range(pd.Timestamp(TARGET_DATE), periods=24, freq="h")
    assert day.equals(expected)


def test_future_covariates_match_the_horizon(build):
    """A covariate shorter than the horizon gets padded with its last value —
    silent, and wrong across a 42h gap."""
    inp = build.build_for_country("FR", "net_position", TARGET_DATE, as_of=AS_OF)
    for name, arr in (inp.get("future_covariates") or {}).items():
        assert len(arr) == inp["prediction_length"], name


def test_as_of_hides_later_observations(monkeypatch, tmp_path):
    """Data past `as_of` exists in the DB but a run at that moment had not seen
    it. Without this the backtest would score against leaked information."""
    db = tmp_path / "t.db"
    # Seed all the way to the nominal cutoff — more than the run could know.
    _seed_db(db, last_obs=NOMINAL_CUTOFF)
    ib = _builder(monkeypatch, db)

    bounded = ib.InputBuilder().build_for_country(
        "FR", "net_position", TARGET_DATE, as_of=AS_OF
    )
    assert bounded["prediction_length"] == GAP_HOURS + 24


def test_full_data_keeps_the_plain_24h_horizon(monkeypatch, tmp_path):
    """When observations do reach the nominal cutoff there is no gap to cross,
    and behaviour must collapse back to the original 24h day-ahead shape."""
    db = tmp_path / "t.db"
    _seed_db(db, last_obs=NOMINAL_CUTOFF)
    ib = _builder(monkeypatch, db)

    inp = ib.InputBuilder().build_for_country("FR", "net_position", TARGET_DATE)
    assert inp["prediction_length"] == 24
    assert inp["future_index"][0] == pd.Timestamp(TARGET_DATE)


def test_missing_target_data_raises_rather_than_forecasting_zeros(monkeypatch, tmp_path):
    db = tmp_path / "t.db"
    _seed_db(db)
    ib = _builder(monkeypatch, db)

    with pytest.raises(ValueError, match="No target data"):
        ib.InputBuilder().build_for_country("ZZ", "net_position", TARGET_DATE, as_of=AS_OF)


def test_context_guard_accepts_real_series_that_crosses_zero():
    """A zero-valued point is legitimate; only the whole series may be judged."""
    from src.chronos2.input_builder import _net_position_context_refusal_reasons

    series = pd.Series(np.tile([-2.0, 0.0, 2.0], 224))
    reasons = _net_position_context_refusal_reasons(
        series,
        NOMINAL_CUTOFF - pd.Timedelta(hours=26),
        NOMINAL_CUTOFF,
    )
    assert reasons == []


@pytest.mark.parametrize(
    ("series", "staleness_hours", "expected"),
    [
        (pd.Series(np.full(672, LEVEL)), 73, "stale_context=73h>72h"),
        (pd.Series(np.full(167, LEVEL)), 26, "thin_context=167<168_real_hours"),
        (pd.Series(np.linspace(-0.9, 0.9, 672)), 26,
         "degenerate_context=max_abs_0.9MW<1MW"),
    ],
)
def test_context_guard_records_countable_refusal_reason(
    series, staleness_hours, expected
):
    from src.chronos2.input_builder import _net_position_context_refusal_reasons

    reasons = _net_position_context_refusal_reasons(
        series,
        NOMINAL_CUTOFF - pd.Timedelta(hours=staleness_hours),
        NOMINAL_CUTOFF,
    )
    assert expected in reasons


def test_stale_historical_context_is_refused_before_alignment(monkeypatch, tmp_path):
    """Post-ABL-181 GR shape: real history exists, but it is far too old."""
    db = tmp_path / "t.db"
    historical_last_obs = NOMINAL_CUTOFF - pd.Timedelta(days=100)
    _seed_db(db, last_obs=historical_last_obs)
    ib = _builder(monkeypatch, db)

    with pytest.raises(ib.ContextRefusalError, match="stale_context=2400h>72h"):
        ib.InputBuilder().build_for_country(
            "FR", "net_position", TARGET_DATE, as_of=AS_OF
        )
