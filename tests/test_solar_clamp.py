"""Correctness checks for the ABL-337 serving-path solar clamp."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solar_clamp import ZEROED_NIGHT_MW_THRESHOLD, clamp_solar_forecasts, solar_row_mask
from src.solar_geometry import UndeclaredNightGenerationError

# 2026-08-14 UTC at DE's representative point: 00-02 and 20-23 are night,
# 04-18 are day, 03 and 19 straddle dawn/dusk (see test_solar_geometry).
NIGHT_HOUR = "2026-08-14 01:00:00"
DAY_HOUR = "2026-08-14 12:00:00"

# 2026-08-14 UTC at ES's representative point (39.125, -4.129): 01:00 is well
# below -8 deg, and 12:00 is the middle of the day. Same instants as above, but
# asserted separately because the two countries' terminators differ.
ES_NIGHT_HOUR = "2026-08-14 01:00:00"


def _frame(rows):
    """rows: (country, forecast_type, renewable_type, target, value, model)."""
    return pd.DataFrame([
        {
            "country_code": r[0],
            "forecast_type": r[1],
            "renewable_type": r[2],
            "target_timestamp_utc": pd.Timestamp(r[3]),
            "generated_at": pd.Timestamp("2026-08-12 08:00:00"),
            "horizon_hours": 48,
            "forecast_value": r[4],
            "model_name": r[5],
        }
        for r in rows
    ])


def test_night_hours_are_zeroed_and_negatives_floored():
    df = _frame([
        ("DE", "solar", "solar", NIGHT_HOUR, 231.0, "catboost"),   # ABL-335's DE floor
        ("DE", "solar", "solar", DAY_HOUR, 18000.0, "catboost"),   # real midday output
        ("DE", "solar", "solar", "2026-08-14 05:00:00", -41.2, "catboost"),
    ])
    out, stats = clamp_solar_forecasts(df)

    assert list(out["forecast_value"]) == [0.0, 18000.0, 0.0]
    assert len(stats) == 1
    s = stats[0]
    assert (s.country_code, s.model_name) == ("DE", "catboost")
    assert s.rows_total == 3
    assert s.hours_zeroed_night == 1
    assert s.hours_raised_floor == 1
    assert s.mw_removed_night == pytest.approx(231.0)
    assert s.mw_added_floor == pytest.approx(41.2)
    assert s.mw_removed_total == pytest.approx(231.0 - 41.2)
    assert s.min_forecast_mw == pytest.approx(-41.2)
    assert s.max_night_forecast_mw == pytest.approx(231.0)


def test_daylight_positive_forecasts_are_never_touched():
    values = [0.0, 1e-6, 12.5, 20762.6]
    df = _frame([("FR", "solar", "solar", DAY_HOUR, v, "catboost") for v in values])
    out, stats = clamp_solar_forecasts(df)

    assert list(out["forecast_value"]) == values
    assert stats[0].hours_zeroed_night == 0
    assert stats[0].hours_raised_floor == 0
    assert stats[0].mw_removed_total == pytest.approx(0.0)


def test_a_night_zero_is_not_counted_as_a_zeroed_hour():
    # Already correct at night: nothing was removed, so nothing is reported.
    df = _frame([("BE", "solar", "solar", NIGHT_HOUR, 0.0, "catboost")])
    out, stats = clamp_solar_forecasts(df)

    assert list(out["forecast_value"]) == [0.0]
    assert stats[0].hours_zeroed_night == 0
    assert stats[0].mw_removed_total == pytest.approx(0.0)


def test_near_zero_night_prediction_below_threshold_is_not_counted():
    # ABL-377: log-link models emit exp(margin) which is never exactly 0.0.
    # A well-fitted model still predicts ~0.006 MW/hour at night.  The clamp
    # zeros the row (mw_removed_night still accumulates), but hours_zeroed_night
    # should not count it because the magnitude is below ZEROED_NIGHT_MW_THRESHOLD.
    tiny = ZEROED_NIGHT_MW_THRESHOLD * 0.5   # clearly sub-threshold
    df = _frame([("DE", "solar", "solar", NIGHT_HOUR, tiny, "catboost")])
    out, stats = clamp_solar_forecasts(df)

    assert list(out["forecast_value"]) == [0.0]          # row still zeroed
    s = stats[0]
    assert s.hours_zeroed_night == 0                      # but NOT counted
    assert s.mw_removed_night == pytest.approx(tiny)     # MW still recorded
    assert s.zeroed_night_mw_threshold == pytest.approx(ZEROED_NIGHT_MW_THRESHOLD)

    # Just above the threshold: must count.
    df2 = _frame([("DE", "solar", "solar", NIGHT_HOUR, ZEROED_NIGHT_MW_THRESHOLD + 0.01, "catboost")])
    _, stats2 = clamp_solar_forecasts(df2)
    assert stats2[0].hours_zeroed_night == 1


def test_negative_at_night_counts_once_as_a_night_hour():
    # -17 MW at 01:00 is both impossible signs at once. It is night-masked, and
    # the MW it "removes" is negative — the clamp added 17 MW to reach zero.
    df = _frame([("AT", "solar", "solar", NIGHT_HOUR, -17.2, "xgboost")])
    out, stats = clamp_solar_forecasts(df)

    assert list(out["forecast_value"]) == [0.0]
    s = stats[0]
    assert s.hours_zeroed_night == 1
    assert s.hours_raised_floor == 0          # not double-counted
    assert s.mw_removed_night == pytest.approx(-17.2)
    assert s.mw_removed_total == pytest.approx(-17.2)


def test_only_solar_rows_are_clamped():
    df = _frame([
        ("DE", "wind_onshore", "wind_onshore", NIGHT_HOUR, 4200.0, "catboost"),
        ("DE", "load", None, NIGHT_HOUR, 45000.0, "xgboost"),
        ("DE", "price", None, NIGHT_HOUR, -12.5, "xgboost"),      # negative prices are real
        ("DE", "solar", "solar", NIGHT_HOUR, 231.0, "catboost"),
    ])
    out, stats = clamp_solar_forecasts(df)

    assert list(out["forecast_value"]) == [4200.0, 45000.0, -12.5, 0.0]
    assert [s.rows_total for s in stats] == [1]


def test_solar_rows_with_a_null_renewable_type_are_still_clamped():
    # 6,888 stored solar rows spell it this way (measured 2026-08-12).
    df = _frame([("DE", "solar", None, NIGHT_HOUR, 231.0, "catboost")])
    assert solar_row_mask(df).tolist() == [True]
    out, _ = clamp_solar_forecasts(df)
    assert list(out["forecast_value"]) == [0.0]


def test_frame_without_a_renewable_type_column_still_clamps_solar():
    df = _frame([("DE", "solar", None, NIGHT_HOUR, 231.0, "catboost")]).drop(columns=["renewable_type"])
    out, stats = clamp_solar_forecasts(df)
    assert list(out["forecast_value"]) == [0.0]
    assert stats[0].hours_zeroed_night == 1


def test_countries_and_models_are_reported_separately():
    df = _frame([
        ("AT", "solar", "solar", NIGHT_HOUR, -17.2, "xgboost"),
        ("BE", "solar", "solar", NIGHT_HOUR, 1879.2, "catboost"),
        ("BE", "solar", "solar", NIGHT_HOUR, 12.0, "tso_raw"),
    ])
    _, stats = clamp_solar_forecasts(df)
    assert {(s.country_code, s.model_name) for s in stats} == {
        ("AT", "xgboost"), ("BE", "catboost"), ("BE", "tso_raw")
    }
    assert all(s.rows_total == 1 for s in stats)


def test_timestamp_spellings_are_all_understood():
    # Space separator, 'T' separator, and the '+01:00' offset form — all three
    # exist in this database (ABL-211/ABL-324). 02:00+01:00 is 01:00 UTC, night.
    df = _frame([("DE", "solar", "solar", NIGHT_HOUR, 100.0, "catboost")])
    df["target_timestamp_utc"] = ["2026-08-14 01:00:00"]
    out_space, _ = clamp_solar_forecasts(df)

    df["target_timestamp_utc"] = ["2026-08-14T01:00:00"]
    out_tee, _ = clamp_solar_forecasts(df)

    df["target_timestamp_utc"] = ["2026-08-14T02:00:00+01:00"]
    out_offset, _ = clamp_solar_forecasts(df)

    assert list(out_space["forecast_value"]) == [0.0]
    assert list(out_tee["forecast_value"]) == [0.0]
    assert list(out_offset["forecast_value"]) == [0.0]


def test_an_undeclared_country_aborts_the_save_rather_than_being_night_zeroed():
    # ABL-425. Before the registration this fell through to the fleet-wide hard
    # zero. An undeclared country is one we have not established the physics
    # for, and the silent answer ("cannot generate at night") is the destructive
    # one — so the clamp raises and `save_forecasts` writes nothing, rather than
    # deleting MW and logging the deletion as a correction.
    df = _frame([("XX", "solar", "solar", NIGHT_HOUR, 231.0, "catboost")])
    with pytest.raises(UndeclaredNightGenerationError):
        clamp_solar_forecasts(df)


def test_a_declared_country_without_a_representative_point_keeps_the_floor_only(caplog, monkeypatch):
    # The ABL-337 degradation survives ABL-425: a country that is declared but
    # cannot be placed must not be masked with someone else's latitude, and must
    # not take the whole save down with it. The two states are distinguishable
    # afterwards — night_mask_applied is False for both, night_generation_possible
    # tells them apart.
    from src import solar_geometry
    monkeypatch.setitem(solar_geometry.NIGHT_GENERATION_POSSIBLE, "XX", False)

    df = _frame([
        ("XX", "solar", "solar", NIGHT_HOUR, 231.0, "catboost"),
        ("XX", "solar", "solar", DAY_HOUR, -5.0, "catboost"),
    ])
    with caplog.at_level("ERROR"):
        out, stats = clamp_solar_forecasts(df)

    assert list(out["forecast_value"]) == [231.0, 0.0]
    assert stats[0].hours_zeroed_night == 0
    assert stats[0].hours_raised_floor == 1
    assert stats[0].night_generation_possible is False
    assert stats[0].night_mask_applied is False
    assert any("XX" in r.message for r in caplog.records)


def test_es_night_generation_survives_the_clamp_while_de_is_zeroed():
    # The ABL-425 headline, both halves in one frame on the same instant. ES
    # runs ~2.3 GW of CSP with thermal storage and ABL-411 measured a 263.5 MW
    # mean night level that is real; DE's night floor is the ABL-335 defect.
    df = _frame([
        ("ES", "solar", "solar", ES_NIGHT_HOUR, 515.5, "catboost"),
        ("DE", "solar", "solar", NIGHT_HOUR, 231.0, "catboost"),
    ])
    out, stats = clamp_solar_forecasts(df)

    assert list(out["forecast_value"]) == [515.5, 0.0]

    by_country = {s.country_code: s for s in stats}
    assert by_country["ES"].night_generation_possible is True
    assert by_country["ES"].night_mask_applied is False
    assert by_country["ES"].hours_zeroed_night == 0
    assert by_country["ES"].mw_removed_night == pytest.approx(0.0)
    assert by_country["ES"].mw_removed_total == pytest.approx(0.0)

    assert by_country["DE"].night_generation_possible is False
    assert by_country["DE"].night_mask_applied is True
    assert by_country["DE"].hours_zeroed_night == 1
    assert by_country["DE"].mw_removed_night == pytest.approx(231.0)


def test_an_exempt_country_is_visibly_exempt_in_the_telemetry():
    # Item 5: "0 night hours zeroed" on its own cannot distinguish an exemption
    # from a fit that already returns nothing at night. The night hours are
    # still counted and the peak still reported, so the instrument shows what
    # the clamp let through rather than going quiet.
    df = _frame([
        ("ES", "solar", "solar", ES_NIGHT_HOUR, 663.0, "catboost"),
        ("ES", "solar", "solar", "2026-08-14 02:00:00", 484.0, "catboost"),
        ("ES", "solar", "solar", DAY_HOUR, 12000.0, "catboost"),
    ])
    out, stats = clamp_solar_forecasts(df)

    assert list(out["forecast_value"]) == [663.0, 484.0, 12000.0]
    s = stats[0]
    assert s.rows_total == 3
    assert s.night_hours == 2
    assert s.hours_zeroed_night == 0
    assert s.max_night_forecast_mw == pytest.approx(663.0)


def test_an_exempt_country_still_gets_the_non_negativity_floor_at_night():
    # The floor is not part of the exemption, and not because negative solar is
    # impossible — NL books a real -1.1 MW overnight floor net of consumption
    # (see solar_geometry, "Why the non-negativity floor is fleet-wide"). It is
    # fleet-wide because that excursion is ~1 MW and a CSP country's is not
    # different in kind. A negative night row is raised to zero and counted as a
    # raised hour rather than a zeroed night hour.
    df = _frame([("ES", "solar", "solar", ES_NIGHT_HOUR, -42.0, "catboost")])
    out, stats = clamp_solar_forecasts(df)

    assert list(out["forecast_value"]) == [0.0]
    s = stats[0]
    assert s.hours_zeroed_night == 0
    assert s.hours_raised_floor == 1
    assert s.mw_added_floor == pytest.approx(42.0)
    assert s.mw_removed_total == pytest.approx(-42.0)
    assert s.night_hours == 1


def test_input_frame_is_not_mutated():
    df = _frame([("DE", "solar", "solar", NIGHT_HOUR, 231.0, "catboost")])
    before = df["forecast_value"].copy()
    clamp_solar_forecasts(df)
    assert df["forecast_value"].equals(before)


def test_empty_and_solar_free_frames_are_returned_untouched():
    empty = pd.DataFrame(columns=["country_code", "forecast_type", "forecast_value"])
    out, stats = clamp_solar_forecasts(empty)
    assert out is empty and stats == []

    load_only = _frame([("DE", "load", None, NIGHT_HOUR, 45000.0, "xgboost")])
    out, stats = clamp_solar_forecasts(load_only)
    assert out is load_only and stats == []


def test_threshold_is_configurable_per_call():
    # 2026-08-14 03:00 UTC in DE peaks at about -0.9 deg: dawn under the shipped
    # -8 threshold, night under a 0 deg one.
    df = _frame([("DE", "solar", "solar", "2026-08-14 03:00:00", 50.0, "catboost")])
    assert list(clamp_solar_forecasts(df)[0]["forecast_value"]) == [50.0]
    assert list(clamp_solar_forecasts(df, threshold_deg=0.0)[0]["forecast_value"]) == [0.0]


def test_clamped_output_has_no_negative_or_night_solar():
    # The property the whole module exists for, over a full served day.
    hours = pd.date_range("2026-08-14", periods=24, freq="h")
    rng = np.random.default_rng(0)
    rows = [("DE", "solar", "solar", h, float(v), "catboost")
            for h, v in zip(hours, rng.normal(200, 400, len(hours)))]
    out, stats = clamp_solar_forecasts(_frame(rows))

    values = out["forecast_value"].to_numpy()
    assert (values >= 0).all()
    from src.solar_geometry import is_night_hour
    assert (values[np.asarray(is_night_hour("DE", hours))] == 0).all()
    assert stats[0].rows_total == 24
