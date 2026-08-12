"""Correctness checks for the ABL-337 solar geometry helper.

The clamp built on this module deletes served numbers, so the geometry has to be
right for reasons other than "it looked plausible". These check it against
astronomy that is true independently of this code: solstice/equinox elevations,
and published sunrise/sunset clock times for two cities.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.solar_geometry import (
    NIGHT_ELEVATION_THRESHOLD_DEG,
    SOLAR_REPRESENTATIVE_POINTS,
    is_night_hour,
    max_sun_elevation_over_hour,
    sun_elevation_at_point_deg,
    sun_elevation_deg,
)


def _day_max_elevation(lat, lon, day):
    minutes = pd.date_range(day, periods=24 * 60, freq="min")
    return float(np.max(sun_elevation_at_point_deg(lat, lon, minutes)))


def _horizon_crossings(lat, lon, day, apparent_horizon_deg=-0.833):
    """Sunrise/sunset to the minute, at the apparent horizon (refraction plus
    solar semidiameter put the disc's edge there when the centre is -0.833)."""
    minutes = pd.date_range(day, periods=24 * 60, freq="min")
    up = sun_elevation_at_point_deg(lat, lon, minutes) > apparent_horizon_deg
    return minutes[np.argmax(up)], minutes[len(minutes) - 1 - np.argmax(up[::-1])]


@pytest.mark.parametrize("lat,lon", [(51.0, 11.2), (45.7, 2.5), (39.1, -4.1), (63.6, 24.6)])
def test_solstice_and_equinox_elevations_match_spherical_astronomy(lat, lon):
    # Noon elevation at the equinox is 90 - latitude, and the solstices sit one
    # axial tilt (23.44 deg) either side of it. True for any longitude.
    assert _day_max_elevation(lat, lon, "2026-03-20") == pytest.approx(90 - lat, abs=0.3)
    assert _day_max_elevation(lat, lon, "2026-06-21") == pytest.approx(90 - lat + 23.44, abs=0.3)
    assert _day_max_elevation(lat, lon, "2026-12-21") == pytest.approx(90 - lat - 23.44, abs=0.3)


def test_sunrise_and_sunset_match_published_times():
    # Berlin (52.520 N, 13.405 E): 2026-06-21 sunrise 04:43 / sunset 21:33 CEST
    # = 02:43 / 19:33 UTC; 2026-12-21 sunrise 08:15 / sunset 15:53 CET
    # = 07:15 / 14:53 UTC.
    rise, set_ = _horizon_crossings(52.520, 13.405, "2026-06-21")
    assert abs((rise - pd.Timestamp("2026-06-21 02:43")).total_seconds()) <= 120
    assert abs((set_ - pd.Timestamp("2026-06-21 19:33")).total_seconds()) <= 120

    rise, set_ = _horizon_crossings(52.520, 13.405, "2026-12-21")
    assert abs((rise - pd.Timestamp("2026-12-21 07:15")).total_seconds()) <= 120
    assert abs((set_ - pd.Timestamp("2026-12-21 14:53")).total_seconds()) <= 120

    # Madrid (40.4168 N, 3.7038 W): 2026-06-21 sunrise 06:44 / sunset 21:48 CEST
    # = 04:44 / 19:48 UTC. A negative longitude, and a different time zone
    # offset — a sign error in the hour angle shows up here and nowhere else.
    rise, set_ = _horizon_crossings(40.4168, -3.7038, "2026-06-21")
    assert abs((rise - pd.Timestamp("2026-06-21 04:44")).total_seconds()) <= 120
    assert abs((set_ - pd.Timestamp("2026-06-21 19:48")).total_seconds()) <= 120


def test_elevation_is_lowest_at_local_solar_midnight():
    minutes = pd.date_range("2026-01-15", periods=24 * 60, freq="min")
    elevation = sun_elevation_deg("DE", minutes)
    # DE's representative point is 11.167 E, so solar midnight is ~45 minutes
    # before 00:00 UTC — i.e. at the very end of the day.
    assert minutes[int(np.argmin(elevation))].hour == 23


def test_scalar_and_array_inputs_agree():
    hours = pd.date_range("2026-08-14", periods=24, freq="h")
    vectorized = sun_elevation_deg("FR", hours)
    assert isinstance(vectorized, np.ndarray) and vectorized.shape == (24,)
    for i, h in enumerate(hours):
        one = sun_elevation_deg("FR", h)
        assert isinstance(one, float)
        assert one == pytest.approx(vectorized[i], abs=1e-9)


def test_timezone_aware_input_is_converted_not_ignored():
    naive = sun_elevation_deg("BE", "2026-08-14 12:00:00")
    aware = sun_elevation_deg("BE", pd.Timestamp("2026-08-14 14:00:00", tz="Europe/Brussels"))
    assert aware == pytest.approx(naive, abs=1e-9)


def test_string_spellings_parse_identically():
    space = sun_elevation_deg("DE", "2026-08-14 05:00:00")
    tee = sun_elevation_deg("DE", "2026-08-14T05:00:00")
    offset = sun_elevation_deg("DE", "2026-08-14T06:00:00+01:00")
    assert tee == pytest.approx(space, abs=1e-9)
    assert offset == pytest.approx(space, abs=1e-9)


def test_hour_peak_covers_the_whole_hour_not_just_its_start():
    # 2026-08-14 03:00 UTC in DE: dark at the top of the hour, sun up before it
    # ends. Judging the hour by its first instant would call it night and zero a
    # real dawn hour.
    start = pd.Timestamp("2026-08-14 03:00:00")
    assert sun_elevation_deg("DE", start) < NIGHT_ELEVATION_THRESHOLD_DEG
    assert max_sun_elevation_over_hour("DE", start) > NIGHT_ELEVATION_THRESHOLD_DEG
    assert not is_night_hour("DE", start)


def test_hour_peak_is_never_below_the_instantaneous_elevation():
    hours = pd.date_range("2026-08-14", periods=24, freq="h")
    assert np.all(max_sun_elevation_over_hour("AT", hours) >= sun_elevation_deg("AT", hours) - 1e-9)


def test_deep_night_hours_are_masked_and_midday_is_not():
    hours = pd.date_range("2026-08-14", periods=24, freq="h")
    for country in ("AT", "BE", "DE", "FR"):
        night = np.asarray(is_night_hour(country, hours))
        # The hours ABL-335 measured the defect in: 00-02 and 22-23 UTC.
        assert night[[0, 1, 2, 22, 23]].all(), country
        # Midday is never masked, at any latitude here.
        assert not night[9:15].any(), country


def test_unknown_country_raises_rather_than_guessing():
    with pytest.raises(KeyError):
        sun_elevation_deg("XX", "2026-08-14 12:00:00")


def test_every_served_country_has_a_representative_point():
    # A country that reaches the clamp without a point loses the night mask
    # (see solar_clamp), so this must not silently drift as coverage grows.
    missing = [c for c in config.SUPPORTED_COUNTRIES if c not in SOLAR_REPRESENTATIVE_POINTS]
    assert missing == []


def test_representative_points_are_inside_europe():
    for country, (lat, lon) in SOLAR_REPRESENTATIVE_POINTS.items():
        assert 34.0 <= lat <= 71.0, country
        assert -12.0 <= lon <= 32.0, country
