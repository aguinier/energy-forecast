"""
Solar geometry: sun elevation for a (country, UTC timestamp) pair (ABL-337).

Nothing in this module reads the database or the model artifacts. It exists so
that the serving-path solar clamp (`src/solar_clamp.py`, ABL-337) and the
solar-geometry training feature (ABL-338) compute the *same* number from the
*same* representative point. Import it; do not write a second copy.

Representative point
--------------------
One point per country: the **capacity-weighted centroid of that country's solar
clusters** in the shared database's `weather_location` table
(`zone_type='solar'`, weighted by `weight`), snapshot 2026-08-12. That is the
fleet's centre of mass, not the country's geographic centre — a better answer to
"is the sun up where the panels are" than a land centroid, and it is the same
location family the weather features are already sampled at.

The table is inlined rather than queried per call so this stays a pure function
(ABL-338 needs it per training row, and the clamp needs it with no DB roundtrip).
Regenerate with `scripts/abl335_solar_night_probe.py --print-points` if
`weather_location` changes.

This is an approximation, and here is exactly what it costs. A single point
cannot represent a country that spans longitude: the eastern edge of a country
sees sunrise earlier than its fleet centroid, by roughly 4 minutes per degree of
longitude. For the widest countries here that is ~25 minutes (FR, ES, DE, PL).
Near the horizon the sun's elevation changes ~0.15-0.25 deg/min at these
latitudes, so 25 minutes of longitude spread is worth ~4-6 degrees of elevation.
`NIGHT_ELEVATION_THRESHOLD_DEG` carries that error explicitly as margin, and
`scripts/abl335_solar_night_probe.py --check-actuals` measures the remaining
headroom against every non-zero actual solar hour on record.

Algorithm
---------
The USNO/NOAA low-precision solar position algorithm (better than 0.01 deg for
1950-2050). Elevation is *geometric* — no atmospheric refraction term. Refraction
plus the sun's semidiameter make the disc appear to rise when its geometric
centre is still ~0.83 deg below the horizon; every threshold here sits far below
that, so the omission never shortens the day.
"""

from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

# Capacity-weighted centroid of each country's `weather_location` solar clusters.
# (lat, lon) in degrees, north/east positive. Snapshot 2026-08-12; the trailing
# comment is the summed cluster capacity the weighting came from.
SOLAR_REPRESENTATIVE_POINTS = {
    'AT': (47.743, 15.684),   # 858 MW
    'BE': (50.905, 4.277),    # 905 MW
    'BG': (42.683, 25.234),   # 3529 MW
    'CH': (46.999, 7.671),    # 97 MW
    'CZ': (49.607, 15.434),   # 2814 MW
    'DE': (50.996, 11.167),   # 38026 MW
    'EE': (58.760, 25.294),   # 1046 MW
    'ES': (39.125, -4.129),   # 36231 MW
    'FI': (63.568, 24.644),   # 219 MW
    'FR': (45.702, 2.544),    # 12895 MW
    'GR': (39.741, 22.545),   # 5978 MW
    'HR': (45.242, 16.741),   # 272 MW
    'HU': (47.235, 19.612),   # 4061 MW
    'IT': (41.923, 12.750),   # 5404 MW
    'LT': (55.289, 24.235),   # 872 MW
    'LV': (56.718, 24.086),   # 585 MW
    'NL': (52.309, 5.846),    # 6897 MW
    'NO': (59.996, 9.639),    # 36 MW
    'PL': (52.362, 18.251),   # 8926 MW
    'PT': (38.932, -8.207),   # 3551 MW
    'RO': (45.402, 24.635),   # 3613 MW
    'SE': (57.989, 15.329),   # 663 MW
    'SI': (46.128, 15.107),   # 94 MW
    'SK': (48.472, 19.584),   # 890 MW
}

# Below this geometric elevation the country's solar fleet is treated as dark.
#
# Civil twilight (-6 deg) is the physical argument: the disc is fully set, there
# is no direct beam, and global horizontal irradiance is order 1 W/m^2. -8 is
# where the measurement put it. Zeroing an hour that really generated is
# fabricating a number, which is worse than leaving a small twilight forecast
# alone, so the threshold was chosen from what the actuals actually show rather
# than from the convention:
#
#   threshold   hours with non-zero actual solar the mask would zero, and the
#               largest such actual (AT/BE/DE, full history, 2026-08-12)
#   -6.0 deg    AT 2 (4.0 MW)   BE 37 (0.1 MW)   DE 260 (18.7 MW)
#   -8.0 deg    AT 1 (4.0 MW)   BE 22 (0.1 MW)   DE 219 ( 3.6 MW)
#   -10.0 deg   AT 0            BE  3 (0.0 MW)   DE 204 ( 3.6 MW)
#
# -8 cuts the largest zeroed actual to 3.6 MW of a 38 GW DE fleet (0.009%) while
# still masking 00-02 and 22-23 UTC for all four countries on a served August
# day; -10 stops covering 02:00 UTC for AT and DE, which is one of the hours the
# defect shows up in. FR is excluded from the table above on evidence, not
# convenience: its actuals carry 137-440 MW at sun elevations down to -65 deg on
# 337 distinct days, which no physical threshold can honour (see the ABL-337
# comment and the follow-up filed for it).
#
# Rerun the measurement with `scripts/abl335_solar_night_probe.py --check-actuals`.
NIGHT_ELEVATION_THRESHOLD_DEG = -8.0

# Hourly forecast rows are labelled with the start of the hour they cover, so an
# hour is only dark if the sun is below the threshold for all of it. Sampled
# every 5 minutes: near the horizon the sun moves <=0.25 deg/min, so the sampling
# grid can miss at most ~1.2 deg of a crossing, well inside the 6 deg margin.
_HOUR_SAMPLE_MINUTES = 5

_J2000_UNIX_DAYS = 10957.5  # days from 1970-01-01T00:00Z to 2000-01-01T12:00Z


def _to_days_since_j2000(timestamp_utc) -> np.ndarray:
    """
    Convert UTC timestamp(s) to fractional days since the J2000.0 epoch.

    Naive timestamps are read as UTC (the convention everywhere in this repo);
    tz-aware ones are converted to UTC first.
    """
    idx = pd.DatetimeIndex(pd.Series(timestamp_utc).values) if isinstance(
        timestamp_utc, (list, tuple, np.ndarray, pd.Series, pd.Index)
    ) else pd.DatetimeIndex([pd.Timestamp(timestamp_utc)])

    if idx.tz is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)

    # Subtract-and-divide rather than a raw int64 view: DatetimeIndex resolution
    # is ns on pandas 2 and can be us on pandas 3, and a hardcoded 1e9 turns that
    # difference into a 1000x error in the hour angle.
    unix_seconds = (idx - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1)
    return np.asarray(unix_seconds, dtype=float) / 86400.0 - _J2000_UNIX_DAYS


def sun_elevation_deg(
    country_code: str,
    timestamp_utc: Union[datetime, pd.Timestamp, str, pd.Series, pd.DatetimeIndex, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Geometric sun elevation in degrees at a country's representative solar point.

    Args:
        country_code: ISO 2-letter code present in SOLAR_REPRESENTATIVE_POINTS.
        timestamp_utc: One UTC instant, or an array/Series/DatetimeIndex of them.
                       Naive timestamps are interpreted as UTC.

    Returns:
        float for a scalar input, np.ndarray for an array input. Positive is
        above the horizon.

    Raises:
        KeyError: if the country has no representative point. Deliberately loud —
                  silently falling back to some other latitude would be exactly
                  the confidently-wrong-number failure this module exists to stop.
    """
    if country_code not in SOLAR_REPRESENTATIVE_POINTS:
        raise KeyError(
            f"No solar representative point for {country_code!r}. "
            f"Known: {sorted(SOLAR_REPRESENTATIVE_POINTS)}"
        )
    lat_deg, lon_deg = SOLAR_REPRESENTATIVE_POINTS[country_code]
    return sun_elevation_at_point_deg(lat_deg, lon_deg, timestamp_utc)


def sun_elevation_at_point_deg(
    lat_deg: float,
    lon_deg: float,
    timestamp_utc: Union[datetime, pd.Timestamp, str, pd.Series, pd.DatetimeIndex, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Geometric sun elevation in degrees at an arbitrary point (north/east positive).

    The country-level entry point above is what the clamp uses; this exists for
    callers that already have a coordinate — e.g. a per-zone feature built off
    `weather_location` — and for testing against published sunrise times.
    """
    scalar_input = not isinstance(
        timestamp_utc, (list, tuple, np.ndarray, pd.Series, pd.Index)
    )
    n = _to_days_since_j2000(timestamp_utc)

    # USNO low-precision solar coordinates.
    mean_longitude_deg = 280.460 + 0.9856474 * n
    mean_anomaly_rad = np.radians(357.528 + 0.9856003 * n)
    ecliptic_longitude_rad = np.radians(
        mean_longitude_deg
        + 1.915 * np.sin(mean_anomaly_rad)
        + 0.020 * np.sin(2.0 * mean_anomaly_rad)
    )
    obliquity_rad = np.radians(23.439 - 0.0000004 * n)

    declination_rad = np.arcsin(np.sin(obliquity_rad) * np.sin(ecliptic_longitude_rad))
    right_ascension_deg = np.degrees(
        np.arctan2(
            np.cos(obliquity_rad) * np.sin(ecliptic_longitude_rad),
            np.cos(ecliptic_longitude_rad),
        )
    )

    # Greenwich mean sidereal time -> local hour angle.
    gmst_hours = np.mod(18.697374558 + 24.06570982441908 * n, 24.0)
    local_mean_sidereal_deg = gmst_hours * 15.0 + lon_deg
    hour_angle_rad = np.radians(
        np.mod(local_mean_sidereal_deg - right_ascension_deg + 180.0, 360.0) - 180.0
    )

    lat_rad = np.radians(lat_deg)
    elevation = np.degrees(
        np.arcsin(
            np.sin(lat_rad) * np.sin(declination_rad)
            + np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
        )
    )

    return float(elevation[0]) if scalar_input else np.asarray(elevation)


def max_sun_elevation_over_hour(
    country_code: str,
    hour_start_utc: Union[datetime, pd.Timestamp, str, pd.Series, pd.DatetimeIndex, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Highest sun elevation reached during the hour beginning at hour_start_utc.

    An hourly generation row covers [t, t+1h); using the elevation at t alone
    would call an hour "night" when the sun rises 50 minutes into it.
    """
    scalar_input = not isinstance(
        hour_start_utc, (list, tuple, np.ndarray, pd.Series, pd.Index)
    )
    starts = pd.DatetimeIndex(
        [pd.Timestamp(hour_start_utc)] if scalar_input else pd.Series(hour_start_utc).values
    )

    offsets = np.arange(0, 60 + _HOUR_SAMPLE_MINUTES, _HOUR_SAMPLE_MINUTES)
    per_offset = np.stack([
        np.atleast_1d(sun_elevation_deg(country_code, starts + pd.Timedelta(minutes=int(m))))
        for m in offsets
    ])
    peak = per_offset.max(axis=0)

    return float(peak[0]) if scalar_input else peak


def is_night_hour(
    country_code: str,
    hour_start_utc: Union[datetime, pd.Timestamp, str, pd.Series, pd.DatetimeIndex, np.ndarray],
    threshold_deg: float = NIGHT_ELEVATION_THRESHOLD_DEG,
) -> Union[bool, np.ndarray]:
    """
    True when the sun stays below threshold_deg for the whole hour starting at
    hour_start_utc — i.e. when the country's solar fleet cannot have produced.
    """
    peak = max_sun_elevation_over_hour(country_code, hour_start_utc)
    if isinstance(peak, float):
        return bool(peak < threshold_deg)
    return np.asarray(peak) < threshold_deg
