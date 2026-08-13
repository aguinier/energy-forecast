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


class UndeclaredNightGenerationError(RuntimeError):
    """A country reached a night-sensitive mechanism with no registered fact.

    Deliberately **not** a `KeyError`, and not a `LookupError` either.
    `solar_clamp` catches `KeyError` around the geometry lookup and degrades to
    the non-negativity floor alone; this abort must not be degradable by that
    handler or by any other, because the default it refuses to supply is the
    destructive one.
    """


# Can this country's solar fleet physically generate while the sun is below
# `NIGHT_ELEVATION_THRESHOLD_DEG`?
#
# This registers a **physical fact about the fleet**, not a policy about what to
# do with it. Two mechanisms read this one fact and apply their own policy on
# top, which is what stops them disagreeing about which hours are dark-but-real:
#
#   - `solar_clamp.clamp_solar_forecasts` exempts a True country from the hard
#     night zero. The `max(0, prediction)` floor still applies everywhere, and
#     that half of the clamp is not per-country — but *not* because negative
#     solar is impossible. It is not: see "Why the non-negativity floor is
#     fleet-wide" below the table. The floor rests on a measured bound instead.
#   - `solar_features.exclude_impossible_night_rows` refuses to run at all for a
#     True country: that rule's warrant is "the sun says this row cannot exist",
#     which is not a claim anyone can make about a fleet that dispatches after
#     dark.
#
# One fact and two policies rather than one shared value, and BG is the reason.
# BG's overnight floor is genuine contamination, so the clamp must fire for it
# (False here, and it does) — yet ABL-403 measured the fit-side exclusion rule
# costing BG 1.4-1.9pp of gate WAPE, so that rule stays off for it. A single
# value cannot express both settings; a single *fact* can.
#
# **There is no default and there must not be one.** A country reaching either
# mechanism without an entry raises `UndeclaredNightGenerationError`. The silent
# direction here is the destructive one: an unregistered ES-like country would
# inherit "cannot generate at night", have real MW deleted on the serving path,
# and `forecast_clamp_log` would record the deletion as a successful correction.
# `DEFAULT_FIT_RULES` (`scripts/evaluate_solar_retrain.py:342`) is the same shape
# of table *with* a default, and its own comment records what an absence there
# costs: a merge that misses an entry is textually clean and stays green.
#
# Completeness against `SOLAR_REPRESENTATIVE_POINTS` and
# `config.SUPPORTED_COUNTRIES` is asserted by
# `tests/test_night_generation_registration.py` rather than at import. The
# `check_registration_tables` comment in `scripts/evaluate_solar_retrain.py`
# records what an import-time abort costs every branch already in flight — it
# takes `--help` and the whole suite down on a textually clean merge — and a test
# failure gives the same "a merge that misses an entry does not stay green"
# property without that tax.
#
# Evidence: ABL-396 screened all 24 against `energy_generation` over ABL-348's
# registered fit and gate windows (`reports/abl_396_solar_night_floor_screen.md`);
# ABL-411 verified ES against Red Eléctrica's own published split
# (`reports/abl_411_es_csp_verification.md`).
NIGHT_GENERATION_POSSIBLE = {
    # The only True in the table, and it is measured rather than reasoned.
    # Spain operates ~2.3 GW of concentrated solar power with molten-salt
    # storage — the only large such fleet in Europe — which charges by day and
    # dispatches after sunset. ENTSO-E folds CSP and PV into one production type
    # (B16), so that output lands in `solar_mw` with nothing to distinguish it.
    #
    # ABL-411 fetched Red Eléctrica's own `solar fotovoltaica` / `solar térmica`
    # split and compared it to the replica over 3,196 night hours: the two
    # together account for **98.55%** of the MW booked for ES when the sun is
    # down, at an hourly MAE of **5.55 MW** against a **263.5 MW** mean night
    # level, and **80.1%** of the annual night energy is `solar térmica`. So the
    # night mask would delete roughly 263.5 MW of real ES generation every night.
    #
    # ABL-411's one refinement, recorded here because it is deliberately *not*
    # modelled: the remaining 18.5% is a TSO-side estimation artifact in REE's
    # *PV* series (44-59 MW at sun elevations of -40 to -49 deg), mirrored
    # faithfully by ENTSO-E and by our ingest. Not physical, but not our defect
    # either. This entry is one bit about the fleet, not a per-hour attribution.
    'ES': True,

    # Everything below is False on ABL-396's screen. The four that are not
    # trivially zero are called out; the rest sit at or under 0.28pp of gate
    # energy at night, most of them at 0.000.
    'AT': False,
    'BE': False,
    # BG carries the largest night floor in the fleet — 4.98pp of gate energy,
    # 85.2% of night hours above 1 MW, up to 1,087.9 MW — and it is still False,
    # because size is not the predicate. ABL-396 §3 ran ES's own discriminator on
    # it: the within-month detrended correlation between a day's daylight energy
    # and that same night's energy is **+0.084** for BG against **+0.515** for
    # ES. Storage dispatch tracks the charge; contamination has no reason to know
    # how sunny that particular day was. BG also operates no CSP fleet.
    'BG': False,
    'CH': False,
    'CZ': False,
    'DE': False,
    # EE is the third-largest floor (0.72pp, 79.1% of night hours above 1 MW,
    # max 76.0 MW) and fails the same discriminator at +0.084. Real, small,
    # persistent — and not generation.
    'EE': False,
    'FI': False,
    # FR is the ABL-337 threshold comment's own example of an actuals defect:
    # 137-440 MW at sun elevations down to -65 deg on 337 distinct days, which no
    # physical threshold can honour. France operates no CSP fleet.
    'FR': False,
    'GR': False,
    'HR': False,
    'HU': False,
    'IT': False,
    'LT': False,
    'LV': False,
    # NL's night series is uniformly *negative* — 1,544 of 1,544 night hours
    # across both ABL-348 windows, -1.47 to -0.12 MW. Not generation at all, and
    # the non-negativity floor rather than the night mask is what answers it.
    # That negative is admissible under A75's net-of-consumption semantics
    # rather than demonstrably a defect, and it is why the floor is justified by
    # a measured bound rather than by physics — see "Why the non-negativity
    # floor is fleet-wide" below the table.
    'NL': False,
    'NO': False,
    'PL': False,
    'PT': False,
    'RO': False,
    'SE': False,
    'SI': False,
    'SK': False,
}


# --- Why the non-negativity floor is fleet-wide -----------------------------
#
# `solar_clamp`'s `max(0, prediction)` half is not per-country. The reason is a
# measured bound, **not** the physical claim that negative solar is impossible.
# That claim is false, and this file used to make it ninety lines above the NL
# entry that disproves it.
#
# `energy_generation` is the A75 document, **net of consumption**. Overnight
# auxiliary and inverter draw nets against zero output, so a fleet can book a
# small negative and be correctly reported. NL does, structurally:
#
#   window                      cc  instants    neg     %    min MW  mean level
#   ABL-348 fit+gate            NL    20,064  8,757  43.6%    -1.62      65.4
#   ABL-348 gate only           NL     2,976  1,045  35.1%    -1.05      65.9
#   calendar 2025               NL    35,040 17,373  49.6%    -1.59      56.2
#   calendar 2025               IT    35,040    383   1.1%    -1.00   3,877.5
#
# NL is the only country booking *any* negative instant over the registered
# fit+gate window; IT is the only other one over calendar 2025. It is a flat
# overnight floor rather than scatter — NL is negative at 100% of instants from
# 20Z through 02Z at a -1.0 to -1.1 MW mean, 80.7% at 03Z, tapering across dawn
# and dusk (04Z 46.2%, 19Z 71.5%) and clean 09Z-14Z.
#
# **`energy_renewable` cannot corroborate that, and an earlier revision of this
# block wrongly said it could.** It is not the pre-netting side of the fetch.
# Over the ABL-348 gate window `energy_renewable.solar_mw` is the *zero-clipped
# copy* of `energy_generation.solar_mw` — `ren == max(0, gen)` to 1e-9 at 100.0%
# of instants in 28 of 32 countries and 99.0% for NL (2,946/2,976), with NL
# flipping into that regime on a single day: 41.7% on 2026-07-01, 99.0% on
# 2026-07-02. `db.py`'s ABL-321 note reaches the same place from the other side
# ("the gate truth is byte-identical between the two tables for 9 of 10 pairs").
# So `ren - gen` over that window *is* `max(0, -gen)` — the floor's own
# correction — and asking whether it is non-negative is asking the floor about
# itself. On the 1,045 gate-window NL instants with `gen < 0`, `ren` is exactly
# 0.0 on 1,036 of them, so the subtraction returns `-gen` and nothing else.
#
# Outside that window the two series are not related that way and the difference
# is not a consumption series either: over the fit+gate window `ren - gen` goes
# negative at 8,668 of 19,948 NL instants, to -185.84 MW at midday, which no
# pre-netting series can do. Only 305 of those carry ABL-188's zero-fill, so it
# is a level difference — NL daytime `ren/gen` runs 0.42-0.64 across 2026-01..06
# against 0.98 in July and 1.00 in August — not a fill artifact. Fleet-wide the
# same subtraction goes negative 645 times *inside* the gate window (HU 480,
# PT 150, GR 15, min -239.0 MW).
#
# What survives is weaker than a measurement and is stated here as such: A75's
# net-of-consumption semantics make a small negative **admissible**, and NL's is
# structurally stable rather than sporadic, so metered overnight draw is the
# reasonable reading of it. The replica holds no independent series that isolates
# the cause. Do not reach for `energy_renewable` to close that gap — its ABL-188
# zero-fill and its own FR night defect (137-440 MW at sun elevations to -65 deg
# on 337 distinct days, CLAUDE.md) are two further reasons it cannot arbitrate a
# sign.
#
# So the floor does erase reported MW. It is justified by how little:
#
#   1. **Bounded where it matters.** Over the ABL-348 registered window the
#      deepest excursion anywhere in the fleet is -1.62 MW, below any level a
#      forecast resolves. That is a direct read of `energy_generation` — it never
#      went through the join above, so the correction to `energy_renewable` does
#      not move it. NL's per-year minimum is -1.30/-1.43/-1.74/-1.59/-1.62
#      MW for 2021/22/24/25/26 — the structural floor is stable and ~1 MW deep
#      across five years. 2023 is the one exception and it is the outlier
#      discussed below, not a different floor.
#   2. **The floor barely runs on these rows.** `solar_clamp` applies
#      `np.where(night, 0.0, np.maximum(original, 0.0))`, so the floor is the
#      *non-night* branch, and NL's negatives concentrate 19Z-04Z where the
#      night mask already zeroes them. The residue the floor actually sees is
#      ~700 instants at 15Z-18Z, at a -0.5 to -1.1 MW mean.
#   3. **Nothing is exposed today.** NL is in no registered solar scope and
#      `forecasts` holds solar rows for AT/BE/DE/FR only. IT is in tranche 2c at
#      0.03% of its own level.
#
# One honest exception to (1), because "everywhere measured" would be the same
# shape of absolute this table exists to retire. Screening *all* history
# (2021-01-01 on, 2,834,612 non-null instants across the fleet) still finds
# negatives in NL and IT alone — 288,582 instants between them — but five of
# those go deeper than -2 MW: NL -57.36 MW at 2023-02-14 12:15, midday, an
# isolated spike at ~100% of NL's own level; NL -2.01/-2.02/-2.03 across three
# consecutive quarter-hours on 2023-08-11 evening; and IT -6.00 MW at
# 2021-11-19 18:00. All five are outside every registered window and none
# recurs — the deepest is a lone 2023 instant against a stable ~1 MW floor.
# The bound in (1) is a statement about the structural floor over the registered
# window, not about every row ever ingested.
#
# **Tripwire.** If NL ever enters an ABL-316 tranche, re-read this before
# assuming the floor is free: at NL's 65 MW mean level a 1.1 MW floor is 1.7% of
# level, which is no longer obviously nothing.
#
# Evidence: ABL-411 follow-up by the Forecasting Scientist, reproduced against
# the replica `C:\Code\able\data\energy_dashboard.db` (`mode=ro`, source
# `energy_generation`, direct read of actuals — nothing fitted). The full-history
# tail in the paragraph above is this repo's own read on the same replica. The
# `energy_renewable` correction is too: the Scientist's ABL-425 addendum offered
# the gate-window join as the measured cause and a 1.049 MW bound on the floor's
# cost. Every figure in it reproduces exactly, and the reproduction is what shows
# the comparator is the clipped copy — 1.049 MW is `-min(gen)` over that window,
# i.e. this block's own -1.05 with the sign flipped, not a second observation.


def night_generation_possible(country_code: str) -> bool:
    """
    Whether this country's solar fleet can generate below the night threshold.

    Args:
        country_code: ISO 2-letter code with an entry in
            `NIGHT_GENERATION_POSSIBLE`.

    Returns:
        The registered physical fact. True means the fleet demonstrably produces
        after dark (ES, on ABL-411's measurement) and must not be night-zeroed.

    Raises:
        UndeclaredNightGenerationError: if the country has no entry. There is no
            default — see the comment over the table for why the absence has to
            be fatal rather than resolvable.
    """
    try:
        return NIGHT_GENERATION_POSSIBLE[country_code]
    except KeyError:
        raise UndeclaredNightGenerationError(
            f"{country_code!r} has no entry in "
            f"solar_geometry.NIGHT_GENERATION_POSSIBLE, and there is no default. "
            f"Register whether its solar fleet can physically generate below "
            f"{NIGHT_ELEVATION_THRESHOLD_DEG:g} deg before it is served or "
            f"fitted; defaulting to 'cannot' would delete real generation for a "
            f"country like ES and log the deletion as a correction (ABL-425). "
            f"Declared: {sorted(NIGHT_GENERATION_POSSIBLE)}"
        ) from None


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
    hour_start_utc.

    That is a statement about the sun and nothing else. It is **not** the claim
    that the fleet cannot have produced (ABL-425): ES dispatches ~263.5 MW of
    stored CSP through hours this returns True for. Whether a dark hour's output
    is impossible is the separate registered fact `NIGHT_GENERATION_POSSIBLE`,
    which the two night-sensitive mechanisms consult on top of this predicate —
    `solar_clamp.clamp_solar_forecasts` and
    `solar_features.exclude_impossible_night_rows`. Every other caller (band
    splits, night/daylight reporting) wants the geometry alone and should keep
    calling this unconditionally.
    """
    peak = max_sun_elevation_over_hour(country_code, hour_start_utc)
    if isinstance(peak, float):
        return bool(peak < threshold_deg)
    return np.asarray(peak) < threshold_deg
