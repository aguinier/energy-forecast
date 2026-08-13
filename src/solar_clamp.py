"""
Serving-path clamp for solar forecasts (ABL-337).

ABL-335 measured what the models emit and nothing corrects: 21,582 of 124,468
stored solar rows are negative (17.3%), and 16,574 of 31,084 stored night hours
read above 1 MW (53%) — DE at a 155-231 MW floor through local midnight. A solar
fleet cannot generate a negative number, and *most* fleets cannot generate at
night. This module enforces both physical facts on the way to the `forecasts`
table:

  - sun below `solar_geometry.NIGHT_ELEVATION_THRESHOLD_DEG` for the whole hour
    -> hard zero, **unless the country is registered as able to generate after
    dark** (`solar_geometry.NIGHT_GENERATION_POSSIBLE`)
  - otherwise -> max(0, prediction)

The exemption is ABL-425, and it exists because the fleet-wide premise was
measurably false for one country. ES runs ~2.3 GW of concentrated solar power
with thermal storage; ABL-411 checked Red Eléctrica's own PV/CSP split against
the replica over 3,196 night hours and it accounts for 98.55% of the MW booked
for ES when the sun is down, at a 263.5 MW mean night level. Clamping ES would
delete that every night and record it here as a successful correction. The
non-negativity floor is **not** part of the exemption: it applies everywhere,
because negative solar is impossible in every country.

There is no default in that table. A country reaching this clamp undeclared
raises `UndeclaredNightGenerationError` rather than inheriting "cannot generate
at night", which for an ES-like country is the destructive answer.

Know what that abort costs before you widen the fleet. `save_forecasts` runs
this before its first `INSERT`, and `scripts/forecast_daily.py:504` concatenates
every country and every forecast type into one frame, so one undeclared solar
country loses the whole batch — load and price for everyone else included. That
is the direction we want (a failed run is visible; deleted generation is not),
and it cannot fire today: every `config.SUPPORTED_COUNTRIES` entry is declared,
asserted by `tests/test_night_generation_registration.py`. Adding a country to
the fleet means adding it to that table in the same commit.

It is applied to `renewable_type='solar'` rows only, to **new rows only**, from
`db.save_forecasts()` — the choke point every serving write goes through
(`scripts/forecast_daily.py`, `src/tso_correction_forecaster.py`). Stored history
is not rewritten and no `UPDATE` is issued; the vintage archive stays a faithful
record of what the models actually said. `scripts/forecast_challengers.py` and
`src/chronos_forecaster.py` write to `forecasts` without going through
`save_forecasts`, but neither emits solar (both are net-position paths) — if one
ever does, it needs this call too.

This is a guard, not a fix. The fit defect it covers is ABL-338's, and the point
of `SolarClampStats` is that the cover-up is measurable: every run records how
many hours it zeroed and how much MW it removed, per country, so the fit can be
scored on how little the clamp has left to do. Same reason `src/db.py` writes
those stats to `forecast_clamp_log` in the same database as the clamped rows —
the Forecasting Scientist can read the instrument without running anything.

Precedent: `src/tso_correction_forecaster.py:318` already floors the TSO
correction path at zero. This restores the same invariant on the path that lacks
it.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .solar_geometry import (
    NIGHT_ELEVATION_THRESHOLD_DEG,
    is_night_hour,
    night_generation_possible,
)

logger = logging.getLogger('energy_forecast')

# MW magnitude below which a night prediction is considered already-zero for the
# purposes of hours_zeroed_night. Log-link models predict exp(margin), which is
# never exactly 0.0 — a well-fitted model still emits ~0.006 MW/hour at night.
# Without this tolerance the counter trips on every night hour regardless of
# whether the model has learned to return essentially nothing at night, hiding
# the success the instrument is meant to detect (ABL-377).
ZEROED_NIGHT_MW_THRESHOLD = 1.0


@dataclass
class SolarClampStats:
    """What the clamp removed, for one (country, model) inside one save.

    Counts are per forecast row. One target hour generated at two horizons
    (D+1 and D+2) is two rows, and counts twice — the row is the thing that
    gets served, so the row is the thing that gets counted.
    """
    country_code: str
    model_name: str
    renewable_type: str
    night_threshold_deg: float
    zeroed_night_mw_threshold: float  # |prediction| must exceed this to count in hours_zeroed_night
    # ABL-425. Together these two make an exempt country *visibly* exempt rather
    # than merely quiet: `hours_zeroed_night == 0` alone cannot distinguish "the
    # fit already returns nothing at night" from "the mask was never applied".
    night_generation_possible: bool  # the registered physical fact for this country
    night_mask_applied: bool         # False when exempt, and False when the country has no
                                     # representative point — disambiguated by the flag above
    rows_total: int
    night_hours: int             # rows in a geometric night hour, mask applied or not
    hours_zeroed_night: int      # night rows whose |prediction| exceeded zeroed_night_mw_threshold
    hours_raised_floor: int      # rows the non-negativity floor lifted; night rows too, where
                                 # the night mask did not apply
    mw_removed_night: float      # sum of predictions the night mask discarded
    mw_added_floor: float        # MW the non-negativity floor added back (>= 0)
    mw_removed_total: float      # sum(original - clamped); net, can be negative
    min_forecast_mw: Optional[float]        # most negative prediction seen
    max_night_forecast_mw: Optional[float]  # largest prediction at a night hour
    target_start: Optional[pd.Timestamp]
    target_end: Optional[pd.Timestamp]
    generated_at: Optional[pd.Timestamp]


def _as_naive_utc(values) -> pd.DatetimeIndex:
    """Parse forecast timestamps to naive UTC.

    Handles the three spellings this database carries (space separator, 'T'
    separator, and a '+01:00' offset form — ABL-211/ABL-324) as well as real
    Timestamp objects. Naive input is UTC by this repo's convention.
    """
    s = pd.Series(values).reset_index(drop=True)
    if s.dtype == object:
        parsed = pd.to_datetime(s, format='mixed', utc=True)
    else:
        parsed = pd.to_datetime(s, utc=True)
    return pd.DatetimeIndex(parsed.dt.tz_convert('UTC').dt.tz_localize(None))


def solar_row_mask(df: pd.DataFrame) -> pd.Series:
    """Rows this clamp owns: solar, however this caller spells it.

    `renewable_type` is authoritative, but 6,888 stored solar rows carry
    `renewable_type IS NULL` with `forecast_type='solar'` (measured 2026-08-12),
    so that spelling counts too. Nothing else is touched — wind can legitimately
    generate at night, and load/price are not generation at all.
    """
    is_solar_type = df['forecast_type'].astype(str).str.lower() == 'solar'
    if 'renewable_type' not in df.columns:
        return is_solar_type
    renewable = df['renewable_type']
    return (renewable.astype(str).str.lower() == 'solar') | (renewable.isna() & is_solar_type)


def clamp_solar_forecasts(
    forecasts_df: pd.DataFrame,
    threshold_deg: float = NIGHT_ELEVATION_THRESHOLD_DEG,
    zeroed_mw_threshold: float = ZEROED_NIGHT_MW_THRESHOLD,
) -> Tuple[pd.DataFrame, List[SolarClampStats]]:
    """
    Apply the night mask and the non-negativity floor to solar rows.

    Args:
        forecasts_df: rows headed for `forecasts`; needs at least country_code,
                      forecast_type, target_timestamp_utc, forecast_value,
                      model_name.
        threshold_deg: sun elevation below which the fleet is treated as dark.

    Returns:
        (clamped copy of the frame, one SolarClampStats per country/model).
        Non-solar rows are returned untouched. The input frame is not mutated.
    """
    if forecasts_df.empty:
        return forecasts_df, []

    mask = solar_row_mask(forecasts_df)
    if not mask.any():
        return forecasts_df, []

    out = forecasts_df.copy()
    stats: List[SolarClampStats] = []

    solar = out.loc[mask]
    for (country_code, model_name), group in solar.groupby(
        [solar['country_code'], solar['model_name']], sort=True
    ):
        # ABL-425: the registered physical fact, read before anything else in the
        # loop. An undeclared country aborts the whole save here — deliberately
        # outside the `try` below, and deliberately not a KeyError, so that
        # handler cannot degrade it into a silent default.
        night_possible = night_generation_possible(str(country_code))

        targets = _as_naive_utc(group['target_timestamp_utc'])
        hour_starts = targets.floor('h')

        night_mask_applied = not night_possible
        try:
            night = np.asarray(is_night_hour(country_code, hour_starts, threshold_deg))
        except KeyError:
            # No representative point for this country. Guessing a latitude here
            # would be the confidently-wrong number we are trying to prevent, so
            # the night mask stands down and only the floor applies. Loud, and
            # visible afterwards as night_mask_applied == False in the telemetry.
            logger.error(
                f"Solar night mask skipped for {country_code}: no representative "
                f"point in solar_geometry.SOLAR_REPRESENTATIVE_POINTS. "
                f"Applying the non-negativity floor only."
            )
            night = np.zeros(len(group), dtype=bool)
            night_mask_applied = False

        original = group['forecast_value'].astype(float).to_numpy()

        # `night` stays the geometric fact and keeps feeding the telemetry;
        # `zeroing` is the subset the hard zero actually applies to. For an
        # exempt country they differ, and that gap is the point: night_hours and
        # max_night_forecast_mw still report what the fleet was forecast to
        # dispatch after dark, which is exactly the number the clamp is now
        # letting through.
        zeroing = night if night_mask_applied else np.zeros(len(group), dtype=bool)
        clamped = np.where(zeroing, 0.0, np.maximum(original, 0.0))

        # hours_zeroed_night: only rows whose |prediction| exceeded the threshold.
        # mw_removed_night: all zeroed rows that were non-zero (threshold-free) —
        # the load-bearing MW instrument stays comparable across re-runs.
        zeroed = zeroing & (np.abs(original) > zeroed_mw_threshold)
        night_nonzero = zeroing & (original != 0.0)
        raised = (~zeroing) & (original < 0.0)

        generated_at = None
        if 'generated_at' in group.columns and len(group):
            generated_at = _as_naive_utc(group['generated_at']).min()

        stats.append(SolarClampStats(
            country_code=str(country_code),
            model_name=str(model_name),
            renewable_type='solar',
            night_threshold_deg=float(threshold_deg),
            zeroed_night_mw_threshold=float(zeroed_mw_threshold),
            night_generation_possible=bool(night_possible),
            night_mask_applied=bool(night_mask_applied),
            rows_total=int(len(group)),
            night_hours=int(night.sum()),
            hours_zeroed_night=int(zeroed.sum()),
            hours_raised_floor=int(raised.sum()),
            mw_removed_night=float(original[night_nonzero].sum()) if night_nonzero.any() else 0.0,
            mw_added_floor=float(-original[raised].sum()) if raised.any() else 0.0,
            mw_removed_total=float((original - clamped).sum()),
            min_forecast_mw=float(original.min()) if len(original) else None,
            max_night_forecast_mw=float(original[night].max()) if night.any() else None,
            target_start=targets.min() if len(targets) else None,
            target_end=targets.max() if len(targets) else None,
            generated_at=generated_at,
        ))

        out.loc[group.index, 'forecast_value'] = clamped

    for s in stats:
        if s.night_generation_possible:
            # Say it in as many words. A run whose ES line read "0 night hours
            # zeroed" and nothing else would be indistinguishable from a run
            # where the mask fired and found nothing to do.
            night_part = (
                f"night mask EXEMPT (registered night-generation capable), "
                f"{s.night_hours} night hours left in place"
            )
        else:
            night_part = (
                f"{s.hours_zeroed_night} of {s.night_hours} night hours zeroed "
                f"({s.mw_removed_night:.1f} MW removed)"
            )
        logger.info(
            f"[solar clamp] {s.country_code}/{s.model_name}: {night_part}, "
            f"{s.hours_raised_floor} hours raised to 0 ({s.mw_added_floor:.1f} MW added back), "
            f"net {s.mw_removed_total:.1f} MW removed from {s.rows_total} rows "
            f"(threshold {s.night_threshold_deg:g} deg)"
        )

    return out, stats
