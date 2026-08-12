"""
Serving-path clamp for solar forecasts (ABL-337).

ABL-335 measured what the models emit and nothing corrects: 21,582 of 124,468
stored solar rows are negative (17.3%), and 16,574 of 31,084 stored night hours
read above 1 MW (53%) — DE at a 155-231 MW floor through local midnight. A solar
fleet cannot generate at night and cannot generate a negative number. This module
enforces those two physical facts on the way to the `forecasts` table:

  - sun below `solar_geometry.NIGHT_ELEVATION_THRESHOLD_DEG` for the whole hour
    -> hard zero
  - sun above it -> max(0, prediction)

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

from .solar_geometry import NIGHT_ELEVATION_THRESHOLD_DEG, is_night_hour

logger = logging.getLogger('energy_forecast')


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
    rows_total: int
    hours_zeroed_night: int      # night rows whose prediction was not already 0
    hours_raised_floor: int      # daylight rows whose prediction was negative
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
        targets = _as_naive_utc(group['target_timestamp_utc'])
        hour_starts = targets.floor('h')

        try:
            night = np.asarray(is_night_hour(country_code, hour_starts, threshold_deg))
        except KeyError:
            # No representative point for this country. Guessing a latitude here
            # would be the confidently-wrong number we are trying to prevent, so
            # the night mask stands down and only the floor applies. Loud, and
            # visible afterwards as hours_zeroed_night == 0 in the telemetry.
            logger.error(
                f"Solar night mask skipped for {country_code}: no representative "
                f"point in solar_geometry.SOLAR_REPRESENTATIVE_POINTS. "
                f"Applying the non-negativity floor only."
            )
            night = np.zeros(len(group), dtype=bool)

        original = group['forecast_value'].astype(float).to_numpy()
        clamped = np.where(night, 0.0, np.maximum(original, 0.0))

        zeroed = night & (original != 0.0)
        raised = (~night) & (original < 0.0)

        generated_at = None
        if 'generated_at' in group.columns and len(group):
            generated_at = _as_naive_utc(group['generated_at']).min()

        stats.append(SolarClampStats(
            country_code=str(country_code),
            model_name=str(model_name),
            renewable_type='solar',
            night_threshold_deg=float(threshold_deg),
            rows_total=int(len(group)),
            hours_zeroed_night=int(zeroed.sum()),
            hours_raised_floor=int(raised.sum()),
            mw_removed_night=float(original[zeroed].sum()) if zeroed.any() else 0.0,
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
        logger.info(
            f"[solar clamp] {s.country_code}/{s.model_name}: "
            f"{s.hours_zeroed_night} night hours zeroed ({s.mw_removed_night:.1f} MW removed), "
            f"{s.hours_raised_floor} hours raised to 0 ({s.mw_added_floor:.1f} MW added back), "
            f"net {s.mw_removed_total:.1f} MW removed from {s.rows_total} rows "
            f"(threshold {s.night_threshold_deg:g} deg)"
        )

    return out, stats
