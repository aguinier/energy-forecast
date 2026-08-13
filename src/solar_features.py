"""Solar-geometry model features (ABL-338).

ABL-335 measured what the four solar artifacts emit at night: AT/BE/FR go
negative (down to -292.7 MW at FR), and DE never returns to zero — 7,689 of
7,716 stored night hours above 1 MW, at a 155-231 MW floor. ABL-337 put a clamp
on the serving path. This module is half of the fit-side answer; the other half
is the non-negativity constraint in `Forecaster` (`nonneg_objective`).

Why the models needed this at all
---------------------------------
Measured 2026-08-12 against the live artifacts, reconstructing the exact serve
vectors for target 2026-08-14 (`reports/abl_338_solar_nonneg.md` §1): at every
night hour all three radiation columns read **0.0**, `target_value_lag_1d` and
`lag_7d` read **0.0**, and the models still emit a country-specific constant —
DE +171 to +263 MW, FR +33 to +80, AT -17, BE -13 to -39. Identical inputs,
non-zero output. Nothing in the 25-name feature vector distinguishes "0 W/m2
because the sun is down" from "0 W/m2 at a dark winter dawn", so the ensemble's
value near the origin of the radiation features is just wherever its residual
happened to settle — an incidental country constant, 0.3-4% of fleet capacity,
whose *sign* is incidental too. That is why a sign constraint alone cannot fix
DE: DE's floor is positive.

The two features
----------------
Both are pure functions of (country, target hour) — no database, no actuals — so
they are computable at any horizon, and **identical in training and at serving**
because both paths call this one function. That was the ABL-337 handover's first
requirement, and it is a construction here rather than a convention: the
training pipeline (`features.create_all_features`) and the serve-faithful
builder (`wind_features.RenewableFeatureBuilder.row`) both call
`solar_geometry_frame`.

- ``sun_elevation_deg`` — geometric sun elevation at the hour's **midpoint**, at
  the country's capacity-weighted representative point. The hour's
  representative sun height, and the continuous handle the model was missing.

- ``is_night`` — 1 when the sun stays below `NIGHT_ELEVATION_THRESHOLD_DEG` for
  the **whole** hour. This is `solar_geometry.is_night_hour`, i.e. bit-identical
  to the predicate the serving clamp uses to zero an hour, so the model's notion
  of night and the clamp's cannot drift apart.

These two are deliberately *not* monotone functions of one another, which is why
both earn their place. A tree ensemble is invariant to any monotone transform of
a single feature, so adding a `sin(elevation)` clear-sky factor alongside the
elevation would add exactly zero information to these models — it is not
included for that reason. `is_night` is different: it is a threshold on the
*maximum* elevation over the hour while `sun_elevation_deg` is the *midpoint*
value, and near sunrise and sunset those disagree — the max is at the end of a
rising hour and at the start of a setting one. So `is_night` carries real
information exactly at the shoulder hours ABL-337 flagged as the clamp's blind
spot, rather than being a split the tree could already have found.
"""

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from .solar_geometry import (
    NIGHT_ELEVATION_THRESHOLD_DEG,
    is_night_hour,
    sun_elevation_deg,
)

#: Feature names this module contributes, in the order they are appended to a
#: solar artifact's `feature_columns`. Anything reading or writing solar feature
#: vectors should use this rather than spelling the names out.
SOLAR_GEOMETRY_FEATURES: Tuple[str, ...] = ("sun_elevation_deg", "is_night")

#: Offset from the start of an hourly row to the instant `sun_elevation_deg` is
#: evaluated at. Rows are labelled with the start of the hour they cover, so the
#: midpoint is the hour's representative sun height.
_HOUR_MIDPOINT = pd.Timedelta(minutes=30)


def solar_geometry_frame(
    country_code: str,
    hour_starts: Sequence,
) -> pd.DataFrame:
    """
    The solar-geometry features for a series of hourly target timestamps.

    Args:
        country_code: ISO 2-letter code; must have a representative point in
            `solar_geometry.SOLAR_REPRESENTATIVE_POINTS`. A country without one
            raises from there rather than defaulting to some other latitude.
        hour_starts: UTC timestamps labelling the start of each hourly row.

    Returns:
        DataFrame with columns `SOLAR_GEOMETRY_FEATURES`, indexed 0..n-1 in the
        order given. `is_night` is an int (0/1) rather than a bool so it
        survives the float feature matrix every algorithm here is handed.
    """
    index = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(hour_starts)))).floor("h")

    if len(index) == 0:
        return pd.DataFrame({name: pd.Series(dtype=float) for name in SOLAR_GEOMETRY_FEATURES})

    midpoints = index + _HOUR_MIDPOINT
    elevation = np.asarray(sun_elevation_deg(country_code, midpoints), dtype=float)
    night = np.asarray(is_night_hour(country_code, index), dtype=bool)

    return pd.DataFrame(
        {
            "sun_elevation_deg": elevation,
            "is_night": night.astype(int),
        }
    )


def night_mask(country_code: str, hour_starts: Sequence) -> np.ndarray:
    """
    Boolean night mask for a series of hourly timestamps — the clamp's predicate.

    Shared by the training-row filter and by every evaluation that reports
    daylight and night hours separately, so a single definition of "night"
    covers the fit, the score and the serving clamp.
    """
    index = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(hour_starts)))).floor("h")
    if len(index) == 0:
        return np.zeros(0, dtype=bool)
    return np.asarray(is_night_hour(country_code, index), dtype=bool)


__all__ = [
    "SOLAR_GEOMETRY_FEATURES",
    "NIGHT_ELEVATION_THRESHOLD_DEG",
    "solar_geometry_frame",
    "night_mask",
]
