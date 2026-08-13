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


#: A night actual at or below this many MW is treated as a real zero; above it,
#: the row is physically impossible and is dropped **from the fit only**.
#:
#: 1 MW is ABL-338's threshold, kept rather than re-derived so this rule and the
#: holdout that measured it are the same rule. It is a floor for reporting noise,
#: not a physical claim: `NIGHT_ELEVATION_THRESHOLD_DEG` already carries the
#: geometry's error budget (-8 deg is 6 deg below civil twilight), so an hour
#: reaching this predicate has no direct beam anywhere in the country's fleet and
#: the honest expectation is exactly zero. Measured on the replica 2026-08-13,
#: whole history, latest revision per hour: the largest night actual this admits
#: is DE 3.6 MW against a 38 GW fleet (0.009%), while FR runs to 440.0 MW.
IMPOSSIBLE_NIGHT_THRESHOLD_MW = 1.0


def impossible_night_mask(
    country_code: str,
    hour_starts: Sequence,
    actuals: Sequence,
    threshold_mw: float = IMPOSSIBLE_NIGHT_THRESHOLD_MW,
) -> np.ndarray:
    """
    Rows the sun says cannot exist: night by `night_mask`, actual above threshold.

    Night is `solar_geometry.is_night_hour` — the serving clamp's own predicate,
    reached through `night_mask` — so the fit refuses exactly the hours the clamp
    zeroes. A second definition here is the failure `solar_geometry` exists to
    stop: if the fit and the clamp disagree about which hours are night, both are
    wrong.

    Non-finite actuals are never flagged. They are the missingness audit's
    business (`finite_training_rows`), and double-counting them would make the
    two exclusion counts sum past the rows actually dropped.

    Args:
        country_code: ISO 2-letter code with a representative point in
            `solar_geometry.SOLAR_REPRESENTATIVE_POINTS`.
        hour_starts: UTC timestamps labelling the start of each hourly row.
        actuals: The observed solar MW for those rows, same length and order.
        threshold_mw: Night actuals strictly above this are impossible.

    Returns:
        Boolean array, True where the row is to be excluded from a fit.

    Raises:
        ValueError: If `hour_starts` and `actuals` differ in length. Silently
            broadcasting would mask rows by position against the wrong hours.
    """
    values = np.asarray(pd.Series(list(actuals)).to_numpy(), dtype=float)
    night = night_mask(country_code, hour_starts)
    if len(values) != len(night):
        raise ValueError(
            f"hour_starts and actuals disagree in length: {len(night)} vs {len(values)}"
        )
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    return night & np.isfinite(values) & (values > threshold_mw)


def exclude_impossible_night_rows(
    frame: pd.DataFrame,
    country_code: str,
    timestamp_column: str = "target_ts",
    actual_column: str = "actual",
    threshold_mw: float = IMPOSSIBLE_NIGHT_THRESHOLD_MW,
) -> Tuple[pd.DataFrame, dict]:
    """
    Drop physically impossible night rows from a **fit** frame, with an audit.

    This is a fit-side rule and only a fit-side rule (ABL-376). Never call it on
    a scoring frame: a contaminated actual has to stay visible in the score, or
    the night number measures the filter instead of the model. The asymmetry is
    the whole point — we refuse to train on values the sun says are impossible,
    and we still hold the model to account against whatever the source reports.

    It is stated as a general rule over countries rather than as an FR special
    case, and it is written to be a no-op where the data is clean: measured on
    the replica 2026-08-13 over ABL-253's registered fit window, it removes
    nothing at all for AT and BE, 7 hours for DE, and 114 for FR.

    Args:
        frame: Fit rows, one per (target, vintage). Not mutated.
        country_code: ISO 2-letter code, passed through to the geometry.
        timestamp_column: Column holding each row's target hour start.
        actual_column: Column holding the observed solar MW.
        threshold_mw: Night actuals strictly above this are dropped.

    Returns:
        `(kept, audit)`. `audit` carries the threshold, the night-row
        denominator and what was removed, so a later run can tell a data fix
        (fewer impossible rows on the same rule) from a rule change (a different
        threshold or predicate) rather than having to infer it from a row count.
    """
    if len(frame) == 0:
        return frame.reset_index(drop=True), {
            "threshold_mw": float(threshold_mw),
            "night_rows": 0, "excluded_rows": 0, "excluded_targets": 0,
            "retained_rows": 0, "max_excluded_mw": None,
            "mean_night_actual_mw": None,
        }

    timestamps, values = frame[timestamp_column], frame[actual_column]
    night = night_mask(country_code, timestamps)
    impossible = impossible_night_mask(country_code, timestamps, values, threshold_mw)
    finite_night = night & np.isfinite(np.asarray(values, dtype=float))
    kept = frame.loc[~impossible].reset_index(drop=True)

    return kept, {
        "threshold_mw": float(threshold_mw),
        "night_rows": int(finite_night.sum()),
        "excluded_rows": int(impossible.sum()),
        # Rows are per (target, vintage), so the row count is the vintage
        # multiple of the distinct contaminated hours. Both are reported: the
        # first is what the fit lost, the second is what the source got wrong.
        "excluded_targets": int(frame.loc[impossible, timestamp_column].nunique()),
        "retained_rows": int(len(kept)),
        "max_excluded_mw": float(values[impossible].max()) if impossible.any() else None,
        "mean_night_actual_mw": (float(values[finite_night].mean())
                                 if finite_night.any() else None),
    }


__all__ = [
    "SOLAR_GEOMETRY_FEATURES",
    "NIGHT_ELEVATION_THRESHOLD_DEG",
    "IMPOSSIBLE_NIGHT_THRESHOLD_MW",
    "solar_geometry_frame",
    "night_mask",
    "impossible_night_mask",
    "exclude_impossible_night_rows",
]
