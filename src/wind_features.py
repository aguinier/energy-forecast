"""Serve-faithful feature builder for renewable xgboost forecasts (ABL-183).

Root cause, per `reports/abl_179_wind_diagnosis.md`: `Forecaster.predict_d2`
(`src/forecaster.py:616-690`, pre-fix) built each target hour's feature row by
taking "the most recent historical row having the target hour" and overriding
only calendar fields and the wind-speed weather columns. Every other column —
including `lag_1d`, both rolling windows, and `temperature_c` — kept the proxy
row's own historical values, which have no defined relationship to the actual
target hour. `temperature_c` was never overridden at all, because the wind
weather allow-list (`config.WEATHER_FEATURES['wind_onshore'/'wind_offshore']`)
only lists the two wind-speed columns, and the override loop only touches
columns in that allow-list.

This module replaces that proxy row with one function, `build_feature_row`,
called once per target hour, parameterized explicitly by:

  - `target_timestamp_utc` — the hour being forecast;
  - `observation_as_of`     — the generation instant; actuals after this are
                               not used;
  - `weather_publication_as_of` — the newest weather run instant admitted;
                               defaults to `observation_as_of`.

The same function is meant to be called from a future training-data rebuild
(with `observation_as_of` set to `target_timestamp_utc` minus the real
schedule horizon, per run) so that training and serving learn/predict from
features with an identical definition. **This issue wires it into serving
only** — `scripts/train.py` still uses the original `features.py` pipeline.
Adopting this builder for training is retraining, which is explicitly the
Forecasting Scientist's step, after this builder is reviewed.

## Why `lag_1d` cannot mean "one day before the target" at D+2

`config.FORECAST_HORIZONS['wind_onshore'/'wind_offshore'] = [1, 2]`, and a
target hour's horizon (generation instant to target) runs roughly 6-29h for
D+1 and 30-53h for D+2 (`Forecaster.predict_d2`, `FORECAST_HOUR=18`). A
"1 day before the target" lag needs data from `target - 24h`. For D+2, that
timestamp is always 6-29h *ahead* of the generation instant — it has not
happened yet. This is true of the rolling windows too: both are defined
(`features.py:284-298`) as ending one hour before the target row, so their
near edge is always in the future at any positive horizon, D+1 included.

Both are frozen artifact inputs (`models/*/wind_*/model.joblib`'s 24-name
`feature_columns`), so this issue cannot drop them without retraining. The
fix applied here, for each:

  - **rolling windows**: anchor the window's end at `observation_as_of`
    instead of at the target. `roll_24h_mean` etc. become "the last 24h of
    known actuals as of generation time" — a single value shared by all 24
    target hours of one generation run, always computable, never leaking.
  - **`lag_1d`**: use the true `target - 1d` value when that is at or before
    `observation_as_of`; otherwise fall back to the nearest same-hour actual
    that *is* at or before the cutoff (`target - k*1d` for the smallest
    admissible integer k >= 1). This is a degraded signal for D+2 and the
    late hours of D+1 — it is honest about that by recording the exact
    `source_timestamp` it used (see `FeatureValue`), not by pretending to be
    a true 1-day lag. It is not a fabrication: every value it returns was
    actually observed, at the timestamp it reports.

`lag_7d` and `lag_14d` never hit this problem: 7 and 14 days exceed any
stored horizon (max ~64h, dashboard `CLAUDE.md`), so `target - 7d`/`- 14d`
are always in the past relative to `observation_as_of`.

## `temperature_c`

Diagnosis finding #3: all five served renewable artifacts include
`temperature_c`, but wind's weather allow-list has no temperature column, so
serving never overrode it. This builder always resolves `temperature_c` from
the same weather-forecast row it uses for the allow-listed columns — it is
not gated by `config.WEATHER_FEATURES`, because the artifact contract
(what `feature_columns` names) is not the same list as "which raw columns are
this type's primary drivers".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from .db import get_connection, load_renewable_type_data


class ServeFaithfulnessError(AssertionError):
    """Raised when a feature would need data past its declared cutoff."""


#: Point (same-hour) lags in days. 1 has a defined fallback (see module
#: docstring); 7 and 14 must always be true lags — a violation is a bug, not
#: an expected D+2 condition, so it raises rather than degrading silently.
POINT_LAG_DAYS: Tuple[int, ...] = (1, 7, 14)
STRICT_LAG_DAYS: Tuple[int, ...] = (7, 14)

#: Rolling windows in hours, anchored at `observation_as_of` (see docstring).
ROLLING_WINDOWS_HOURS: Tuple[int, ...] = (24, 168)

#: ABL-183 wires this builder into serving for wind only — the two types
#: ABL-179 diagnosed. Confirmed against the real frozen artifacts, 2026-08-11:
#: BE/FR wind_offshore (xgboost) and BE/DE/FR wind_onshore (catboost) plus AT
#: wind_onshore (xgboost) all report exactly this builder's 24 feature names —
#: 10 calendar + 3 point lags + 8 rolling stats + 2 wind-speed + temperature_c,
#: no holiday or heating/cooling-degree columns.
#:
#: solar/hydro_total/biomass/renewable are deliberately excluded even though
#: the calendar/lag/rolling logic below is generic across renewable types:
#: `_WEATHER_RAW_COLUMNS` below only fetches temperature and wind speed, not
#: the shortwave/direct/diffuse radiation columns solar's and renewable's
#: WEATHER_FEATURES require — adding those types here without also extending
#: the weather query would silently NaN every solar weather feature. Solar is
#: ABL-185 (separate diagnosis, in progress); if it lands on the same
#: train/serve mismatch, extend `_WEATHER_RAW_COLUMNS` and this tuple together
#: rather than assuming today's wind-only query already covers it. load/price
#: are out of scope for a different reason: different artifact shape, not
#: diagnosed by ABL-179, left on the original code path.
SUPPORTED_FORECAST_TYPES: Tuple[str, ...] = ("wind_onshore", "wind_offshore")


@dataclass(frozen=True)
class FeatureValue:
    """One feature's value plus the provenance a golden test checks.

    `source_timestamp` is the observation the value came from (a lag's source
    hour, the rolling window's anchor, the weather row's valid-at hour, or the
    target hour itself for pure calendar features). `published_at` is set
    only for weather-sourced features: the `forecast_run_time` of the run
    actually used. `degraded` marks a value produced by a fallback rule
    (currently: `lag_1d` outside true D-1 reach) rather than the feature's
    nominal definition.
    """

    value: float
    source_timestamp: Optional[pd.Timestamp]
    published_at: Optional[pd.Timestamp] = None
    degraded: bool = False


@dataclass(frozen=True)
class FeatureRequest:
    country_code: str
    forecast_type: str
    target_timestamp_utc: pd.Timestamp
    observation_as_of: pd.Timestamp
    weather_publication_as_of: pd.Timestamp

    @classmethod
    def build(
        cls,
        country_code: str,
        forecast_type: str,
        target_timestamp_utc,
        observation_as_of,
        weather_publication_as_of=None,
    ) -> "FeatureRequest":
        obs = pd.Timestamp(observation_as_of)
        pub = pd.Timestamp(weather_publication_as_of) if weather_publication_as_of is not None else obs
        return cls(
            country_code=country_code,
            forecast_type=forecast_type,
            target_timestamp_utc=pd.Timestamp(target_timestamp_utc),
            observation_as_of=obs,
            weather_publication_as_of=pub,
        )


# ---------------------------------------------------------------------------
# Calendar features — pure functions of the target hour, always available.
# ---------------------------------------------------------------------------


def _calendar_features(target_ts: pd.Timestamp) -> Dict[str, FeatureValue]:
    hour = target_ts.hour
    dow = target_ts.dayofweek
    month = target_ts.month
    values = {
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "is_weekend": int(dow >= 5),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_sin": np.sin(2 * np.pi * dow / 7),
        "day_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }
    return {name: FeatureValue(value=float(val), source_timestamp=target_ts) for name, val in values.items()}


# ---------------------------------------------------------------------------
# Actuals loading — one read per builder, sliced in memory per target hour.
# ---------------------------------------------------------------------------


def _load_actuals_series(country_code: str, forecast_type: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    df = load_renewable_type_data(
        country_code, forecast_type, start.strftime("%Y-%m-%d"), (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    )
    if df.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    series = pd.Series(df["target_value"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["timestamp_utc"]))
    return series.sort_index()


def _lookup_hour(series: pd.Series, ts: pd.Timestamp) -> float:
    """Exact hourly lookup. No interpolation, no nearest-neighbour — a missing
    hour is NaN, not a fabricated value (dashboard-wide convention; see
    top-level CLAUDE.md 'Never extrapolate')."""
    floored = ts.floor("h")
    if floored in series.index:
        return float(series.loc[floored])
    return float("nan")


def _min_admissible_lag_days(target_ts: pd.Timestamp, as_of: pd.Timestamp) -> int:
    """Smallest integer k >= 1 such that `target_ts - k days <= as_of`.

    k=1 is the true one-day lag whenever that is already in the past; larger
    k is the documented fallback for horizons where it is not (see module
    docstring). This is the same formula regardless of whether the caller is
    reconstructing a historical vintage or serving live.
    """
    gap_days = (target_ts - as_of).total_seconds() / 86400.0
    return max(1, int(np.ceil(gap_days)))


def _point_lags(actuals: pd.Series, req: FeatureRequest) -> Dict[str, FeatureValue]:
    out: Dict[str, FeatureValue] = {}
    for days in POINT_LAG_DAYS:
        col = f"target_value_lag_{days}d"
        if days == 1:
            k = _min_admissible_lag_days(req.target_timestamp_utc, req.observation_as_of)
            source_ts = (req.target_timestamp_utc - pd.Timedelta(days=k)).floor("h")
            out[col] = FeatureValue(
                value=_lookup_hour(actuals, source_ts),
                source_timestamp=source_ts,
                degraded=(k != 1),
            )
            continue

        source_ts = (req.target_timestamp_utc - pd.Timedelta(days=days)).floor("h")
        if source_ts > req.observation_as_of:
            raise ServeFaithfulnessError(
                f"{col}: source {source_ts} is after observation_as_of {req.observation_as_of}; "
                f"a {days}-day lag should never reach the future at a D+1/D+2 horizon."
            )
        out[col] = FeatureValue(value=_lookup_hour(actuals, source_ts), source_timestamp=source_ts)
    return out


def _rolling_features(actuals: pd.Series, req: FeatureRequest) -> Dict[str, FeatureValue]:
    """Rolling stats anchored at `observation_as_of` (inclusive), not at the
    target. Identical for every target hour of one generation run — see
    module docstring for why the target-anchored definition cannot be served."""
    anchor = req.observation_as_of.floor("h")
    bounded = actuals[actuals.index <= anchor]
    out: Dict[str, FeatureValue] = {}
    for window_hours in ROLLING_WINDOWS_HOURS:
        window_start = anchor - pd.Timedelta(hours=window_hours - 1)
        chunk = bounded[bounded.index >= window_start]
        prefix = f"target_value_roll_{window_hours}h"
        stats = {
            "mean": chunk.mean() if len(chunk) else np.nan,
            "std": chunk.std() if len(chunk) > 1 else np.nan,
            "min": chunk.min() if len(chunk) else np.nan,
            "max": chunk.max() if len(chunk) else np.nan,
        }
        for stat_name, val in stats.items():
            out[f"{prefix}_{stat_name}"] = FeatureValue(
                value=float(val) if pd.notna(val) else float("nan"),
                source_timestamp=anchor,
            )
    return out


# ---------------------------------------------------------------------------
# Weather — resolved at the target hour, bounded by weather_publication_as_of.
# ---------------------------------------------------------------------------

_WEATHER_RAW_COLUMNS: Tuple[str, ...] = ("temperature_2m_k", "wind_speed_10m_ms", "wind_speed_100m_ms")


def _load_weather_archive(country_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cols = ", ".join(_WEATHER_RAW_COLUMNS)
    query = f"""
        SELECT timestamp_utc, forecast_run_time, {cols}
        FROM weather_data
        WHERE country_code = ?
          AND data_quality = 'forecast'
          AND timestamp_utc >= ?
          AND timestamp_utc <= ?
        ORDER BY forecast_run_time
    """
    with get_connection() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(country_code, start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")),
        )
    if df.empty:
        return df.assign(timestamp_utc=pd.Series(dtype="datetime64[ns]"), forecast_run_time=pd.Series(dtype="datetime64[ns]"))
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    df["forecast_run_time"] = pd.to_datetime(df["forecast_run_time"], format="mixed", utc=True).dt.tz_localize(None)
    return df


def _weather_row(archive: pd.DataFrame, req: FeatureRequest) -> Optional[pd.Series]:
    if archive.empty:
        return None
    target = req.target_timestamp_utc.floor("h")
    candidates = archive[(archive["timestamp_utc"] == target) & (archive["forecast_run_time"] <= req.weather_publication_as_of)]
    if candidates.empty:
        return None
    return candidates.sort_values("forecast_run_time").iloc[-1]


def _weather_features(archive: pd.DataFrame, req: FeatureRequest) -> Dict[str, FeatureValue]:
    row = _weather_row(archive, req)
    weather_cols = list(config.WEATHER_FEATURES.get(req.forecast_type, []))
    out: Dict[str, FeatureValue] = {}

    if row is None:
        for col in weather_cols:
            out[col] = FeatureValue(value=float("nan"), source_timestamp=None)
        out["temperature_c"] = FeatureValue(value=float("nan"), source_timestamp=None)
        return out

    published_at = pd.Timestamp(row["forecast_run_time"])
    for col in weather_cols:
        raw = row.get(col)
        out[col] = FeatureValue(
            value=float(raw) if pd.notna(raw) else float("nan"),
            source_timestamp=req.target_timestamp_utc.floor("h"),
            published_at=published_at,
        )
    temp_k = row.get("temperature_2m_k")
    out["temperature_c"] = FeatureValue(
        value=float(temp_k) - 273.15 if pd.notna(temp_k) else float("nan"),
        source_timestamp=req.target_timestamp_utc.floor("h"),
        published_at=published_at,
    )
    return out


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


class RenewableFeatureBuilder:
    """Loads actuals + weather once for a (country, forecast_type) span, then
    answers many `row()` calls in memory — the shape `predict_d2` needs for a
    24-hour target day, and the shape a future training rebuild needs across
    many run days.
    """

    def __init__(self, country_code: str, forecast_type: str, span_start, span_end):
        if forecast_type not in SUPPORTED_FORECAST_TYPES:
            raise ValueError(
                f"RenewableFeatureBuilder does not support forecast_type={forecast_type!r}; "
                f"supported: {SUPPORTED_FORECAST_TYPES}"
            )
        self.country_code = country_code
        self.forecast_type = forecast_type
        self._span_start = pd.Timestamp(span_start)
        self._span_end = pd.Timestamp(span_end)
        self._actuals = _load_actuals_series(country_code, forecast_type, self._span_start, self._span_end)
        self._weather = _load_weather_archive(country_code, self._span_start, self._span_end + pd.Timedelta(days=3))

    def row(
        self,
        target_timestamp_utc,
        observation_as_of,
        weather_publication_as_of=None,
    ) -> Dict[str, FeatureValue]:
        req = FeatureRequest.build(
            self.country_code, self.forecast_type, target_timestamp_utc, observation_as_of, weather_publication_as_of
        )
        out: Dict[str, FeatureValue] = {}
        out.update(_calendar_features(req.target_timestamp_utc))
        out.update(_point_lags(self._actuals, req))
        out.update(_rolling_features(self._actuals, req))
        out.update(_weather_features(self._weather, req))
        return out


def build_feature_row(
    country_code: str,
    forecast_type: str,
    target_timestamp_utc,
    observation_as_of,
    weather_publication_as_of=None,
    lookback_days: Optional[int] = None,
) -> Dict[str, FeatureValue]:
    """Single-call convenience wrapper around `RenewableFeatureBuilder`.

    Prefer constructing `RenewableFeatureBuilder` directly and calling `.row()`
    per target hour when building a whole day (24 calls) or a training frame
    (many run days) — this wrapper re-queries the database on every call.
    """
    if lookback_days is None:
        lookback_days = max(config.LAG_DAYS) + 7
    target_ts = pd.Timestamp(target_timestamp_utc)
    obs = pd.Timestamp(observation_as_of)
    span_start = min(target_ts, obs) - pd.Timedelta(days=lookback_days)
    span_end = max(target_ts, obs)
    builder = RenewableFeatureBuilder(country_code, forecast_type, span_start, span_end)
    return builder.row(target_ts, obs, weather_publication_as_of)


def to_vector(row: Dict[str, FeatureValue], columns: Iterable[str]) -> Dict[str, float]:
    """Strip provenance for model consumption, in the artifact's own column
    order. A column the builder does not know how to build raises rather than
    silently omitting a feature the model expects."""
    result = {}
    for col in columns:
        if col not in row:
            raise KeyError(f"builder does not produce feature {col!r}; known: {sorted(row)}")
        result[col] = row[col].value
    return result
