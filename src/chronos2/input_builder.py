"""Input builder for Chronos-2 forecasting.

Loads target time series and covariates from the energy dashboard database,
aligns everything to an hourly grid, and builds the input dicts that the
Chronos-2 engine expects.

Ported from netpredict2's input_builder.py, adapted from:
- CSV file loading -> SQLite database queries
- Zone-based targets -> country-based targets
- Meteologica covariates -> ENTSO-E + Open-Meteo weather

Time logic (for D+2 forecast, same as netpredict2):
- Past cutoff: target_date - 1 day + 23 hours (D+1 23:00)
- Context start: past_cutoff - (context_length - 1) hours
- Future window: target_date 00:00 to target_date 23:00
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import holidays

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.chronos2.covariate_mapper import build_covariate_map
from ..tso_plausibility import guard_tso_series

logger = logging.getLogger("energy_forecast.chronos2")


# A healthy scheduled net-position run has 672/672 real context hours and ends
# 26h before the nominal cutoff (measured across all 19 live countries on
# 2026-08-12).  Allow one missed daily refresh (50h), but refuse before a
# second missed refresh would extend the gap to 74h.  One week is the minimum
# real history accepted: it is a full weekly seasonal cycle and sits safely
# between the old GR failure (24 real hours) and the live cohort (672).
NET_POSITION_MAX_STALENESS_HOURS = 72
NET_POSITION_MIN_REAL_OBSERVATIONS = 168
NET_POSITION_MIN_MAX_ABS_MW = 1.0


class ContextRefusalError(ValueError):
    """The observed target context is not safe to forecast or publish from."""


def _net_position_context_refusal_reasons(
    target_series: pd.Series,
    past_cutoff: pd.Timestamp,
    nominal_cutoff: pd.Timestamp,
) -> list[str]:
    """Return countable reasons a net-position context must be refused.

    Zero is a legitimate point value for net position, so degeneracy is judged
    over the entire real series.  Alignment happens only after this check: zero
    padding must never make missing observations look real.
    """
    real_values = pd.to_numeric(target_series, errors="coerce").dropna()
    real_observations = int(real_values.size)
    staleness_hours = int(
        (nominal_cutoff - past_cutoff) / pd.Timedelta(hours=1)
    )
    max_abs_mw = float(real_values.abs().max()) if real_observations else 0.0

    reasons = []
    if staleness_hours > NET_POSITION_MAX_STALENESS_HOURS:
        reasons.append(
            f"stale_context={staleness_hours}h>"
            f"{NET_POSITION_MAX_STALENESS_HOURS}h"
        )
    if real_observations < NET_POSITION_MIN_REAL_OBSERVATIONS:
        reasons.append(
            f"thin_context={real_observations}<"
            f"{NET_POSITION_MIN_REAL_OBSERVATIONS}_real_hours"
        )
    if max_abs_mw < NET_POSITION_MIN_MAX_ABS_MW:
        reasons.append(
            f"degenerate_context=max_abs_{max_abs_mw:g}MW<"
            f"{NET_POSITION_MIN_MAX_ABS_MW:g}MW"
        )
    return reasons


# Table/column mapping for target loading (same as db.py)
TARGET_TABLE_MAP = {
    "load": ("energy_load", "load_mw"),
    "price": ("energy_price", "price_eur_mwh"),
    "renewable": ("energy_renewable", "total_renewable_mw"),
    "solar": ("energy_renewable", "solar_mw"),
    "wind_onshore": ("energy_renewable", "wind_onshore_mw"),
    "wind_offshore": ("energy_renewable", "wind_offshore_mw"),
    "hydro_total": ("energy_renewable", "(hydro_run_mw + hydro_reservoir_mw)"),
    "biomass": ("energy_renewable", "biomass_mw"),
    "net_position": ("net_position", "net_position_mw"),
}


def _get_connection() -> sqlite3.Connection:
    """Get a database connection."""
    conn = sqlite3.connect(str(config.DATABASE_PATH), timeout=30.0)
    return conn


def _load_target_series(
    country_code: str,
    forecast_type: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Load target time series from DB, resampled to hourly.

    Returns a pandas Series indexed by datetime (tz-naive UTC).
    """
    if forecast_type not in TARGET_TABLE_MAP:
        raise ValueError(f"Unknown forecast type: {forecast_type}")

    table, value_col = TARGET_TABLE_MAP[forecast_type]

    # For renewable types that use energy_renewable table,
    # there's no data_quality filter needed
    quality_filter = ""
    if table in ("energy_load", "energy_price"):
        quality_filter = "AND data_quality = 'actual'"

    query = f"""
        SELECT timestamp_utc, {value_col} as target_value
        FROM {table}
        WHERE country_code = ?
          AND timestamp_utc >= ?
          AND timestamp_utc < ?
          {quality_filter}
        ORDER BY timestamp_utc
    """

    conn = _get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(country_code, start_date, end_date))
    finally:
        conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    df = df.set_index("timestamp_utc")
    # Resample to hourly
    series = df["target_value"].resample("h").mean()
    return series


def _last_available_timestamp(
    country_code: str,
    forecast_type: str,
    before: str,
) -> Optional[pd.Timestamp]:
    """Last hour with an actual observation for this target, strictly before `before`.

    The D+2 schedule asks for context up to D+1 23:00, but a run firing at 06:00
    on day D cannot have observations for D+1 — that is still ~42h away. This
    measures where the data really stops so the caller can end the context there
    instead of padding the hole with zeros (see build_for_country).
    """
    if forecast_type not in TARGET_TABLE_MAP:
        raise ValueError(f"Unknown forecast type: {forecast_type}")

    table, value_col = TARGET_TABLE_MAP[forecast_type]
    quality_filter = ""
    if table in ("energy_load", "energy_price"):
        quality_filter = "AND data_quality = 'actual'"

    query = f"""
        SELECT MAX(timestamp_utc)
        FROM {table}
        WHERE country_code = ?
          AND timestamp_utc < ?
          AND {value_col} IS NOT NULL
          {quality_filter}
    """
    conn = _get_connection()
    try:
        row = conn.execute(query, (country_code, before)).fetchone()
    finally:
        conn.close()

    if not row or row[0] is None:
        return None
    return pd.Timestamp(row[0]).floor("h")


def _load_weather_series(
    country_code: str,
    columns: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load weather data columns from the weather_data table."""
    cols_sql = ", ".join(columns)
    query = f"""
        SELECT timestamp_utc, {cols_sql}
        FROM weather_data
        WHERE country_code = ?
          AND timestamp_utc >= ?
          AND timestamp_utc < ?
          AND data_quality = 'actual'
        ORDER BY timestamp_utc
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(country_code, start_date, end_date))
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    df = df.set_index("timestamp_utc").resample("h").mean()
    return df


def _load_weather_forecast_series(
    country_code: str,
    columns: list[str],
    target_date: str,
) -> pd.DataFrame:
    """Load weather forecast for target date from the weather_data table."""
    cols_sql = ", ".join(columns)
    query = f"""
        SELECT timestamp_utc, {cols_sql}
        FROM weather_data
        WHERE country_code = ?
          AND DATE(timestamp_utc) = DATE(?)
          AND data_quality = 'forecast'
          AND forecast_run_time = (
              SELECT MAX(forecast_run_time) FROM weather_data
              WHERE country_code = ?
                AND DATE(timestamp_utc) = DATE(?)
                AND data_quality = 'forecast'
          )
        ORDER BY timestamp_utc
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(country_code, target_date, country_code, target_date))
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    df = df.set_index("timestamp_utc").resample("h").mean()
    return df


def _load_weather_forecast_range(
    country_code: str,
    columns: list[str],
    start: str,
    end: str,
    as_of: Optional[str] = None,
) -> pd.DataFrame:
    """Weather forecast over an arbitrary [start, end) window, freshest run per hour.

    The single-day variant above cannot serve the extended horizon: once the
    context ends at the last real observation, the forecast window spans the
    remaining gap plus the whole target day, which straddles calendar days.

    `as_of` bounds `forecast_run_time` so a backtest sees only the weather runs
    that existed when the forecast would have fired. Live callers leave it None,
    where the freshest run is by definition already in the past.
    """
    cols_sql = ", ".join(columns)
    as_of_filter = "AND forecast_run_time <= ?" if as_of else ""
    query = f"""
        SELECT timestamp_utc, {cols_sql} FROM (
            SELECT timestamp_utc, {cols_sql},
                   ROW_NUMBER() OVER (
                       PARTITION BY timestamp_utc ORDER BY forecast_run_time DESC
                   ) AS rn
            FROM weather_data
            WHERE country_code = ?
              AND timestamp_utc >= ?
              AND timestamp_utc < ?
              AND data_quality = 'forecast'
              {as_of_filter}
        ) WHERE rn = 1
        ORDER BY timestamp_utc
    """
    params = [country_code, start, end] + ([as_of] if as_of else [])
    conn = _get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=tuple(params))
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    df = df.set_index("timestamp_utc").resample("h").mean()
    return df


def _guarded_tso_hourly(df: pd.DataFrame, conn, country_code: str,
                        table: str, column: str) -> pd.Series:
    """Shared tail of both TSO reads: plausibility guard, then hourly mean.

    These two series are suffix-1 covariates, so a value three orders of
    magnitude out does not merely mislead the model for its own hour — it sets
    the scale the whole covariate is normalised on. The guard runs at the
    published resolution and *before* the resample, so a bad quarter cannot be
    smeared across its hour first (ABL-431).
    """
    if df.empty:
        return pd.Series(dtype=float)

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed",
                                         utc=True).dt.tz_localize(None)
    df = df.drop_duplicates(subset="timestamp_utc", keep="last")
    series = df.set_index("timestamp_utc")["value"]
    series = guard_tso_series(series, conn, country_code, table, column,
                              context="chronos2.input_builder")
    return series.resample("h").mean()


def _load_tso_load_forecast(
    country_code: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Load TSO load forecast from energy_load_forecast table."""
    query = """
        SELECT target_timestamp_utc as timestamp_utc, forecast_value_mw as value
        FROM energy_load_forecast
        WHERE country_code = ?
          AND target_timestamp_utc >= ?
          AND target_timestamp_utc < ?
        ORDER BY target_timestamp_utc
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(country_code, start_date, end_date))
        return _guarded_tso_hourly(df, conn, country_code,
                                   "energy_load_forecast", "forecast_value_mw")
    finally:
        conn.close()


def _load_generation_forecast(
    country_code: str,
    column: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Load TSO generation forecast (solar/wind) from energy_generation_forecast table."""
    query = f"""
        SELECT target_timestamp_utc as timestamp_utc, {column} as value
        FROM energy_generation_forecast
        WHERE country_code = ?
          AND target_timestamp_utc >= ?
          AND target_timestamp_utc < ?
        ORDER BY target_timestamp_utc
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(country_code, start_date, end_date))
        return _guarded_tso_hourly(df, conn, country_code,
                                   "energy_generation_forecast", column)
    finally:
        conn.close()


def _load_price_series(
    country_code: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Load day-ahead price series."""
    query = """
        SELECT timestamp_utc, price_eur_mwh as value
        FROM energy_price
        WHERE country_code = ?
          AND timestamp_utc >= ?
          AND timestamp_utc < ?
          AND data_quality = 'actual'
        ORDER BY timestamp_utc
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(country_code, start_date, end_date))
    finally:
        conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    df = df.set_index("timestamp_utc")
    return df["value"].resample("h").mean()


def _load_load_series(
    country_code: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Load realized load series for neighbor features."""
    query = """
        SELECT timestamp_utc, load_mw as value
        FROM energy_load
        WHERE country_code = ?
          AND timestamp_utc >= ?
          AND timestamp_utc < ?
          AND data_quality = 'actual'
        ORDER BY timestamp_utc
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(country_code, start_date, end_date))
    finally:
        conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    df = df.set_index("timestamp_utc")
    return df["value"].resample("h").mean()


# A D+2 forecast cannot see recent cross-border flows. Derivation (re-derived
# 2026-07-25 from MEASURED data, after fixing the broken ingest that made the lag
# look like 26h -- it is actually ~1h):
#   - ENTSO-E publishes physical flows at ~H+1 (measured: max timestamp 14:00 at
#     14:58 UTC across BE/FR/DE).
#   - The binding constraint is INGEST CADENCE, not publication: prod cron runs
#     00:30/06:30/13:30/18:30 UTC, and the workstation replica syncs 05:00 UTC.
#   - Forecast runs 06:00 UTC (08:00 Brussels) for target date T, so origin =
#     T-42h, and the freshest flow in the replica then came from the 00:30 run
#     => flows only to ~T-48h.
#   - Required lag is 48h for target hour T 00:00, rising to 71h for T 23:00.
# One uniform lag >= the 71h worst case, applied identically in training and
# inference, so train-lag == eval-lag == serve-lag and no target hour can see
# unpublished data. See docs/superpowers/specs/2026-07-25-crossborder-lag-parity-design.md
CROSSBORDER_SERVE_LAG_HOURS = 72  # 3 days; >= 71h worst case


def _load_crossborder_flow_covariates(
    country_code: str,
    start_date: str,
    end_date: str,
) -> dict[str, pd.Series]:
    """Load cross-border flows as 3 homogeneous aggregate covariates.

    Regardless of which neighbours a country has, ALWAYS returns the same 3
    keys, so Chronos-2 global fine-tuning receives identical covariate keys for
    every series (per-neighbour keys were heterogeneous and broke fine-tuning):

        flow__total_export_mw = sum over borders of max(flow_mw, 0)
        flow__total_import_mw = sum over borders of max(-flow_mw, 0)
        flow__net_mw          = sum over borders of flow_mw

    Sign convention: flow_mw > 0 means physical flow FROM country_code TO the
    neighbour (export). Each series is hourly MW indexed by datetime. A country
    with no cross-border data returns the 3 keys as empty series (never {}), so
    homogeneity holds even for countries without flow data.

    KNOWN DEFECT (measured 2026-08-06, ABL-28) — on this database `flow_mw` is
    never negative: 0 negative rows out of 3,543,250, range 0.0..6,500.87. The
    import leg is stored as separate rows keyed `country_to = X`, which this
    query does not read. So in practice `flow__total_import_mw` is a CONSTANT
    ZERO for every country and hour, `flow__net_mw` is a duplicate of
    `flow__total_export_mw`, and a net-position model receives gross export
    where this docstring promises net flow (FR 2026-08-01: +12,022 MW here vs
    a true net position of +8,191). Fixing it means reading the `country_to`
    leg too. Left as-is deliberately for now: A/B'd over 14 vintages it is
    worth 0.8% of MAE, so it is filed rather than changed in flight.
    """
    query = """
        SELECT country_to, timestamp_utc, flow_mw
        FROM crossborder_flows
        WHERE country_from = ?
          AND timestamp_utc >= ?
          AND timestamp_utc < ?
        ORDER BY country_to, timestamp_utc
    """
    # Shift the QUERY window back by the serve lag so that, after lagging the series
    # forward below, the caller's [start_date, end_date] window is fully covered with
    # real (lagged) data instead of losing its first LAG hours to an empty gap.
    lag = pd.Timedelta(hours=CROSSBORDER_SERVE_LAG_HOURS)
    query_start = str(pd.Timestamp(start_date) - lag)
    query_end = str(pd.Timestamp(end_date) - lag)

    conn = _get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(country_code, query_start, query_end))
    finally:
        conn.close()

    if df.empty:
        return {
            "flow__total_export_mw": pd.Series(dtype="float64"),
            "flow__total_import_mw": pd.Series(dtype="float64"),
            "flow__net_mw": pd.Series(dtype="float64"),
        }

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    df["_export"] = df["flow_mw"].clip(lower=0)     # max(flow, 0)
    df["_import"] = (-df["flow_mw"]).clip(lower=0)  # max(-flow, 0)

    agg = df.groupby("timestamp_utc").agg(
        total_export=("_export", "sum"),
        total_import=("_import", "sum"),
        net=("flow_mw", "sum"),
    )

    # Lag every flow series by the serve lag: the value observed at t-LAG is what
    # the model is allowed to see at t. Applied here (the single choke point for
    # both the training and inference paths) so train/eval/serve stay identical.
    lag = pd.Timedelta(hours=CROSSBORDER_SERVE_LAG_HOURS)
    return {
        "flow__total_export_mw": agg["total_export"].resample("h").mean().shift(freq=lag),
        "flow__total_import_mw": agg["total_import"].resample("h").mean().shift(freq=lag),
        "flow__net_mw": agg["net"].resample("h").mean().shift(freq=lag),
    }


def _load_neighbor_net_position(
    country_code: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Load a country's net position as a covariate (for neighbor features)."""
    query = """
        SELECT timestamp_utc, net_position_mw as value
        FROM net_position
        WHERE country_code = ?
          AND timestamp_utc >= ?
          AND timestamp_utc < ?
        ORDER BY timestamp_utc
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(country_code, start_date, end_date))
    finally:
        conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    return df.set_index("timestamp_utc")["value"].resample("h").mean()


def _build_calendar_series(index: pd.DatetimeIndex, country_code: str) -> dict[str, np.ndarray]:
    """Build calendar features (hour, dayofweek, month, is_holiday) as normalized arrays."""
    hour_norm = (index.hour / 23.0).values.astype(np.float32)
    dow_norm = (index.dayofweek / 6.0).values.astype(np.float32)
    month_norm = ((index.month - 1) / 11.0).values.astype(np.float32)

    # Holiday detection
    try:
        country_holidays = holidays.country_holidays(country_code)
        is_holiday = np.array(
            [1.0 if d.date() in country_holidays else 0.0 for d in index],
            dtype=np.float32,
        )
    except Exception:
        is_holiday = np.zeros(len(index), dtype=np.float32)

    return {
        "cal__hour": hour_norm,
        "cal__dayofweek": dow_norm,
        "cal__month": month_norm,
        "cal__is_holiday": is_holiday,
    }


def _align_to_index(series: pd.Series, target_index: pd.DatetimeIndex) -> np.ndarray:
    """Align a series to a target index with ffill/bfill (limit=6h), fill NaN with 0."""
    aligned = series.reindex(target_index).ffill(limit=6).bfill(limit=6).fillna(0.0)
    return aligned.values.astype(np.float32)


class InputBuilder:
    """Builds Chronos-2 input dicts from the energy dashboard database.

    Handles both inference (build_for_country) and training (build_training_input)
    modes with proper temporal alignment and covariate loading.
    """

    def __init__(
        self,
        context_length: int | None = None,
        prediction_length: int | None = None,
    ):
        self.context_length = context_length or config.CHRONOS2_CONTEXT_LENGTH
        self.prediction_length = prediction_length or config.CHRONOS2_PREDICTION_LENGTH

    def build_for_country(
        self,
        country_code: str,
        forecast_type: str,
        target_date: str,
        include_neighbors: bool = False,
        as_of: Optional[str] = None,
        publication_as_of: Optional[str] = None,
    ) -> dict:
        """Build inference input for a specific country/type/date.

        The context ends at the last hour that actually has an observation, not
        at the nominal D+1 23:00. A D+2 run fires at ~06:00 on day D, so the
        nominal cutoff sits ~42h in the future and no data exists for it. The
        previous version still built the context out to that cutoff, where
        `_align_to_index` forward-filled 6h and wrote 0.0 into the remaining ~36 —
        handing the model a block of zeros as the most recent thing it had seen.
        Net position is signed and centred near zero, so those zeros read as
        plausible values and dragged every forecast toward zero (measured: FR at
        6% of actual, DE sign-flipped).

        The forecast window therefore spans the gap plus the target day, and
        `prediction_length` in the returned dict says how long it is. Callers
        take the last 24 points — `future_index` names their timestamps.

        Args:
            country_code: ISO 2-letter country code
            forecast_type: load, price, renewable, solar, etc.
            target_date: D+2 target date (YYYY-MM-DD) -- the day to forecast
            include_neighbors: Whether to include neighbor country features
            as_of: How far the *observations* reach — bounds the target series
                and anything else read by timestamp. None (live) means no bound:
                the data simply stops where it stops.
            publication_as_of: When the run *fires* — bounds a covariate by the
                time its own run was issued, not by the timestamp it describes.
                Defaults to `as_of`, which is right whenever the two coincide.

        Two bounds, because for a day-ahead-published target they are not the
        same instant and one value cannot express both. `net_position` for day X
        appears ~12:45 CET on X-1, so a 06:00Z run on D observes actuals through
        D 21:00 — `as_of = D 22:00`. But that run could not see a *weather* run
        issued at 12:00Z on D, so its publication bound is D 06:00. Passing
        D 22:00 for both leaks 16h of fresher weather forecasts into a backtest;
        passing D 06:00 for both truncates the context 16h short, which is what
        `compare_experiments.py` does and why its net_position numbers understate
        the pipeline. Leaking is the more dangerous direction: it is how offline
        scores come to measure information production never had (ABL-28).

        Not every covariate can be bounded this way. Suffix-1 sources (TSO load
        forecast, DA prices, cross-border flows) are bounded by timestamp only,
        because `publication_timestamp_utc` records when we fetched rather than
        when the value was published and is NULL on these rows. A reconstruction
        can therefore still see a late-revised suffix-1 value; that residual leak
        is documented rather than papered over.

        Returns:
            Input dict with target, past_covariates, future_covariates,
            plus prediction_length and future_index describing the horizon.
        """
        target_dt = pd.Timestamp(target_date)
        if publication_as_of is None:
            publication_as_of = as_of

        # Nominal cutoff the schedule implies: D+1 23:00.
        nominal_cutoff = target_dt - pd.Timedelta(hours=1)
        # ...and the upper bound on anything a run at `as_of` could observe.
        query_end = nominal_cutoff + pd.Timedelta(hours=1)
        if as_of is not None:
            query_end = min(query_end, pd.Timestamp(as_of).floor("h"))

        # Where the data really stops. Clamped to the nominal cutoff so a
        # backfilled database cannot pull the context past the schedule.
        last_seen = _last_available_timestamp(
            country_code, forecast_type, query_end.strftime("%Y-%m-%d %H:%M:%S")
        )
        if last_seen is None:
            raise ValueError(f"No target data for {country_code}/{forecast_type} "
                             f"before {query_end}")
        past_cutoff = min(nominal_cutoff, last_seen)
        context_start = past_cutoff - pd.Timedelta(hours=self.context_length - 1)

        # Future window: from the first unobserved hour through the target day.
        # Equals the target day exactly when the data reaches the nominal cutoff.
        future_start = past_cutoff + pd.Timedelta(hours=1)
        future_end = target_dt + pd.Timedelta(hours=23)

        # Date strings for queries
        context_start_str = context_start.strftime("%Y-%m-%d")
        past_cutoff_str = (past_cutoff + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        future_start_str = future_start.strftime("%Y-%m-%d %H:%M:%S")
        future_end_str = (future_end + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

        # --- Load target history ---
        target_series = _load_target_series(
            country_code, forecast_type, context_start_str, past_cutoff_str
        )
        if target_series.empty:
            raise ValueError(f"No target data for {country_code}/{forecast_type} "
                           f"from {context_start_str} to {past_cutoff_str}")

        # Exact context index (the query starts at midnight for compatibility,
        # which can return a few earlier hours that must not count as context).
        past_index = pd.date_range(context_start, past_cutoff, freq="h")
        if forecast_type == "net_position":
            refusal_reasons = _net_position_context_refusal_reasons(
                target_series.reindex(past_index), past_cutoff, nominal_cutoff
            )
            if refusal_reasons:
                raise ContextRefusalError(
                    f"Refusing {country_code}/{forecast_type} target={target_date}: "
                    + "; ".join(refusal_reasons)
                )

        target_aligned = _align_to_index(target_series, past_index)

        # Create hourly index for future
        future_index = pd.date_range(future_start, future_end, freq="h")
        prediction_length = len(future_index)

        # --- Build covariates ---
        cov_map = build_covariate_map(country_code, forecast_type, include_neighbors)
        past_covariates = {}
        future_covariates = {}

        # Process suffix-0 (future-known) covariates
        calendar_added = False
        for cov_spec in cov_map["suffix_0"]:
            source = cov_spec["source"]
            cov_name = cov_spec["cov_name"]

            if source == "calendar" and not calendar_added:
                # Calendar features — available for both past and future
                cal_past = _build_calendar_series(past_index, country_code)
                cal_future = _build_calendar_series(future_index, country_code)
                for cal_name in cal_past:
                    past_covariates[cal_name] = cal_past[cal_name]
                    future_covariates[cal_name] = cal_future[cal_name]
                calendar_added = True

            elif source == "weather_data":
                column = cov_spec["column"]
                # Past: actual weather
                weather_past = _load_weather_series(
                    country_code, [column], context_start_str, past_cutoff_str
                )
                if not weather_past.empty and column in weather_past.columns:
                    past_covariates[cov_name] = _align_to_index(weather_past[column], past_index)
                else:
                    past_covariates[cov_name] = np.zeros(len(past_index), dtype=np.float32)

                # Future: weather forecast. Range-based, because the horizon now
                # starts at the last observed hour and straddles calendar days.
                weather_future = _load_weather_forecast_range(
                    country_code, [column], future_start_str, future_end_str,
                    as_of=publication_as_of,
                )
                if not weather_future.empty and column in weather_future.columns:
                    future_covariates[cov_name] = _align_to_index(weather_future[column], future_index)
                else:
                    # Fallback: use last known value
                    future_covariates[cov_name] = np.full(
                        len(future_index),
                        past_covariates[cov_name][-1] if len(past_covariates.get(cov_name, [])) > 0 else 0.0,
                        dtype=np.float32
                    )

        # Process suffix-1 (past-only) covariates
        for cov_spec in cov_map["suffix_1"]:
            source = cov_spec["source"]
            cov_name = cov_spec["cov_name"]
            cc = cov_spec.get("country_override", country_code)

            if source == "energy_load_forecast":
                series = _load_tso_load_forecast(cc, context_start_str, past_cutoff_str)
                past_covariates[cov_name] = _align_to_index(series, past_index) if not series.empty else np.zeros(len(past_index), dtype=np.float32)

            elif source == "energy_generation_forecast":
                column = cov_spec["column"]
                series = _load_generation_forecast(cc, column, context_start_str, past_cutoff_str)
                past_covariates[cov_name] = _align_to_index(series, past_index) if not series.empty else np.zeros(len(past_index), dtype=np.float32)

            elif source == "energy_price":
                series = _load_price_series(cc, context_start_str, past_cutoff_str)
                past_covariates[cov_name] = _align_to_index(series, past_index) if not series.empty else np.zeros(len(past_index), dtype=np.float32)

            elif source == "energy_load":
                series = _load_load_series(cc, context_start_str, past_cutoff_str)
                past_covariates[cov_name] = _align_to_index(series, past_index) if not series.empty else np.zeros(len(past_index), dtype=np.float32)

            elif source == "crossborder_flows":
                flow_dict = _load_crossborder_flow_covariates(
                    country_code, context_start_str, past_cutoff_str
                )
                for flow_name, flow_series in flow_dict.items():
                    past_covariates[flow_name] = _align_to_index(flow_series, past_index) if not flow_series.empty else np.zeros(len(past_index), dtype=np.float32)

            elif source == "net_position":
                series = _load_neighbor_net_position(cc, context_start_str, past_cutoff_str)
                past_covariates[cov_name] = _align_to_index(series, past_index) if not series.empty else np.zeros(len(past_index), dtype=np.float32)

        input_dict = {
            "target": target_aligned,
            # The horizon is data-dependent: the caller cannot assume 24.
            "prediction_length": prediction_length,
            "future_index": future_index,
        }
        if past_covariates:
            input_dict["past_covariates"] = past_covariates
        if future_covariates:
            input_dict["future_covariates"] = future_covariates

        gap = prediction_length - 24
        logger.info(f"Built inference input for {country_code}/{forecast_type} "
                    f"target={target_date}: {len(target_aligned)} context ending "
                    f"{past_cutoff}, horizon {prediction_length}h ({gap}h gap + 24h day), "
                    f"{len(past_covariates)} past covs, {len(future_covariates)} future covs")
        if gap > 0:
            logger.info(f"  data stops {gap}h short of the nominal cutoff "
                        f"{nominal_cutoff} — forecasting across the gap rather "
                        f"than zero-filling it")

        return input_dict

    def build_training_input(
        self,
        country_code: str,
        forecast_type: str,
        start_date: str,
        end_date: str,
        exclude_dates: list[tuple[str, str]] | None = None,
        include_neighbors: bool = False,
    ) -> dict | None:
        """Build ONE long training input for a (country, forecast_type) pair.

        Chronos2Pipeline.fit() uses internal random-window cropping, so we pass
        the full continuous series.

        Args:
            country_code: ISO 2-letter country code
            forecast_type: load, price, renewable, solar, etc.
            start_date: Training data start date
            end_date: Training data end date
            exclude_dates: Backtest weeks to NaN-mask [(start, end), ...]
            include_neighbors: Include neighbor features

        Returns:
            Input dict for training, or None if insufficient data
        """
        # Load full target series
        target_series = _load_target_series(country_code, forecast_type, start_date, end_date)
        if target_series.empty:
            logger.warning(f"No target data for {country_code}/{forecast_type}")
            return None

        # NaN-mask excluded dates (backtest weeks)
        # NaN values are preserved — Chronos-2 skips NaN windows during training
        if exclude_dates:
            n_masked = 0
            for exc_start, exc_end in exclude_dates:
                mask = (target_series.index >= pd.Timestamp(exc_start)) & (
                    target_series.index <= pd.Timestamp(exc_end) + pd.Timedelta(hours=23)
                )
                n_masked += mask.sum()
                target_series.loc[mask] = np.nan
            if n_masked > 0:
                logger.info(f"  Masked {n_masked} hours ({n_masked // 24} days) for backtest exclusion")

        # Check minimum non-NaN length
        non_nan_count = target_series.notna().sum()
        min_length = self.context_length + self.prediction_length
        if non_nan_count < min_length:
            logger.warning(f"Skipping {country_code}/{forecast_type}: "
                         f"only {non_nan_count} non-NaN points (need {min_length})")
            return None

        # Keep NaN values in the target — Chronos skips NaN windows during training
        target = target_series.values.astype(np.float32)
        series_index = target_series.index

        # Build covariate map
        cov_map = build_covariate_map(country_code, forecast_type, include_neighbors)
        past_covariates = {}
        future_covariates = {}

        # --- Suffix-0: Weather + calendar (future-known) ---
        weather_cols_needed = []
        weather_cov_names = []
        for cov_spec in cov_map["suffix_0"]:
            if cov_spec["source"] == "weather_data":
                weather_cols_needed.append(cov_spec["column"])
                weather_cov_names.append(cov_spec["cov_name"])
            elif cov_spec["source"] == "calendar":
                # Calendar covariates
                cal_data = _build_calendar_series(series_index, country_code)
                for cal_name, cal_values in cal_data.items():
                    if len(cal_values) == len(target):
                        past_covariates[cal_name] = cal_values
                        future_covariates[cal_name] = None  # Mark as future-known

        # Load weather data in one query
        if weather_cols_needed:
            weather_df = _load_weather_series(country_code, weather_cols_needed, start_date, end_date)
            for col, cov_name in zip(weather_cols_needed, weather_cov_names):
                if not weather_df.empty and col in weather_df.columns:
                    aligned = _align_to_index(weather_df[col], series_index)
                    if len(aligned) == len(target):
                        past_covariates[cov_name] = aligned
                        future_covariates[cov_name] = None  # Future-known

        # --- Suffix-1: TSO forecasts, DA prices, neighbor features (past-only) ---
        for cov_spec in cov_map["suffix_1"]:
            source = cov_spec["source"]
            cov_name = cov_spec["cov_name"]
            cc = cov_spec.get("country_override", country_code)

            series_data = pd.Series(dtype=float)

            if source == "energy_load_forecast":
                series_data = _load_tso_load_forecast(cc, start_date, end_date)
            elif source == "energy_generation_forecast":
                series_data = _load_generation_forecast(cc, cov_spec["column"], start_date, end_date)
            elif source == "energy_price":
                series_data = _load_price_series(cc, start_date, end_date)
            elif source == "energy_load":
                series_data = _load_load_series(cc, start_date, end_date)

            elif source == "crossborder_flows":
                flow_dict = _load_crossborder_flow_covariates(
                    country_code, start_date, end_date
                )
                for flow_name, flow_series in flow_dict.items():
                    if not flow_series.empty:
                        aligned = _align_to_index(flow_series, series_index)
                        if len(aligned) == len(target):
                            past_covariates[flow_name] = aligned
                continue

            elif source == "net_position":
                series_data = _load_neighbor_net_position(cc, start_date, end_date)
                if not series_data.empty:
                    aligned = _align_to_index(series_data, series_index)
                    if len(aligned) == len(target):
                        past_covariates[cov_name] = aligned
                continue

            if not series_data.empty:
                aligned = _align_to_index(series_data, series_index)
                if len(aligned) == len(target):
                    past_covariates[cov_name] = aligned

        # Build input dict
        input_dict = {"target": target}
        if past_covariates:
            input_dict["past_covariates"] = past_covariates
        if future_covariates:
            input_dict["future_covariates"] = future_covariates

        logger.info(f"Training input {country_code}/{forecast_type}: "
                    f"{len(target)} points, {len(past_covariates)} past covs, "
                    f"{len(future_covariates)} future covs")

        return input_dict

    def build_batch_training_inputs(
        self,
        countries: list[str],
        forecast_types: list[str],
        start_date: str,
        end_date: str,
        exclude_dates: list[tuple[str, str]] | None = None,
        include_neighbors: bool = False,
        val_fraction: float = 0.0,
    ) -> tuple[list[dict], list[dict] | None, list[tuple[str, str]]]:
        """Build training inputs for multiple countries and forecast types.

        Args:
            countries: List of country codes (or ["all"] for all supported)
            forecast_types: List of forecast types (or ["all"])
            start_date: Training start date
            end_date: Training end date
            exclude_dates: Backtest weeks to exclude
            include_neighbors: Include neighbor features
            val_fraction: Fraction of series to hold out for validation

        Returns:
            (train_inputs, val_inputs, series_labels) where series_labels
            is a list of (country_code, forecast_type) tuples
        """
        if countries == ["all"]:
            countries = config.SUPPORTED_COUNTRIES
        if forecast_types == ["all"]:
            forecast_types = config.FORECAST_TYPES + config.RENEWABLE_TYPES

        all_inputs = []
        all_labels = []

        for cc in countries:
            skip_types = config.SKIP_RENEWABLE_TYPES.get(cc, [])
            for ft in forecast_types:
                if ft in skip_types:
                    continue
                inp = self.build_training_input(
                    cc, ft, start_date, end_date,
                    exclude_dates=exclude_dates,
                    include_neighbors=include_neighbors,
                )
                if inp is not None:
                    all_inputs.append(inp)
                    all_labels.append((cc, ft))

        logger.info(f"Built {len(all_inputs)} training inputs from "
                    f"{len(countries)} countries x {len(forecast_types)} types")

        # Split into train/val
        if val_fraction > 0 and len(all_inputs) > 1:
            import random
            random.seed(42)
            n_val = max(1, int(len(all_inputs) * val_fraction))
            indices = list(range(len(all_inputs)))
            random.shuffle(indices)
            val_indices = set(indices[:n_val])

            train_inputs = [all_inputs[i] for i in range(len(all_inputs)) if i not in val_indices]
            val_inputs = [all_inputs[i] for i in range(len(all_inputs)) if i in val_indices]

            logger.info(f"Split: {len(train_inputs)} train, {len(val_inputs)} validation")
            return train_inputs, val_inputs, all_labels

        return all_inputs, None, all_labels
