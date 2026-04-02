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

logger = logging.getLogger("energy_forecast.chronos2")


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
    finally:
        conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    df = df.drop_duplicates(subset="timestamp_utc", keep="last")
    return df.set_index("timestamp_utc")["value"].resample("h").mean()


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
    finally:
        conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    df = df.drop_duplicates(subset="timestamp_utc", keep="last")
    return df.set_index("timestamp_utc")["value"].resample("h").mean()


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


def _load_crossborder_flow_covariates(
    country_code: str,
    start_date: str,
    end_date: str,
) -> dict[str, pd.Series]:
    """Load per-border flow series as individual covariates.

    For country DE with neighbors FR, NL, PL, CZ, AT, CH, returns:
        {"flow__FR": pd.Series, "flow__NL": pd.Series, ...}

    Each series is hourly MW indexed by datetime.
    """
    query = """
        SELECT country_to, timestamp_utc, flow_mw
        FROM crossborder_flows
        WHERE country_from = ?
          AND timestamp_utc >= ?
          AND timestamp_utc < ?
        ORDER BY country_to, timestamp_utc
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(country_code, start_date, end_date))
    finally:
        conn.close()

    if df.empty:
        return {}

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)

    result = {}
    for neighbor, group in df.groupby("country_to"):
        series = group.set_index("timestamp_utc")["flow_mw"].resample("h").mean()
        result[f"flow__{neighbor}"] = series

    return result


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
    ) -> dict:
        """Build inference input for a specific country/type/date.

        Args:
            country_code: ISO 2-letter country code
            forecast_type: load, price, renewable, solar, etc.
            target_date: D+2 target date (YYYY-MM-DD) -- the day to forecast
            include_neighbors: Whether to include neighbor country features

        Returns:
            Input dict with target, past_covariates, future_covariates
            suitable for ChronosEngine.forecast()
        """
        target_dt = pd.Timestamp(target_date)

        # Time boundaries (same as netpredict2)
        # Past cutoff: D+1 23:00 (one day before target, end of day)
        past_cutoff = target_dt - pd.Timedelta(hours=1)  # target_date 00:00 - 1h = D+1 23:00
        context_start = past_cutoff - pd.Timedelta(hours=self.context_length - 1)

        # Future window: target_date 00:00 to 23:00
        future_start = target_dt
        future_end = target_dt + pd.Timedelta(hours=23)

        # Date strings for queries
        context_start_str = context_start.strftime("%Y-%m-%d")
        past_cutoff_str = (past_cutoff + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        future_end_str = (future_end + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

        # --- Load target history ---
        target_series = _load_target_series(
            country_code, forecast_type, context_start_str, past_cutoff_str
        )
        if target_series.empty:
            raise ValueError(f"No target data for {country_code}/{forecast_type} "
                           f"from {context_start_str} to {past_cutoff_str}")

        # Create hourly index for past context
        past_index = pd.date_range(context_start, past_cutoff, freq="h")
        target_aligned = _align_to_index(target_series, past_index)

        # Create hourly index for future
        future_index = pd.date_range(future_start, future_end, freq="h")

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

                # Future: weather forecast
                weather_future = _load_weather_forecast_series(
                    country_code, [column], target_date
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

        input_dict = {"target": target_aligned}
        if past_covariates:
            input_dict["past_covariates"] = past_covariates
        if future_covariates:
            input_dict["future_covariates"] = future_covariates

        logger.info(f"Built inference input for {country_code}/{forecast_type} "
                    f"target={target_date}: {len(target_aligned)} context, "
                    f"{len(past_covariates)} past covs, {len(future_covariates)} future covs")

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
