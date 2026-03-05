"""
Chronos-Bolt Forecaster for Energy Price Prediction (Belgium)

Uses a fine-tuned Chronos-Bolt-small model via AutoGluon TimeSeriesPredictor
to generate 24h price forecasts for Belgium. Designed to run alongside the
existing LightGBM/XGBoost pipeline (Option B: dual model).

IMPORTANT — All timestamps are UTC:
  - Energy prices from ENTSO-E / market coupling are in UTC
  - DB columns (target_timestamp_utc, generated_at) are UTC
  - Forecast targets cover hours 00:00–23:00 UTC of the target day
  - Belgium is CET (UTC+1) / CEST (UTC+2), so "D+1" at 11:00 CET
    means reference_date = today UTC, target = tomorrow 00:00–23:00 UTC
  - The model's prediction_length=24 covers exactly one delivery day

Horizon behavior (matches XGBoost):
  - D+1: target_date = reference_date + 1 day → hours 00–23 UTC
  - D+2: target_date = reference_date + 2 days → hours 00–23 UTC
  - History is padded (interpolated) up to target_date 00:00 if needed

The Chronos model was fine-tuned with known covariates:
  - load_forecast_mw, solar_forecast_mw, wind_onshore_forecast_mw,
    wind_offshore_forecast_mw, total_gen_forecast_mw, hour, dayofweek, month

Requirements:
  - Python from chronos-venv (has autogluon-timeseries, torch, etc.)
  - Fine-tuned model at CHRONOS_MODEL_PATH

Author: OpenClaw integration
Date: 2026-02-22 (updated 2026-02-24: aligned D+N horizon with XGBoost, UTC docs)
"""

import logging
import sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("energy_forecast.chronos")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to the fine-tuned AutoGluon Chronos model
CHRONOS_MODEL_PATH = Path(
    r"C:\Users\guill\.openclaw\workspace\experiments\ag_chronos2_finetune"
)

# Database path
DATABASE_PATH = Path(os.getenv('ENERGY_DB_PATH', '/data/energy_dashboard.db'))

# Known covariates expected by the model (must match training)
KNOWN_COVARIATES = [
    "load_forecast_mw",
    "solar_forecast_mw",
    "wind_onshore_forecast_mw",
    "wind_offshore_forecast_mw",
    "total_gen_forecast_mw",
    "hour",
    "dayofweek",
    "month",
]

# Past covariates (auto-detected by AutoGluon, must be in history data)
PAST_COVARIATES = [
    "temperature_2m_k",
    "wind_speed_100m_ms",
    "shortwave_radiation_wm2",
]

# All covariates that must appear in the input data
ALL_COVARIATES = KNOWN_COVARIATES + PAST_COVARIATES

# Model metadata
MODEL_NAME = "chronos-bolt-small"
PREDICTION_LENGTH = 24  # 24 hourly steps


# ============================================================================
# DATA LOADING
# ============================================================================


def _get_connection() -> sqlite3.Connection:
    """Get a database connection."""
    conn = sqlite3.connect(str(DATABASE_PATH), timeout=30.0)
    return conn


def load_price_history(
    country_code: str = "BE",
    lookback_days: int = 90,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Load historical price data from the DB.

    Returns DataFrame with columns: [timestamp, target] indexed for AutoGluon.
    """
    if end_date is None:
        end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=lookback_days)

    query = """
        SELECT timestamp_utc as timestamp, price_eur_mwh as target
        FROM energy_price
        WHERE country_code = ?
          AND timestamp_utc >= ?
          AND timestamp_utc < ?
          AND data_quality = 'actual'
        ORDER BY timestamp_utc
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query(
            query, conn, params=(country_code, start_date.isoformat(), end_date.isoformat())
        )
    finally:
        conn.close()

    if df.empty:
        raise ValueError(f"No price data for {country_code} in [{start_date}, {end_date})")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True).dt.tz_localize(None)
    # Floor to hourly (data may be 15-min) and aggregate
    df["timestamp"] = df["timestamp"].dt.floor("h")
    df["target"] = df["target"].astype(float)
    df = df.groupby("timestamp", as_index=False)["target"].mean()
    return df


def load_covariates(
    country_code: str = "BE",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Load covariate data from DB: TSO load forecast + generation forecast + time features.

    Returns DataFrame with columns: [timestamp] + KNOWN_COVARIATES
    """
    conn = _get_connection()
    try:
        # Load TSO load forecast
        load_df = pd.read_sql_query(
            """
            SELECT target_timestamp_utc as timestamp,
                   forecast_value_mw as load_forecast_mw
            FROM energy_load_forecast
            WHERE country_code = ?
              AND target_timestamp_utc >= ?
              AND target_timestamp_utc < ?
            ORDER BY target_timestamp_utc
            """,
            conn, params=(country_code, start_date.isoformat(), end_date.isoformat()),
        )

        # Load TSO generation forecast
        gen_df = pd.read_sql_query(
            """
            SELECT target_timestamp_utc as timestamp,
                   solar_mw as solar_forecast_mw,
                   wind_onshore_mw as wind_onshore_forecast_mw,
                   wind_offshore_mw as wind_offshore_forecast_mw,
                   total_forecast_mw as total_gen_forecast_mw
            FROM energy_generation_forecast
            WHERE country_code = ?
              AND target_timestamp_utc >= ?
              AND target_timestamp_utc < ?
            ORDER BY target_timestamp_utc
            """,
            conn, params=(country_code, start_date.isoformat(), end_date.isoformat()),
        )

        # Load weather data (past covariates)
        weather_df = pd.read_sql_query(
            """
            SELECT timestamp_utc as timestamp,
                   temperature_2m_k,
                   wind_speed_100m_ms,
                   shortwave_radiation_wm2
            FROM weather_data
            WHERE country_code = ?
              AND timestamp_utc >= ?
              AND timestamp_utc < ?
            ORDER BY timestamp_utc
            """,
            conn, params=(country_code, start_date.isoformat(), end_date.isoformat()),
        )
    finally:
        conn.close()

    # Parse timestamps and floor to hourly
    for df in [load_df, gen_df, weather_df]:
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True).dt.tz_localize(None)
            df["timestamp"] = df["timestamp"].dt.floor("h")

    # Deduplicate all (after flooring)
    for df in [load_df, gen_df, weather_df]:
        if not df.empty:
            df.drop_duplicates(subset="timestamp", keep="last", inplace=True)

    # Start with whichever has data
    frames = [df for df in [load_df, gen_df, weather_df] if not df.empty]
    if not frames:
        raise ValueError(f"No covariate data for {country_code} in [{start_date}, {end_date})")

    cov_df = frames[0]
    for f in frames[1:]:
        cov_df = pd.merge(cov_df, f, on="timestamp", how="outer")

    cov_df = cov_df.sort_values("timestamp").reset_index(drop=True)

    # Add time features
    cov_df["hour"] = cov_df["timestamp"].dt.hour
    cov_df["dayofweek"] = cov_df["timestamp"].dt.dayofweek
    cov_df["month"] = cov_df["timestamp"].dt.month

    # Fill missing covariate columns with 0
    for col in ALL_COVARIATES:
        if col not in cov_df.columns:
            cov_df[col] = 0.0

    return cov_df


def _save_forecasts_direct(forecast_df: pd.DataFrame) -> int:
    """
    Save forecasts directly to DB without importing the project's db module
    (avoids dependency on python-dotenv in the chronos-venv).
    """
    conn = _get_connection()
    cursor = conn.cursor()

    # Ensure table exists (UNIQUE includes model_name for multi-model support)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country_code TEXT NOT NULL,
            forecast_type TEXT NOT NULL,
            target_timestamp_utc TIMESTAMP NOT NULL,
            generated_at TIMESTAMP NOT NULL,
            horizon_hours INTEGER NOT NULL,
            forecast_value REAL NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT,
            UNIQUE(country_code, forecast_type, target_timestamp_utc, horizon_hours, model_name, generated_at)
        )
    """)

    count = 0
    for _, row in forecast_df.iterrows():
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO forecasts
                (country_code, forecast_type, target_timestamp_utc,
                 generated_at, horizon_hours, forecast_value, model_name, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["country_code"],
                    row["forecast_type"],
                    str(row["target_timestamp_utc"]),
                    str(row["generated_at"]),
                    int(row["horizon_hours"]),
                    float(row["forecast_value"]),
                    row["model_name"],
                    row.get("model_version", ""),
                ),
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to insert row: {e}")

    conn.commit()
    conn.close()
    return count


# ============================================================================
# FORECASTER CLASS
# ============================================================================


class ChronosForecaster:
    """
    Wrapper around the fine-tuned Chronos-Bolt model for price forecasting.

    Usage:
        forecaster = ChronosForecaster()
        forecaster.load_model()
        forecast_df = forecaster.predict()  # Returns DB-ready DataFrame
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        country_code: str = "BE",
    ):
        self.model_path = model_path or CHRONOS_MODEL_PATH
        self.country_code = country_code
        self.predictor = None
        self.model_version = ""

    def load_model(self):
        """Load the AutoGluon TimeSeriesPredictor."""
        from autogluon.timeseries import TimeSeriesPredictor

        if not self.model_path.exists():
            raise FileNotFoundError(f"Chronos model not found at {self.model_path}")

        self.predictor = TimeSeriesPredictor.load(str(self.model_path))
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Chronos model loaded from {self.model_path}")

    def predict(
        self,
        reference_date: Optional[date] = None,
        horizon_days: int = 2,
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """
        Generate price forecast for 24 hours of the target day (D+horizon_days).

        Matches XGBoost behavior: uses reference_date + horizon_days to determine
        the target day, then forecasts hours 00:00-23:00 UTC of that day using
        market coupling covariates (Elia load/generation forecasts).

        Args:
            reference_date: Date of forecast generation (default: today)
            horizon_days: Days ahead to forecast (D+1 or D+2)
            lookback_days: Days of history to feed the model

        Returns:
            DataFrame with columns matching the forecasts table:
                country_code, forecast_type, target_timestamp_utc,
                forecast_value, model_name, model_version,
                horizon_hours, generated_at
        """
        if self.predictor is None:
            self.load_model()

        if reference_date is None:
            reference_date = date.today()

        generated_at = datetime.now()

        # Target day = reference_date + horizon_days (same as XGBoost D+N)
        target_date = reference_date + timedelta(days=horizon_days)

        logger.info(
            f"Generating Chronos D+{horizon_days} price forecast "
            f"(ref: {reference_date}, target: {target_date})"
        )

        # --- Load historical price data ---
        # History ends at end of reference_date (we don't have future actuals)
        history_end = datetime.combine(reference_date + timedelta(days=1), datetime.min.time())
        history_start = history_end - timedelta(days=lookback_days)

        price_df = load_price_history(
            self.country_code, lookback_days=lookback_days, end_date=history_end
        )
        logger.info(f"  Loaded {len(price_df)} price history rows")

        # Forecast covers 00:00-23:00 UTC of the target day
        forecast_start = datetime.combine(target_date, datetime.min.time())
        forecast_end_ts = forecast_start + timedelta(hours=PREDICTION_LENGTH)

        last_history_ts = price_df["timestamp"].max()
        logger.info(f"  History ends: {last_history_ts}, Forecast: {forecast_start} → {forecast_end_ts}")

        # --- Load covariates (history + future) ---
        cov_df = load_covariates(
            self.country_code,
            start_date=history_start,
            end_date=forecast_end_ts + timedelta(days=1),
        )
        logger.info(f"  Loaded {len(cov_df)} covariate rows")

        # --- Build AutoGluon TimeSeriesDataFrame ---
        from autogluon.timeseries import TimeSeriesDataFrame

        # Merge price with covariates for the history period
        price_df = price_df.drop_duplicates(subset="timestamp", keep="last")
        merged = pd.merge(price_df, cov_df, on="timestamp", how="left")

        # Fill missing covariates with forward-fill then 0
        for col in ALL_COVARIATES:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = merged[col].ffill().fillna(0.0)

        # Extend history up to forecast_start - 1h so AutoGluon can predict
        # from forecast_start. For D+2 there may be a gap we fill with interpolation.
        desired_end = forecast_start - timedelta(hours=1)
        full_range = pd.date_range(merged["timestamp"].min(), desired_end, freq="h")
        merged = merged.set_index("timestamp").reindex(full_range)
        merged.index.name = "timestamp"
        if merged["target"].isna().any():
            merged["target"] = merged["target"].interpolate(method="linear")
        for col in ALL_COVARIATES:
            merged[col] = merged[col].ffill().fillna(0.0)
        merged = merged.reset_index().rename(columns={"index": "timestamp"})

        # Add item_id (required by AutoGluon)
        merged["item_id"] = "BE_price"

        # Build TimeSeriesDataFrame (include all covariates: known + past)
        ts_df = TimeSeriesDataFrame.from_data_frame(
            merged[["item_id", "timestamp", "target"] + ALL_COVARIATES],
            id_column="item_id",
            timestamp_column="timestamp",
        )

        # --- Build known_covariates for forecast horizon ---
        # Must cover exactly the next prediction_length timesteps after history
        future_timestamps = pd.date_range(
            forecast_start, periods=PREDICTION_LENGTH, freq="h"
        )

        future_cov = cov_df[cov_df["timestamp"].isin(future_timestamps)].copy()

        # If we don't have enough future covariates, generate synthetic ones
        if len(future_cov) < PREDICTION_LENGTH:
            logger.warning(
                f"  Only {len(future_cov)} future covariate rows, generating synthetic for missing"
            )
            existing_ts = set(future_cov["timestamp"]) if not future_cov.empty else set()
            missing_ts = [t for t in future_timestamps if t not in existing_ts]
            if missing_ts:
                synth = pd.DataFrame({"timestamp": missing_ts})
                synth["hour"] = synth["timestamp"].dt.hour
                synth["dayofweek"] = synth["timestamp"].dt.dayofweek
                synth["month"] = synth["timestamp"].dt.month
                for col in ["load_forecast_mw", "solar_forecast_mw",
                            "wind_onshore_forecast_mw", "wind_offshore_forecast_mw",
                            "total_gen_forecast_mw"]:
                    synth[col] = merged[col].iloc[-1] if col in merged.columns else 0.0
                future_cov = pd.concat([future_cov, synth], ignore_index=True)

        future_cov = future_cov.sort_values("timestamp").head(PREDICTION_LENGTH)
        future_cov["item_id"] = "BE_price"

        # Fill any missing columns
        for col in KNOWN_COVARIATES:
            if col not in future_cov.columns:
                future_cov[col] = 0.0

        known_covariates_df = TimeSeriesDataFrame.from_data_frame(
            future_cov[["item_id", "timestamp"] + KNOWN_COVARIATES],
            id_column="item_id",
            timestamp_column="timestamp",
        )

        # --- Run prediction ---
        predictions = self.predictor.predict(
            ts_df, known_covariates=known_covariates_df
        )
        logger.info(f"  Chronos prediction complete: {len(predictions)} values")

        # --- Format output to match forecasts table ---
        # AutoGluon predictions are indexed by timestamp — use those directly
        # and filter to the target day (00:00-23:00 UTC) to match XGBoost behavior
        forecasts = []

        # Extract timestamp → value pairs from predictions
        pred_timestamps = predictions.index.get_level_values("timestamp")
        pred_values = predictions["mean"].values

        # Filter to target day only (00:00-23:00 UTC)
        target_day_start = forecast_start  # already datetime.combine(target_date, 00:00)
        target_day_end = target_day_start + timedelta(hours=23)

        for ts, val in zip(pred_timestamps, pred_values):
            ts_naive = ts.to_pydatetime().replace(tzinfo=None) if hasattr(ts, 'to_pydatetime') else ts
            if ts_naive < target_day_start or ts_naive > target_day_end:
                continue
            hours_until = (ts_naive - datetime.now()).total_seconds() / 3600
            horizon_hours = int(max(1, hours_until))

            forecasts.append({
                "country_code": self.country_code,
                "forecast_type": "price",
                "target_timestamp_utc": ts_naive,
                "generated_at": generated_at,
                "horizon_hours": horizon_hours,
                "forecast_value": float(val),
                "model_name": MODEL_NAME,
                "model_version": self.model_version,
            })

        result_df = pd.DataFrame(forecasts)
        logger.info(
            f"  Forecast range: {result_df['forecast_value'].min():.2f} - "
            f"{result_df['forecast_value'].max():.2f} EUR/MWh"
        )
        return result_df


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def run_chronos_forecast(
    country_code: str = "BE",
    reference_date: Optional[date] = None,
    horizon_days: int = 2,
    save_to_db: bool = False,
) -> pd.DataFrame:
    """
    Convenience function to run a Chronos forecast.

    Args:
        country_code: Country code (default: BE)
        reference_date: Reference date (default: today)
        horizon_days: Forecast horizon in days
        save_to_db: Whether to save results to the database

    Returns:
        DataFrame with forecast results
    """
    forecaster = ChronosForecaster(country_code=country_code)
    forecaster.load_model()
    forecast_df = forecaster.predict(
        reference_date=reference_date,
        horizon_days=horizon_days,
    )

    if save_to_db:
        records = _save_forecasts_direct(forecast_df)
        logger.info(f"Saved {records} Chronos forecast records to DB")

    return forecast_df


if __name__ == "__main__":
    import argparse
    import sys as _sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run Chronos price forecast")
    parser.add_argument("--country", default="BE", help="Country code")
    parser.add_argument("--horizon", type=int, default=2, help="Forecast horizon (days)")
    parser.add_argument("--save", action="store_true", help="Save to database")
    parser.add_argument("--date", type=str, default=None, help="Reference date YYYY-MM-DD")
    args = parser.parse_args()

    ref_date = None
    if args.date:
        ref_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    try:
        df = run_chronos_forecast(
            country_code=args.country,
            reference_date=ref_date,
            horizon_days=args.horizon,
            save_to_db=args.save,
        )
        print(f"\nForecast ({len(df)} rows):")
        print(df[["target_timestamp_utc", "forecast_value", "horizon_hours"]].to_string(index=False))
        _sys.exit(0)
    except Exception as exc:
        logger.error(f"Chronos forecast failed: {exc}")
        _sys.exit(1)
