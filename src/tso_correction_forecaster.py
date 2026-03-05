"""
TSO Error Correction Forecaster for Renewable Generation.

Loads Elia's day-ahead TSO forecasts for solar/wind_onshore/wind_offshore,
applies a trained LightGBM correction model (residual learning), and saves
corrected forecasts alongside the raw TSO forecast to the `forecasts` table.

Two model_names are written:
  - "tso_raw"       — the original Elia TSO forecast (as-is)
  - "tso_corrected"  — TSO + predicted error correction

This ensures the dashboard can show both, and the frontend's
getAvailableMLModels() picks them up automatically.

Usage:
    python src/tso_correction_forecaster.py --country BE --horizon 1 --save
    python src/tso_correction_forecaster.py --country BE --horizon 1 --date 2026-02-25 --save
    python src/tso_correction_forecaster.py --retrain --country BE

Author: Aurora / OpenClaw
Date: 2026-02-25
"""

import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import config
from db import get_connection, save_forecasts
from evaluation.tso_correction import (
    TSOCorrectionModel,
    load_tso_vs_actual,
    load_weather_for_correction,
    create_correction_features,
    get_correction_feature_cols,
    MODELS_DIR,
)

logger = logging.getLogger("energy_forecast.tso_correction_forecaster")

RENEWABLE_TYPES = ["solar", "wind_onshore", "wind_offshore"]


# ============================================================================
# TSO FORECAST LOADING (for future dates — no actuals yet)
# ============================================================================

def load_tso_forecast_for_date(
    country_code: str,
    renewable_type: str,
    target_date: date,
) -> pd.DataFrame:
    """
    Load Elia's TSO day-ahead forecast for a specific target date.

    Returns DataFrame with columns:
        timestamp_utc, tso_forecast_mw
    (24 rows, hours 00:00–23:00 UTC of target_date)
    """
    col_map = {
        "solar": "solar_mw",
        "wind_onshore": "wind_onshore_mw",
        "wind_offshore": "wind_offshore_mw",
    }
    col = col_map[renewable_type]

    start = target_date.strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

    with get_connection() as conn:
        df = pd.read_sql_query(f"""
            SELECT target_timestamp_utc as timestamp_utc, {col} as tso_forecast_mw
            FROM energy_generation_forecast
            WHERE country_code = ?
              AND target_timestamp_utc >= ? AND target_timestamp_utc < ?
            ORDER BY target_timestamp_utc
        """, conn, params=(country_code, start, end))

    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"], format="mixed", utc=True
        ).dt.tz_localize(None)

    return df


def load_recent_errors(
    country_code: str,
    renewable_type: str,
    before_date: date,
    lookback_days: int = 14,
) -> pd.DataFrame:
    """
    Load recent TSO errors (actual - forecast) for lag features.
    We need the last ~7 days of errors to compute lag/rolling features.
    """
    start = (before_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = before_date.strftime("%Y-%m-%d")

    df = load_tso_vs_actual(country_code, renewable_type, start, end)
    return df


def load_weather_forecast_for_date(
    country_code: str,
    target_date: date,
) -> pd.DataFrame:
    """Load weather forecast for the target date."""
    start = target_date.strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

    # Use latest forecast run to avoid duplicate rows per hour
    with get_connection() as conn:
        df = pd.read_sql_query("""
            SELECT timestamp_utc,
                   temperature_2m_k, wind_speed_10m_ms, wind_speed_100m_ms,
                   shortwave_radiation_wm2, direct_radiation_wm2, diffuse_radiation_wm2
            FROM weather_data
            WHERE country_code = ? AND data_quality = 'forecast'
              AND timestamp_utc >= ? AND timestamp_utc < ?
              AND forecast_run_time = (
                  SELECT MAX(forecast_run_time) FROM weather_data
                  WHERE country_code = ? AND data_quality = 'forecast'
                    AND timestamp_utc >= ? AND timestamp_utc < ?
              )
            ORDER BY timestamp_utc
        """, conn, params=(country_code, start, end, country_code, start, end))

    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"], format="mixed", utc=True
        ).dt.tz_localize(None)

    # If no forecast weather, try actual weather (for backtesting on past dates)
    if df.empty:
        with get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT timestamp_utc,
                       temperature_2m_k, wind_speed_10m_ms, wind_speed_100m_ms,
                       shortwave_radiation_wm2, direct_radiation_wm2, diffuse_radiation_wm2
                FROM weather_data
                WHERE country_code = ? AND data_quality = 'actual'
                  AND timestamp_utc >= ? AND timestamp_utc < ?
                ORDER BY timestamp_utc
            """, conn, params=(country_code, start, end))

        if not df.empty:
            df["timestamp_utc"] = pd.to_datetime(
                df["timestamp_utc"], format="mixed", utc=True
            ).dt.tz_localize(None)

    # Deduplicate by timestamp (safety net)
    if not df.empty:
        df = df.drop_duplicates(subset=["timestamp_utc"], keep="last")

    return df


# ============================================================================
# FORECAST GENERATION
# ============================================================================

def run_tso_correction_forecast(
    country_code: str = "BE",
    reference_date: Optional[date] = None,
    horizon_days: int = 1,
    renewable_types: Optional[List[str]] = None,
    save_to_db: bool = False,
) -> pd.DataFrame:
    """
    Generate corrected renewable forecasts.

    For each renewable type:
    1. Load TSO forecast for the target date
    2. Build features (weather, time, recent error lags)
    3. Apply correction model
    4. Return both raw TSO and corrected forecasts

    Args:
        country_code: ISO country code
        reference_date: Date of forecast generation (default: today)
        horizon_days: Days ahead (1=D+1, 2=D+2)
        renewable_types: Types to forecast (default: all 3)
        save_to_db: Whether to save to the forecasts table

    Returns:
        DataFrame with all forecast rows (both tso_raw and tso_corrected)
    """
    if reference_date is None:
        reference_date = date.today()

    target_date = reference_date + timedelta(days=horizon_days)
    generated_at = datetime.utcnow()

    if renewable_types is None:
        renewable_types = RENEWABLE_TYPES

    all_rows = []

    for rtype in renewable_types:
        logger.info(f"Processing {country_code}/{rtype} D+{horizon_days} -> {target_date}")

        # 1. Load TSO forecast for target date
        tso_df = load_tso_forecast_for_date(country_code, rtype, target_date)
        if tso_df.empty:
            logger.warning(f"No TSO forecast for {rtype} on {target_date}, skipping")
            continue

        logger.info(f"  TSO forecast: {len(tso_df)} hours")

        # 2. Try to load correction model
        model_path = MODELS_DIR / country_code / rtype
        try:
            model = TSOCorrectionModel.load(country_code, rtype, model_path)
            logger.info(f"  Loaded correction model (trained {model.trained_at})")
            has_model = True
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"  No correction model for {rtype}: {e}")
            has_model = False

        # 3. Build features for correction
        if has_model:
            # Load recent errors for lag features
            recent = load_recent_errors(country_code, rtype, target_date, lookback_days=14)

            # Load weather for target date
            weather = load_weather_forecast_for_date(country_code, target_date)

            # Merge weather with TSO forecast
            if not weather.empty:
                tso_with_weather = pd.merge(tso_df, weather, on="timestamp_utc", how="left")
            else:
                tso_with_weather = tso_df.copy()
                logger.warning(f"  No weather data for {target_date}")

            # Combine recent history + target day for feature creation
            # (we need the history for lag/rolling features, but only predict on target day)
            if not recent.empty:
                # Add placeholder tso_error = 0 for target day (we don't know actuals yet)
                tso_with_weather["tso_error"] = 0.0
                tso_with_weather["actual_mw"] = tso_with_weather["tso_forecast_mw"]  # placeholder

                # Align columns
                for col in recent.columns:
                    if col not in tso_with_weather.columns:
                        tso_with_weather[col] = np.nan

                combined = pd.concat([recent, tso_with_weather], ignore_index=True)
                combined = combined.sort_values("timestamp_utc").reset_index(drop=True)
            else:
                tso_with_weather["tso_error"] = 0.0
                combined = tso_with_weather

            # Create features
            featured = create_correction_features(combined, rtype)

            # Extract only target day rows
            target_start = pd.Timestamp(target_date)
            target_end = pd.Timestamp(target_date + timedelta(days=1))
            target_mask = (featured["timestamp_utc"] >= target_start) & (featured["timestamp_utc"] < target_end)
            target_rows = featured[target_mask].drop_duplicates(subset=["timestamp_utc"], keep="last").copy()

            if len(target_rows) > 0:
                # Predict correction
                corrected_values = model.correct(target_rows)
                # Build lookup: timestamp string -> corrected value
                correction_lookup = dict(zip(
                    target_rows["timestamp_utc"].astype(str).values,
                    corrected_values,
                ))
                logger.info(f"  Corrected {len(target_rows)} hours")
            else:
                logger.warning(f"  No target rows after feature creation")
                correction_lookup = None
        else:
            correction_lookup = None

        # 4. Build forecast rows
        for _, row in tso_df.iterrows():
            ts = row["timestamp_utc"]
            tso_val = float(row["tso_forecast_mw"])

            # Compute horizon_hours relative to reference_date
            ref_start = pd.Timestamp(reference_date)
            h_hours = int((ts - ref_start).total_seconds() / 3600)

            base_row = {
                "country_code": country_code,
                "forecast_type": rtype,
                "renewable_type": rtype,
                "target_timestamp_utc": ts,
                "generated_at": generated_at,
                "horizon_hours": h_hours,
            }

            # Raw TSO forecast
            all_rows.append({
                **base_row,
                "forecast_value": tso_val,
                "model_name": "tso_raw",
                "model_version": "elia_day_ahead",
            })

            # Corrected forecast (lookup by timestamp)
            if correction_lookup is not None:
                ts_key = str(pd.Timestamp(ts))
                if ts_key in correction_lookup:
                    corr_val = float(correction_lookup[ts_key])
                    corr_val = max(0.0, corr_val)  # generation can't be negative
                    all_rows.append({
                        **base_row,
                        "forecast_value": corr_val,
                        "model_name": "tso_corrected",
                        "model_version": f"lgbm_{model.trained_at.strftime('%Y%m%d') if model.trained_at else 'unknown'}",
                    })

    if not all_rows:
        logger.warning("No forecasts generated")
        return pd.DataFrame()

    forecast_df = pd.DataFrame(all_rows)
    logger.info(f"Generated {len(forecast_df)} forecast rows ({len(forecast_df)//2} per model)")

    if save_to_db:
        records = save_forecasts(forecast_df)
        logger.info(f"Saved {records} forecast records to DB")

    return forecast_df


# ============================================================================
# RETRAINING
# ============================================================================

def retrain_models(
    country_code: str = "BE",
    train_end: Optional[str] = None,
):
    """
    Retrain all TSO correction models with latest data and save to disk.
    Intended for monthly cron execution.
    """
    from evaluation.tso_correction import train_and_save_all

    if train_end is None:
        train_end = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Retraining TSO correction models for {country_code} (data up to {train_end})...")
    train_and_save_all(country_code, train_end=train_end)
    print("Retraining complete.")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="TSO Error Correction Forecaster")
    parser.add_argument("--country", default="BE", help="Country code")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon (days)")
    parser.add_argument("--save", action="store_true", help="Save to database")
    parser.add_argument("--date", type=str, default=None, help="Reference date YYYY-MM-DD")
    parser.add_argument("--retrain", action="store_true", help="Retrain models instead of forecasting")
    parser.add_argument("--types", type=str, default=None, help="Comma-separated renewable types")
    args = parser.parse_args()

    if args.retrain:
        retrain_models(args.country, train_end=args.date)
        sys.exit(0)

    ref_date = None
    if args.date:
        ref_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    rtypes = None
    if args.types:
        rtypes = [t.strip() for t in args.types.split(",")]

    try:
        df = run_tso_correction_forecast(
            country_code=args.country,
            reference_date=ref_date,
            horizon_days=args.horizon,
            renewable_types=rtypes,
            save_to_db=args.save,
        )
        if not df.empty:
            print(f"\nForecast ({len(df)} rows):")
            summary = df.groupby("model_name").agg(
                rows=("forecast_value", "count"),
                mean_mw=("forecast_value", "mean"),
                min_mw=("forecast_value", "min"),
                max_mw=("forecast_value", "max"),
            )
            print(summary.to_string())
            print(f"\nSaved {len(df)} forecast records" if args.save else "\n(dry run — not saved)")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"TSO correction forecast failed: {exc}", exc_info=True)
        sys.exit(1)
