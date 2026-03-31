#!/usr/bin/env python3
"""Generate energy forecasts using Chronos-2 models.

Usage:
    # Generate D+2 forecasts with experiment V003
    python scripts/forecast_chronos2.py --experiment V003 --countries DE,FR --types load,price

    # All countries, save to DB
    python scripts/forecast_chronos2.py --experiment V003 --save-to-db

    # Specific target date
    python scripts/forecast_chronos2.py --experiment V002 --target-date 2024-01-15 --countries DE --types load

    # Dry run (print but don't save)
    python scripts/forecast_chronos2.py --experiment V003 --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.chronos2.engine import ChronosEngine
from src.chronos2.input_builder import InputBuilder
from src.db import get_connection, save_quantile_forecasts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("energy_forecast.chronos2")


def save_point_forecasts(
    country_code: str,
    forecast_type: str,
    target_timestamps: list,
    forecast_values: np.ndarray,
    model_name: str,
    model_version: str,
    generated_at: datetime,
    horizon_days: int = 2,
) -> int:
    """Save point forecasts to the forecasts table."""
    count = 0
    with get_connection(readonly=False) as conn:
        cursor = conn.cursor()
        for ts, val in zip(target_timestamps, forecast_values):
            hours_ahead = int((pd.Timestamp(ts) - pd.Timestamp(generated_at)).total_seconds() / 3600)
            horizon_hours = max(1, hours_ahead)
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO forecasts
                    (country_code, forecast_type, target_timestamp_utc,
                     generated_at, horizon_hours, forecast_value, model_name, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    country_code,
                    forecast_type,
                    str(ts),
                    str(generated_at),
                    horizon_hours,
                    float(val),
                    model_name,
                    model_version,
                ))
                count += 1
            except Exception as e:
                logger.warning(f"Failed to insert forecast: {e}")
    return count


def run_forecast(
    experiment_id: str,
    countries: list[str],
    forecast_types: list[str],
    target_date: str,
    device: str = "cuda",
    save_to_db: bool = False,
    include_neighbors: bool = False,
) -> pd.DataFrame:
    """Generate Chronos-2 forecasts for specified countries and types.

    Args:
        experiment_id: Experiment ID (e.g., V002, V003)
        countries: List of country codes
        forecast_types: List of forecast types
        target_date: Target date to forecast (YYYY-MM-DD)
        device: cuda or cpu
        save_to_db: Whether to save results to database
        include_neighbors: Include neighbor features

    Returns:
        DataFrame with all forecast results
    """
    # Load experiment config
    exp_dir = config.EXPERIMENTS_DIR / experiment_id
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    with open(config_path) as f:
        exp_config = json.load(f)

    model_config = exp_config.get("model", {})
    training_config = exp_config.get("training", {})

    # Determine model path
    if training_config.get("fine_tune", False):
        model_path = str(config.MODELS_DIR / "chronos2" / experiment_id / "finetuned-ckpt")
    else:
        model_path = None  # Use pretrained

    context_length = model_config.get("context_length", config.CHRONOS2_CONTEXT_LENGTH)
    prediction_length = model_config.get("prediction_length", config.CHRONOS2_PREDICTION_LENGTH)

    # Initialize engine and input builder
    engine = ChronosEngine(
        model_path=model_path,
        device=device,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    input_builder = InputBuilder(
        context_length=context_length,
        prediction_length=prediction_length,
    )

    generated_at = datetime.utcnow()
    model_name = f"chronos-2-{experiment_id}"
    model_version = generated_at.strftime("%Y%m%d_%H%M%S")

    # Target timestamps (24 hours of the target date)
    target_dt = pd.Timestamp(target_date)
    target_timestamps = pd.date_range(target_dt, periods=24, freq="h")

    all_results = []

    for cc in countries:
        skip_types = config.SKIP_RENEWABLE_TYPES.get(cc, [])
        for ft in forecast_types:
            if ft in skip_types:
                continue

            try:
                # Build input
                inp = input_builder.build_for_country(
                    cc, ft, target_date,
                    include_neighbors=include_neighbors,
                )

                # Run forecast
                result = engine.forecast(
                    target=inp["target"],
                    past_covariates=inp.get("past_covariates"),
                    future_covariates=inp.get("future_covariates"),
                )

                point_forecast = result["median"]
                quantiles = result["quantiles"]  # (n_quantiles, 24)

                logger.info(
                    f"{cc}/{ft}: range [{point_forecast.min():.1f}, {point_forecast.max():.1f}]"
                )

                # Collect results
                for i, (ts, val) in enumerate(zip(target_timestamps, point_forecast)):
                    all_results.append({
                        "country_code": cc,
                        "forecast_type": ft,
                        "target_timestamp_utc": ts,
                        "forecast_value": float(val),
                        "model_name": model_name,
                        "model_version": model_version,
                        "generated_at": generated_at,
                    })

                # Save to DB if requested
                if save_to_db:
                    # Save point forecasts
                    n_saved = save_point_forecasts(
                        cc, ft, target_timestamps, point_forecast,
                        model_name, model_version, generated_at,
                    )
                    logger.info(f"  Saved {n_saved} point forecasts to DB")

                    # Save quantile forecasts
                    quantile_dict = {}
                    for qi, q_level in enumerate(config.CHRONOS2_QUANTILE_LEVELS):
                        quantile_dict[q_level] = quantiles[qi]

                    n_q = save_quantile_forecasts(
                        cc, ft, target_timestamps.tolist(),
                        quantile_dict, model_name, generated_at,
                    )
                    logger.info(f"  Saved {n_q} quantile forecasts to DB")

            except Exception as e:
                logger.error(f"Failed to forecast {cc}/{ft}: {e}")
                continue

    results_df = pd.DataFrame(all_results)
    logger.info(f"Total: {len(results_df)} forecast points generated")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Generate Chronos-2 energy forecasts")
    parser.add_argument("--experiment", required=True, help="Experiment ID (e.g., V003)")
    parser.add_argument("--countries", default="all", help="Comma-separated country codes, or 'all'")
    parser.add_argument("--types", default="all", help="Comma-separated forecast types, or 'all'")
    parser.add_argument("--target-date", default=None, help="Target date (YYYY-MM-DD). Default: D+2 from today")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--save-to-db", action="store_true", help="Save results to database")
    parser.add_argument("--dry-run", action="store_true", help="Print results without saving")
    parser.add_argument("--include-neighbors", action="store_true", help="Include neighbor features")
    args = parser.parse_args()

    # Parse countries and types
    if args.countries == "all":
        countries = config.SUPPORTED_COUNTRIES
    else:
        countries = args.countries.split(",")

    if args.types == "all":
        forecast_types = config.FORECAST_TYPES + config.RENEWABLE_TYPES
    else:
        forecast_types = args.types.split(",")

    # Default target date: D+2 from today
    if args.target_date:
        target_date = args.target_date
    else:
        target_date = (date.today() + timedelta(days=2)).isoformat()

    save = args.save_to_db and not args.dry_run

    logger.info(f"=== Chronos-2 Forecast: {args.experiment} ===")
    logger.info(f"Target date: {target_date}")
    logger.info(f"Countries: {countries}")
    logger.info(f"Types: {forecast_types}")
    logger.info(f"Save to DB: {save}")

    results_df = run_forecast(
        experiment_id=args.experiment,
        countries=countries,
        forecast_types=forecast_types,
        target_date=target_date,
        device=args.device,
        save_to_db=save,
        include_neighbors=args.include_neighbors,
    )

    if not results_df.empty:
        # Print summary
        print(f"\nForecast Summary ({len(results_df)} points):")
        summary = results_df.groupby(["country_code", "forecast_type"]).agg(
            mean=("forecast_value", "mean"),
            min=("forecast_value", "min"),
            max=("forecast_value", "max"),
        ).round(2)
        print(summary.to_string())

        if args.dry_run:
            print("\n[DRY RUN] Results NOT saved to database")
    else:
        print("No forecasts generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
