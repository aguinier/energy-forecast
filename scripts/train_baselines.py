#!/usr/bin/env python3
"""
Baseline Metrics Computation Script

Computes and stores baseline metrics (persistence, seasonal naive)
for all country/type combinations. These serve as reference points
for evaluating ML model improvements.

Usage:
    python scripts/train_baselines.py
    python scripts/train_baselines.py --countries DE,FR
    python scripts/train_baselines.py --types load,price
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config
from db import (
    initialize_all_tables,
    load_training_data,
    save_model_evaluation,
)
from metrics import calculate_all_metrics, mae, rmse, mape
from baselines import (
    PersistenceBaseline,
    SeasonalNaiveBaseline,
    WeeklyAverageBaseline,
)


def setup_logging():
    """Setup logging to file and console."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / f"train_baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger('energy_forecast')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute baseline metrics for all country/type combinations',
    )

    parser.add_argument(
        '--countries',
        type=str,
        default='all',
        help='Comma-separated country codes or "all" (default: all)'
    )

    parser.add_argument(
        '--types',
        type=str,
        default='load,price,solar,wind_onshore,wind_offshore',
        help='Comma-separated forecast types (default: load,price,solar,wind_onshore,wind_offshore)'
    )

    parser.add_argument(
        '--start',
        type=str,
        default=None,
        help='Evaluation start date (default: 6 months ago)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='Evaluation end date (default: today)'
    )

    parser.add_argument(
        '--horizon',
        type=int,
        default=48,
        help='Forecast horizon in hours (default: 48 for D+2)'
    )

    return parser.parse_args()


def get_countries(countries_arg: str) -> List[str]:
    """Parse countries argument."""
    if countries_arg.lower() == 'all':
        return config.SUPPORTED_COUNTRIES
    return [c.strip().upper() for c in countries_arg.split(',')]


def get_forecast_types(types_arg: str) -> List[str]:
    """Parse forecast types argument."""
    return [t.strip().lower() for t in types_arg.split(',')]


def compute_baseline_for_country_type(
    country_code: str,
    forecast_type: str,
    start_date: str,
    end_date: str,
    horizon_hours: int,
    logger: logging.Logger,
) -> Optional[dict]:
    """
    Compute baseline metrics for a single country/type combination.

    Returns:
        Dictionary with baseline metrics, or None if insufficient data
    """
    try:
        # Load data
        df = load_training_data(country_code, forecast_type, start_date, end_date)

        if df.empty or len(df) < horizon_hours * 2:
            logger.warning(f"Insufficient data for {country_code}/{forecast_type}")
            return None

        # Ensure timestamp column exists
        if 'timestamp_utc' not in df.columns:
            logger.warning(f"No timestamp column for {country_code}/{forecast_type}")
            return None

        # Create series for baselines
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        df = df.sort_values('timestamp_utc')

        hist_series = pd.Series(
            df['target_value'].values,
            index=df['timestamp_utc']
        )

        # Get target timestamps (skip first horizon_hours to have baseline data)
        target_timestamps = df['timestamp_utc'].iloc[horizon_hours:]
        y_true = df['target_value'].iloc[horizon_hours:].values

        if len(y_true) < 100:
            logger.warning(f"Insufficient evaluation data for {country_code}/{forecast_type}")
            return None

        # Initialize baselines
        persistence = PersistenceBaseline(horizon_hours=horizon_hours)
        seasonal = SeasonalNaiveBaseline(horizon_hours=horizon_hours)
        weekly = WeeklyAverageBaseline(horizon_hours=horizon_hours)

        results = {}

        # Compute each baseline
        for name, baseline in [
            ('persistence', persistence),
            ('seasonal_naive', seasonal),
            ('weekly_average', weekly),
        ]:
            y_pred = baseline.predict_for_target(hist_series, target_timestamps)

            # Filter valid predictions
            valid = ~np.isnan(y_pred)
            if valid.sum() < 100:
                logger.debug(f"Insufficient valid predictions for {name}")
                continue

            y_t = y_true[valid]
            y_p = y_pred[valid]

            results[name] = {
                'mae': mae(y_t, y_p),
                'rmse': rmse(y_t, y_p),
                'mape': mape(y_t, y_p),
                'sample_count': int(valid.sum()),
            }

        if not results:
            return None

        return {
            'country_code': country_code,
            'forecast_type': forecast_type,
            'horizon_hours': horizon_hours,
            'evaluation_period': f"{start_date} to {end_date}",
            'baselines': results,
        }

    except Exception as e:
        logger.error(f"Error computing baselines for {country_code}/{forecast_type}: {e}")
        return None


def main():
    """Main baseline computation loop."""
    args = parse_args()
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("Baseline Metrics Computation")
    logger.info("=" * 60)

    # Validate config
    try:
        config.validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Initialize database tables
    initialize_all_tables()

    # Parse arguments
    countries = get_countries(args.countries)
    forecast_types = get_forecast_types(args.types)

    # Default date range: last 6 months
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')
    start_date = args.start or (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

    total = len(countries) * len(forecast_types)

    logger.info(f"Countries: {len(countries)}")
    logger.info(f"Forecast types: {forecast_types}")
    logger.info(f"Evaluation period: {start_date} to {end_date}")
    logger.info(f"Forecast horizon: {args.horizon} hours")
    logger.info(f"Total combinations: {total}")
    logger.info("")

    results = []
    completed = 0
    success = 0

    for country in countries:
        for ftype in forecast_types:
            completed += 1

            logger.info(f"[{completed}/{total}] Computing baselines for {country}/{ftype}...")

            result = compute_baseline_for_country_type(
                country, ftype, start_date, end_date, args.horizon, logger
            )

            if result:
                results.append(result)
                success += 1

                # Save to database for each baseline
                for baseline_name, metrics in result['baselines'].items():
                    save_model_evaluation(
                        country_code=country,
                        forecast_type=ftype,
                        model_version=f"baseline_{baseline_name}",
                        metrics=metrics,
                        skill_scores={},  # Baselines don't have skill scores
                        training_samples=0,
                        test_samples=metrics.get('sample_count', 0),
                        evaluation_periods=result['evaluation_period'],
                        is_baseline=True,
                        model_location="baseline",
                    )

                # Log baseline metrics
                for name, metrics in result['baselines'].items():
                    logger.info(f"    {name}: MAE={metrics['mae']:.1f}, MAPE={metrics['mape']:.2f}%")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Baseline Computation Summary")
    logger.info("=" * 60)
    logger.info(f"Total combinations: {total}")
    logger.info(f"Successful: {success}")
    logger.info(f"Failed/Skipped: {total - success}")

    # Show average baseline performance by type
    if results:
        logger.info("")
        logger.info("Average Baseline Performance by Type:")

        for ftype in forecast_types:
            type_results = [r for r in results if r['forecast_type'] == ftype]
            if not type_results:
                continue

            logger.info(f"\n  {ftype.upper()}:")
            for baseline in ['persistence', 'seasonal_naive', 'weekly_average']:
                maes = [r['baselines'].get(baseline, {}).get('mae') for r in type_results]
                maes = [m for m in maes if m is not None]
                if maes:
                    avg_mae = np.mean(maes)
                    logger.info(f"    {baseline:20s}: Avg MAE = {avg_mae:.1f}")

    logger.info("")
    logger.info("[DONE] Baseline computation complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
