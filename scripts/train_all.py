#!/usr/bin/env python3
"""
Batch Training Script for All Energy Forecasting Models

Trains 120 models (5 types x 24 countries) with progress tracking,
automatic evaluation, and optional auto-promotion.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --auto-promote
    python scripts/train_all.py --types load,price --countries DE,FR
    python scripts/train_all.py --parallel 4
"""

import argparse
import logging
import sys
import time
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config
from db import initialize_all_tables


def setup_logging():
    """Setup logging to file and console."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / f"train_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger('energy_forecast'), log_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch train all energy forecasting models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--countries',
        type=str,
        default='all',
        help='Comma-separated country codes or "all" (default: all 24 countries)'
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
        default='2023-01-01',
        help='Training start date YYYY-MM-DD (default: 2023-01-01)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='Training end date YYYY-MM-DD (default: latest available)'
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        default='xgboost',
        choices=['xgboost', 'lightgbm', 'catboost'],
        help='Algorithm to use (default: xgboost)'
    )

    parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Enable grid search hyperparameter tuning'
    )

    parser.add_argument(
        '--auto-promote',
        action='store_true',
        help='Auto-promote models that beat baselines and production'
    )

    parser.add_argument(
        '--min-skill',
        type=float,
        default=0.0,
        help='Minimum skill score vs persistence for auto-promotion (default: 0.0)'
    )

    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel training jobs (default: 1, max: 4)'
    )

    parser.add_argument(
        '--report-file',
        type=str,
        default=None,
        help='Output CSV file for training report (default: logs/train_all_report_TIMESTAMP.csv)'
    )

    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        default=True,
        help='Continue training other models if one fails (default: True)'
    )

    return parser.parse_args()


def get_countries(countries_arg: str) -> List[str]:
    """Parse countries argument."""
    if countries_arg.lower() == 'all':
        return config.SUPPORTED_COUNTRIES
    return [c.strip().upper() for c in countries_arg.split(',')]


def get_forecast_types(types_arg: str) -> List[str]:
    """Parse forecast types argument."""
    if types_arg.lower() == 'all':
        return ['load', 'price', 'solar', 'wind_onshore', 'wind_offshore']
    return [t.strip().lower() for t in types_arg.split(',')]


def train_single_model(
    country: str,
    forecast_type: str,
    start_date: str,
    end_date: Optional[str],
    algorithm: str,
    grid_search: bool,
    auto_promote: bool,
    min_skill: float,
) -> Dict:
    """
    Train a single model (for parallel execution).

    Returns result dictionary.
    """
    # Import here to avoid multiprocessing issues
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

    import logging
    logging.basicConfig(level=logging.WARNING)  # Quiet mode for workers
    logger = logging.getLogger('energy_forecast')

    from train import train_model, should_train_renewable_type
    import config

    # Check if we should skip this renewable type
    if forecast_type in config.RENEWABLE_TYPES:
        if not should_train_renewable_type(country, forecast_type, logger):
            return {
                'country_code': country,
                'forecast_type': forecast_type,
                'status': 'skipped',
                'reason': 'Insufficient data',
            }

    result = train_model(
        country_code=country,
        forecast_type=forecast_type,
        start_date=start_date,
        end_date=end_date,
        algorithm=algorithm,
        hyperparams=None,
        grid_search=grid_search,
        grid_params=None,
        logger=logger,
        run_evaluation=True,
        auto_promote=auto_promote,
        min_skill=min_skill,
    )

    return result


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    """Main batch training loop."""
    args = parse_args()
    logger, log_file = setup_logging()

    start_time = time.time()

    logger.info("=" * 70)
    logger.info("BATCH TRAINING - Energy Forecasting Models")
    logger.info("=" * 70)

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
    parallel = min(args.parallel, 4)  # Cap at 4

    total_models = len(countries) * len(forecast_types)

    logger.info(f"Countries: {len(countries)} ({', '.join(countries[:5])}{'...' if len(countries) > 5 else ''})")
    logger.info(f"Forecast types: {forecast_types}")
    logger.info(f"Total models to train: {total_models}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Grid search: {'ENABLED' if args.grid_search else 'DISABLED'}")
    logger.info(f"Auto-promotion: {'ENABLED' if args.auto_promote else 'DISABLED'}")
    logger.info(f"Parallel jobs: {parallel}")
    logger.info(f"Date range: {args.start} to {args.end or 'latest'}")
    logger.info("")

    # Build job list
    jobs = []
    for country in countries:
        for ftype in forecast_types:
            jobs.append((country, ftype))

    results = []
    completed = 0
    skipped = 0
    failed = 0
    promoted = 0

    if parallel > 1:
        # Parallel training
        logger.info(f"Starting parallel training with {parallel} workers...")

        with ProcessPoolExecutor(max_workers=parallel) as executor:
            future_to_job = {
                executor.submit(
                    train_single_model,
                    country, ftype,
                    args.start, args.end,
                    args.algorithm,
                    args.grid_search,
                    args.auto_promote,
                    args.min_skill,
                ): (country, ftype)
                for country, ftype in jobs
            }

            for future in as_completed(future_to_job):
                country, ftype = future_to_job[future]
                completed += 1

                try:
                    result = future.result()
                    results.append(result)

                    if result['status'] == 'skipped':
                        skipped += 1
                        logger.info(f"[{completed}/{total_models}] SKIP {country}/{ftype}")
                    elif result['status'] == 'failed':
                        failed += 1
                        logger.info(f"[{completed}/{total_models}] FAIL {country}/{ftype}: {result.get('error')}")
                    else:
                        if result.get('promoted'):
                            promoted += 1
                        skill = result.get('skill_scores', {}).get('skill_vs_persistence', 0)
                        status = "[PROMOTED]" if result.get('promoted') else ""
                        logger.info(f"[{completed}/{total_models}] OK {country}/{ftype}: "
                                   f"MAE={result['metrics'].get('mae', 0):.1f}, skill={skill:+.3f} {status}")

                except Exception as e:
                    failed += 1
                    logger.error(f"[{completed}/{total_models}] ERROR {country}/{ftype}: {e}")
                    results.append({
                        'country_code': country,
                        'forecast_type': ftype,
                        'status': 'failed',
                        'error': str(e),
                    })
    else:
        # Sequential training
        from train import train_model, should_train_renewable_type

        for country, ftype in jobs:
            completed += 1

            # Check if we should skip this renewable type
            if ftype in config.RENEWABLE_TYPES:
                if not should_train_renewable_type(country, ftype, logger):
                    skipped += 1
                    results.append({
                        'country_code': country,
                        'forecast_type': ftype,
                        'status': 'skipped',
                        'reason': 'Insufficient data',
                    })
                    continue

            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (total_models - completed) if completed > 0 else 0

            logger.info(f"[{completed}/{total_models}] Training {country}/{ftype} "
                       f"(elapsed: {format_duration(elapsed)}, ETA: {format_duration(eta)})")

            try:
                result = train_model(
                    country_code=country,
                    forecast_type=ftype,
                    start_date=args.start,
                    end_date=args.end,
                    algorithm=args.algorithm,
                    hyperparams=None,
                    grid_search=args.grid_search,
                    grid_params=None,
                    logger=logger,
                    run_evaluation=True,
                    auto_promote=args.auto_promote,
                    min_skill=args.min_skill,
                )
                results.append(result)

                if result['status'] == 'failed':
                    failed += 1
                elif result.get('promoted'):
                    promoted += 1

            except Exception as e:
                failed += 1
                logger.error(f"    ERROR: {e}")
                results.append({
                    'country_code': country,
                    'forecast_type': ftype,
                    'status': 'failed',
                    'error': str(e),
                })

                if not args.continue_on_error:
                    logger.error("Stopping due to error (use --continue-on-error to keep going)")
                    break

    # Calculate final stats
    total_time = time.time() - start_time
    success = sum(1 for r in results if r['status'] == 'success')

    # Write report CSV
    report_file = args.report_file
    if not report_file:
        report_file = config.LOGS_DIR / f"train_all_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(report_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'country_code', 'forecast_type', 'status', 'mae', 'mape', 'rmse',
            'skill_vs_persistence', 'skill_vs_seasonal', 'promoted', 'model_version', 'error'
        ])

        for r in results:
            writer.writerow([
                r.get('country_code'),
                r.get('forecast_type'),
                r.get('status'),
                r.get('metrics', {}).get('mae'),
                r.get('metrics', {}).get('mape'),
                r.get('metrics', {}).get('rmse'),
                r.get('skill_scores', {}).get('skill_vs_persistence'),
                r.get('skill_scores', {}).get('skill_vs_seasonal_naive'),
                r.get('promoted', False),
                r.get('model_version'),
                r.get('error'),
            ])

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("BATCH TRAINING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total models:     {total_models}")
    logger.info(f"Successful:       {success}")
    logger.info(f"Failed:           {failed}")
    logger.info(f"Skipped:          {skipped}")
    if args.auto_promote:
        logger.info(f"Promoted:         {promoted}")
    logger.info(f"Total time:       {format_duration(total_time)}")
    logger.info(f"Avg per model:    {format_duration(total_time / max(success, 1))}")
    logger.info(f"Report saved:     {report_file}")
    logger.info(f"Log file:         {log_file}")

    # Show top performers
    if success > 0:
        logger.info("")
        logger.info("Top 10 Models by Skill Score:")
        sorted_results = sorted(
            [r for r in results if r['status'] == 'success'],
            key=lambda x: x.get('skill_scores', {}).get('skill_vs_persistence', -999),
            reverse=True,
        )[:10]

        for i, r in enumerate(sorted_results, 1):
            skill = r.get('skill_scores', {}).get('skill_vs_persistence', 0)
            mae = r.get('metrics', {}).get('mae', 0)
            logger.info(f"  {i:2d}. {r['country_code']}/{r['forecast_type']}: "
                       f"skill={skill:+.4f}, MAE={mae:.1f}")

    logger.info("")
    logger.info("[DONE] Batch training complete!")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
