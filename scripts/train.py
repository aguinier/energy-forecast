#!/usr/bin/env python3
"""
Training Script for Energy Forecasting Models

Enhanced with:
- Walk-forward validation
- Optuna hyperparameter optimization
- Multi-algorithm comparison
- Evaluation pipeline and auto-promotion

Usage:
    python scripts/train.py --countries all --types all
    python scripts/train.py --countries DE,FR --types load
    python scripts/train.py --countries DE --types load --algorithm lightgbm
    python scripts/train.py --countries DE --types load --grid-search
    python scripts/train.py --countries DE --types load --optuna --n-trials 50
    python scripts/train.py --countries DE --types load --walk-forward
    python scripts/train.py --countries DE --types load --algorithms xgboost,lightgbm,catboost
    python scripts/train.py --countries DE --types load --auto-promote
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config
import db
from db import (
    create_forecasts_table,
    initialize_all_tables,
    save_model_evaluation,
    get_latest_evaluation,
    get_latest_data_timestamp,
)
from forecaster import Forecaster, CascadeForecaster
from metrics import format_metrics, calculate_all_metrics
from baselines import compute_baseline_metrics, PersistenceBaseline, SeasonalNaiveBaseline
from model_registry import get_registry
from deployment import auto_promote_if_better
from validation import WalkForwardValidator, TimeSeriesValidator, format_validation_report
from hyperopt import OptunaOptimizer, compare_algorithms as optuna_compare_algorithms
from feature_selection import FeatureSelector
from features import create_all_features, get_feature_columns
from datetime import timedelta
import pytz


def setup_logging():
    """Setup logging to file and console."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = config.LOGS_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
        description='Train energy forecasting models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train all models for all countries
    python scripts/train.py --countries all --types all

    # Train load model for Germany and France
    python scripts/train.py --countries DE,FR --types load

    # Train all models for Germany
    python scripts/train.py --countries DE --types all

    # Train with custom date range
    python scripts/train.py --countries DE --types load --start 2023-01-01

    # Train with LightGBM algorithm
    python scripts/train.py --countries DE --types load --algorithm lightgbm

    # Train with custom hyperparameters
    python scripts/train.py --countries DE --types load --hyperparams '{"n_estimators": 300, "max_depth": 6}'

    # Train with grid search
    python scripts/train.py --countries DE --types load --grid-search

    # Train with custom grid search parameters
    python scripts/train.py --countries DE --types load --grid-search --grid-params '{"max_depth": [4, 6, 8]}'
        """
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
        default='all',
        help='Comma-separated forecast types (load,price,renewable) or "all" (default: all)'
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
        '--hyperparams',
        type=str,
        default=None,
        help='JSON string of hyperparameters to override defaults'
    )

    parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Enable grid search hyperparameter tuning'
    )

    parser.add_argument(
        '--grid-params',
        type=str,
        default=None,
        help='JSON string of grid search parameter space'
    )

    parser.add_argument(
        '--auto-promote',
        action='store_true',
        help='Automatically promote models that beat baselines and production'
    )

    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip baseline evaluation (faster training, no promotion possible)'
    )

    parser.add_argument(
        '--min-skill',
        type=float,
        default=0.0,
        help='Minimum skill score vs persistence for auto-promotion (default: 0.0)'
    )

    # Walk-forward validation
    parser.add_argument(
        '--walk-forward',
        action='store_true',
        help='Use walk-forward validation instead of simple holdout'
    )

    parser.add_argument(
        '--n-folds',
        type=int,
        default=6,
        help='Number of walk-forward validation folds (default: 6)'
    )

    # Optuna hyperparameter optimization
    parser.add_argument(
        '--optuna',
        action='store_true',
        help='Use Optuna for Bayesian hyperparameter optimization'
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of Optuna optimization trials (default: 50)'
    )

    # Multi-algorithm comparison
    parser.add_argument(
        '--algorithms',
        type=str,
        default=None,
        help='Comma-separated list of algorithms to compare (e.g., xgboost,lightgbm,catboost)'
    )

    # Feature selection
    parser.add_argument(
        '--feature-selection',
        action='store_true',
        help='Enable automated feature selection'
    )

    # Cascade architecture
    parser.add_argument(
        '--cascade',
        action='store_true',
        help='Use cascade architecture for price: Stage 1 (load+renewable) → Stage 2 (price)'
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
        return config.FORECAST_TYPES + config.RENEWABLE_TYPES
    return [t.strip().lower() for t in types_arg.split(',')]


def should_train_renewable_type(country_code: str, renewable_type: str, logger: logging.Logger) -> bool:
    """
    Determine if we should train this renewable type for this country

    Returns:
        True if data availability is sufficient, False otherwise
    """
    # Check config-based exclusions
    if country_code in config.SKIP_RENEWABLE_TYPES:
        if renewable_type in config.SKIP_RENEWABLE_TYPES[country_code]:
            logger.info(f"Skipping {renewable_type} for {country_code} (configured skip)")
            return False

    # Check data availability (last 30 days)
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=30)

    try:
        df = db.load_renewable_type_data(
            country_code,
            renewable_type,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        if df.empty:
            logger.warning(f"No data for {country_code} {renewable_type}")
            return False

        # Require sufficient non-zero values
        # Solar has lower threshold because it naturally produces zero at night
        threshold = 0.30 if renewable_type == 'solar' else 0.50
        non_zero_pct = (df['target_value'] > 0).sum() / len(df)

        if non_zero_pct < threshold:
            logger.info(f"Skipping {country_code} {renewable_type} (only {non_zero_pct:.1%} non-zero, threshold {threshold:.0%})")
            return False

        logger.info(f"Training {country_code} {renewable_type} ({non_zero_pct:.1%} non-zero data)")
        return True

    except Exception as e:
        logger.warning(f"Error checking data for {country_code} {renewable_type}: {e}")
        return False


def train_model(
    country_code: str,
    forecast_type: str,
    start_date: str,
    end_date: str,
    algorithm: str,
    hyperparams: Optional[Dict[str, Any]],
    grid_search: bool,
    grid_params: Optional[Dict[str, List]],
    logger: logging.Logger,
    run_evaluation: bool = True,
    auto_promote: bool = False,
    min_skill: float = 0.0,
    walk_forward: bool = False,
    n_folds: int = 6,
    use_optuna: bool = False,
    n_trials: int = 50,
    feature_selection: bool = False,
    cascade: bool = False,
) -> dict:
    """
    Train a single model with evaluation and optional auto-promotion.

    Args:
        country_code: ISO 2-letter country code
        forecast_type: 'load', 'price', or 'renewable'
        start_date: Training start date (YYYY-MM-DD)
        end_date: Training end date (YYYY-MM-DD)
        algorithm: 'xgboost', 'lightgbm', or 'catboost'
        hyperparams: Custom hyperparameters to override defaults
        grid_search: Enable grid search hyperparameter tuning
        grid_params: Custom grid search parameter space
        run_evaluation: Run baseline evaluation after training
        auto_promote: Auto-promote if model beats production
        min_skill: Minimum skill score for auto-promotion
        walk_forward: Use walk-forward validation
        n_folds: Number of walk-forward folds
        use_optuna: Use Optuna for hyperparameter optimization
        n_trials: Number of Optuna trials
        feature_selection: Enable automated feature selection

    Returns:
        Dictionary with training results
    """
    result = {
        'country_code': country_code,
        'forecast_type': forecast_type,
        'algorithm': algorithm,
        'status': 'failed',
        'metrics': {},
        'skill_scores': {},
        'promoted': False,
        'error': None,
        'walk_forward_results': None,
        'optuna_results': None,
        'selected_features': None,
    }

    registry = get_registry()

    try:
        logger.info(f"Training {forecast_type} for {country_code} with {algorithm}...")

        # Log enabled features
        features_enabled = []
        if walk_forward:
            features_enabled.append(f"walk-forward ({n_folds} folds)")
        if use_optuna:
            features_enabled.append(f"Optuna ({n_trials} trials)")
        if grid_search:
            features_enabled.append("grid-search")
        if feature_selection:
            features_enabled.append("feature-selection")
        if cascade:
            features_enabled.append("cascade (load+renewable→price)")
        if features_enabled:
            logger.info(f"  Enabled: {', '.join(features_enabled)}")

        # ── Cascade architecture (price only) ──────────────────────
        if cascade:
            if forecast_type != 'price':
                logger.warning(f"  Cascade only applies to 'price', skipping for '{forecast_type}'")
            else:
                logger.info("  Using CASCADE architecture: Stage 1 (load+renewable) → Stage 2 (price)")
                cascade_forecaster = CascadeForecaster(
                    country_code=country_code,
                    algorithm=algorithm,
                    hyperparams=hyperparams,
                )
                metrics = cascade_forecaster.train(
                    start_date=start_date,
                    end_date=end_date,
                )

                # Save cascade model
                cascade_path = cascade_forecaster.save()
                result['status'] = 'success'
                result['metrics'] = metrics
                result['model_version'] = cascade_forecaster.model_version
                result['cascade'] = True
                result['stage1_metrics'] = cascade_forecaster.stage1_metrics

                logger.info(f"[OK] CASCADE {country_code} price: "
                             f"MAE={metrics.get('mae', 0):.2f}, MAPE={metrics.get('mape', 0):.2f}%")

                # Run evaluation with cascade-aware predict
                if run_evaluation:
                    logger.info("    Running cascade baseline evaluation...")
                    skill_scores = evaluate_cascade_against_baselines(
                        cascade_forecaster, country_code, start_date, end_date, logger
                    )
                    result['skill_scores'] = skill_scores
                    skill_vs_persist = skill_scores.get('skill_vs_persistence', 0)
                    logger.info(f"    Skill vs persistence: {skill_vs_persist:.4f}")

                return result

        # Load and prepare data for potential Optuna/feature selection
        if use_optuna or feature_selection:
            # Handle None end_date - get latest available data
            effective_end_date = end_date
            if effective_end_date is None:
                latest = get_latest_data_timestamp(country_code, forecast_type)
                if latest:
                    effective_end_date = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    effective_end_date = datetime.now().strftime("%Y-%m-%d")

            df = db.load_training_data(country_code, forecast_type, start_date, effective_end_date)
            if df.empty:
                raise ValueError(f"No training data for {country_code} {forecast_type}")

            df = create_all_features(df, forecast_type, country_code=country_code)
            feature_cols = [c for c in get_feature_columns(forecast_type) if c in df.columns]

            X = df[feature_cols].values
            y = df['target_value'].values

            # Feature selection
            if feature_selection:
                logger.info("  Running feature selection...")
                selector = FeatureSelector(algorithm=algorithm, min_features=10, patience=3)
                fs_result = selector.select_features(X, y, feature_cols, forecast_type)
                selected_feature_names = fs_result.selected_features
                result['selected_features'] = selected_feature_names
                logger.info(f"  Selected {len(selected_feature_names)}/{len(feature_cols)} features")

                # Update feature columns for subsequent training
                feature_cols = selected_feature_names
                X = df[feature_cols].values

            # Optuna hyperparameter optimization
            if use_optuna:
                logger.info(f"  Running Optuna optimization ({n_trials} trials)...")
                optimizer = OptunaOptimizer(
                    algorithm=algorithm,
                    n_trials=n_trials,
                    metric='mae',
                )
                optuna_result = optimizer.optimize(X, y, forecast_type, show_progress=False)
                hyperparams = optuna_result.best_params
                result['optuna_results'] = {
                    'best_params': optuna_result.best_params,
                    'best_score': optuna_result.best_score,
                    'n_trials': optuna_result.n_trials,
                }
                logger.info(f"  Optuna best MAE: {optuna_result.best_score:.2f}")

        # Create forecaster with potentially optimized hyperparameters
        forecaster = Forecaster(
            country_code,
            forecast_type,
            algorithm=algorithm,
            hyperparams=hyperparams
        )

        # Train with walk-forward validation or standard training
        if walk_forward:
            metrics = forecaster.train_with_walk_forward(
                start_date=start_date,
                end_date=end_date,
                n_folds=n_folds,
            )
            result['walk_forward_results'] = {
                'n_folds': n_folds,
                'aggregate_metrics': metrics,
            }
        else:
            metrics = forecaster.train(
                start_date=start_date,
                end_date=end_date,
                grid_search=grid_search,
                grid_params=grid_params
            )

        # Save as candidate (not directly to production)
        model_data = forecaster._get_model_data()
        model_path = registry.save_model(model_data, country_code, forecast_type, "candidate")

        result['status'] = 'success'
        result['metrics'] = metrics
        result['model_version'] = forecaster.model_version
        result['hyperparams'] = forecaster.hyperparams

        if forecaster.grid_search_results:
            result['grid_search_results'] = forecaster.grid_search_results

        logger.info(f"[OK] {country_code} {forecast_type}: MAE={metrics.get('mae', 0):.2f}, MAPE={metrics.get('mape', 0):.2f}%")

        # Run baseline evaluation
        if run_evaluation:
            logger.info(f"    Running baseline evaluation...")
            skill_scores = evaluate_against_baselines(
                forecaster, country_code, forecast_type, start_date, end_date, logger
            )
            result['skill_scores'] = skill_scores

            # Save evaluation to database
            save_model_evaluation(
                country_code=country_code,
                forecast_type=forecast_type,
                model_version=forecaster.model_version,
                metrics=metrics,
                skill_scores=skill_scores,
                training_samples=forecaster.training_samples if hasattr(forecaster, 'training_samples') else 0,
                test_samples=forecaster.test_samples if hasattr(forecaster, 'test_samples') else 0,
                is_baseline=False,
                model_location="candidate",
            )

            skill_vs_persist = skill_scores.get('skill_vs_persistence', 0)
            logger.info(f"    Skill vs persistence: {skill_vs_persist:.4f}")

            # Auto-promote if enabled and model qualifies
            if auto_promote:
                promotion_result = auto_promote_if_better(
                    country_code, forecast_type,
                    min_skill_threshold=min_skill,
                )
                result['promoted'] = promotion_result.promoted
                if promotion_result.promoted:
                    logger.info(f"    [PROMOTED] {promotion_result.reason}")
                else:
                    logger.info(f"    [NOT PROMOTED] {promotion_result.reason}")

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"[FAIL] {country_code} {forecast_type}: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return result


def evaluate_against_baselines(
    forecaster: Forecaster,
    country_code: str,
    forecast_type: str,
    start_date: str,
    end_date: str,
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    Evaluate a trained model against baseline forecasts.

    Returns:
        Dictionary with skill scores vs each baseline
    """
    skill_scores = {
        'skill_vs_persistence': 0.0,
        'skill_vs_seasonal_naive': 0.0,
    }

    try:
        # Load validation data (last 30 days of training period)
        val_end = end_date or datetime.now().strftime('%Y-%m-%d')
        val_start = (datetime.strptime(val_end, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')

        val_df = db.load_training_data(country_code, forecast_type, val_start, val_end)

        if val_df.empty or len(val_df) < 48:  # Need at least 48 hours
            logger.warning(f"    Insufficient validation data for evaluation")
            return skill_scores

        # Ensure we have timestamp column
        if 'timestamp_utc' not in val_df.columns:
            logger.warning(f"    No timestamp column in validation data")
            return skill_scores

        # Create features and predict
        from features import create_all_features
        val_df = create_all_features(val_df, forecast_type)
        val_df = val_df.dropna()

        if val_df.empty:
            return skill_scores

        X_val = val_df[forecaster.feature_columns].values
        y_true = val_df['target_value'].values

        y_pred = forecaster.model.predict(X_val)

        # Create historical series for baselines
        hist_series = pd.Series(
            val_df['target_value'].values,
            index=pd.to_datetime(val_df['timestamp_utc'])
        )
        target_timestamps = pd.to_datetime(val_df['timestamp_utc'])

        # Compute baseline predictions
        persistence = PersistenceBaseline(horizon_hours=48)
        seasonal = SeasonalNaiveBaseline(horizon_hours=48)

        y_persistence = persistence.predict_for_target(hist_series, target_timestamps)
        y_seasonal = seasonal.predict_for_target(hist_series, target_timestamps)

        # Filter valid predictions
        valid_persist = ~np.isnan(y_persistence)
        valid_seasonal = ~np.isnan(y_seasonal)

        # Compute skill scores
        from metrics import skill_score as compute_skill

        if valid_persist.sum() > 0:
            skill_scores['skill_vs_persistence'] = compute_skill(
                y_true[valid_persist],
                y_pred[valid_persist],
                y_persistence[valid_persist],
                metric='mae'
            )

        if valid_seasonal.sum() > 0:
            skill_scores['skill_vs_seasonal_naive'] = compute_skill(
                y_true[valid_seasonal],
                y_pred[valid_seasonal],
                y_seasonal[valid_seasonal],
                metric='mae'
            )

    except Exception as e:
        logger.warning(f"    Evaluation error: {e}")

    return skill_scores


def evaluate_cascade_against_baselines(
    cascade_forecaster: CascadeForecaster,
    country_code: str,
    start_date: str,
    end_date: str,
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    Evaluate a cascade model against baselines.

    Unlike standard evaluation, this generates Stage 1 predictions first,
    then uses them as features for the Stage 2 price model.
    """
    skill_scores = {
        'skill_vs_persistence': 0.0,
        'skill_vs_seasonal_naive': 0.0,
    }

    try:
        # Load recent validation data (last 30 days)
        val_end = end_date or datetime.now().strftime('%Y-%m-%d')
        val_start = (datetime.strptime(val_end, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')

        # Load price data with features
        val_df = db.load_training_data(country_code, "price", val_start, val_end)
        if val_df.empty or len(val_df) < 48:
            logger.warning("    Insufficient validation data")
            return skill_scores

        val_df = create_all_features(val_df, "price", country_code=country_code)
        val_df = val_df.dropna()

        if val_df.empty:
            return skill_scores

        # Load load/renewable data for cascade features
        load_val = db.load_training_data(country_code, "load", val_start, val_end)
        ren_val = db.load_training_data(country_code, "renewable", val_start, val_end)

        if load_val.empty or ren_val.empty:
            logger.warning("    Missing load/renewable data for cascade evaluation")
            return skill_scores

        load_val = create_all_features(load_val, "load", country_code=country_code)
        ren_val = create_all_features(ren_val, "renewable", country_code=country_code)

        # Generate Stage 1 predictions (cascade features)
        # We need to merge by timestamp
        load_val["_ts"] = pd.to_datetime(load_val["timestamp_utc"])
        ren_val["_ts"] = pd.to_datetime(ren_val["timestamp_utc"])
        val_df["_ts"] = pd.to_datetime(val_df["timestamp_utc"])

        # Build a combined features DataFrame for predict_stage1
        # Use val_df as base and add load/renewable columns
        load_indexed = load_val.set_index("_ts")
        ren_indexed = ren_val.set_index("_ts")

        # For each price timestamp, get load and renewable features
        combined = val_df.copy()

        # Add load/renewable features that Stage 1 models need
        for col in cascade_forecaster.load_model.feature_columns:
            if col not in combined.columns and col in load_indexed.columns:
                combined[col] = combined["_ts"].map(
                    load_indexed[col].to_dict() if not load_indexed.index.duplicated().any()
                    else load_indexed[~load_indexed.index.duplicated()][col].to_dict()
                )

        for col in cascade_forecaster.renewable_model.feature_columns:
            if col not in combined.columns and col in ren_indexed.columns:
                combined[col] = combined["_ts"].map(
                    ren_indexed[col].to_dict() if not ren_indexed.index.duplicated().any()
                    else ren_indexed[~ren_indexed.index.duplicated()][col].to_dict()
                )

        combined = combined.fillna(0)

        # Generate cascade features via Stage 1 models
        cascade_preds = cascade_forecaster.predict_stage1(combined)
        combined["cascade_load_prediction"] = cascade_preds["cascade_load_prediction"]
        combined["cascade_renewable_prediction"] = cascade_preds["cascade_renewable_prediction"]
        combined["cascade_residual_load"] = cascade_preds["cascade_residual_load"]

        # Predict with Stage 2 model
        price_features = cascade_forecaster.price_model.feature_columns
        missing_cols = [c for c in price_features if c not in combined.columns]
        for col in missing_cols:
            combined[col] = 0

        X_val = combined[price_features].values
        y_true = combined["target_value"].values
        y_pred = cascade_forecaster.price_model.model.predict(X_val)

        # Compute baselines
        hist_series = pd.Series(
            combined["target_value"].values,
            index=pd.to_datetime(combined["timestamp_utc"])
        )
        target_timestamps = pd.to_datetime(combined["timestamp_utc"])

        persistence = PersistenceBaseline(horizon_hours=48)
        seasonal = SeasonalNaiveBaseline(horizon_hours=48)

        y_persistence = persistence.predict_for_target(hist_series, target_timestamps)
        y_seasonal = seasonal.predict_for_target(hist_series, target_timestamps)

        from metrics import skill_score as compute_skill

        valid_persist = ~np.isnan(y_persistence)
        valid_seasonal = ~np.isnan(y_seasonal)

        if valid_persist.sum() > 0:
            skill_scores['skill_vs_persistence'] = compute_skill(
                y_true[valid_persist], y_pred[valid_persist],
                y_persistence[valid_persist], metric='mae'
            )
        if valid_seasonal.sum() > 0:
            skill_scores['skill_vs_seasonal_naive'] = compute_skill(
                y_true[valid_seasonal], y_pred[valid_seasonal],
                y_seasonal[valid_seasonal], metric='mae'
            )

    except Exception as e:
        logger.warning(f"    Cascade evaluation error: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return skill_scores


def main():
    """Main training loop."""
    args = parse_args()
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("Energy Forecasting Model Training")
    logger.info("=" * 60)

    # Validate config
    try:
        config.validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Ensure all tables exist
    initialize_all_tables()

    # Parse arguments
    countries = get_countries(args.countries)
    forecast_types = get_forecast_types(args.types)

    # Parse JSON arguments
    hyperparams = None
    if args.hyperparams:
        try:
            hyperparams = json.loads(args.hyperparams)
            logger.info(f"Custom hyperparams: {hyperparams}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --hyperparams: {e}")
            sys.exit(1)

    grid_params = None
    if args.grid_params:
        try:
            grid_params = json.loads(args.grid_params)
            logger.info(f"Custom grid params: {grid_params}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --grid-params: {e}")
            sys.exit(1)

    # Parse algorithms list
    algorithms = [args.algorithm]
    if args.algorithms:
        algorithms = [a.strip().lower() for a in args.algorithms.split(',')]

    logger.info(f"Countries: {countries}")
    logger.info(f"Forecast types: {forecast_types}")
    logger.info(f"Algorithm(s): {algorithms}")
    logger.info(f"Date range: {args.start} to {args.end or 'latest'}")
    if args.grid_search:
        logger.info("Grid search: ENABLED")
    if args.walk_forward:
        logger.info(f"Walk-forward validation: ENABLED ({args.n_folds} folds)")
    if args.optuna:
        logger.info(f"Optuna optimization: ENABLED ({args.n_trials} trials)")
    if args.feature_selection:
        logger.info("Feature selection: ENABLED")
    if args.auto_promote:
        logger.info(f"Auto-promotion: ENABLED (min_skill={args.min_skill})")
    if args.skip_evaluation:
        logger.info("Evaluation: DISABLED")
    logger.info("")

    # Track results
    results = []
    total = len(countries) * len(forecast_types) * len(algorithms)
    completed = 0
    skipped = 0

    for country in countries:
        for forecast_type in forecast_types:
            # Check if we should skip this renewable type
            if forecast_type in config.RENEWABLE_TYPES:
                if not should_train_renewable_type(country, forecast_type, logger):
                    skipped += len(algorithms)
                    completed += len(algorithms)
                    continue

            for algorithm in algorithms:
                completed += 1
                logger.info(f"[{completed}/{total}] Training {country} {forecast_type} ({algorithm})")

                result = train_model(
                    country,
                    forecast_type,
                    args.start,
                    args.end,
                    algorithm,
                    hyperparams,
                    args.grid_search,
                    grid_params,
                    logger,
                    run_evaluation=not args.skip_evaluation,
                    auto_promote=args.auto_promote,
                    min_skill=args.min_skill,
                    walk_forward=args.walk_forward,
                    n_folds=args.n_folds,
                    use_optuna=args.optuna,
                    n_trials=args.n_trials,
                    feature_selection=args.feature_selection,
                    cascade=args.cascade,
                )
                results.append(result)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)

    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    promoted = sum(1 for r in results if r.get('promoted', False))

    logger.info(f"Total: {len(results)}, Success: {success}, Failed: {failed}, Skipped: {skipped}")

    if args.auto_promote:
        logger.info(f"Promoted to production: {promoted}")

    # Show skill score summary
    if not args.skip_evaluation and success > 0:
        logger.info("\nSkill Scores (vs Persistence):")
        for r in results:
            if r['status'] == 'success':
                skill = r.get('skill_scores', {}).get('skill_vs_persistence', 0)
                status = "[BETTER]" if skill > 0 else "[WORSE]"
                promoted_str = " [PROMOTED]" if r.get('promoted') else ""
                logger.info(f"  {r['country_code']}/{r['forecast_type']}: {skill:+.4f} {status}{promoted_str}")

    if failed > 0:
        logger.info("\nFailed models:")
        for r in results:
            if r['status'] == 'failed':
                logger.info(f"  - {r['country_code']} {r['forecast_type']}: {r['error']}")

    logger.info("\n[DONE] Training complete!")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
