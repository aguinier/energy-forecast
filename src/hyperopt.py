"""
Hyperparameter Optimization with Optuna

Provides Bayesian hyperparameter optimization for energy forecasting models.
Supports XGBoost, LightGBM, and CatBoost with proper time-series cross-validation.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from metrics import mae, rmse, mape

logger = logging.getLogger("energy_forecast")


# ============================================================================
# SEARCH SPACES
# ============================================================================

def get_xgboost_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Define XGBoost hyperparameter search space."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0,
        'early_stopping_rounds': 50,
    }


def get_lightgbm_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Define LightGBM hyperparameter search space."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'objective': 'regression',
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1,
    }


def get_catboost_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Define CatBoost hyperparameter search space."""
    return {
        'iterations': trial.suggest_int('iterations', 200, 1000, step=100),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'random_seed': 42,
        'verbose': 0,
        'thread_count': -1,
    }


SEARCH_SPACE_FUNCTIONS = {
    'xgboost': get_xgboost_search_space,
    'lightgbm': get_lightgbm_search_space,
    'catboost': get_catboost_search_space,
}


# ============================================================================
# OPTUNA OPTIMIZER
# ============================================================================

@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    optimization_history: List[Dict]
    study: optuna.Study


class OptunaOptimizer:
    """
    Bayesian hyperparameter optimizer using Optuna.

    Uses Tree-structured Parzen Estimator (TPE) for efficient search
    and MedianPruner for early stopping of unpromising trials.

    Example:
        optimizer = OptunaOptimizer('xgboost', n_trials=50)
        result = optimizer.optimize(X_train, y_train, X_val, y_val)
        best_model = optimizer.get_best_model(result.best_params)
    """

    def __init__(
        self,
        algorithm: str,
        n_trials: int = 50,
        n_cv_splits: int = 3,
        metric: str = 'mae',
        seed: int = 42,
        timeout: Optional[int] = None,
    ):
        """
        Initialize Optuna optimizer.

        Args:
            algorithm: 'xgboost', 'lightgbm', or 'catboost'
            n_trials: Number of optimization trials (default: 50)
            n_cv_splits: Number of cross-validation splits
            metric: Optimization metric ('mae', 'rmse', 'mape')
            seed: Random seed for reproducibility
            timeout: Maximum optimization time in seconds (optional)
        """
        if algorithm not in SEARCH_SPACE_FUNCTIONS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self.algorithm = algorithm
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.metric = metric
        self.seed = seed
        self.timeout = timeout

        self._search_space_fn = SEARCH_SPACE_FUNCTIONS[algorithm]

    def _create_model(self, params: Dict[str, Any]) -> Any:
        """Create model instance with given parameters."""
        if self.algorithm == 'xgboost':
            # Remove early_stopping_rounds from params for model creation
            model_params = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
            return XGBRegressor(**model_params)
        elif self.algorithm == 'lightgbm':
            return LGBMRegressor(**params)
        elif self.algorithm == 'catboost':
            return CatBoostRegressor(**params)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute optimization metric."""
        if self.metric == 'mae':
            return mae(y_true, y_pred)
        elif self.metric == 'rmse':
            return rmse(y_true, y_pred)
        elif self.metric == 'mape':
            return mape(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _objective(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray,
        forecast_type: str,
    ) -> float:
        """Optuna objective function with time-series cross-validation."""
        # Sample hyperparameters
        params = self._search_space_fn(trial)

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create and train model
            model = self._create_model(params)

            if self.algorithm == 'xgboost':
                early_stop_rounds = params.get('early_stopping_rounds', 50)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            elif self.algorithm == 'lightgbm':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(50, verbose=False)],
                )
            elif self.algorithm == 'catboost':
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False,
                )

            # Predict and compute metric
            y_pred = model.predict(X_val)
            score = self._compute_metric(y_val, y_pred)
            scores.append(score)

            # Report intermediate result for pruning
            trial.report(np.mean(scores), fold_idx)

            # Prune unpromising trials
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        forecast_type: str = "load",
        show_progress: bool = True,
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.

        Args:
            X: Feature matrix
            y: Target vector
            forecast_type: Type of forecast for metric thresholds
            show_progress: Whether to show progress bar

        Returns:
            OptimizationResult with best parameters and study details
        """
        # Create sampler and pruner
        sampler = TPESampler(seed=self.seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
        )

        # Suppress Optuna logging if not showing progress
        if not show_progress:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        study.optimize(
            lambda trial: self._objective(trial, X, y, forecast_type),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=show_progress,
        )

        # Collect results
        optimization_history = [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state),
            }
            for t in study.trials
        ]

        logger.info(f"Optuna optimization complete: best {self.metric}={study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            optimization_history=optimization_history,
            study=study,
        )

    def get_best_model(self, params: Dict[str, Any]) -> Any:
        """
        Create a model instance with the best parameters.

        Args:
            params: Best parameters from optimization

        Returns:
            Untrained model instance with optimal hyperparameters
        """
        # Merge with default params for any missing values
        default_params = config.get_default_params(self.algorithm)
        full_params = {**default_params, **params}
        return self._create_model(full_params)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def optimize_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    algorithm: str = 'xgboost',
    n_trials: int = 50,
    forecast_type: str = 'load',
    metric: str = 'mae',
) -> Dict[str, Any]:
    """
    Convenience function to optimize hyperparameters.

    Args:
        X: Feature matrix
        y: Target vector
        algorithm: 'xgboost', 'lightgbm', or 'catboost'
        n_trials: Number of optimization trials
        forecast_type: Type of forecast
        metric: Optimization metric

    Returns:
        Dictionary with best parameters
    """
    optimizer = OptunaOptimizer(
        algorithm=algorithm,
        n_trials=n_trials,
        metric=metric,
    )
    result = optimizer.optimize(X, y, forecast_type)
    return result.best_params


def compare_algorithms(
    X: np.ndarray,
    y: np.ndarray,
    algorithms: List[str] = None,
    n_trials_per_algo: int = 30,
    forecast_type: str = 'load',
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple algorithms with optimized hyperparameters.

    Args:
        X: Feature matrix
        y: Target vector
        algorithms: List of algorithms to compare (default: all supported)
        n_trials_per_algo: Number of optimization trials per algorithm
        forecast_type: Type of forecast

    Returns:
        Dictionary mapping algorithm name to {best_params, best_score}
    """
    if algorithms is None:
        algorithms = ['xgboost', 'lightgbm', 'catboost']

    results = {}
    for algo in algorithms:
        logger.info(f"Optimizing {algo}...")
        optimizer = OptunaOptimizer(algorithm=algo, n_trials=n_trials_per_algo)
        result = optimizer.optimize(X, y, forecast_type, show_progress=True)
        results[algo] = {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'n_trials': result.n_trials,
        }

    # Log comparison
    logger.info("\n=== Algorithm Comparison ===")
    for algo, data in sorted(results.items(), key=lambda x: x[1]['best_score']):
        logger.info(f"{algo}: MAE={data['best_score']:.4f}")

    return results


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    print("Testing Optuna hyperparameter optimization...")

    # Create sample data
    np.random.seed(42)
    n_samples = 5000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5] * [1.0, 2.0, 0.5, -1.0, 1.5], axis=1) + np.random.randn(n_samples) * 0.1

    print(f"\n1. Sample data: {X.shape}")

    # Test XGBoost optimization
    print("\n2. Testing XGBoost optimization (10 trials)...")
    optimizer = OptunaOptimizer('xgboost', n_trials=10)
    result = optimizer.optimize(X, y, forecast_type='load', show_progress=True)
    print(f"   Best MAE: {result.best_score:.4f}")
    print(f"   Best params: {result.best_params}")

    # Test algorithm comparison
    print("\n3. Testing algorithm comparison (5 trials each)...")
    comparison = compare_algorithms(X, y, n_trials_per_algo=5)
    for algo, data in comparison.items():
        print(f"   {algo}: MAE={data['best_score']:.4f}")

    print("\n[OK] Hyperparameter optimization tests complete!")
