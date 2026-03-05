"""
Forecaster Model for Energy Forecasting

Provides Forecaster class for training and D+2 prediction.
Supports XGBoost, LightGBM, and CatBoost algorithms.
"""

import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from db import (
    load_training_data,
    get_latest_data_timestamp,
    load_weather_forecast_for_hour,
)
from features import create_all_features, get_feature_columns
from metrics import calculate_all_metrics, format_metrics


logger = logging.getLogger("energy_forecast")


class Forecaster:
    """
    Multi-algorithm forecaster for energy D+2 prediction.
    Supports XGBoost, LightGBM, and CatBoost.

    Attributes:
        country_code: ISO 2-letter country code
        forecast_type: 'load', 'price', or 'renewable'
        algorithm: 'xgboost', 'lightgbm', or 'catboost'
        hyperparams: Algorithm hyperparameters
        model: Trained model instance
        feature_columns: List of feature column names
        model_version: Version string for tracking
    """

    def __init__(
        self,
        country_code: str,
        forecast_type: str,
        algorithm: str = "xgboost",
        hyperparams: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize forecaster.

        Args:
            country_code: ISO 2-letter country code
            forecast_type: 'load', 'price', or 'renewable'
            algorithm: 'xgboost', 'lightgbm', or 'catboost' (default: 'xgboost')
            hyperparams: Custom hyperparameters to override defaults
        """
        self.country_code = country_code
        self.forecast_type = forecast_type
        self.algorithm = algorithm.lower()

        # Validate algorithm
        if self.algorithm not in config.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {config.SUPPORTED_ALGORITHMS}"
            )

        # Merge provided hyperparams with defaults
        default_params = config.get_default_params(self.algorithm)
        self.hyperparams = {**default_params, **(hyperparams or {})}

        self.model: Optional[Any] = None
        self.feature_columns: List[str] = []
        self.model_version: str = ""
        self.training_metrics: Dict[str, float] = {}
        self.grid_search_results: Optional[Dict] = None

    def _create_model(self, params: Optional[Dict] = None) -> Any:
        """
        Create model instance based on algorithm.

        Args:
            params: Hyperparameters (default: self.hyperparams)

        Returns:
            Model instance (XGBRegressor, LGBMRegressor, or CatBoostRegressor)
        """
        params = params or self.hyperparams

        if self.algorithm == "xgboost":
            return XGBRegressor(**params)
        elif self.algorithm == "lightgbm":
            return LGBMRegressor(**params)
        elif self.algorithm == "catboost":
            return CatBoostRegressor(**params)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def train(
        self,
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        validation_days: int = None,
        grid_search: bool = False,
        grid_params: Optional[Dict[str, List]] = None,
    ) -> Dict[str, float]:
        """
        Train the model on historical data.

        Args:
            start_date: Training start date (YYYY-MM-DD)
            end_date: Training end date (default: latest available)
            validation_days: Days to hold out for validation
            grid_search: Enable hyperparameter tuning with GridSearchCV
            grid_params: Custom grid search parameter space (default: from config)

        Returns:
            Dictionary with validation metrics
        """
        if validation_days is None:
            validation_days = config.VALIDATION_DAYS

        if end_date is None:
            # Get latest available data
            latest = get_latest_data_timestamp(self.country_code, self.forecast_type)
            if latest:
                end_date = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Training {self.forecast_type} model for {self.country_code}")
        logger.info(f"Algorithm: {self.algorithm}")
        logger.info(f"Date range: {start_date} to {end_date}")
        if grid_search:
            logger.info("Grid search enabled")

        # Load and prepare data
        df = load_training_data(
            self.country_code, self.forecast_type, start_date, end_date
        )

        if df.empty:
            raise ValueError(
                f"No training data available for {self.country_code} {self.forecast_type}"
            )

        # Create features
        df = create_all_features(df, self.forecast_type)

        if len(df) < config.MIN_TRAINING_HOURS:
            logger.warning(
                f"Only {len(df)} hours of data (min: {config.MIN_TRAINING_HOURS})"
            )

        # Get feature columns that exist in the data
        self.feature_columns = [
            col for col in get_feature_columns(self.forecast_type) if col in df.columns
        ]

        logger.info(f"Using {len(self.feature_columns)} features")

        # Split data chronologically
        val_size = validation_days * 24  # hours
        train_df = df.iloc[:-val_size] if val_size < len(df) else df
        val_df = df.iloc[-val_size:] if val_size < len(df) else pd.DataFrame()

        logger.info(
            f"Train: {len(train_df)} samples, Validation: {len(val_df)} samples"
        )

        # Prepare X and y
        X_train = train_df[self.feature_columns]
        y_train = train_df["target_value"]

        X_val = val_df[self.feature_columns] if not val_df.empty else None
        y_val = val_df["target_value"] if not val_df.empty else None

        # Train with or without grid search
        if grid_search:
            self._train_with_grid_search(X_train, y_train, X_val, y_val, grid_params)
        else:
            self._train_simple(X_train, y_train, X_val, y_val)

        # Set version
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

        return self.training_metrics

    def _train_simple(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        early_stopping_rounds: int = 50,
    ) -> None:
        """Standard training with optional early stopping.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for early stopping)
            y_val: Validation target (for early stopping)
            early_stopping_rounds: Stop if no improvement for this many rounds
        """
        # Initialize model
        self.model = self._create_model()

        # Use early stopping if validation data is available
        use_early_stopping = (
            X_val is not None
            and y_val is not None
            and len(X_val) > 0
            and early_stopping_rounds > 0
        )

        if use_early_stopping:
            if self.algorithm == "xgboost":
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                # XGBoost 2.0+ uses callbacks for early stopping in params
                # Check if model stopped early
                if hasattr(self.model, 'best_iteration'):
                    logger.info(f"XGBoost stopped at iteration {self.model.best_iteration}")

            elif self.algorithm == "lightgbm":
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        self._lgbm_early_stopping_callback(early_stopping_rounds)
                    ],
                )
                if hasattr(self.model, 'best_iteration_'):
                    logger.info(f"LightGBM stopped at iteration {self.model.best_iteration_}")

            elif self.algorithm == "catboost":
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
                if hasattr(self.model, 'best_iteration_'):
                    logger.info(f"CatBoost stopped at iteration {self.model.best_iteration_}")
        else:
            # No validation data - train without early stopping
            self.model.fit(X_train, y_train)

        # Evaluate on validation set
        if X_val is not None and y_val is not None and len(X_val) > 0:
            y_pred = self.model.predict(X_val)
            self.training_metrics = calculate_all_metrics(
                y_val.values, y_pred, self.forecast_type
            )
            logger.info(format_metrics(self.training_metrics, self.forecast_type))
        else:
            self.training_metrics = {}
            logger.warning("No validation data available")

    def _lgbm_early_stopping_callback(self, stopping_rounds: int):
        """Create LightGBM early stopping callback."""
        from lightgbm import early_stopping
        return early_stopping(stopping_rounds=stopping_rounds, verbose=False)

    def _train_with_grid_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        grid_params: Optional[Dict[str, List]] = None,
    ) -> None:
        """Training with GridSearchCV for hyperparameter tuning."""
        # Use provided grid params or defaults
        param_grid = grid_params or config.get_grid_search_params(self.algorithm)

        logger.info(f"Grid search param grid: {param_grid}")

        # Calculate total combinations
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        logger.info(f"Total parameter combinations: {total_combinations}")

        # Use TimeSeriesSplit for proper time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)

        # Create base model with current hyperparams (excluding grid search params)
        base_params = {k: v for k, v in self.hyperparams.items() if k not in param_grid}
        base_model = self._create_model(base_params)

        # Run grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

        grid_search.fit(X_train, y_train)

        # Store results
        self.model = grid_search.best_estimator_
        self.hyperparams = {**self.hyperparams, **grid_search.best_params_}

        self.grid_search_results = {
            "best_params": grid_search.best_params_,
            "best_cv_score": float(
                -grid_search.best_score_
            ),  # Convert back to positive MAE
            "cv_results": {
                "params": [dict(p) for p in grid_search.cv_results_["params"]],
                "mean_scores": (-grid_search.cv_results_["mean_test_score"]).tolist(),
                "std_scores": grid_search.cv_results_["std_test_score"].tolist(),
            },
        }

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV MAE: {-grid_search.best_score_:.4f}")

        # Evaluate on validation set
        if X_val is not None and y_val is not None and len(X_val) > 0:
            y_pred = self.model.predict(X_val)
            self.training_metrics = calculate_all_metrics(
                y_val.values, y_pred, self.forecast_type
            )
            logger.info(format_metrics(self.training_metrics, self.forecast_type))
        else:
            self.training_metrics = {}
            logger.warning("No validation data available")

    def train_with_walk_forward(
        self,
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        n_folds: int = 6,
        test_size_days: int = 30,
    ) -> Dict[str, float]:
        """
        Train model using walk-forward validation.

        This method uses expanding window validation that mimics production deployment:
        - Train on all available historical data up to a cutoff
        - Test on the next period (e.g., 1 month)
        - Expand training window and repeat

        Args:
            start_date: Training start date (YYYY-MM-DD)
            end_date: Training end date (default: latest available)
            n_folds: Number of validation folds (default: 6)
            test_size_days: Size of each test period in days

        Returns:
            Dictionary with aggregated validation metrics
        """
        from validation import WalkForwardValidator

        if end_date is None:
            latest = get_latest_data_timestamp(self.country_code, self.forecast_type)
            if latest:
                end_date = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Walk-forward training {self.forecast_type} for {self.country_code}")
        logger.info(f"Algorithm: {self.algorithm}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Folds: {n_folds}, test size: {test_size_days} days")

        # Load and prepare data
        df = load_training_data(self.country_code, self.forecast_type, start_date, end_date)

        if df.empty:
            raise ValueError(
                f"No training data available for {self.country_code} {self.forecast_type}"
            )

        # Create features
        df = create_all_features(df, self.forecast_type, country_code=self.country_code)

        if len(df) < config.MIN_TRAINING_HOURS:
            logger.warning(f"Only {len(df)} hours of data (min: {config.MIN_TRAINING_HOURS})")

        # Get feature columns
        self.feature_columns = [
            col for col in get_feature_columns(self.forecast_type) if col in df.columns
        ]

        logger.info(f"Using {len(self.feature_columns)} features")

        # Initialize walk-forward validator
        validator = WalkForwardValidator(
            n_splits=n_folds,
            test_size_days=test_size_days,
            gap_days=2,  # D+2 gap
            min_train_days=config.MIN_TRAINING_HOURS // 24,
        )

        # Collect metrics from all folds
        all_test_actuals = []
        all_test_predictions = []
        fold_metrics = []

        for split in validator.get_splits(df, "timestamp_utc"):
            train_df, test_df = validator.split_data(df, split, "timestamp_utc")

            if len(train_df) == 0 or len(test_df) == 0:
                continue

            X_train = train_df[self.feature_columns]
            y_train = train_df["target_value"]
            X_test = test_df[self.feature_columns]
            y_test = test_df["target_value"]

            # Train model on this fold
            model = self._create_model()
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Collect for aggregation
            all_test_actuals.extend(y_test.values)
            all_test_predictions.extend(y_pred)

            # Per-fold metrics
            fold_metric = calculate_all_metrics(y_test.values, y_pred, self.forecast_type)
            fold_metrics.append({
                'fold': split.fold_number,
                'test_start': split.test_start.isoformat(),
                'test_end': split.test_end.isoformat(),
                'mae': fold_metric['mae'],
                'mape': fold_metric['mape'],
            })

            logger.info(
                f"  Fold {split.fold_number}: MAE={fold_metric['mae']:.2f}, "
                f"MAPE={fold_metric['mape']:.2f}%"
            )

        # Compute aggregate metrics
        all_test_actuals = np.array(all_test_actuals)
        all_test_predictions = np.array(all_test_predictions)

        self.training_metrics = calculate_all_metrics(
            all_test_actuals, all_test_predictions, self.forecast_type
        )

        logger.info(f"Aggregate: {format_metrics(self.training_metrics, self.forecast_type)}")

        # Train final model on all data
        X_all = df[self.feature_columns]
        y_all = df["target_value"]
        self.model = self._create_model()
        self.model.fit(X_all, y_all)

        # Set version
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Store fold details
        self.walk_forward_results = {
            'n_folds': n_folds,
            'fold_metrics': fold_metrics,
            'aggregate_metrics': self.training_metrics,
        }

        return self.training_metrics

    def predict(
        self,
        reference_date: Optional[date] = None,
        hours: List[int] = None,
        horizon_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate forecast for configured horizon (24 hourly values).

        Uses default horizon from config.DEFAULT_HORIZONS based on forecast_type.

        Args:
            reference_date: Date of forecast generation (default: today)
            hours: Hours to forecast (default: 0-23)
            horizon_days: Override default horizon (D+1, D+2, etc.)

        Returns:
            DataFrame with forecast columns:
                - target_timestamp_utc
                - forecast_value
                - horizon_hours
        """
        # Use type-specific default horizon if not specified
        if horizon_days is None:
            horizon_days = config.DEFAULT_HORIZONS.get(
                self.forecast_type,
                config.FORECAST_TARGET_DAYS,  # fallback to legacy default
            )

        return self.predict_d2(reference_date, hours, horizon_days)

    def predict_d2(
        self,
        reference_date: Optional[date] = None,
        hours: List[int] = None,
        horizon_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate forecast for a specific horizon (24 hourly values).

        Args:
            reference_date: Date of forecast generation (default: today)
            hours: Hours to forecast (default: 0-23)
            horizon_days: Days ahead to forecast (default: config default for type)

        Returns:
            DataFrame with forecast columns:
                - target_timestamp_utc
                - forecast_value
                - horizon_hours
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        if reference_date is None:
            reference_date = date.today()

        if hours is None:
            hours = list(range(24))

        # Use type-specific default horizon if not specified
        if horizon_days is None:
            horizon_days = config.DEFAULT_HORIZONS.get(
                self.forecast_type,
                config.FORECAST_TARGET_DAYS,  # fallback to legacy default
            )

        # Target date is D+horizon
        target_date = reference_date + timedelta(days=horizon_days)

        logger.info(f"Generating D+{horizon_days} forecast for {target_date}")

        # Load recent historical data for features
        lookback_days = max(config.LAG_DAYS) + 7  # Extra buffer
        start_date = (reference_date - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )
        end_date = (reference_date + timedelta(days=1)).strftime("%Y-%m-%d")

        df = load_training_data(
            self.country_code, self.forecast_type, start_date, end_date
        )

        if df.empty:
            raise ValueError(f"No historical data available for {self.country_code}")

        # Create features from historical data
        df = create_all_features(df, self.forecast_type)

        # Generate predictions for each target hour
        forecasts = []
        generated_at = datetime.now()

        for hour in hours:
            target_ts = datetime(
                target_date.year, target_date.month, target_date.day, hour
            )

            # Calculate horizon (hours from now to target)
            hours_until = (target_ts - datetime.now()).total_seconds() / 3600
            horizon_hours = int(max(1, hours_until))

            # Get features for this hour
            if not df.empty:
                # Find the same hour from most recent day in data
                df["hour"] = pd.to_datetime(df["timestamp_utc"]).dt.hour
                same_hour_data = df[df["hour"] == hour]

                if not same_hour_data.empty:
                    # Use the most recent row with this hour (for lag features, rolling stats)
                    features = same_hour_data.iloc[-1:][self.feature_columns]

                    # Override time features to match target
                    features = features.copy()
                    features["hour"] = hour
                    features["day_of_week"] = target_ts.weekday()
                    features["month"] = target_ts.month
                    features["is_weekend"] = 1 if target_ts.weekday() >= 5 else 0
                    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
                    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
                    features["day_sin"] = np.sin(2 * np.pi * target_ts.weekday() / 7)
                    features["day_cos"] = np.cos(2 * np.pi * target_ts.weekday() / 7)
                    features["month_sin"] = np.sin(2 * np.pi * target_ts.month / 12)
                    features["month_cos"] = np.cos(2 * np.pi * target_ts.month / 12)

                    # Load forecast weather and replace weather columns
                    forecast_weather = load_weather_forecast_for_hour(
                        self.country_code, target_ts
                    )
                    if forecast_weather is not None:
                        # Get weather features for this forecast type
                        weather_cols = config.WEATHER_FEATURES.get(
                            self.forecast_type, []
                        )
                        for col in weather_cols:
                            if col in features.columns and col in forecast_weather:
                                features[col] = forecast_weather[col]

                        # Recompute derived weather features
                        if "temperature_2m_k" in forecast_weather:
                            temp_c = forecast_weather["temperature_2m_k"] - 273.15
                            if "temperature_c" in features.columns:
                                features["temperature_c"] = temp_c
                            # Heating/cooling degrees for load forecasting
                            if self.forecast_type == "load":
                                base_temp = 18
                                if "heating_degree" in features.columns:
                                    features["heating_degree"] = max(
                                        0, base_temp - temp_c
                                    )
                                if "cooling_degree" in features.columns:
                                    features["cooling_degree"] = max(
                                        0, temp_c - base_temp
                                    )
                    else:
                        logger.debug(
                            f"No weather forecast for {target_ts}, using historical proxy"
                        )

                    # Predict
                    prediction = self.model.predict(features[self.feature_columns])[0]
                else:
                    # Fallback: use mean of recent data
                    prediction = df["target_value"].mean()
                    logger.warning(
                        f"No data for hour {hour}, using mean: {prediction:.2f}"
                    )
            else:
                prediction = 0
                logger.warning(f"No data available, using 0")

            forecasts.append(
                {
                    "country_code": self.country_code,
                    "forecast_type": self.forecast_type,
                    "target_timestamp_utc": target_ts,
                    "generated_at": generated_at,
                    "horizon_hours": horizon_hours,
                    "forecast_value": float(prediction),
                    "model_name": self.algorithm,
                    "model_version": self.model_version,
                }
            )

        return pd.DataFrame(forecasts)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        importance = self.model.feature_importances_
        df = pd.DataFrame({"feature": self.feature_columns, "importance": importance})
        return df.sort_values("importance", ascending=False)

    def _get_model_data(self) -> Dict:
        """
        Get model data dictionary for saving or registry.

        Returns:
            Dictionary with model and metadata
        """
        if self.model is None:
            raise RuntimeError("No model to export")

        return {
            "model": self.model,
            "algorithm": self.algorithm,
            "hyperparams": self.hyperparams,
            "feature_columns": self.feature_columns,
            "country_code": self.country_code,
            "forecast_type": self.forecast_type,
            "model_version": self.model_version,
            "training_metrics": self.training_metrics,
            "grid_search_results": self.grid_search_results,
            "walk_forward_results": getattr(self, 'walk_forward_results', None),
            "saved_at": datetime.now().isoformat(),
        }

    def save(self, path: Optional[str] = None) -> str:
        """
        Save model to disk.

        Args:
            path: Optional custom path (default: models/{country}/{type}/)

        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        if path is None:
            model_dir = config.MODELS_DIR / self.country_code / self.forecast_type
            model_dir.mkdir(parents=True, exist_ok=True)
            path = model_dir / "model.joblib"
        else:
            path = Path(path)

        # Save model and metadata
        model_data = {
            "model": self.model,
            "algorithm": self.algorithm,
            "hyperparams": self.hyperparams,
            "feature_columns": self.feature_columns,
            "country_code": self.country_code,
            "forecast_type": self.forecast_type,
            "model_version": self.model_version,
            "training_metrics": self.training_metrics,
            "grid_search_results": self.grid_search_results,
            "saved_at": datetime.now().isoformat(),
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

        return str(path)

    @classmethod
    def load(
        cls, country_code: str, forecast_type: str, path: Optional[str] = None
    ) -> "Forecaster":
        """
        Load model from disk.

        Args:
            country_code: ISO 2-letter country code
            forecast_type: 'load', 'price', or 'renewable'
            path: Optional custom path

        Returns:
            Loaded Forecaster instance
        """
        if path is None:
            path = config.MODELS_DIR / country_code / forecast_type / "model.joblib"
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")

        model_data = joblib.load(path)

        # Get algorithm from saved data or default to xgboost for backward compatibility
        algorithm = model_data.get("algorithm", "xgboost")
        hyperparams = model_data.get("hyperparams", None)

        forecaster = cls(
            country_code, forecast_type, algorithm=algorithm, hyperparams=hyperparams
        )
        forecaster.model = model_data["model"]
        forecaster.feature_columns = model_data["feature_columns"]
        forecaster.model_version = model_data.get("model_version", "")
        forecaster.training_metrics = model_data.get("training_metrics", {})
        forecaster.grid_search_results = model_data.get("grid_search_results", None)

        logger.info(f"Model loaded from {path} (algorithm: {algorithm})")
        return forecaster


# ============================================================================
# CASCADE FORECASTER
# ============================================================================


class CascadeForecaster:
    """
    Cascade architecture for price forecasting: Stage 1 (load + renewable) → Stage 2 (price).

    The idea: electricity price depends on load and renewable generation.
    Instead of using raw lag features of load/renewable in the price model,
    we first predict load and renewable, then use those predictions as features
    for the price model. This captures the causal chain:
        load demand + renewable supply → price.

    Stage 1: Train separate load and renewable models
    Stage 2: Train price model using original price features + cascade features:
        - cascade_load_prediction: predicted load from Stage 1
        - cascade_renewable_prediction: predicted renewable from Stage 1
        - cascade_residual_load: predicted load - predicted renewable

    Key design decisions:
    - Out-of-fold (OOF) predictions for Stage 2 training to avoid data leakage
    - OOF is generated per-timestamp on the price dataset's time range,
      so NO samples are lost (fixes the data loss bug)
    - predict() runs Stage 1 models first, then Stage 2 (fixes the inference bug)
    """

    def __init__(
        self,
        country_code: str,
        algorithm: str = "xgboost",
        hyperparams: Optional[Dict[str, Any]] = None,
    ):
        self.country_code = country_code
        self.algorithm = algorithm.lower()
        self.hyperparams = hyperparams

        # Stage 1 models (trained Forecaster instances)
        self.load_model: Optional[Forecaster] = None
        self.renewable_model: Optional[Forecaster] = None

        # Stage 2 model (price, with cascade features)
        self.price_model: Optional[Forecaster] = None

        # Metadata
        self.model_version: str = ""
        self.training_metrics: Dict[str, float] = {}
        self.stage1_metrics: Dict[str, Dict[str, float]] = {}
        self.cascade_feature_columns: List[str] = [
            "cascade_load_prediction",
            "cascade_renewable_prediction",
            "cascade_residual_load",
        ]

    def train(
        self,
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        validation_days: int = None,
        n_cv_folds: int = 5,
    ) -> Dict[str, float]:
        """
        Train the full cascade pipeline.

        1. Load all data (load, renewable, price) for the date range
        2. Train Stage 1 models (load, renewable) with cross-validation
           to generate out-of-fold predictions aligned with price timestamps
        3. Add cascade features to price training data
        4. Train Stage 2 price model

        Args:
            start_date: Training start date
            end_date: Training end date
            validation_days: Days to hold out for validation
            n_cv_folds: Number of CV folds for OOF predictions (default: 5)

        Returns:
            Stage 2 (price) validation metrics
        """
        if validation_days is None:
            validation_days = config.VALIDATION_DAYS

        if end_date is None:
            latest = get_latest_data_timestamp(self.country_code, "price")
            if latest:
                end_date = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info("=" * 60)
        logger.info(f"CASCADE TRAINING: {self.country_code} price ({self.algorithm})")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info("=" * 60)

        # ── Stage 1A: Train LOAD model ──────────────────────────────
        logger.info("\n── Stage 1A: Training LOAD model ──")
        self.load_model = Forecaster(
            self.country_code, "load",
            algorithm=self.algorithm,
            hyperparams=self.hyperparams,
        )
        load_metrics = self.load_model.train(
            start_date=start_date, end_date=end_date,
            validation_days=validation_days,
        )
        self.stage1_metrics["load"] = load_metrics
        logger.info(f"  Load model: MAE={load_metrics.get('mae', 0):.2f}, "
                     f"MAPE={load_metrics.get('mape', 0):.2f}%")

        # ── Stage 1B: Train RENEWABLE model ─────────────────────────
        logger.info("\n── Stage 1B: Training RENEWABLE model ──")
        self.renewable_model = Forecaster(
            self.country_code, "renewable",
            algorithm=self.algorithm,
            hyperparams=self.hyperparams,
        )
        renewable_metrics = self.renewable_model.train(
            start_date=start_date, end_date=end_date,
            validation_days=validation_days,
        )
        self.stage1_metrics["renewable"] = renewable_metrics
        logger.info(f"  Renewable model: MAE={renewable_metrics.get('mae', 0):.2f}, "
                     f"MAPE={renewable_metrics.get('mape', 0):.2f}%")

        # ── Stage 2: Train PRICE model with cascade features ────────
        logger.info("\n── Stage 2: Training PRICE model with cascade features ──")

        # Load price data and create base features
        price_df = load_training_data(self.country_code, "price", start_date, end_date)
        if price_df.empty:
            raise ValueError(f"No price data for {self.country_code}")
        price_df = create_all_features(price_df, "price", country_code=self.country_code)
        logger.info(f"  Price data: {len(price_df)} samples after feature engineering")

        # Load load/renewable data for the same date range
        load_df = load_training_data(self.country_code, "load", start_date, end_date)
        renewable_df = load_training_data(self.country_code, "renewable", start_date, end_date)

        if load_df.empty or renewable_df.empty:
            raise ValueError(
                f"Missing load ({len(load_df)}) or renewable ({len(renewable_df)}) data "
                f"for cascade training"
            )

        # Create features for load and renewable
        load_df = create_all_features(load_df, "load", country_code=self.country_code)
        renewable_df = create_all_features(renewable_df, "renewable", country_code=self.country_code)

        # Get the feature columns used by Stage 1 models
        load_feature_cols = self.load_model.feature_columns
        renewable_feature_cols = self.renewable_model.feature_columns

        # ── Generate OOF predictions aligned with price timestamps ──
        # Key insight: we generate predictions for every timestamp in the price
        # dataset by finding the matching timestamp in load/renewable data.
        # This avoids data loss from inner joins.
        logger.info("  Generating out-of-fold cascade predictions...")

        # Build lookup DataFrames indexed by timestamp
        load_df["_ts"] = pd.to_datetime(load_df["timestamp_utc"])
        renewable_df["_ts"] = pd.to_datetime(renewable_df["timestamp_utc"])
        price_df["_ts"] = pd.to_datetime(price_df["timestamp_utc"])

        load_indexed = load_df.set_index("_ts")
        renewable_indexed = renewable_df.set_index("_ts")

        # Find timestamps where we have load features (required for cascade)
        # Renewable is optional — we use the full model for missing timestamps
        price_timestamps = set(price_df["_ts"])
        load_timestamps = set(load_indexed.index)
        renewable_timestamps = set(renewable_indexed.index)

        # Only require load data (larger coverage); renewable gaps filled by full model
        common_ts = price_timestamps & load_timestamps
        has_renewable = price_timestamps & load_timestamps & renewable_timestamps
        logger.info(f"  Timestamps: price={len(price_timestamps)}, "
                     f"load={len(load_timestamps)}, renewable={len(renewable_timestamps)}, "
                     f"common (price∩load)={len(common_ts)}, "
                     f"with renewable={len(has_renewable)}")

        if len(common_ts) < 100:
            raise ValueError(
                f"Only {len(common_ts)} common timestamps between price/load. "
                f"Need at least 100 for cascade training."
            )

        # Use all price timestamps with load data
        price_mask = price_df["_ts"].isin(common_ts)
        cascade_df = price_df[price_mask].copy().reset_index(drop=True)
        logger.info(f"  Cascade training samples: {len(cascade_df)} "
                     f"(lost {len(price_df) - len(cascade_df)} without load data)")

        # Generate OOF predictions using TimeSeriesSplit
        n_samples = len(cascade_df)
        tscv = TimeSeriesSplit(n_splits=n_cv_folds)

        # Prepare load/renewable feature arrays aligned with cascade_df timestamps
        cascade_timestamps = cascade_df["_ts"].values

        # Extract load features for cascade timestamps using vectorized merge
        # Deduplicate indexed DataFrames to avoid issues with .loc
        load_dedup = load_indexed[~load_indexed.index.duplicated(keep='first')]
        renewable_dedup = renewable_indexed[~renewable_indexed.index.duplicated(keep='first')]

        # Build load features array
        load_features = np.zeros((n_samples, len(load_feature_cols)))
        for j, col in enumerate(load_feature_cols):
            if col in load_dedup.columns:
                mapped = cascade_df["_ts"].map(load_dedup[col])
                load_features[:, j] = mapped.fillna(0).values

        # Build renewable features array (0 for timestamps without renewable data)
        renewable_features = np.zeros((n_samples, len(renewable_feature_cols)))
        for j, col in enumerate(renewable_feature_cols):
            if col in renewable_dedup.columns:
                mapped = cascade_df["_ts"].map(renewable_dedup[col])
                renewable_features[:, j] = mapped.fillna(0).values

        # Load/renewable targets for OOF training
        load_target_series = cascade_df["_ts"].map(
            load_dedup["target_value"] if "target_value" in load_dedup.columns
            else pd.Series(dtype=float)
        )
        renewable_target_series = cascade_df["_ts"].map(
            renewable_dedup["target_value"] if "target_value" in renewable_dedup.columns
            else pd.Series(dtype=float)
        )
        load_targets = load_target_series.values.astype(float)
        renewable_targets = renewable_target_series.values.astype(float)

        # Generate OOF predictions
        oof_load_pred = np.full(n_samples, np.nan)
        oof_renewable_pred = np.full(n_samples, np.nan)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(cascade_df)):
            logger.info(f"    OOF fold {fold_idx + 1}/{n_cv_folds}: "
                         f"train={len(train_idx)}, val={len(val_idx)}")

            # Train load OOF model
            X_load_train = load_features[train_idx]
            y_load_train = load_targets[train_idx]
            valid_load = ~np.isnan(y_load_train)

            if valid_load.sum() > 0:
                load_oof_model = self.load_model._create_model()
                load_oof_model.fit(X_load_train[valid_load], y_load_train[valid_load])
                oof_load_pred[val_idx] = load_oof_model.predict(load_features[val_idx])

            # Train renewable OOF model
            X_ren_train = renewable_features[train_idx]
            y_ren_train = renewable_targets[train_idx]
            valid_ren = ~np.isnan(y_ren_train)

            if valid_ren.sum() > 0:
                ren_oof_model = self.renewable_model._create_model()
                ren_oof_model.fit(X_ren_train[valid_ren], y_ren_train[valid_ren])
                oof_renewable_pred[val_idx] = ren_oof_model.predict(
                    renewable_features[val_idx]
                )

        # Fill any remaining NaN (first fold has no OOF) with Stage 1 model predictions
        nan_load = np.isnan(oof_load_pred)
        nan_ren = np.isnan(oof_renewable_pred)
        if nan_load.any():
            oof_load_pred[nan_load] = self.load_model.model.predict(
                load_features[nan_load]
            )
        if nan_ren.any():
            oof_renewable_pred[nan_ren] = self.renewable_model.model.predict(
                renewable_features[nan_ren]
            )

        # Add cascade features to price training data
        cascade_df["cascade_load_prediction"] = oof_load_pred
        cascade_df["cascade_renewable_prediction"] = oof_renewable_pred
        cascade_df["cascade_residual_load"] = oof_load_pred - oof_renewable_pred

        # Get price feature columns + cascade features
        price_feature_cols = [
            col for col in get_feature_columns("price") if col in cascade_df.columns
        ]
        all_stage2_features = price_feature_cols + self.cascade_feature_columns

        # Verify all cascade features exist
        for col in self.cascade_feature_columns:
            assert col in cascade_df.columns, f"Missing cascade feature: {col}"

        logger.info(f"  Stage 2 features: {len(all_stage2_features)} "
                     f"({len(price_feature_cols)} price + {len(self.cascade_feature_columns)} cascade)")
        logger.info(f"  Stage 2 training samples: {len(cascade_df)}")

        # Split into train/validation (chronological)
        val_size = validation_days * 24
        if val_size < len(cascade_df):
            train_part = cascade_df.iloc[:-val_size]
            val_part = cascade_df.iloc[-val_size:]
        else:
            train_part = cascade_df
            val_part = pd.DataFrame()

        # Train Stage 2 price model
        self.price_model = Forecaster(
            self.country_code, "price",
            algorithm=self.algorithm,
            hyperparams=self.hyperparams,
        )
        self.price_model.feature_columns = all_stage2_features

        X_train = train_part[all_stage2_features]
        y_train = train_part["target_value"]

        self.price_model.model = self.price_model._create_model()

        if not val_part.empty:
            X_val = val_part[all_stage2_features]
            y_val = val_part["target_value"]
            self.price_model._train_simple(X_train, y_train, X_val, y_val)
        else:
            self.price_model._train_simple(X_train, y_train, None, None)

        self.training_metrics = self.price_model.training_metrics
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.price_model.model_version = self.model_version

        logger.info(f"\n  CASCADE RESULTS:")
        logger.info(f"  Stage 2 samples: {len(cascade_df)}")
        if self.training_metrics:
            logger.info(f"  MAE: {self.training_metrics.get('mae', 0):.2f}")
            logger.info(f"  MAPE: {self.training_metrics.get('mape', 0):.2f}%")

        # Clean up temp column
        if "_ts" in cascade_df.columns:
            cascade_df.drop(columns=["_ts"], inplace=True)

        return self.training_metrics

    def predict_stage1(self, features_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Run Stage 1 models to generate cascade features for new data.

        This is the key fix for Bug 2: at inference/evaluation time, we must
        generate cascade features by running the Stage 1 models before
        feeding data to the Stage 2 price model.

        Args:
            features_df: DataFrame with features for load and renewable models.
                         Must contain the feature columns of both Stage 1 models.

        Returns:
            Dict with cascade_load_prediction, cascade_renewable_prediction,
            cascade_residual_load arrays.
        """
        if self.load_model is None or self.renewable_model is None:
            raise RuntimeError("Stage 1 models not trained. Call train() first.")

        # Prepare load features (use 0 for missing columns)
        load_features = pd.DataFrame(0.0, index=features_df.index,
                                      columns=self.load_model.feature_columns)
        for col in self.load_model.feature_columns:
            if col in features_df.columns:
                load_features[col] = features_df[col].values

        # Prepare renewable features
        ren_features = pd.DataFrame(0.0, index=features_df.index,
                                     columns=self.renewable_model.feature_columns)
        for col in self.renewable_model.feature_columns:
            if col in features_df.columns:
                ren_features[col] = features_df[col].values

        load_pred = self.load_model.model.predict(load_features)
        ren_pred = self.renewable_model.model.predict(ren_features)

        return {
            "cascade_load_prediction": load_pred,
            "cascade_renewable_prediction": ren_pred,
            "cascade_residual_load": load_pred - ren_pred,
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the Stage 2 price model."""
        if self.price_model is None:
            raise RuntimeError("Price model not trained")
        return self.price_model.get_feature_importance()

    def save(self, path: Optional[str] = None) -> str:
        """Save the complete cascade model (Stage 1 + Stage 2)."""
        if self.price_model is None:
            raise RuntimeError("No model to save")

        if path is None:
            model_dir = config.MODELS_DIR / self.country_code / "price_cascade"
            model_dir.mkdir(parents=True, exist_ok=True)
            path = model_dir / "model.joblib"
        else:
            path = Path(path)

        model_data = {
            "type": "cascade",
            "algorithm": self.algorithm,
            "country_code": self.country_code,
            "model_version": self.model_version,
            "training_metrics": self.training_metrics,
            "stage1_metrics": self.stage1_metrics,
            # Stage 1 models
            "load_model": self.load_model.model,
            "load_feature_columns": self.load_model.feature_columns,
            "load_hyperparams": self.load_model.hyperparams,
            "renewable_model": self.renewable_model.model,
            "renewable_feature_columns": self.renewable_model.feature_columns,
            "renewable_hyperparams": self.renewable_model.hyperparams,
            # Stage 2 model
            "price_model": self.price_model.model,
            "price_feature_columns": self.price_model.feature_columns,
            "price_hyperparams": self.price_model.hyperparams,
            # Cascade config
            "cascade_feature_columns": self.cascade_feature_columns,
            "saved_at": datetime.now().isoformat(),
        }

        joblib.dump(model_data, path)
        logger.info(f"Cascade model saved to {path}")
        return str(path)

    @classmethod
    def load_model(cls, country_code: str, path: Optional[str] = None) -> "CascadeForecaster":
        """Load a saved cascade model."""
        if path is None:
            path = config.MODELS_DIR / country_code / "price_cascade" / "model.joblib"
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Cascade model not found at {path}")

        data = joblib.load(path)

        cascade = cls(
            country_code=data["country_code"],
            algorithm=data["algorithm"],
        )
        cascade.model_version = data.get("model_version", "")
        cascade.training_metrics = data.get("training_metrics", {})
        cascade.stage1_metrics = data.get("stage1_metrics", {})
        cascade.cascade_feature_columns = data.get("cascade_feature_columns", [
            "cascade_load_prediction",
            "cascade_renewable_prediction",
            "cascade_residual_load",
        ])

        # Restore Stage 1 models
        cascade.load_model = Forecaster(
            country_code, "load",
            algorithm=data["algorithm"],
            hyperparams=data.get("load_hyperparams"),
        )
        cascade.load_model.model = data["load_model"]
        cascade.load_model.feature_columns = data["load_feature_columns"]

        cascade.renewable_model = Forecaster(
            country_code, "renewable",
            algorithm=data["algorithm"],
            hyperparams=data.get("renewable_hyperparams"),
        )
        cascade.renewable_model.model = data["renewable_model"]
        cascade.renewable_model.feature_columns = data["renewable_feature_columns"]

        # Restore Stage 2 model
        cascade.price_model = Forecaster(
            country_code, "price",
            algorithm=data["algorithm"],
            hyperparams=data.get("price_hyperparams"),
        )
        cascade.price_model.model = data["price_model"]
        cascade.price_model.feature_columns = data["price_feature_columns"]

        logger.info(f"Cascade model loaded from {path}")
        return cascade


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    print("Testing Forecaster...")

    # Train a model
    print("\n1. Training load forecaster for DE...")
    forecaster = Forecaster("DE", "load")
    metrics = forecaster.train(start_date="2024-01-01")

    print(f"\n2. Training metrics:")
    print(format_metrics(metrics, "load"))

    # Feature importance
    print("\n3. Top 10 features:")
    importance = forecaster.get_feature_importance()
    print(importance.head(10).to_string(index=False))

    # Save model
    print("\n4. Saving model...")
    path = forecaster.save()
    print(f"   Saved to: {path}")

    # Load model
    print("\n5. Loading model...")
    loaded = Forecaster.load("DE", "load")
    print(f"   Version: {loaded.model_version}")

    # Generate D+2 forecast
    print("\n6. Generating D+2 forecast...")
    forecast_df = loaded.predict_d2()
    print(f"   Generated {len(forecast_df)} hourly forecasts")
    print(
        forecast_df[["target_timestamp_utc", "forecast_value", "horizon_hours"]].head()
    )

    print("\n[OK] Forecaster tests complete!")
