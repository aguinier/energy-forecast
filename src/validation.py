"""
Walk-Forward Validation for Energy Forecasting

Implements proper time-series cross-validation that respects temporal order
and mimics production deployment scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Generator, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from metrics import calculate_all_metrics
from baselines import get_all_baseline_predictions, compute_baseline_metrics


@dataclass
class ValidationSplit:
    """Represents a single train/test split for walk-forward validation."""
    fold_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int = 0
    test_size: int = 0


@dataclass
class ValidationResult:
    """Results from a single validation fold."""
    fold_number: int
    train_period: Tuple[datetime, datetime]
    test_period: Tuple[datetime, datetime]
    metrics: Dict[str, float]
    baseline_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    skill_scores: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None


@dataclass
class WalkForwardResults:
    """Aggregated results from walk-forward validation."""
    fold_results: List[ValidationResult]
    aggregate_metrics: Dict[str, float]
    aggregate_skill_scores: Dict[str, float]
    n_folds: int
    total_train_samples: int
    total_test_samples: int


class WalkForwardValidator:
    """
    Walk-Forward Validation for Time Series

    Implements expanding window validation that mimics production deployment:
    - Train on all available historical data up to a cutoff
    - Test on the next period (e.g., 1 month)
    - Expand training window and repeat

    This is the gold standard for time series model evaluation because:
    1. No future information leakage
    2. Multiple test periods capture seasonality effects
    3. Mimics how model would be deployed in production
    """

    def __init__(
        self,
        n_splits: int = 6,
        test_size_days: int = 30,
        gap_days: int = 0,
        min_train_days: int = 365,
    ):
        """
        Args:
            n_splits: Number of validation folds (default 6 = 6 months of testing)
            test_size_days: Size of each test period in days
            gap_days: Gap between train end and test start (to simulate forecast horizon)
            min_train_days: Minimum training data required (default 365 = 1 year)
        """
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.gap_days = gap_days
        self.min_train_days = min_train_days

    def get_splits(
        self,
        data: pd.DataFrame,
        timestamp_col: str = "timestamp_utc",
    ) -> Generator[ValidationSplit, None, None]:
        """
        Generate train/test split indices for walk-forward validation.

        Args:
            data: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Yields:
            ValidationSplit objects with train/test boundaries
        """
        if timestamp_col not in data.columns:
            raise ValueError(f"Column '{timestamp_col}' not found in data")

        timestamps = pd.to_datetime(data[timestamp_col])
        min_date = timestamps.min()
        max_date = timestamps.max()

        total_days = (max_date - min_date).days
        test_days_total = self.n_splits * self.test_size_days
        available_for_training = total_days - test_days_total

        if available_for_training < self.min_train_days:
            raise ValueError(
                f"Insufficient data for {self.n_splits} splits. "
                f"Need at least {self.min_train_days + test_days_total} days, "
                f"have {total_days} days."
            )

        # Start from the end and work backwards to define test periods
        test_end = max_date
        for fold in range(self.n_splits, 0, -1):
            test_start = test_end - timedelta(days=self.test_size_days)
            train_end = test_start - timedelta(days=self.gap_days)
            train_start = min_date

            # Count samples in each period
            train_mask = (timestamps >= train_start) & (timestamps < train_end)
            test_mask = (timestamps >= test_start) & (timestamps <= test_end)

            yield ValidationSplit(
                fold_number=fold,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_size=train_mask.sum(),
                test_size=test_mask.sum(),
            )

            test_end = test_start

    def split_data(
        self,
        data: pd.DataFrame,
        split: ValidationSplit,
        timestamp_col: str = "timestamp_utc",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data according to validation split boundaries.

        Args:
            data: Full dataset
            split: ValidationSplit defining boundaries
            timestamp_col: Name of timestamp column

        Returns:
            Tuple of (train_data, test_data)
        """
        timestamps = pd.to_datetime(data[timestamp_col])

        train_mask = (timestamps >= split.train_start) & (timestamps < split.train_end)
        test_mask = (timestamps >= split.test_start) & (timestamps <= split.test_end)

        return data[train_mask].copy(), data[test_mask].copy()


class TimeSeriesValidator:
    """
    High-level validation orchestrator for energy forecasting models.

    Combines walk-forward validation with baseline comparisons and
    comprehensive metric computation.
    """

    def __init__(
        self,
        n_splits: int = 6,
        test_size_days: int = 30,
        horizon_hours: int = 48,
        min_train_days: int = 365,
    ):
        """
        Args:
            n_splits: Number of validation folds
            test_size_days: Size of each test period
            horizon_hours: Forecast horizon for baseline computation
            min_train_days: Minimum training period
        """
        self.walk_forward = WalkForwardValidator(
            n_splits=n_splits,
            test_size_days=test_size_days,
            gap_days=horizon_hours // 24,  # Convert hours to days
            min_train_days=min_train_days,
        )
        self.horizon_hours = horizon_hours

    def validate_model(
        self,
        model: Any,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        timestamp_col: str = "timestamp_utc",
        forecast_type: str = "load",
        fit_on_each_fold: bool = True,
        store_predictions: bool = False,
    ) -> WalkForwardResults:
        """
        Run full walk-forward validation on a model.

        Args:
            model: Model with fit() and predict() methods
            data: Full dataset with features and target
            target_col: Name of target column
            feature_cols: List of feature column names
            timestamp_col: Name of timestamp column
            forecast_type: Type for MAPE threshold selection
            fit_on_each_fold: Whether to retrain on each fold (recommended)
            store_predictions: Whether to store predictions in results

        Returns:
            WalkForwardResults with aggregate and per-fold metrics
        """
        fold_results = []
        all_test_actuals = []
        all_test_predictions = []
        all_test_timestamps = []

        for split in self.walk_forward.get_splits(data, timestamp_col):
            train_data, test_data = self.walk_forward.split_data(
                data, split, timestamp_col
            )

            if len(train_data) == 0 or len(test_data) == 0:
                continue

            # Prepare features and target
            X_train = train_data[feature_cols].values
            y_train = train_data[target_col].values
            X_test = test_data[feature_cols].values
            y_test = test_data[target_col].values
            test_timestamps = pd.to_datetime(test_data[timestamp_col])

            # Train model on this fold's training data
            if fit_on_each_fold:
                model.fit(X_train, y_train)

            # Generate predictions
            y_pred = model.predict(X_test)

            # Compute metrics
            metrics = calculate_all_metrics(
                y_test, y_pred, forecast_type=forecast_type
            )

            # Compute baseline metrics if we have historical data
            historical_series = pd.Series(
                train_data[target_col].values,
                index=pd.to_datetime(train_data[timestamp_col]),
            )

            baseline_metrics = compute_baseline_metrics(
                y_true=y_test,
                y_pred=y_pred,
                historical_data=historical_series,
                target_timestamps=test_timestamps,
                forecast_type=forecast_type,
                horizon_hours=self.horizon_hours,
            )

            # Extract skill scores
            skill_scores = {
                k: v for k, v in baseline_metrics.items()
                if k.startswith("skill_vs_")
            }

            result = ValidationResult(
                fold_number=split.fold_number,
                train_period=(split.train_start, split.train_end),
                test_period=(split.test_start, split.test_end),
                metrics=metrics,
                baseline_metrics=baseline_metrics,
                skill_scores=skill_scores,
                predictions=y_pred if store_predictions else None,
                actuals=y_test if store_predictions else None,
            )
            fold_results.append(result)

            # Collect for aggregate metrics
            all_test_actuals.extend(y_test)
            all_test_predictions.extend(y_pred)
            all_test_timestamps.extend(test_timestamps)

        # Compute aggregate metrics across all folds
        all_test_actuals = np.array(all_test_actuals)
        all_test_predictions = np.array(all_test_predictions)

        aggregate_metrics = calculate_all_metrics(
            all_test_actuals,
            all_test_predictions,
            forecast_type=forecast_type,
        )

        # Aggregate skill scores (mean across folds)
        aggregate_skill_scores = {}
        skill_keys = [k for k in fold_results[0].skill_scores.keys()] if fold_results else []
        for key in skill_keys:
            scores = [r.skill_scores.get(key, np.nan) for r in fold_results]
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                aggregate_skill_scores[key] = np.mean(valid_scores)
            else:
                aggregate_skill_scores[key] = float("nan")

        return WalkForwardResults(
            fold_results=fold_results,
            aggregate_metrics=aggregate_metrics,
            aggregate_skill_scores=aggregate_skill_scores,
            n_folds=len(fold_results),
            total_train_samples=sum(
                r.train_period[1].timestamp() - r.train_period[0].timestamp()
                for r in fold_results
            ) // 3600,  # Approximate hours
            total_test_samples=len(all_test_actuals),
        )


def create_validation_summary(results: WalkForwardResults) -> Dict[str, Any]:
    """
    Create a summary dictionary from validation results.

    Suitable for storing in database or JSON.
    """
    return {
        "n_folds": results.n_folds,
        "total_test_samples": results.total_test_samples,
        "aggregate_metrics": results.aggregate_metrics,
        "aggregate_skill_scores": results.aggregate_skill_scores,
        "per_fold_metrics": [
            {
                "fold": r.fold_number,
                "test_period": f"{r.test_period[0].isoformat()} to {r.test_period[1].isoformat()}",
                "mae": r.metrics.get("mae"),
                "mape": r.metrics.get("mape"),
                "skill_vs_persistence": r.skill_scores.get("skill_vs_persistence"),
                "skill_vs_seasonal_naive": r.skill_scores.get("skill_vs_seasonal_naive"),
            }
            for r in results.fold_results
        ],
    }


def format_validation_report(results: WalkForwardResults) -> str:
    """
    Format validation results as a human-readable report.
    """
    lines = [
        "=" * 60,
        "Walk-Forward Validation Report",
        "=" * 60,
        "",
        f"Number of folds: {results.n_folds}",
        f"Total test samples: {results.total_test_samples}",
        "",
        "Aggregate Metrics:",
        f"  MAE:   {results.aggregate_metrics.get('mae', float('nan')):.2f}",
        f"  RMSE:  {results.aggregate_metrics.get('rmse', float('nan')):.2f}",
        f"  MAPE:  {results.aggregate_metrics.get('mape', float('nan')):.2f}%",
        f"  SMAPE: {results.aggregate_metrics.get('smape', float('nan')):.2f}%",
        f"  MASE:  {results.aggregate_metrics.get('mase', float('nan')):.3f}",
        f"  Dir. Accuracy: {results.aggregate_metrics.get('directional_accuracy', float('nan')):.1f}%",
        "",
        "Skill Scores (vs Baselines):",
    ]

    for key, value in results.aggregate_skill_scores.items():
        baseline_name = key.replace("skill_vs_", "").replace("_", " ").title()
        status = "better" if value > 0 else "worse" if value < 0 else "equal"
        lines.append(f"  vs {baseline_name}: {value:.4f} ({status})")

    lines.extend([
        "",
        "Per-Fold Results:",
        "-" * 60,
    ])

    for r in results.fold_results:
        lines.extend([
            f"Fold {r.fold_number}: {r.test_period[0].strftime('%Y-%m-%d')} to {r.test_period[1].strftime('%Y-%m-%d')}",
            f"  MAE: {r.metrics.get('mae', float('nan')):.2f}, "
            f"MAPE: {r.metrics.get('mape', float('nan')):.2f}%, "
            f"Skill: {r.skill_scores.get('skill_vs_persistence', float('nan')):.4f}",
        ])

    lines.append("=" * 60)
    return "\n".join(lines)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from sklearn.linear_model import Ridge

    print("Testing walk-forward validation...")

    # Create sample data spanning 2 years
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="H")

    # Simulate load data with patterns
    hours = dates.hour.values
    day_of_year = dates.dayofyear.values

    # Daily pattern + yearly pattern + trend + noise
    daily_pattern = 10000 * np.sin(2 * np.pi * hours / 24)
    yearly_pattern = 5000 * np.sin(2 * np.pi * day_of_year / 365)
    trend = np.linspace(0, 2000, len(dates))
    noise = np.random.normal(0, 1000, len(dates))

    load = 45000 + daily_pattern + yearly_pattern + trend + noise

    # Create features
    data = pd.DataFrame({
        "timestamp_utc": dates,
        "load_mw": load,
        "hour": hours,
        "day_of_week": dates.dayofweek.values,
        "month": dates.month.values,
        "hour_sin": np.sin(2 * np.pi * hours / 24),
        "hour_cos": np.cos(2 * np.pi * hours / 24),
    })

    print(f"\n1. Sample data shape: {data.shape}")
    print(f"   Date range: {dates[0]} to {dates[-1]}")

    # Test walk-forward splits
    print("\n2. Testing walk-forward splits:")
    validator = WalkForwardValidator(n_splits=6, test_size_days=30, min_train_days=180)

    for split in validator.get_splits(data, "timestamp_utc"):
        print(f"   Fold {split.fold_number}: "
              f"Train {split.train_start.strftime('%Y-%m-%d')} to {split.train_end.strftime('%Y-%m-%d')} "
              f"({split.train_size} samples), "
              f"Test {split.test_start.strftime('%Y-%m-%d')} to {split.test_end.strftime('%Y-%m-%d')} "
              f"({split.test_size} samples)")

    # Test full validation with a simple model
    print("\n3. Testing full validation with Ridge model:")
    ts_validator = TimeSeriesValidator(
        n_splits=3,
        test_size_days=30,
        horizon_hours=48,
        min_train_days=180,
    )

    feature_cols = ["hour", "day_of_week", "month", "hour_sin", "hour_cos"]
    model = Ridge(alpha=1.0)

    results = ts_validator.validate_model(
        model=model,
        data=data,
        target_col="load_mw",
        feature_cols=feature_cols,
        timestamp_col="timestamp_utc",
        forecast_type="load",
        fit_on_each_fold=True,
    )

    print(format_validation_report(results))

    print("\n[OK] Walk-forward validation tests complete!")
