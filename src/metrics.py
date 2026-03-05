"""
Evaluation Metrics for Energy Forecasting

Provides MAE, RMSE, MAPE for forecast evaluation.
"""

import numpy as np
from typing import Dict, Optional


# Minimum thresholds for MAPE calculation by forecast type
# Values below these thresholds are excluded to prevent division by near-zero
MAPE_THRESHOLDS = {
    "load": 100.0,  # MW - exclude very low load readings
    "price": 5.0,  # EUR/MWh - prices can be near-zero or negative
    "renewable": 10.0,  # MW - exclude night-time solar zeros
    "solar": 10.0,  # MW - exclude night-time zeros
    "wind_onshore": 10.0,  # MW - low wind periods
    "wind_offshore": 10.0,  # MW - low wind periods
    "hydro_total": 10.0,  # MW - small installations
    "biomass": 5.0,  # MW - typically small values
}
DEFAULT_MAPE_THRESHOLD = 1.0


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Error

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, min_threshold: float = 1.0) -> float:
    """
    Mean Absolute Percentage Error with minimum threshold.

    Values where |y_true| <= min_threshold are excluded to prevent
    division by near-zero values producing astronomical MAPE.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        min_threshold: Minimum absolute value to include in calculation

    Returns:
        MAPE value as percentage, or NaN if no valid values
    """
    mask = np.abs(y_true) > min_threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error

    More robust than MAPE for values near zero.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        SMAPE value as percentage
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return float(100 * np.mean(numerator / denominator))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonality: int = 24,
) -> float:
    """
    Mean Absolute Scaled Error (MASE)

    Scale-independent metric that compares forecast error to naive seasonal baseline.
    MASE < 1 means the model beats the naive baseline.
    MASE > 1 means the model is worse than the naive baseline.

    Args:
        y_true: Actual values (test set)
        y_pred: Predicted values
        y_train: Training data for computing naive baseline MAE.
                 If None, uses y_true with shifted values.
        seasonality: Seasonal period (default 24 for hourly data = daily cycle)

    Returns:
        MASE value (lower is better, <1 beats naive baseline)
    """
    # Compute MAE of predictions
    mae_pred = np.mean(np.abs(y_true - y_pred))

    # Compute naive seasonal baseline MAE from training data
    if y_train is not None and len(y_train) > seasonality:
        naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
        mae_naive = np.mean(naive_errors)
    elif len(y_true) > seasonality:
        # Fallback: use test data for naive baseline
        naive_errors = np.abs(y_true[seasonality:] - y_true[:-seasonality])
        mae_naive = np.mean(naive_errors)
    else:
        # Cannot compute MASE without sufficient data
        return float("nan")

    if mae_naive == 0:
        return float("nan")

    return float(mae_pred / mae_naive)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy (DA)

    Measures the percentage of times the model correctly predicts the direction
    of change (up or down) compared to the previous value.

    Args:
        y_true: Actual values (must have at least 2 values)
        y_pred: Predicted values

    Returns:
        Directional accuracy as percentage (0-100)
    """
    if len(y_true) < 2:
        return float("nan")

    # Direction of actual changes
    actual_direction = np.sign(np.diff(y_true))
    # Direction of predicted changes
    pred_direction = np.sign(np.diff(y_pred))

    # Count correct directions (including both predicting no change correctly)
    correct = np.sum(actual_direction == pred_direction)
    total = len(actual_direction)

    return float(100 * correct / total)


def peak_hour_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hours: Optional[np.ndarray] = None,
    peak_hours: tuple = (8, 9, 10, 11, 17, 18, 19, 20),
) -> Dict[str, float]:
    """
    Peak Hour Accuracy

    Evaluates model performance specifically during peak demand hours.
    Peak hours typically have higher load and are more critical for grid operations.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        hours: Array of hour values (0-23) corresponding to each observation.
               If None, assumes sequential hours starting from 0.
        peak_hours: Tuple of peak hour values (default: morning and evening peaks)

    Returns:
        Dictionary with peak and off-peak metrics:
        - peak_mae: MAE during peak hours
        - peak_mape: MAPE during peak hours
        - offpeak_mae: MAE during off-peak hours
        - offpeak_mape: MAPE during off-peak hours
        - peak_ratio: Ratio of peak to off-peak MAE (>1 means worse during peaks)
    """
    if hours is None:
        hours = np.arange(len(y_true)) % 24

    peak_mask = np.isin(hours, peak_hours)
    offpeak_mask = ~peak_mask

    result = {
        "peak_mae": float("nan"),
        "peak_mape": float("nan"),
        "offpeak_mae": float("nan"),
        "offpeak_mape": float("nan"),
        "peak_ratio": float("nan"),
    }

    if peak_mask.sum() > 0:
        result["peak_mae"] = mae(y_true[peak_mask], y_pred[peak_mask])
        result["peak_mape"] = mape(y_true[peak_mask], y_pred[peak_mask])

    if offpeak_mask.sum() > 0:
        result["offpeak_mae"] = mae(y_true[offpeak_mask], y_pred[offpeak_mask])
        result["offpeak_mape"] = mape(y_true[offpeak_mask], y_pred[offpeak_mask])

    if result["peak_mae"] and result["offpeak_mae"]:
        if not np.isnan(result["offpeak_mae"]) and result["offpeak_mae"] > 0:
            result["peak_ratio"] = result["peak_mae"] / result["offpeak_mae"]

    return result


def skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
    metric: str = "mae",
) -> float:
    """
    Skill Score

    Measures the improvement of the model over a baseline forecast.
    Skill = 1 - (error_model / error_baseline)

    - skill = 1.0: Perfect forecast (no error)
    - skill = 0.0: Model equals baseline (no improvement)
    - skill < 0.0: Model is worse than baseline
    - skill > 0.0: Model improves over baseline

    Args:
        y_true: Actual values
        y_pred: Model predictions
        y_baseline: Baseline predictions (e.g., persistence, seasonal naive)
        metric: Which error metric to use ('mae', 'rmse', 'mape')

    Returns:
        Skill score (higher is better, 0 = same as baseline, negative = worse)
    """
    if metric == "mae":
        error_model = mae(y_true, y_pred)
        error_baseline = mae(y_true, y_baseline)
    elif metric == "rmse":
        error_model = rmse(y_true, y_pred)
        error_baseline = rmse(y_true, y_baseline)
    elif metric == "mape":
        error_model = mape(y_true, y_pred)
        error_baseline = mape(y_true, y_baseline)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if error_baseline == 0 or np.isnan(error_baseline):
        return float("nan")

    return float(1 - (error_model / error_baseline))


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    forecast_type: Optional[str] = None,
    y_train: Optional[np.ndarray] = None,
    y_baseline: Optional[np.ndarray] = None,
    hours: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate all metrics at once.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        forecast_type: Type of forecast for MAPE threshold selection
        y_train: Training data for MASE calculation (optional)
        y_baseline: Baseline predictions for skill score (optional)
        hours: Hour values for peak hour analysis (optional)

    Returns:
        Dictionary with all metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mape_threshold = (
        MAPE_THRESHOLDS.get(forecast_type, DEFAULT_MAPE_THRESHOLD)
        if forecast_type
        else DEFAULT_MAPE_THRESHOLD
    )

    result = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred, min_threshold=mape_threshold),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train=y_train),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }

    # Add skill score if baseline provided
    if y_baseline is not None:
        y_baseline = np.asarray(y_baseline)
        result["skill_score"] = skill_score(y_true, y_pred, y_baseline, metric="mae")
    else:
        result["skill_score"] = float("nan")

    # Add peak hour metrics if hours provided
    if hours is not None:
        peak_metrics = peak_hour_accuracy(y_true, y_pred, hours)
        result.update(peak_metrics)

    return result


def format_metrics(metrics: Dict[str, float], target_name: str = "") -> str:
    """
    Format metrics for display.

    Args:
        metrics: Dictionary of metric values
        target_name: Name of target variable (for units)

    Returns:
        Formatted string
    """
    lines = []
    if target_name:
        lines.append(f"Metrics for {target_name}:")
    else:
        lines.append("Metrics:")

    lines.append(f"  MAE:   {metrics['mae']:.2f}")
    lines.append(f"  RMSE:  {metrics['rmse']:.2f}")
    lines.append(f"  MAPE:  {metrics['mape']:.2f}%")
    lines.append(f"  SMAPE: {metrics['smape']:.2f}%")

    return "\n".join(lines)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing metrics...")

    # Create sample data
    np.random.seed(42)
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 380, 520])

    print("\n1. Individual metrics:")
    print(f"   MAE:   {mae(y_true, y_pred):.2f}")
    print(f"   RMSE:  {rmse(y_true, y_pred):.2f}")
    print(f"   MAPE:  {mape(y_true, y_pred):.2f}%")
    print(f"   SMAPE: {smape(y_true, y_pred):.2f}%")

    print("\n2. All metrics at once:")
    metrics = calculate_all_metrics(y_true, y_pred)
    print(format_metrics(metrics, "test"))

    # Test with near-zero values
    print("\n3. Testing with near-zero values:")
    y_true_zero = np.array([0.1, 0.01, 0.001, 100])
    y_pred_zero = np.array([0.2, 0.02, 0.002, 110])
    metrics_zero = calculate_all_metrics(y_true_zero, y_pred_zero)
    print(format_metrics(metrics_zero, "near-zero test"))

    print("\n[OK] Metrics tests complete!")
