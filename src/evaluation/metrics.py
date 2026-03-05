"""
Evaluation metrics for energy forecasting.

Point forecast metrics: MAE, RMSE, MAPE, sMAPE
Probabilistic metrics: CRPS, Pinball Loss (for quantile forecasts)
Statistical tests: Diebold-Mariano test

All metrics operate on aligned (forecast, actual) arrays.
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple


# ============================================================================
# POINT FORECAST METRICS
# ============================================================================

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%). Skips near-zero actuals."""
    mask = np.abs(actual) > epsilon
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    denom = np.abs(actual) + np.abs(predicted) + epsilon
    return float(np.mean(2 * np.abs(actual - predicted) / denom) * 100)


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Bias (predicted - actual). Positive = over-forecast."""
    return float(np.mean(predicted - actual))


def skill_score(model_metric: float, baseline_metric: float) -> float:
    """
    Skill score: 1 - (model / baseline).
    Positive = model is better. 0 = same. Negative = model is worse.
    """
    if baseline_metric == 0:
        return 0.0
    return float(1.0 - model_metric / baseline_metric)


# ============================================================================
# PROBABILISTIC METRICS
# ============================================================================

def pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    """
    Pinball (quantile) loss for a single quantile.
    
    Args:
        actual: Actual values
        predicted: Predicted quantile values
        quantile: Quantile level (0-1), e.g. 0.1 for P10
    
    Returns:
        Mean pinball loss
    """
    diff = actual - predicted
    return float(np.mean(np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)))


def crps_empirical(actual: np.ndarray, quantile_forecasts: Dict[float, np.ndarray]) -> float:
    """
    Approximate CRPS using quantile forecasts (trapezoidal rule).
    
    Args:
        actual: Actual values (n,)
        quantile_forecasts: Dict mapping quantile level → predicted values
            e.g. {0.1: array, 0.25: array, 0.5: array, 0.75: array, 0.9: array}
    
    Returns:
        Mean CRPS across all timesteps
    """
    quantiles = sorted(quantile_forecasts.keys())
    n = len(actual)
    crps_values = np.zeros(n)
    
    for i in range(len(quantiles) - 1):
        q_lo = quantiles[i]
        q_hi = quantiles[i + 1]
        dq = q_hi - q_lo
        
        pred_lo = quantile_forecasts[q_lo]
        pred_hi = quantile_forecasts[q_hi]
        
        # For each quantile level q, the integrand is (F(x) - 1{x <= y})^2
        # Using trapezoidal approximation
        for j, q in enumerate([q_lo, q_hi]):
            pred = quantile_forecasts[q]
            indicator = (actual <= pred).astype(float)
            integrand = (q - indicator) ** 2
            if j == 0:
                crps_values += integrand * dq / 2
            else:
                crps_values += integrand * dq / 2
    
    return float(np.mean(crps_values))


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def diebold_mariano_test(
    actual: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    loss: str = "squared",
    horizon: int = 1,
) -> Tuple[float, float, str]:
    """
    Diebold-Mariano test for comparing two forecasts.
    
    Tests H0: both forecasts have equal predictive accuracy.
    
    Args:
        actual: Actual values
        pred_a: Predictions from model A
        pred_b: Predictions from model B
        loss: Loss function — "squared" (MSE) or "absolute" (MAE)
        horizon: Forecast horizon (for HAC standard errors)
    
    Returns:
        (dm_statistic, p_value, interpretation)
        Negative DM → model A is better. Positive DM → model B is better.
    """
    if loss == "squared":
        loss_a = (actual - pred_a) ** 2
        loss_b = (actual - pred_b) ** 2
    elif loss == "absolute":
        loss_a = np.abs(actual - pred_a)
        loss_b = np.abs(actual - pred_b)
    else:
        raise ValueError(f"Unknown loss: {loss}")
    
    d = loss_a - loss_b  # Positive = A worse, B better
    n = len(d)
    d_mean = np.mean(d)
    
    # HAC variance estimator (Newey-West with h-1 lags)
    gamma_0 = np.mean((d - d_mean) ** 2)
    gamma_sum = 0.0
    for k in range(1, horizon):
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += 2 * gamma_k
    
    var_d = (gamma_0 + gamma_sum) / n
    
    if var_d <= 0:
        return 0.0, 1.0, "inconclusive (zero variance)"
    
    dm_stat = d_mean / np.sqrt(var_d)
    
    # Two-sided test using t-distribution with n-1 df
    p_value = 2 * stats.t.sf(np.abs(dm_stat), df=n - 1)
    
    if p_value < 0.01:
        sig = "***"
    elif p_value < 0.05:
        sig = "**"
    elif p_value < 0.10:
        sig = "*"
    else:
        sig = ""
    
    if dm_stat < 0:
        interpretation = f"Model A is better {sig} (DM={dm_stat:.3f}, p={p_value:.4f})"
    elif dm_stat > 0:
        interpretation = f"Model B is better {sig} (DM={dm_stat:.3f}, p={p_value:.4f})"
    else:
        interpretation = f"No difference (DM={dm_stat:.3f}, p={p_value:.4f})"
    
    return float(dm_stat), float(p_value), interpretation


# ============================================================================
# CONVENIENCE
# ============================================================================

def calculate_point_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate all point forecast metrics at once."""
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
        "smape": smape(actual, predicted),
        "bias": bias(actual, predicted),
        "n": len(actual),
    }


def format_metrics_table(metrics: Dict[str, float], name: str = "") -> str:
    """Format metrics as a readable string."""
    lines = [f"--- {name} ---"] if name else []
    for k, v in metrics.items():
        if k == "n":
            lines.append(f"  {k:>8s}: {v:.0f}")
        else:
            lines.append(f"  {k:>8s}: {v:.4f}")
    return "\n".join(lines)
