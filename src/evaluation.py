"""
Evaluation Report Generator for Energy Forecasting

Generates comprehensive evaluation reports including:
- Per-hour performance breakdown
- Seasonal analysis
- Error distribution analysis
- Comparison vs baselines
- Feature importance stability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from .metrics import (
    calculate_all_metrics,
    mae,
    rmse,
    mape,
    skill_score,
    peak_hour_accuracy,
)
from .baselines import compute_baseline_metrics, get_all_baseline_predictions


@dataclass
class HourlyBreakdown:
    """Metrics broken down by hour of day."""
    hour: int
    mae: float
    mape: float
    rmse: float
    sample_count: int
    mean_actual: float
    mean_predicted: float
    bias: float  # mean(pred - actual)


@dataclass
class SeasonalBreakdown:
    """Metrics broken down by season."""
    season: str  # 'winter', 'spring', 'summer', 'fall'
    months: List[int]
    mae: float
    mape: float
    rmse: float
    sample_count: int
    skill_vs_persistence: float


@dataclass
class ErrorDistribution:
    """Statistical distribution of forecast errors."""
    mean_error: float
    std_error: float
    median_error: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    skewness: float
    kurtosis: float
    max_overestimate: float
    max_underestimate: float


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for a forecast model."""
    # Metadata
    country_code: str
    forecast_type: str
    model_version: str
    evaluation_date: str
    test_period_start: str
    test_period_end: str
    sample_count: int

    # Core metrics
    aggregate_metrics: Dict[str, float]

    # Baseline comparisons
    baseline_metrics: Dict[str, Dict[str, float]]
    skill_scores: Dict[str, float]

    # Detailed breakdowns
    hourly_breakdown: List[HourlyBreakdown]
    seasonal_breakdown: List[SeasonalBreakdown]
    error_distribution: ErrorDistribution

    # Peak hour analysis
    peak_metrics: Dict[str, float]

    # Feature importance (if available)
    feature_importance: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "metadata": {
                "country_code": self.country_code,
                "forecast_type": self.forecast_type,
                "model_version": self.model_version,
                "evaluation_date": self.evaluation_date,
                "test_period_start": self.test_period_start,
                "test_period_end": self.test_period_end,
                "sample_count": self.sample_count,
            },
            "aggregate_metrics": self.aggregate_metrics,
            "baseline_metrics": self.baseline_metrics,
            "skill_scores": self.skill_scores,
            "hourly_breakdown": [
                {
                    "hour": h.hour,
                    "mae": h.mae,
                    "mape": h.mape,
                    "rmse": h.rmse,
                    "sample_count": h.sample_count,
                    "mean_actual": h.mean_actual,
                    "mean_predicted": h.mean_predicted,
                    "bias": h.bias,
                }
                for h in self.hourly_breakdown
            ],
            "seasonal_breakdown": [
                {
                    "season": s.season,
                    "months": s.months,
                    "mae": s.mae,
                    "mape": s.mape,
                    "rmse": s.rmse,
                    "sample_count": s.sample_count,
                    "skill_vs_persistence": s.skill_vs_persistence,
                }
                for s in self.seasonal_breakdown
            ],
            "error_distribution": {
                "mean_error": self.error_distribution.mean_error,
                "std_error": self.error_distribution.std_error,
                "median_error": self.error_distribution.median_error,
                "percentile_5": self.error_distribution.percentile_5,
                "percentile_25": self.error_distribution.percentile_25,
                "percentile_75": self.error_distribution.percentile_75,
                "percentile_95": self.error_distribution.percentile_95,
                "skewness": self.error_distribution.skewness,
                "kurtosis": self.error_distribution.kurtosis,
                "max_overestimate": self.error_distribution.max_overestimate,
                "max_underestimate": self.error_distribution.max_underestimate,
            },
            "peak_metrics": self.peak_metrics,
            "feature_importance": self.feature_importance,
        }

    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


def compute_hourly_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hours: np.ndarray,
    forecast_type: str = "load",
) -> List[HourlyBreakdown]:
    """
    Compute metrics broken down by hour of day.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        hours: Hour values (0-23) for each observation
        forecast_type: Type for MAPE threshold

    Returns:
        List of HourlyBreakdown for each hour 0-23
    """
    breakdowns = []

    for hour in range(24):
        mask = hours == hour
        if mask.sum() == 0:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]
        errors = y_p - y_t

        breakdowns.append(HourlyBreakdown(
            hour=hour,
            mae=mae(y_t, y_p),
            mape=mape(y_t, y_p),
            rmse=rmse(y_t, y_p),
            sample_count=int(mask.sum()),
            mean_actual=float(np.mean(y_t)),
            mean_predicted=float(np.mean(y_p)),
            bias=float(np.mean(errors)),
        ))

    return breakdowns


def compute_seasonal_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    months: np.ndarray,
    historical_data: Optional[pd.Series] = None,
    target_timestamps: Optional[pd.DatetimeIndex] = None,
    forecast_type: str = "load",
    horizon_hours: int = 48,
) -> List[SeasonalBreakdown]:
    """
    Compute metrics broken down by season.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        months: Month values (1-12) for each observation
        historical_data: For baseline computation
        target_timestamps: Timestamps for baseline computation
        forecast_type: Type for MAPE threshold
        horizon_hours: Forecast horizon for baselines

    Returns:
        List of SeasonalBreakdown for each season
    """
    seasons = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "fall": [9, 10, 11],
    }

    breakdowns = []

    for season_name, season_months in seasons.items():
        mask = np.isin(months, season_months)
        if mask.sum() == 0:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        # Compute skill vs persistence if we have historical data
        skill = float("nan")
        if historical_data is not None and target_timestamps is not None:
            from .baselines import PersistenceBaseline
            persistence = PersistenceBaseline(horizon_hours=horizon_hours)
            ts_mask = target_timestamps.month.isin(season_months)
            if ts_mask.sum() > 0:
                y_baseline = persistence.predict_for_target(
                    historical_data,
                    target_timestamps[ts_mask],
                )
                valid = ~np.isnan(y_baseline)
                if valid.sum() > 0:
                    # Align with masked y_t and y_p
                    skill = skill_score(
                        y_t[valid[:len(y_t)]] if len(valid) >= len(y_t) else y_t,
                        y_p[valid[:len(y_p)]] if len(valid) >= len(y_p) else y_p,
                        y_baseline[valid][:len(y_t)],
                    )

        breakdowns.append(SeasonalBreakdown(
            season=season_name,
            months=season_months,
            mae=mae(y_t, y_p),
            mape=mape(y_t, y_p),
            rmse=rmse(y_t, y_p),
            sample_count=int(mask.sum()),
            skill_vs_persistence=skill,
        ))

    return breakdowns


def compute_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ErrorDistribution:
    """
    Compute statistical distribution of forecast errors.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        ErrorDistribution with comprehensive statistics
    """
    errors = y_pred - y_true

    # Compute skewness and kurtosis manually to avoid scipy dependency
    mean_err = np.mean(errors)
    std_err = np.std(errors)

    if std_err > 0:
        normalized = (errors - mean_err) / std_err
        skewness = float(np.mean(normalized ** 3))
        kurtosis = float(np.mean(normalized ** 4) - 3)  # Excess kurtosis
    else:
        skewness = 0.0
        kurtosis = 0.0

    return ErrorDistribution(
        mean_error=float(mean_err),
        std_error=float(std_err),
        median_error=float(np.median(errors)),
        percentile_5=float(np.percentile(errors, 5)),
        percentile_25=float(np.percentile(errors, 25)),
        percentile_75=float(np.percentile(errors, 75)),
        percentile_95=float(np.percentile(errors, 95)),
        skewness=skewness,
        kurtosis=kurtosis,
        max_overestimate=float(np.max(errors)),
        max_underestimate=float(np.min(errors)),
    )


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    historical_data: pd.Series,
    country_code: str,
    forecast_type: str,
    model_version: str,
    horizon_hours: int = 48,
    feature_importance: Optional[Dict[str, float]] = None,
) -> EvaluationReport:
    """
    Generate a comprehensive evaluation report.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        timestamps: Timestamps for each observation
        historical_data: Historical data for baseline computation
        country_code: Country code
        forecast_type: Type of forecast
        model_version: Model version string
        horizon_hours: Forecast horizon
        feature_importance: Optional feature importance dict

    Returns:
        Complete EvaluationReport
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    hours = timestamps.hour.values
    months = timestamps.month.values

    # Aggregate metrics
    aggregate_metrics = calculate_all_metrics(y_true, y_pred, forecast_type=forecast_type)

    # Baseline comparisons
    baseline_results = compute_baseline_metrics(
        y_true=y_true,
        y_pred=y_pred,
        historical_data=historical_data,
        target_timestamps=timestamps,
        forecast_type=forecast_type,
        horizon_hours=horizon_hours,
    )

    baseline_metrics = {
        k: v for k, v in baseline_results.items()
        if isinstance(v, dict)
    }
    skill_scores = {
        k: v for k, v in baseline_results.items()
        if k.startswith("skill_vs_") and not isinstance(v, dict)
    }

    # Hourly breakdown
    hourly_breakdown = compute_hourly_breakdown(y_true, y_pred, hours, forecast_type)

    # Seasonal breakdown
    seasonal_breakdown = compute_seasonal_breakdown(
        y_true, y_pred, months,
        historical_data, timestamps,
        forecast_type, horizon_hours,
    )

    # Error distribution
    error_dist = compute_error_distribution(y_true, y_pred)

    # Peak hour metrics
    peak_metrics = peak_hour_accuracy(y_true, y_pred, hours)

    return EvaluationReport(
        country_code=country_code,
        forecast_type=forecast_type,
        model_version=model_version,
        evaluation_date=datetime.now().isoformat(),
        test_period_start=timestamps.min().isoformat(),
        test_period_end=timestamps.max().isoformat(),
        sample_count=len(y_true),
        aggregate_metrics=aggregate_metrics,
        baseline_metrics=baseline_metrics,
        skill_scores=skill_scores,
        hourly_breakdown=hourly_breakdown,
        seasonal_breakdown=seasonal_breakdown,
        error_distribution=error_dist,
        peak_metrics=peak_metrics,
        feature_importance=feature_importance,
    )


def format_evaluation_report(report: EvaluationReport) -> str:
    """
    Format evaluation report as human-readable text.
    """
    lines = [
        "=" * 70,
        f"EVALUATION REPORT: {report.country_code} - {report.forecast_type}",
        "=" * 70,
        "",
        "METADATA",
        "-" * 40,
        f"  Model Version:    {report.model_version}",
        f"  Evaluation Date:  {report.evaluation_date}",
        f"  Test Period:      {report.test_period_start[:10]} to {report.test_period_end[:10]}",
        f"  Sample Count:     {report.sample_count:,}",
        "",
        "AGGREGATE METRICS",
        "-" * 40,
        f"  MAE:    {report.aggregate_metrics.get('mae', float('nan')):,.2f}",
        f"  RMSE:   {report.aggregate_metrics.get('rmse', float('nan')):,.2f}",
        f"  MAPE:   {report.aggregate_metrics.get('mape', float('nan')):.2f}%",
        f"  SMAPE:  {report.aggregate_metrics.get('smape', float('nan')):.2f}%",
        f"  MASE:   {report.aggregate_metrics.get('mase', float('nan')):.3f}",
        f"  Dir. Accuracy: {report.aggregate_metrics.get('directional_accuracy', float('nan')):.1f}%",
        "",
        "SKILL SCORES (vs Baselines)",
        "-" * 40,
    ]

    for name, score in report.skill_scores.items():
        baseline = name.replace("skill_vs_", "").replace("_", " ").title()
        status = "[BETTER]" if score > 0 else "[WORSE]" if score < 0 else "[EQUAL]"
        lines.append(f"  vs {baseline:20s}: {score:+.4f} {status}")

    lines.extend([
        "",
        "PEAK HOUR ANALYSIS",
        "-" * 40,
        f"  Peak MAE:     {report.peak_metrics.get('peak_mae', float('nan')):,.2f}",
        f"  Off-Peak MAE: {report.peak_metrics.get('offpeak_mae', float('nan')):,.2f}",
        f"  Peak Ratio:   {report.peak_metrics.get('peak_ratio', float('nan')):.2f}x",
        "",
        "ERROR DISTRIBUTION",
        "-" * 40,
        f"  Mean Error:   {report.error_distribution.mean_error:+,.2f}",
        f"  Std Error:    {report.error_distribution.std_error:,.2f}",
        f"  Median Error: {report.error_distribution.median_error:+,.2f}",
        f"  5th-95th:     [{report.error_distribution.percentile_5:+,.0f}, {report.error_distribution.percentile_95:+,.0f}]",
        f"  Max Over:     {report.error_distribution.max_overestimate:+,.2f}",
        f"  Max Under:    {report.error_distribution.max_underestimate:+,.2f}",
        "",
        "SEASONAL BREAKDOWN",
        "-" * 40,
    ])

    for s in report.seasonal_breakdown:
        skill_str = f"skill={s.skill_vs_persistence:+.3f}" if not np.isnan(s.skill_vs_persistence) else "skill=N/A"
        lines.append(
            f"  {s.season.capitalize():8s}: MAE={s.mae:,.0f}, MAPE={s.mape:.1f}%, {skill_str}"
        )

    lines.extend([
        "",
        "HOURLY BREAKDOWN (Best/Worst Hours)",
        "-" * 40,
    ])

    # Show best and worst 3 hours by MAPE
    sorted_hours = sorted(report.hourly_breakdown, key=lambda h: h.mape if not np.isnan(h.mape) else float('inf'))
    best_3 = sorted_hours[:3]
    worst_3 = sorted_hours[-3:]

    lines.append("  Best Hours:")
    for h in best_3:
        lines.append(f"    Hour {h.hour:02d}: MAE={h.mae:,.0f}, MAPE={h.mape:.1f}%, Bias={h.bias:+,.0f}")

    lines.append("  Worst Hours:")
    for h in worst_3:
        lines.append(f"    Hour {h.hour:02d}: MAE={h.mae:,.0f}, MAPE={h.mape:.1f}%, Bias={h.bias:+,.0f}")

    if report.feature_importance:
        lines.extend([
            "",
            "FEATURE IMPORTANCE (Top 10)",
            "-" * 40,
        ])
        sorted_features = sorted(
            report.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        for name, importance in sorted_features:
            lines.append(f"  {name:30s}: {importance:.4f}")

    lines.append("=" * 70)
    return "\n".join(lines)


def compare_models(
    reports: List[EvaluationReport],
) -> pd.DataFrame:
    """
    Create a comparison table of multiple models/versions.

    Args:
        reports: List of evaluation reports to compare

    Returns:
        DataFrame with comparison metrics
    """
    rows = []
    for r in reports:
        row = {
            "country_code": r.country_code,
            "forecast_type": r.forecast_type,
            "model_version": r.model_version,
            "mae": r.aggregate_metrics.get("mae"),
            "rmse": r.aggregate_metrics.get("rmse"),
            "mape": r.aggregate_metrics.get("mape"),
            "mase": r.aggregate_metrics.get("mase"),
            "directional_accuracy": r.aggregate_metrics.get("directional_accuracy"),
            "skill_vs_persistence": r.skill_scores.get("skill_vs_persistence"),
            "skill_vs_seasonal_naive": r.skill_scores.get("skill_vs_seasonal_naive"),
            "peak_ratio": r.peak_metrics.get("peak_ratio"),
            "sample_count": r.sample_count,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing evaluation report generation...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2024-06-30", freq="H")

    # Simulate load with patterns
    hours = dates.hour.values
    base_load = 40000 + 10000 * np.sin(2 * np.pi * hours / 24)
    noise = np.random.normal(0, 1000, len(dates))
    y_true = base_load + noise

    # Simulate predictions (slightly biased and noisy)
    pred_noise = np.random.normal(500, 800, len(dates))  # Slight overestimate
    y_pred = y_true + pred_noise

    # Create historical data series
    historical_data = pd.Series(y_true, index=dates)

    print(f"\n1. Sample data: {len(y_true):,} observations")
    print(f"   Date range: {dates[0]} to {dates[-1]}")

    # Generate evaluation report
    print("\n2. Generating evaluation report...")
    report = generate_evaluation_report(
        y_true=y_true,
        y_pred=y_pred,
        timestamps=dates,
        historical_data=historical_data,
        country_code="DE",
        forecast_type="load",
        model_version="20240115_180000",
        horizon_hours=48,
        feature_importance={
            "hour_sin": 0.15,
            "hour_cos": 0.12,
            "lag_24h": 0.25,
            "lag_168h": 0.18,
            "temperature": 0.10,
            "day_of_week": 0.08,
            "rolling_mean_24h": 0.07,
            "rolling_std_24h": 0.05,
        },
    )

    # Print formatted report
    print("\n" + format_evaluation_report(report))

    # Test JSON export
    print("\n3. Testing JSON export...")
    json_str = report.to_json()
    print(f"   JSON size: {len(json_str):,} bytes")

    print("\n[OK] Evaluation report tests complete!")
