"""
Baseline Models for Energy Forecasting

Provides simple baseline forecasts for comparison with ML models.
A good ML model should beat these baselines.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3

try:
    from .metrics import calculate_all_metrics, skill_score
except ImportError:
    from metrics import calculate_all_metrics, skill_score


class PersistenceBaseline:
    """
    Persistence Baseline: Tomorrow = Today (same hour)

    The simplest baseline - assumes the value at the same hour
    yesterday will be the value tomorrow.

    For D+2 forecasting: value(t+48) = value(t)
    """

    def __init__(self, horizon_hours: int = 48):
        """
        Args:
            horizon_hours: How many hours ahead to forecast (default 48 for D+2)
        """
        self.horizon_hours = horizon_hours
        self.name = "persistence"

    def predict(self, historical_data: pd.Series) -> pd.Series:
        """
        Generate persistence forecast.

        Args:
            historical_data: Series with datetime index containing historical values

        Returns:
            Series with forecasted values (shifted by horizon_hours)
        """
        # Shift data by horizon hours
        forecast = historical_data.shift(self.horizon_hours)
        return forecast.dropna()

    def predict_for_target(
        self,
        historical_data: pd.Series,
        target_timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Generate predictions for specific target timestamps.

        Args:
            historical_data: Series with datetime index
            target_timestamps: Timestamps to forecast for

        Returns:
            Array of forecast values aligned with target_timestamps
        """
        predictions = []
        for ts in target_timestamps:
            # Look back by horizon_hours
            source_ts = ts - timedelta(hours=self.horizon_hours)
            if source_ts in historical_data.index:
                predictions.append(historical_data.loc[source_ts])
            else:
                predictions.append(np.nan)
        return np.array(predictions)


class SeasonalNaiveBaseline:
    """
    Seasonal Naive Baseline: Tomorrow = Same day last week (same hour)

    Uses weekly seasonality - assumes the value at the same hour
    on the same weekday last week will be the value.

    For D+2 forecasting: value(t+48) = value(t+48 - 168) = value(t - 120)
    """

    def __init__(self, seasonality_hours: int = 168, horizon_hours: int = 48):
        """
        Args:
            seasonality_hours: Seasonal period (default 168 = 1 week)
            horizon_hours: How many hours ahead to forecast
        """
        self.seasonality_hours = seasonality_hours
        self.horizon_hours = horizon_hours
        self.name = "seasonal_naive"

    def predict(self, historical_data: pd.Series) -> pd.Series:
        """
        Generate seasonal naive forecast.

        Args:
            historical_data: Series with datetime index

        Returns:
            Series with forecasted values
        """
        # For D+2: look back 168h (1 week) then forward 48h = net -120h
        shift = self.seasonality_hours - self.horizon_hours
        forecast = historical_data.shift(shift)
        return forecast.dropna()

    def predict_for_target(
        self,
        historical_data: pd.Series,
        target_timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Generate predictions for specific target timestamps.
        """
        predictions = []
        shift = self.seasonality_hours - self.horizon_hours
        for ts in target_timestamps:
            source_ts = ts - timedelta(hours=shift)
            if source_ts in historical_data.index:
                predictions.append(historical_data.loc[source_ts])
            else:
                predictions.append(np.nan)
        return np.array(predictions)


class WeeklyAverageBaseline:
    """
    Weekly Average Baseline: Average of last 7 days at same hour

    Uses rolling weekly average for each hour of day.
    More robust than single-day lookback.
    """

    def __init__(self, window_days: int = 7, horizon_hours: int = 48):
        """
        Args:
            window_days: Number of days to average (default 7)
            horizon_hours: How many hours ahead to forecast
        """
        self.window_days = window_days
        self.horizon_hours = horizon_hours
        self.name = "weekly_average"

    def predict(self, historical_data: pd.Series) -> pd.Series:
        """
        Generate weekly average forecast.

        For each target hour, average the same hour over the past week.
        """
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            raise ValueError("historical_data must have DatetimeIndex")

        # Group by hour of day and compute rolling mean
        df = historical_data.to_frame(name='value')
        df['hour'] = df.index.hour

        # For each timestamp, compute mean of same hour in last window_days
        predictions = []
        for idx in historical_data.index:
            hour = idx.hour
            # Look at past window_days * 24 hours, filter to same hour
            start = idx - timedelta(days=self.window_days)
            mask = (historical_data.index >= start) & (historical_data.index < idx)
            same_hour = historical_data.loc[mask]
            same_hour = same_hour[same_hour.index.hour == hour]
            if len(same_hour) > 0:
                predictions.append(same_hour.mean())
            else:
                predictions.append(np.nan)

        result = pd.Series(predictions, index=historical_data.index)
        # Shift forward by horizon to align with forecast targets
        return result.shift(-self.horizon_hours).dropna()

    def predict_for_target(
        self,
        historical_data: pd.Series,
        target_timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Generate predictions for specific target timestamps.
        """
        predictions = []
        for ts in target_timestamps:
            hour = ts.hour
            # Look back from forecast generation time (ts - horizon)
            gen_time = ts - timedelta(hours=self.horizon_hours)
            start = gen_time - timedelta(days=self.window_days)

            mask = (historical_data.index >= start) & (historical_data.index < gen_time)
            same_hour = historical_data.loc[mask]
            same_hour = same_hour[same_hour.index.hour == hour]

            if len(same_hour) > 0:
                predictions.append(same_hour.mean())
            else:
                predictions.append(np.nan)

        return np.array(predictions)


class TSOBaseline:
    """
    TSO Baseline: Use ENTSO-E TSO forecasts as baseline

    Compares ML models against official TSO day-ahead forecasts.
    TSO forecasts are stored in energy_load_forecast and energy_generation_forecast tables.
    """

    def __init__(self, db_path: str, forecast_type: str = "day_ahead"):
        """
        Args:
            db_path: Path to SQLite database
            forecast_type: 'day_ahead' or 'week_ahead'
        """
        self.db_path = db_path
        self.forecast_type = forecast_type
        self.name = f"tso_{forecast_type}"

    def get_load_forecast(
        self,
        country_code: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.Series:
        """
        Get TSO load forecast from database.
        """
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT timestamp_utc, forecast_value_mw
            FROM energy_load_forecast
            WHERE country_code = ?
              AND forecast_type = ?
              AND timestamp_utc >= ?
              AND timestamp_utc <= ?
            ORDER BY timestamp_utc
        """
        df = pd.read_sql_query(
            query,
            conn,
            params=(country_code, self.forecast_type, start_date.isoformat(), end_date.isoformat()),
            parse_dates=['timestamp_utc'],
            index_col='timestamp_utc',
        )
        conn.close()
        return df['forecast_value_mw'] if not df.empty else pd.Series(dtype=float)

    def get_generation_forecast(
        self,
        country_code: str,
        start_date: datetime,
        end_date: datetime,
        generation_type: str = "solar",
    ) -> pd.Series:
        """
        Get TSO generation forecast from database.

        Args:
            generation_type: 'solar', 'wind_onshore', 'wind_offshore', or 'total'
        """
        conn = sqlite3.connect(self.db_path)

        column_map = {
            "solar": "solar_mw",
            "wind_onshore": "wind_onshore_mw",
            "wind_offshore": "wind_offshore_mw",
            "total": "total_forecast_mw",
        }
        column = column_map.get(generation_type, "total_forecast_mw")

        query = f"""
            SELECT timestamp_utc, {column}
            FROM energy_generation_forecast
            WHERE country_code = ?
              AND timestamp_utc >= ?
              AND timestamp_utc <= ?
            ORDER BY timestamp_utc
        """
        df = pd.read_sql_query(
            query,
            conn,
            params=(country_code, start_date.isoformat(), end_date.isoformat()),
            parse_dates=['timestamp_utc'],
            index_col='timestamp_utc',
        )
        conn.close()
        return df[column] if not df.empty else pd.Series(dtype=float)


def compute_baseline_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    historical_data: pd.Series,
    target_timestamps: pd.DatetimeIndex,
    forecast_type: str,
    horizon_hours: int = 48,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for all baseline models and compare with model predictions.

    Args:
        y_true: Actual values
        y_pred: Model predictions
        historical_data: Historical data for baseline computation
        target_timestamps: Timestamps corresponding to y_true
        forecast_type: Type of forecast for MAPE thresholds
        horizon_hours: Forecast horizon

    Returns:
        Dictionary with metrics for each baseline and skill scores
    """
    baselines = {
        "persistence": PersistenceBaseline(horizon_hours=horizon_hours),
        "seasonal_naive": SeasonalNaiveBaseline(horizon_hours=horizon_hours),
        "weekly_average": WeeklyAverageBaseline(horizon_hours=horizon_hours),
    }

    results = {
        "model": calculate_all_metrics(y_true, y_pred, forecast_type=forecast_type),
    }

    for name, baseline in baselines.items():
        y_baseline = baseline.predict_for_target(historical_data, target_timestamps)

        # Filter out NaN values
        valid_mask = ~np.isnan(y_baseline)
        if valid_mask.sum() > 0:
            y_true_valid = y_true[valid_mask]
            y_baseline_valid = y_baseline[valid_mask]
            y_pred_valid = y_pred[valid_mask]

            results[name] = calculate_all_metrics(
                y_true_valid, y_baseline_valid, forecast_type=forecast_type
            )
            # Compute skill score of model vs this baseline
            results[f"skill_vs_{name}"] = skill_score(
                y_true_valid, y_pred_valid, y_baseline_valid, metric="mae"
            )
        else:
            results[name] = {"mae": float("nan"), "rmse": float("nan")}
            results[f"skill_vs_{name}"] = float("nan")

    return results


def get_all_baseline_predictions(
    historical_data: pd.Series,
    target_timestamps: pd.DatetimeIndex,
    horizon_hours: int = 48,
) -> Dict[str, np.ndarray]:
    """
    Get predictions from all baseline models.

    Args:
        historical_data: Historical data with datetime index
        target_timestamps: Target timestamps to forecast
        horizon_hours: Forecast horizon

    Returns:
        Dictionary mapping baseline name to prediction array
    """
    baselines = {
        "persistence": PersistenceBaseline(horizon_hours=horizon_hours),
        "seasonal_naive": SeasonalNaiveBaseline(horizon_hours=horizon_hours),
        "weekly_average": WeeklyAverageBaseline(horizon_hours=horizon_hours),
    }

    predictions = {}
    for name, baseline in baselines.items():
        predictions[name] = baseline.predict_for_target(
            historical_data, target_timestamps
        )

    return predictions


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing baseline models...")

    # Create sample hourly data for 2 weeks
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=24 * 14, freq="H")

    # Simulate load data with daily pattern
    hours = dates.hour
    base_load = 40000 + 10000 * np.sin(2 * np.pi * hours / 24)  # Daily cycle
    noise = np.random.normal(0, 1000, len(dates))
    load_data = pd.Series(base_load + noise, index=dates)

    print(f"\n1. Sample data shape: {load_data.shape}")
    print(f"   Date range: {dates[0]} to {dates[-1]}")

    # Test persistence baseline
    print("\n2. Testing Persistence Baseline:")
    persistence = PersistenceBaseline(horizon_hours=48)
    target_times = dates[-24:]  # Last day
    preds = persistence.predict_for_target(load_data, target_times)
    print(f"   Predictions shape: {preds.shape}")
    print(f"   First few predictions: {preds[:3]}")

    # Test seasonal naive baseline
    print("\n3. Testing Seasonal Naive Baseline:")
    seasonal = SeasonalNaiveBaseline(horizon_hours=48)
    preds = seasonal.predict_for_target(load_data, target_times)
    print(f"   Predictions shape: {preds.shape}")
    print(f"   First few predictions: {preds[:3]}")

    # Test weekly average baseline
    print("\n4. Testing Weekly Average Baseline:")
    weekly = WeeklyAverageBaseline(horizon_hours=48)
    preds = weekly.predict_for_target(load_data, target_times)
    print(f"   Predictions shape: {preds.shape}")
    print(f"   First few predictions: {preds[:3]}")

    # Test baseline metrics computation
    print("\n5. Testing baseline metrics computation:")
    y_true = load_data[-24:].values
    y_pred = y_true * 1.05  # Simulate model predictions (5% overestimate)
    metrics = compute_baseline_metrics(
        y_true=y_true,
        y_pred=y_pred,
        historical_data=load_data,
        target_timestamps=target_times,
        forecast_type="load",
        horizon_hours=48,
    )
    print("   Metrics computed:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"   {key}: MAE={value.get('mae', 'N/A'):.2f}")
        else:
            print(f"   {key}: {value:.4f}")

    print("\n[OK] Baseline tests complete!")
