"""
Feature Engineering for Energy Forecasting

Creates features for D+2 forecasting including:
- Time features (hour, day of week, month, etc.)
- Holiday features (European calendar)
- Historical patterns (same hour from D-1, D-7, D-14)
- Rolling statistics
- Weather features
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import logging
import holidays

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


logger = logging.getLogger("energy_forecast")


# ============================================================================
# TIME FEATURES
# ============================================================================


def create_time_features(
    df: pd.DataFrame, timestamp_col: str = "timestamp_utc"
) -> pd.DataFrame:
    """
    Create time-based features from timestamp.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])

    # Basic time features
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek  # 0=Monday, 6=Sunday
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear

    # Binary features
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding for periodic features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


# ============================================================================
# HOLIDAY FEATURES
# ============================================================================

# Mapping of country codes to holidays library country codes
COUNTRY_HOLIDAY_MAP = {
    'AT': 'AT',  # Austria
    'BE': 'BE',  # Belgium
    'BG': 'BG',  # Bulgaria
    'CH': 'CH',  # Switzerland
    'CZ': 'CZ',  # Czech Republic
    'DE': 'DE',  # Germany
    'EE': 'EE',  # Estonia
    'ES': 'ES',  # Spain
    'FI': 'FI',  # Finland
    'FR': 'FR',  # France
    'GR': 'GR',  # Greece
    'HR': 'HR',  # Croatia
    'HU': 'HU',  # Hungary
    'IT': 'IT',  # Italy
    'LT': 'LT',  # Lithuania
    'LV': 'LV',  # Latvia
    'NL': 'NL',  # Netherlands
    'NO': 'NO',  # Norway
    'PL': 'PL',  # Poland
    'PT': 'PT',  # Portugal
    'RO': 'RO',  # Romania
    'SE': 'SE',  # Sweden
    'SI': 'SI',  # Slovenia
    'SK': 'SK',  # Slovakia
}


def _get_holiday_calendar(country_code: str, years: List[int]) -> Dict:
    """
    Get holiday calendar for a country and set of years.

    Args:
        country_code: ISO 2-letter country code
        years: List of years to fetch holidays for

    Returns:
        Dictionary mapping dates to holiday names
    """
    holiday_country = COUNTRY_HOLIDAY_MAP.get(country_code.upper())
    if holiday_country is None:
        logger.warning(f"No holiday calendar for {country_code}, using Germany as fallback")
        holiday_country = 'DE'

    try:
        return holidays.country_holidays(holiday_country, years=years)
    except Exception as e:
        logger.warning(f"Failed to load holidays for {country_code}: {e}")
        return {}


def create_holiday_features(
    df: pd.DataFrame,
    country_code: str,
    timestamp_col: str = "timestamp_utc"
) -> pd.DataFrame:
    """
    Create holiday-related features from timestamp and country.

    Features:
    - is_holiday: Binary flag (1 if holiday, 0 otherwise)
    - days_to_holiday: Days until next holiday (0-7, capped)
    - days_from_holiday: Days since last holiday (0-7, capped)
    - is_bridge_day: 1 if day between holiday and weekend

    Args:
        df: DataFrame with timestamp column
        country_code: ISO 2-letter country code
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with added holiday features
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])
    dates = ts.dt.date

    # Get unique years in the data
    years = list(set(ts.dt.year.unique()) | {ts.dt.year.min() - 1, ts.dt.year.max() + 1})
    holiday_cal = _get_holiday_calendar(country_code, years)

    # Create holiday lookup set for fast access
    holiday_dates = set(holiday_cal.keys())

    # is_holiday: binary flag
    df["is_holiday"] = dates.apply(lambda d: 1 if d in holiday_dates else 0)

    # Compute days_to_holiday and days_from_holiday
    # Sort holiday dates for efficient lookup
    sorted_holidays = sorted(holiday_dates)

    if sorted_holidays:
        days_to = []
        days_from = []

        for d in dates:
            # Days to next holiday
            future = [h for h in sorted_holidays if h > d]
            if future:
                days_to.append(min(7, (future[0] - d).days))
            else:
                days_to.append(7)  # Cap at 7 if no holiday found

            # Days from last holiday
            past = [h for h in sorted_holidays if h < d]
            if past:
                days_from.append(min(7, (d - past[-1]).days))
            else:
                days_from.append(7)  # Cap at 7 if no holiday found

        df["days_to_holiday"] = days_to
        df["days_from_holiday"] = days_from
    else:
        df["days_to_holiday"] = 7
        df["days_from_holiday"] = 7

    # is_bridge_day: day between holiday and weekend
    # A bridge day is typically a Friday when Thursday is a holiday,
    # or Monday when Tuesday is a holiday
    day_of_week = ts.dt.dayofweek  # 0=Monday, 4=Friday

    bridge_days = []
    for i, d in enumerate(dates):
        is_bridge = 0
        dow = day_of_week.iloc[i]

        # Friday bridge: Thursday is holiday
        if dow == 4:  # Friday
            import datetime
            thursday = d - datetime.timedelta(days=1)
            if thursday in holiday_dates:
                is_bridge = 1

        # Monday bridge: Tuesday is holiday
        elif dow == 0:  # Monday
            import datetime
            tuesday = d + datetime.timedelta(days=1)
            if tuesday in holiday_dates:
                is_bridge = 1

        bridge_days.append(is_bridge)

    df["is_bridge_day"] = bridge_days

    return df


# ============================================================================
# LAG FEATURES
# ============================================================================


def create_lag_features(
    df: pd.DataFrame, target_col: str = "target_value", lag_days: List[int] = None
) -> pd.DataFrame:
    """
    Create lag features for the target variable.

    For D+2 forecasting, we use same-hour values from previous days.

    Args:
        df: DataFrame with target column (must be sorted by timestamp)
        target_col: Name of target column
        lag_days: Days to lag (default: [1, 7, 14])

    Returns:
        DataFrame with lag features
    """
    if lag_days is None:
        lag_days = config.LAG_DAYS

    df = df.copy()

    for days in lag_days:
        lag_hours = days * 24
        col_name = f"{target_col}_lag_{days}d"
        df[col_name] = df[target_col].shift(lag_hours)

    return df


# ============================================================================
# ROLLING FEATURES
# ============================================================================


def create_rolling_features(
    df: pd.DataFrame, target_col: str = "target_value", windows: List[int] = None
) -> pd.DataFrame:
    """
    Create rolling window statistics.

    Args:
        df: DataFrame with target column (must be sorted by timestamp)
        target_col: Name of target column
        windows: Window sizes in hours (default: [24, 168])

    Returns:
        DataFrame with rolling features
    """
    if windows is None:
        windows = config.ROLLING_WINDOWS

    df = df.copy()

    for window in windows:
        prefix = f"{target_col}_roll_{window}h"

        # Mean
        df[f"{prefix}_mean"] = (
            df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        )

        # Standard deviation
        df[f"{prefix}_std"] = (
            df[target_col].shift(1).rolling(window=window, min_periods=1).std()
        )

        # Min/Max
        df[f"{prefix}_min"] = (
            df[target_col].shift(1).rolling(window=window, min_periods=1).min()
        )
        df[f"{prefix}_max"] = (
            df[target_col].shift(1).rolling(window=window, min_periods=1).max()
        )

    return df


# ============================================================================
# WEATHER FEATURES
# ============================================================================


def create_weather_features(df: pd.DataFrame, forecast_type: str) -> pd.DataFrame:
    """
    Process weather features for a specific forecast type.

    Includes:
    - Temperature conversion (K to C)
    - Heating/cooling degree days for load
    - Feature selection based on forecast type

    Args:
        df: DataFrame with weather columns
        forecast_type: 'load', 'price', or 'renewable'

    Returns:
        DataFrame with processed weather features
    """
    df = df.copy()

    # Temperature conversion (Kelvin to Celsius)
    if "temperature_2m_k" in df.columns:
        df["temperature_c"] = df["temperature_2m_k"] - 273.15

        # Heating and cooling degree days (for load forecasting)
        if forecast_type == "load":
            base_temp = 18  # Reference temperature
            df["heating_degree"] = np.maximum(0, base_temp - df["temperature_c"])
            df["cooling_degree"] = np.maximum(0, df["temperature_c"] - base_temp)

    # Select relevant weather features based on forecast type
    weather_cols = config.WEATHER_FEATURES.get(forecast_type, [])

    # Keep only relevant weather columns plus derived features
    keep_cols = []
    for col in weather_cols:
        if col in df.columns:
            keep_cols.append(col)

    # Add derived columns
    if "temperature_c" in df.columns:
        keep_cols.append("temperature_c")
    if "heating_degree" in df.columns:
        keep_cols.extend(["heating_degree", "cooling_degree"])

    # Log which weather features are available
    logger.debug(f"Weather features for {forecast_type}: {keep_cols}")

    return df


# ============================================================================
# COMPLETE FEATURE PIPELINE
# ============================================================================


def create_all_features(
    df: pd.DataFrame,
    forecast_type: str,
    target_col: str = "target_value",
    country_code: Optional[str] = None
) -> pd.DataFrame:
    """
    Create all features for model training.

    Args:
        df: Raw DataFrame with timestamp_utc, target_value, and weather columns
        forecast_type: 'load', 'price', or 'renewable'
        target_col: Name of target column
        country_code: ISO 2-letter country code for holiday features (optional)

    Returns:
        DataFrame with all features (rows with NaN in essential columns removed)
    """
    logger.info(f"Creating features for {forecast_type}")

    # Ensure sorted by timestamp
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Fill NaN in weather columns first (forward fill then backward fill)
    weather_cols = [c for c in df.columns if c not in ["timestamp_utc", target_col, "country_code"]]
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Create time features
    df = create_time_features(df)

    # Create holiday features if country code is provided
    if country_code is not None:
        df = create_holiday_features(df, country_code)
    elif "country_code" in df.columns:
        # Try to get country code from data
        cc = df["country_code"].iloc[0] if len(df) > 0 else None
        if cc:
            df = create_holiday_features(df, cc)

    # Create lag features
    df = create_lag_features(df, target_col)

    # Create rolling features
    df = create_rolling_features(df, target_col)

    # Process weather features
    df = create_weather_features(df, forecast_type)

    # Identify essential columns that must not be NaN
    essential_cols = [target_col]
    essential_cols += [c for c in df.columns if "_lag_" in c]
    essential_cols += [c for c in df.columns if "_roll_" in c and "_mean" in c]

    # Drop rows with NaN only in essential columns
    initial_len = len(df)
    mask = df[essential_cols].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    final_len = len(df)

    # Fill any remaining NaN with 0 (for weather features)
    df = df.fillna(0)

    logger.info(
        f"Features created. Rows: {initial_len} -> {final_len} (dropped {initial_len - final_len} with NaN)"
    )

    return df


def get_feature_columns(forecast_type: str, include_holidays: bool = True) -> List[str]:
    """
    Get list of feature column names for a forecast type.

    Args:
        forecast_type: 'load', 'price', or 'renewable'
        include_holidays: Whether to include holiday features (default: True)

    Returns:
        List of feature column names
    """
    # Time features
    time_features = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
    ]

    # Holiday features (high impact for load forecasting)
    holiday_features = []
    if include_holidays:
        holiday_features = [
            "is_holiday",
            "days_to_holiday",
            "days_from_holiday",
            "is_bridge_day",
        ]

    # Lag features
    lag_features = [f"target_value_lag_{d}d" for d in config.LAG_DAYS]

    # Rolling features
    rolling_features = []
    for window in config.ROLLING_WINDOWS:
        rolling_features.extend(
            [
                f"target_value_roll_{window}h_mean",
                f"target_value_roll_{window}h_std",
                f"target_value_roll_{window}h_min",
                f"target_value_roll_{window}h_max",
            ]
        )

    # Weather features (based on type)
    weather_features = config.WEATHER_FEATURES.get(forecast_type, [])

    # Add derived weather features
    if forecast_type == "load":
        weather_features = weather_features + [
            "temperature_c",
            "heating_degree",
            "cooling_degree",
        ]
    else:
        weather_features = weather_features + ["temperature_c"]

    all_features = (
        time_features + holiday_features + lag_features + rolling_features + weather_features
    )

    return all_features


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    # Import db module
    from db import load_training_data

    print("Testing feature engineering...")

    # Load sample data
    print("\n1. Loading sample data...")
    df = load_training_data("DE", "load", "2024-01-01", "2024-03-01")
    print(f"   Raw data: {len(df)} rows, {len(df.columns)} columns")

    # Create features
    print("\n2. Creating features...")
    df_features = create_all_features(df, "load")
    print(
        f"   With features: {len(df_features)} rows, {len(df_features.columns)} columns"
    )
    print(f"   Columns: {list(df_features.columns)}")

    # Get feature column names
    print("\n3. Feature columns for 'load':")
    feature_cols = get_feature_columns("load")
    print(f"   {len(feature_cols)} features")
    for col in feature_cols:
        exists = col in df_features.columns
        print(f"   {'[OK]' if exists else '[--]'} {col}")

    # Check for NaN
    print("\n4. Checking for NaN values...")
    nan_counts = df_features.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"   Columns with NaN: {dict(nan_cols)}")
    else:
        print("   No NaN values found")

    print("\n[OK] Feature engineering tests complete!")
