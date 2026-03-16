"""
Database operations for Energy Forecasting Module
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional, Dict, List
from contextlib import contextmanager
import pandas as pd
import pytz

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


logger = logging.getLogger('energy_forecast')


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

@contextmanager
def get_connection(readonly: bool = True):
    """
    Context manager for database connections

    Args:
        readonly: If True, open in read-only mode (safer for queries)

    Yields:
        sqlite3.Connection: Database connection
    """
    conn = None
    try:
        # Always use standard connection (SQLite URI mode can be unreliable)
        conn = sqlite3.connect(str(config.DATABASE_PATH), timeout=30.0)
        conn.row_factory = sqlite3.Row
        yield conn

        if not readonly:
            conn.commit()
    except Exception as e:
        if conn and not readonly:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()


# ============================================================================
# SCHEMA CREATION
# ============================================================================

def create_forecasts_table():
    """
    Create the forecasts table if it doesn't exist.
    Run this once during setup.
    """
    with get_connection(readonly=False) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country_code TEXT NOT NULL,
                forecast_type TEXT NOT NULL,
                target_timestamp_utc TIMESTAMP NOT NULL,
                generated_at TIMESTAMP NOT NULL,
                horizon_hours INTEGER NOT NULL,
                forecast_value REAL NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT,

                UNIQUE(country_code, forecast_type, target_timestamp_utc, horizon_hours, model_name, generated_at)
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_forecasts_lookup
            ON forecasts(country_code, forecast_type, target_timestamp_utc)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_forecasts_generated
            ON forecasts(generated_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_forecasts_model_lookup
            ON forecasts(country_code, forecast_type, model_name, target_timestamp_utc)
        """)

        logger.info("Forecasts table created/verified")

    # Run migration for existing databases
    migrate_forecasts_add_model_name_unique()


def migrate_forecasts_add_model_name_unique():
    """
    Migrate forecasts table to include model_name in UNIQUE constraint.

    This prevents two models generating at the same timestamp from overwriting
    each other. Idempotent — checks if already migrated before running.
    """
    with get_connection(readonly=False) as conn:
        cursor = conn.cursor()

        # Check if already migrated by inspecting the table's CREATE SQL
        cursor.execute("""
            SELECT sql FROM sqlite_master
            WHERE type='table' AND name='forecasts'
        """)
        row = cursor.fetchone()
        if row is None:
            return  # Table doesn't exist yet, will be created fresh

        create_sql = row['sql']
        # If model_name is already in the UNIQUE constraint, skip migration
        if 'model_name, generated_at)' in create_sql:
            logger.debug("Forecasts table already migrated — skipping")
            return

        logger.info("Migrating forecasts table: adding model_name to UNIQUE constraint...")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecasts_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country_code TEXT NOT NULL,
                forecast_type TEXT NOT NULL,
                target_timestamp_utc TIMESTAMP NOT NULL,
                generated_at TIMESTAMP NOT NULL,
                horizon_hours INTEGER NOT NULL,
                forecast_value REAL NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT,

                UNIQUE(country_code, forecast_type, target_timestamp_utc, horizon_hours, model_name, generated_at)
            )
        """)

        cursor.execute("""
            INSERT INTO forecasts_new
                (id, country_code, forecast_type, target_timestamp_utc,
                 generated_at, horizon_hours, forecast_value, model_name, model_version)
            SELECT
                id, country_code, forecast_type, target_timestamp_utc,
                generated_at, horizon_hours, forecast_value,
                COALESCE(model_name, 'xgboost'), model_version
            FROM forecasts
        """)

        cursor.execute("DROP TABLE forecasts")
        cursor.execute("ALTER TABLE forecasts_new RENAME TO forecasts")

        # Recreate indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_forecasts_lookup
            ON forecasts(country_code, forecast_type, target_timestamp_utc)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_forecasts_generated
            ON forecasts(generated_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_forecasts_model_lookup
            ON forecasts(country_code, forecast_type, model_name, target_timestamp_utc)
        """)

        logger.info("Forecasts table migration complete")


# ============================================================================
# DATA LOADING FOR TRAINING
# ============================================================================

def load_energy_data(
    country_code: str,
    forecast_type: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Load energy data (load, price, or renewable) for training.

    Args:
        country_code: ISO 2-letter country code
        forecast_type: 'load', 'price', 'renewable', or individual renewable types
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with timestamp_utc and target value columns
    """
    # Check if this is an individual renewable type
    if forecast_type in config.RENEWABLE_TYPES:
        return load_renewable_type_data(country_code, forecast_type, start_date, end_date)

    table_map = {
        'load': ('energy_load', 'load_mw'),
        'price': ('energy_price', 'price_eur_mwh'),
        'renewable': ('energy_renewable', 'total_renewable_mw')
    }

    if forecast_type not in table_map:
        raise ValueError(f"Unknown forecast type: {forecast_type}")

    table, value_col = table_map[forecast_type]

    query = f"""
        SELECT timestamp_utc, {value_col} as target_value
        FROM {table}
        WHERE country_code = ?
          AND timestamp_utc >= ?
          AND timestamp_utc < ?
          AND data_quality = 'actual'
        ORDER BY timestamp_utc
    """

    with get_connection() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(country_code, start_date, end_date),
        )

    # Parse timestamps handling mixed formats (some may have TZ offsets)
    if not df.empty:
        df['timestamp_utc'] = pd.to_datetime(
            df['timestamp_utc'], format='mixed', utc=True
        ).dt.tz_localize(None)

    logger.info(f"Loaded {len(df)} {forecast_type} records for {country_code}")
    return df


def load_renewable_type_data(
    country_code: str,
    renewable_type: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Load specific renewable type data as training target

    Args:
        country_code: ISO 2-letter country code
        renewable_type: 'solar', 'wind_onshore', 'wind_offshore', 'hydro_total', 'biomass'
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD

    Returns:
        DataFrame with timestamp_utc and target_value columns
    """
    # Map renewable type to database columns
    column_map = {
        'solar': 'solar_mw',
        'wind_onshore': 'wind_onshore_mw',
        'wind_offshore': 'wind_offshore_mw',
        'hydro_total': '(hydro_run_mw + hydro_reservoir_mw)',
        'biomass': 'biomass_mw'
    }

    if renewable_type not in column_map:
        raise ValueError(f"Unknown renewable type: {renewable_type}")

    target_col = column_map[renewable_type]

    query = f"""
        SELECT timestamp_utc, {target_col} as target_value
        FROM energy_renewable
        WHERE country_code = ?
          AND timestamp_utc >= ?
          AND timestamp_utc < ?
        ORDER BY timestamp_utc
    """

    with get_connection() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(country_code, start_date, end_date),
        )

    if df.empty:
        logger.warning(f"No {renewable_type} data for {country_code}")
        return df

    # Parse timestamps handling mixed formats (some may have TZ offsets)
    df['timestamp_utc'] = pd.to_datetime(
        df['timestamp_utc'], format='mixed', utc=True
    ).dt.tz_localize(None)

    logger.info(f"Loaded {len(df)} {renewable_type} records for {country_code}")
    return df


def load_weather_data(
    country_code: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Load historical (actual) weather data for a country.

    Filters for data_quality='actual' to exclude forecast records.
    Use load_weather_forecast() for forecast data.

    Args:
        country_code: ISO 2-letter country code
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with weather features (historical/actual only)
    """
    query = """
        SELECT
            timestamp_utc,
            temperature_2m_k,
            relative_humidity_2m_frac,
            wind_speed_10m_ms,
            wind_speed_100m_ms,
            shortwave_radiation_wm2,
            direct_radiation_wm2,
            diffuse_radiation_wm2
        FROM weather_data
        WHERE country_code = ?
          AND timestamp_utc >= ?
          AND timestamp_utc < ?
          AND data_quality = 'actual'
        ORDER BY timestamp_utc
    """

    with get_connection() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(country_code, start_date, end_date),
        )

    # Parse timestamps handling mixed formats (some may have TZ offsets)
    if not df.empty:
        df['timestamp_utc'] = pd.to_datetime(
            df['timestamp_utc'], format='mixed', utc=True
        ).dt.tz_localize(None)

    logger.info(f"Loaded {len(df)} actual weather records for {country_code}")
    return df


def load_weather_forecast(
    country_code: str,
    target_date: str,
    forecast_run_time: Optional[str] = None
) -> pd.DataFrame:
    """
    Load weather forecast data for a target date.

    Args:
        country_code: ISO 2-letter country code
        target_date: Target date (YYYY-MM-DD) to get forecast for
        forecast_run_time: Optional specific forecast run time.
                          If None, uses the most recent forecast available.

    Returns:
        DataFrame with weather forecast features for the target date (hourly).
        Returns empty DataFrame if no forecast available.
    """
    if forecast_run_time:
        # Use specific forecast run
        query = """
            SELECT
                timestamp_utc,
                temperature_2m_k,
                relative_humidity_2m_frac,
                wind_speed_10m_ms,
                wind_speed_100m_ms,
                shortwave_radiation_wm2,
                direct_radiation_wm2,
                diffuse_radiation_wm2,
                forecast_run_time
            FROM weather_data
            WHERE country_code = ?
              AND DATE(timestamp_utc) = DATE(?)
              AND data_quality = 'forecast'
              AND forecast_run_time = ?
            ORDER BY timestamp_utc
        """
        params = (country_code, target_date, forecast_run_time)
    else:
        # Use most recent forecast for the target date
        query = """
            SELECT
                timestamp_utc,
                temperature_2m_k,
                relative_humidity_2m_frac,
                wind_speed_10m_ms,
                wind_speed_100m_ms,
                shortwave_radiation_wm2,
                direct_radiation_wm2,
                diffuse_radiation_wm2,
                forecast_run_time
            FROM weather_data
            WHERE country_code = ?
              AND DATE(timestamp_utc) = DATE(?)
              AND data_quality = 'forecast'
              AND forecast_run_time = (
                  SELECT MAX(forecast_run_time)
                  FROM weather_data
                  WHERE country_code = ?
                    AND DATE(timestamp_utc) = DATE(?)
                    AND data_quality = 'forecast'
              )
            ORDER BY timestamp_utc
        """
        params = (country_code, target_date, country_code, target_date)

    with get_connection() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=params,
            parse_dates=['timestamp_utc', 'forecast_run_time']
        )

    if not df.empty:
        run_time = df['forecast_run_time'].iloc[0]
        logger.info(f"Loaded {len(df)} forecast weather records for {country_code} "
                   f"(target: {target_date}, run: {run_time})")
    else:
        logger.warning(f"No weather forecast found for {country_code} on {target_date}")

    return df


def load_weather_forecast_for_hour(
    country_code: str,
    target_datetime: datetime
) -> Optional[dict]:
    """
    Load weather forecast for a specific hour.

    Args:
        country_code: ISO 2-letter country code
        target_datetime: Target datetime (will be rounded to hour)

    Returns:
        Dictionary with weather features for that hour, or None if not available.
    """
    # Round to hour
    target_hour = target_datetime.replace(minute=0, second=0, microsecond=0)
    target_str = target_hour.strftime('%Y-%m-%d %H:%M:%S')

    query = """
        SELECT
            temperature_2m_k,
            relative_humidity_2m_frac,
            wind_speed_10m_ms,
            wind_speed_100m_ms,
            shortwave_radiation_wm2,
            direct_radiation_wm2,
            diffuse_radiation_wm2
        FROM weather_data
        WHERE country_code = ?
          AND timestamp_utc = ?
          AND data_quality = 'forecast'
        ORDER BY forecast_run_time DESC
        LIMIT 1
    """

    with get_connection() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(country_code, target_str)
        )

    if df.empty:
        logger.debug(f"No weather forecast for {country_code} at {target_str}")
        return None

    return df.iloc[0].to_dict()


def load_training_data(
    country_code: str,
    forecast_type: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Load merged dataset with energy target and weather features.

    Resamples energy data to hourly and merges with weather.

    Args:
        country_code: ISO 2-letter country code
        forecast_type: 'load', 'price', or 'renewable'
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Merged DataFrame with target and weather features (hourly)
    """
    # Load energy data
    energy_df = load_energy_data(country_code, forecast_type, start_date, end_date)

    # Load weather data
    weather_df = load_weather_data(country_code, start_date, end_date)

    if energy_df.empty:
        logger.warning(f"No energy data for {country_code} {forecast_type}")
        return energy_df

    # Resample energy data to hourly (mean of sub-hourly values)
    energy_df = energy_df.set_index('timestamp_utc')
    energy_df = energy_df.resample('h').mean().reset_index()
    energy_df = energy_df.dropna()

    logger.info(f"Resampled to {len(energy_df)} hourly records")

    # Merge with weather if available
    if not weather_df.empty:
        df = pd.merge(
            energy_df,
            weather_df,
            on='timestamp_utc',
            how='left'
        )
    else:
        df = energy_df

    logger.info(f"Training data: {len(df)} records for {country_code} {forecast_type}")
    return df


def load_training_data_multipoint(
    country_code: str,
    forecast_type: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Load merged dataset using multipoint weather data instead of centroid data.

    Args:
        country_code: ISO 2-letter country code
        forecast_type: 'load', 'price', or 'renewable'  
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Merged DataFrame with target and multipoint weather features (hourly)
    """
    # Load energy data using existing function (unchanged)
    energy_df = load_energy_data(country_code, forecast_type, start_date, end_date)
    
    # Load weather data from multipoint table
    weather_df = load_weather_data_multipoint(country_code, forecast_type, start_date, end_date)

    if energy_df.empty:
        logger.warning(f"No energy data for {country_code} {forecast_type}")
        return energy_df

    # Resample energy data to hourly (mean of sub-hourly values)
    energy_df = energy_df.set_index('timestamp_utc')
    energy_df = energy_df.resample('h').mean().reset_index()
    energy_df = energy_df.dropna()

    logger.info(f"Resampled to {len(energy_df)} hourly records")

    # Merge with weather if available
    if not weather_df.empty:
        df = pd.merge(
            energy_df,
            weather_df,
            on='timestamp_utc',
            how='left'
        )
    else:
        df = energy_df

    logger.info(f"Multipoint training data: {len(df)} records for {country_code} {forecast_type}")
    return df


def load_weather_data_multipoint(
    country_code: str,
    forecast_type: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Load multipoint weather data for a specific country and forecast type.

    Args:
        country_code: ISO 2-letter country code
        forecast_type: Type of forecast ('wind_onshore', 'wind_offshore', 'solar', 'load')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with multipoint weather data
    """
    query = """
    SELECT 
        timestamp_utc,
        temperature_2m_k,
        dew_point_2m_k,
        relative_humidity_2m_frac,
        pressure_msl_hpa,
        wind_speed_10m_ms,
        wind_gusts_10m_ms,
        wind_direction_10m_deg,
        wind_speed_100m_ms,
        wind_direction_100m_deg,
        wind_speed_80m_ms,
        wind_speed_120m_ms,
        precip_mm,
        rain_mm,
        snowfall_mm,
        shortwave_radiation_wm2,
        direct_radiation_wm2,
        direct_normal_irradiance_wm2,
        diffuse_radiation_wm2,
        n_points
    FROM weather_data_multipoint 
    WHERE country_code = ?
      AND forecast_type = ?
      AND timestamp_utc >= ?
      AND timestamp_utc <= ?
    ORDER BY timestamp_utc
    """

    try:
        with get_connection() as conn:
            df = pd.read_sql_query(
                query, 
                conn, 
                params=[country_code, forecast_type, start_date, end_date]
            )

        if not df.empty:
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
            logger.info(f"Loaded {len(df)} multipoint weather records for {country_code}-{forecast_type}")
        else:
            logger.warning(f"No multipoint weather data found for {country_code}-{forecast_type} from {start_date} to {end_date}")

        return df

    except Exception as e:
        logger.error(f"Error loading multipoint weather data: {e}")
        return pd.DataFrame()


# ============================================================================
# FORECAST STORAGE
# ============================================================================

def save_forecasts(forecasts_df: pd.DataFrame) -> int:
    """
    Save forecasts to the database.

    Args:
        forecasts_df: DataFrame with columns:
            - country_code
            - forecast_type
            - target_timestamp_utc
            - generated_at
            - horizon_hours
            - forecast_value
            - model_name
            - model_version (optional)

    Returns:
        Number of records inserted
    """
    if forecasts_df.empty:
        logger.warning("Empty DataFrame, nothing to save")
        return 0

    required_cols = [
        'country_code', 'forecast_type', 'target_timestamp_utc',
        'generated_at', 'horizon_hours', 'forecast_value', 'model_name'
    ]
    for col in required_cols:
        if col not in forecasts_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Filter out all-zero forecasts for types where zero is never valid
    zero_invalid_types = {'load', 'price'}
    if forecasts_df['forecast_type'].iloc[0] in zero_invalid_types:
        zero_mask = forecasts_df['forecast_value'] == 0
        if zero_mask.all():
            logger.warning(f"All forecast values are zero for {forecasts_df['forecast_type'].iloc[0]} - skipping save")
            return 0

    records_inserted = 0

    with get_connection(readonly=False) as conn:
        cursor = conn.cursor()

        for _, row in forecasts_df.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO forecasts
                    (country_code, forecast_type, renewable_type, target_timestamp_utc,
                     generated_at, horizon_hours, forecast_value,
                     model_name, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['country_code'],
                    row['forecast_type'],
                    row.get('renewable_type'),  # Can be NULL for load/price
                    row['target_timestamp_utc'].isoformat() if hasattr(row['target_timestamp_utc'], 'isoformat') else row['target_timestamp_utc'],
                    row['generated_at'].isoformat() if hasattr(row['generated_at'], 'isoformat') else row['generated_at'],
                    int(row['horizon_hours']),
                    float(row['forecast_value']),
                    row['model_name'],
                    row.get('model_version')
                ))
                records_inserted += 1
            except Exception as e:
                logger.error(f"Failed to insert forecast: {e}")

    logger.info(f"Saved {records_inserted} forecasts to database")
    return records_inserted


def get_forecasts(
    country_code: str,
    forecast_type: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Retrieve forecasts for evaluation.

    Args:
        country_code: ISO 2-letter country code
        forecast_type: 'load', 'price', or 'renewable'
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with forecast records
    """
    query = """
        SELECT
            country_code,
            forecast_type,
            target_timestamp_utc,
            generated_at,
            horizon_hours,
            forecast_value,
            model_name,
            model_version
        FROM forecasts
        WHERE country_code = ?
          AND forecast_type = ?
          AND target_timestamp_utc >= ?
          AND target_timestamp_utc < ?
        ORDER BY target_timestamp_utc, horizon_hours
    """

    with get_connection() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(country_code, forecast_type, start_date, end_date),
            parse_dates=['target_timestamp_utc', 'generated_at']
        )

    return df


def get_latest_data_timestamp(country_code: str, data_type: str) -> Optional[datetime]:
    """
    Get the most recent data timestamp for a country/type.

    Args:
        country_code: ISO 2-letter country code
        data_type: 'load', 'price', 'renewable', or 'weather'

    Returns:
        Most recent timestamp or None
    """
    # Map renewable types to energy_renewable table
    if data_type in config.RENEWABLE_TYPES:
        table = 'energy_renewable'
    else:
        table_map = {
            'load': 'energy_load',
            'price': 'energy_price',
            'renewable': 'energy_renewable',
            'weather': 'weather_data'
        }

        if data_type not in table_map:
            raise ValueError(f"Unknown data type: {data_type}")

        table = table_map[data_type]

    query = f"""
        SELECT MAX(timestamp_utc) as latest
        FROM {table}
        WHERE country_code = ?
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (country_code,))
        result = cursor.fetchone()

        if result and result['latest']:
            return pd.to_datetime(result['latest'])

    return None


# ============================================================================
# MODEL EVALUATIONS
# ============================================================================

def create_model_evaluations_table():
    """
    Create the model_evaluations table for storing evaluation metrics.
    """
    with get_connection(readonly=False) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country_code TEXT NOT NULL,
                forecast_type TEXT NOT NULL,
                model_version TEXT NOT NULL,
                evaluation_date TEXT NOT NULL,
                -- Core metrics
                mae REAL,
                rmse REAL,
                mape REAL,
                smape REAL,
                mase REAL,
                directional_accuracy REAL,
                -- Baseline comparisons
                skill_vs_persistence REAL,
                skill_vs_seasonal REAL,
                skill_vs_tso REAL,
                -- Metadata
                training_samples INTEGER,
                test_samples INTEGER,
                evaluation_periods TEXT,
                is_baseline BOOLEAN DEFAULT FALSE,
                model_location TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(country_code, forecast_type, model_version, evaluation_date)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_evaluations_lookup
            ON model_evaluations(country_code, forecast_type, model_version)
        """)

        logger.info("Model evaluations table created/verified")


def save_model_evaluation(
    country_code: str,
    forecast_type: str,
    model_version: str,
    metrics: Dict,
    skill_scores: Dict,
    training_samples: int = 0,
    test_samples: int = 0,
    evaluation_periods: Optional[str] = None,
    is_baseline: bool = False,
    model_location: str = "candidate",
) -> int:
    """
    Save model evaluation metrics to database.

    Args:
        country_code: Country code
        forecast_type: Type of forecast
        model_version: Model version string
        metrics: Dictionary with mae, rmse, mape, smape, mase, directional_accuracy
        skill_scores: Dictionary with skill_vs_persistence, skill_vs_seasonal, skill_vs_tso
        training_samples: Number of training samples
        test_samples: Number of test samples
        evaluation_periods: JSON string of evaluation periods
        is_baseline: Whether this is a baseline evaluation
        model_location: 'candidate', 'production', or 'history'

    Returns:
        ID of inserted record
    """
    with get_connection(readonly=False) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO model_evaluations
            (country_code, forecast_type, model_version, evaluation_date,
             mae, rmse, mape, smape, mase, directional_accuracy,
             skill_vs_persistence, skill_vs_seasonal, skill_vs_tso,
             training_samples, test_samples, evaluation_periods,
             is_baseline, model_location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            country_code,
            forecast_type,
            model_version,
            datetime.now().strftime('%Y-%m-%d'),
            metrics.get('mae'),
            metrics.get('rmse'),
            metrics.get('mape'),
            metrics.get('smape'),
            metrics.get('mase'),
            metrics.get('directional_accuracy'),
            skill_scores.get('skill_vs_persistence'),
            skill_scores.get('skill_vs_seasonal_naive') or skill_scores.get('skill_vs_seasonal'),
            skill_scores.get('skill_vs_tso'),
            training_samples,
            test_samples,
            evaluation_periods,
            is_baseline,
            model_location,
        ))

        return cursor.lastrowid


def get_model_evaluations(
    country_code: str,
    forecast_type: str,
    model_version: Optional[str] = None,
    include_baselines: bool = False,
) -> pd.DataFrame:
    """
    Get model evaluations from database.

    Args:
        country_code: Country code
        forecast_type: Type of forecast
        model_version: Optional specific version (if None, returns all)
        include_baselines: Whether to include baseline evaluations

    Returns:
        DataFrame with evaluation records
    """
    query = """
        SELECT *
        FROM model_evaluations
        WHERE country_code = ?
          AND forecast_type = ?
    """
    params = [country_code, forecast_type]

    if model_version:
        query += " AND model_version = ?"
        params.append(model_version)

    if not include_baselines:
        query += " AND is_baseline = FALSE"

    query += " ORDER BY evaluation_date DESC"

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    return df


def get_latest_evaluation(
    country_code: str,
    forecast_type: str,
    model_location: str = "candidate",
) -> Optional[Dict]:
    """
    Get the most recent evaluation for a country/type/location.
    """
    query = """
        SELECT *
        FROM model_evaluations
        WHERE country_code = ?
          AND forecast_type = ?
          AND model_location = ?
          AND is_baseline = FALSE
        ORDER BY evaluation_date DESC
        LIMIT 1
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (country_code, forecast_type, model_location))
        row = cursor.fetchone()

        if row:
            return dict(row)

    return None


# ============================================================================
# DEPLOYED MODELS
# ============================================================================

def create_deployed_models_table():
    """
    Create the deployed_models table for tracking production deployments.
    """
    with get_connection(readonly=False) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deployed_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country_code TEXT NOT NULL,
                forecast_type TEXT NOT NULL,
                model_version TEXT NOT NULL,
                deployed_at TIMESTAMP NOT NULL,
                deployed_by TEXT DEFAULT 'system',
                previous_version TEXT,
                deployment_reason TEXT,
                status TEXT DEFAULT 'active',
                -- Performance at deployment
                mae_at_deployment REAL,
                mape_at_deployment REAL,
                skill_score_at_deployment REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_deployed_lookup
            ON deployed_models(country_code, forecast_type, status)
        """)

        logger.info("Deployed models table created/verified")


def get_deployed_model(
    country_code: str,
    forecast_type: str,
) -> Optional[Dict]:
    """
    Get the currently deployed (active) model for a country/type.

    Returns:
        Dictionary with deployment info or None if not deployed
    """
    query = """
        SELECT *
        FROM deployed_models
        WHERE country_code = ?
          AND forecast_type = ?
          AND status = 'active'
        ORDER BY deployed_at DESC
        LIMIT 1
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (country_code, forecast_type))
        row = cursor.fetchone()

        if row:
            return dict(row)

    return None


def save_deployment(
    country_code: str,
    forecast_type: str,
    model_version: str,
    deployed_by: str = "system",
    deployment_reason: str = "",
    mae_at_deployment: Optional[float] = None,
    mape_at_deployment: Optional[float] = None,
    skill_score_at_deployment: Optional[float] = None,
) -> int:
    """
    Record a new model deployment.

    This will:
    1. Mark any existing active deployment as 'inactive'
    2. Insert new deployment record as 'active'

    Returns:
        ID of new deployment record
    """
    with get_connection(readonly=False) as conn:
        cursor = conn.cursor()

        # Get previous version
        cursor.execute("""
            SELECT model_version FROM deployed_models
            WHERE country_code = ? AND forecast_type = ? AND status = 'active'
        """, (country_code, forecast_type))
        prev_row = cursor.fetchone()
        previous_version = prev_row['model_version'] if prev_row else None

        # Mark previous as inactive
        cursor.execute("""
            UPDATE deployed_models
            SET status = 'inactive'
            WHERE country_code = ? AND forecast_type = ? AND status = 'active'
        """, (country_code, forecast_type))

        # Insert new deployment
        cursor.execute("""
            INSERT INTO deployed_models
            (country_code, forecast_type, model_version, deployed_at,
             deployed_by, previous_version, deployment_reason, status,
             mae_at_deployment, mape_at_deployment, skill_score_at_deployment)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
        """, (
            country_code,
            forecast_type,
            model_version,
            datetime.now().isoformat(),
            deployed_by,
            previous_version,
            deployment_reason,
            mae_at_deployment,
            mape_at_deployment,
            skill_score_at_deployment,
        ))

        logger.info(f"Deployed {country_code}/{forecast_type} version {model_version}")
        return cursor.lastrowid


def rollback_deployment(
    country_code: str,
    forecast_type: str,
    rollback_by: str = "system",
) -> bool:
    """
    Rollback to the previous model version.

    Returns:
        True if rollback successful, False if no previous version
    """
    with get_connection(readonly=False) as conn:
        cursor = conn.cursor()

        # Get current active deployment
        cursor.execute("""
            SELECT id, model_version, previous_version
            FROM deployed_models
            WHERE country_code = ? AND forecast_type = ? AND status = 'active'
        """, (country_code, forecast_type))
        current = cursor.fetchone()

        if not current or not current['previous_version']:
            logger.warning(f"No previous version to rollback for {country_code}/{forecast_type}")
            return False

        # Mark current as rolled_back
        cursor.execute("""
            UPDATE deployed_models
            SET status = 'rolled_back'
            WHERE id = ?
        """, (current['id'],))

        # Find the previous deployment and reactivate it
        cursor.execute("""
            UPDATE deployed_models
            SET status = 'active'
            WHERE country_code = ? AND forecast_type = ?
              AND model_version = ? AND status = 'inactive'
        """, (country_code, forecast_type, current['previous_version']))

        if cursor.rowcount == 0:
            # Previous version not in history, create new deployment
            cursor.execute("""
                INSERT INTO deployed_models
                (country_code, forecast_type, model_version, deployed_at,
                 deployed_by, deployment_reason, status)
                VALUES (?, ?, ?, ?, ?, 'Rollback', 'active')
            """, (
                country_code,
                forecast_type,
                current['previous_version'],
                datetime.now().isoformat(),
                rollback_by,
            ))

        logger.info(f"Rolled back {country_code}/{forecast_type} to {current['previous_version']}")
        return True


def get_all_deployed_models() -> pd.DataFrame:
    """
    Get all currently deployed models.

    Returns:
        DataFrame with all active deployments
    """
    query = """
        SELECT
            country_code,
            forecast_type,
            model_version,
            deployed_at,
            deployed_by,
            mae_at_deployment,
            mape_at_deployment,
            skill_score_at_deployment
        FROM deployed_models
        WHERE status = 'active'
        ORDER BY country_code, forecast_type
    """

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, parse_dates=['deployed_at'])

    return df


def get_deployed_models() -> List[Dict]:
    """
    Get all currently deployed models as a list of dicts.

    Returns:
        List of deployed model dictionaries
    """
    df = get_all_deployed_models()
    return df.to_dict('records') if not df.empty else []


def get_deployment_history(
    country_code: Optional[str] = None,
    forecast_type: Optional[str] = None,
    limit: int = 100,
) -> pd.DataFrame:
    """
    Get deployment history.
    """
    query = """
        SELECT *
        FROM deployed_models
        WHERE 1=1
    """
    params = []

    if country_code:
        query += " AND country_code = ?"
        params.append(country_code)

    if forecast_type:
        query += " AND forecast_type = ?"
        params.append(forecast_type)

    query += " ORDER BY deployed_at DESC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params, parse_dates=['deployed_at'])

    return df


# ============================================================================
# FORECAST RUNS
# ============================================================================

def create_forecast_runs_table():
    """
    Create the forecast_runs table for tracking scheduled executions.
    """
    with get_connection(readonly=False) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TIMESTAMP NOT NULL,
                run_type TEXT NOT NULL,
                trigger_source TEXT,
                status TEXT NOT NULL,
                countries_requested TEXT,
                countries_completed TEXT,
                types_requested TEXT,
                types_completed TEXT,
                forecasts_generated INTEGER DEFAULT 0,
                execution_time_seconds REAL,
                error_message TEXT,
                log_file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_forecast_runs_timestamp
            ON forecast_runs(run_timestamp DESC)
        """)

        logger.info("Forecast runs table created/verified")


def start_forecast_run(
    run_type: str = "scheduled",
    trigger_source: str = "bat_file",
    countries: Optional[List[str]] = None,
    forecast_types: Optional[List[str]] = None,
) -> int:
    """
    Record the start of a forecast run.

    Returns:
        Run ID for updating later
    """
    with get_connection(readonly=False) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO forecast_runs
            (run_timestamp, run_type, trigger_source, status,
             countries_requested, types_requested)
            VALUES (?, ?, ?, 'running', ?, ?)
        """, (
            datetime.now().isoformat(),
            run_type,
            trigger_source,
            ','.join(countries) if countries else None,
            ','.join(forecast_types) if forecast_types else None,
        ))

        return cursor.lastrowid


def complete_forecast_run(
    run_id: int,
    status: str = "completed",
    countries_completed: Optional[List[str]] = None,
    types_completed: Optional[List[str]] = None,
    forecasts_generated: int = 0,
    execution_time_seconds: float = 0,
    error_message: Optional[str] = None,
    log_file_path: Optional[str] = None,
):
    """
    Update a forecast run with completion info.
    """
    with get_connection(readonly=False) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE forecast_runs
            SET status = ?,
                countries_completed = ?,
                types_completed = ?,
                forecasts_generated = ?,
                execution_time_seconds = ?,
                error_message = ?,
                log_file_path = ?
            WHERE id = ?
        """, (
            status,
            ','.join(countries_completed) if countries_completed else None,
            ','.join(types_completed) if types_completed else None,
            forecasts_generated,
            execution_time_seconds,
            error_message,
            log_file_path,
            run_id,
        ))


def get_latest_forecast_run() -> Optional[Dict]:
    """
    Get the most recent forecast run.
    """
    query = """
        SELECT *
        FROM forecast_runs
        ORDER BY run_timestamp DESC
        LIMIT 1
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        row = cursor.fetchone()

        if row:
            return dict(row)

    return None


def get_forecast_runs(
    limit: int = 50,
    status: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get forecast run history.
    """
    query = "SELECT * FROM forecast_runs WHERE 1=1"
    params = []

    if status:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY run_timestamp DESC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    return df


# ============================================================================
# SCHEMA INITIALIZATION
# ============================================================================

def initialize_all_tables():
    """
    Create all required tables for the forecasting system.
    """
    create_forecasts_table()
    create_model_evaluations_table()
    create_deployed_models_table()
    create_forecast_runs_table()
    logger.info("All tables initialized")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    print("Testing database operations...")

    # Test connection
    print("\n1. Testing connection...")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM countries")
        result = cursor.fetchone()
        print(f"   Countries in database: {result['count']}")

    # Test loading energy data
    print("\n2. Testing load_energy_data...")
    df = load_energy_data('DE', 'load', '2024-01-01', '2024-01-08')
    print(f"   Loaded {len(df)} records")
    if not df.empty:
        print(f"   Date range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")

    # Test loading weather data
    print("\n3. Testing load_weather_data...")
    df = load_weather_data('DE', '2024-01-01', '2024-01-08')
    print(f"   Loaded {len(df)} records")

    # Test loading training data
    print("\n4. Testing load_training_data...")
    df = load_training_data('DE', 'load', '2024-01-01', '2024-01-08')
    print(f"   Loaded {len(df)} records with columns: {list(df.columns)}")

    # Test creating all tables
    print("\n5. Testing initialize_all_tables...")
    initialize_all_tables()

    # Test model evaluations
    print("\n6. Testing model_evaluations operations...")
    eval_id = save_model_evaluation(
        country_code="DE",
        forecast_type="load",
        model_version="TEST_20240115",
        metrics={"mae": 500.0, "rmse": 650.0, "mape": 3.5, "smape": 3.2, "mase": 0.85},
        skill_scores={"skill_vs_persistence": 0.15, "skill_vs_seasonal_naive": 0.12},
        training_samples=8760,
        test_samples=720,
        is_baseline=False,
        model_location="candidate",
    )
    print(f"   Saved evaluation with ID: {eval_id}")

    latest_eval = get_latest_evaluation("DE", "load", "candidate")
    print(f"   Latest evaluation MAE: {latest_eval['mae'] if latest_eval else 'None'}")

    # Test deployment operations
    print("\n7. Testing deployment operations...")
    deploy_id = save_deployment(
        country_code="DE",
        forecast_type="load",
        model_version="TEST_20240115",
        deployed_by="test",
        deployment_reason="Initial deployment",
        mae_at_deployment=500.0,
        mape_at_deployment=3.5,
        skill_score_at_deployment=0.15,
    )
    print(f"   Deployed with ID: {deploy_id}")

    deployed = get_deployed_model("DE", "load")
    print(f"   Deployed version: {deployed['model_version'] if deployed else 'None'}")

    all_deployed = get_all_deployed_models()
    print(f"   Total deployed models: {len(all_deployed)}")

    # Test forecast runs
    print("\n8. Testing forecast_runs operations...")
    run_id = start_forecast_run(
        run_type="manual",
        trigger_source="test",
        countries=["DE", "FR"],
        forecast_types=["load", "price"],
    )
    print(f"   Started run with ID: {run_id}")

    complete_forecast_run(
        run_id=run_id,
        status="completed",
        countries_completed=["DE", "FR"],
        types_completed=["load", "price"],
        forecasts_generated=96,
        execution_time_seconds=120.5,
    )

    latest_run = get_latest_forecast_run()
    print(f"   Latest run status: {latest_run['status'] if latest_run else 'None'}")

    # Test get latest timestamp
    print("\n9. Testing get_latest_data_timestamp...")
    for data_type in ['load', 'price', 'renewable', 'weather']:
        latest = get_latest_data_timestamp('DE', data_type)
        print(f"   {data_type}: {latest}")

    print("\n[OK] Database tests complete!")
