"""
Configuration for Energy Forecasting Module
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent

# Database path: use ENERGY_DB_PATH env var (set by Docker), or fallback to /data/
DATABASE_PATH = Path(os.getenv('ENERGY_DB_PATH', '/data/energy_dashboard.db'))

MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# ============================================================================
# FORECAST CONFIGURATION
# ============================================================================
FORECAST_HOUR = 18  # Run daily at 18:00
FORECAST_TARGET_DAYS = 2  # D+2 (day after tomorrow) - default for backwards compatibility
FORECAST_TYPES = ['load', 'price', 'renewable', 'net_position']

# Default horizons by forecast type (backward compatibility - single horizon)
# D+1 for load/renewable (day-ahead forecasts)
# D+2 for price (since D+1 price is known from day-ahead market)
DEFAULT_HORIZONS = {
    'load': 1,        # D+1 - load forecast for tomorrow
    'price': 2,       # D+2 - price forecast (D+1 price known from auction)
    'renewable': 1,   # D+1 - renewable generation forecast
    'solar': 1,
    'wind_onshore': 1,
    'wind_offshore': 1,
    'hydro_total': 1,
    'biomass': 1,
    'net_position': 2,    # D+2 - net position forecast
}

# Multi-horizon configuration: each type generates forecasts for all specified horizons
# This enables comparing D+1 vs D+2 accuracy in the frontend
FORECAST_HORIZONS = {
    'load': [1, 2],           # D+1 and D+2
    'price': [1, 2],          # D+1 and D+2
    'renewable': [1, 2],      # D+1 and D+2
    'solar': [1, 2],          # D+1 and D+2
    'wind_onshore': [1, 2],   # D+1 and D+2
    'wind_offshore': [1, 2],  # D+1 and D+2
    'hydro_total': [1, 2],    # D+1 and D+2
    'biomass': [1, 2],        # D+1 and D+2
    'net_position': [1, 2],   # D+1 and D+2
}


def get_horizons_for_type(forecast_type: str) -> list:
    """Get all horizons to generate for a forecast type.

    Args:
        forecast_type: Type of forecast (load, price, renewable, etc.)

    Returns:
        List of horizon days [1, 2] for D+1 and D+2
    """
    return FORECAST_HORIZONS.get(forecast_type, [1, 2])

# Maximum forecast horizon (limited by weather forecast availability)
MAX_FORECAST_HORIZON = 14  # 14 days weather forecast available

# Individual renewable types for detailed forecasting
RENEWABLE_TYPES = [
    'solar',
    'wind_onshore',
    'wind_offshore',
    'hydro_total',  # combined hydro_run + hydro_reservoir
    'biomass'
]

# ============================================================================
# SUPPORTED COUNTRIES
# ============================================================================
# 24 countries with complete data across all 4 types (load, price, renewable, weather)
SUPPORTED_COUNTRIES = [
    'AT',  # Austria
    'BE',  # Belgium
    'BG',  # Bulgaria
    'CH',  # Switzerland
    'CZ',  # Czech Republic
    'DE',  # Germany
    'EE',  # Estonia
    'ES',  # Spain
    'FI',  # Finland
    'FR',  # France
    'GR',  # Greece
    'HR',  # Croatia
    'HU',  # Hungary
    'IT',  # Italy
    'LT',  # Lithuania
    'LV',  # Latvia
    'NL',  # Netherlands
    'NO',  # Norway
    'PL',  # Poland
    'PT',  # Portugal
    'RO',  # Romania
    'SE',  # Sweden
    'SI',  # Slovenia
    'SK',  # Slovakia
]

# Country-specific renewable type exclusions (e.g., landlocked countries have no offshore wind)
SKIP_RENEWABLE_TYPES = {
    'AT': ['wind_offshore'],  # Austria - Landlocked
    'CH': ['wind_offshore'],  # Switzerland - Landlocked
    'CZ': ['wind_offshore'],  # Czech Republic - Landlocked
    'HU': ['wind_offshore'],  # Hungary - Landlocked
    'SK': ['wind_offshore'],  # Slovakia - Landlocked
    'BG': ['wind_offshore'],  # Bulgaria - Limited offshore
    'RO': ['wind_offshore'],  # Romania - Limited offshore
    'LT': ['wind_offshore'],  # Lithuania - Limited offshore
    'LV': ['wind_offshore'],  # Latvia - Limited offshore
    'EE': ['wind_offshore'],  # Estonia - Limited offshore
    'SI': ['wind_offshore'],  # Slovenia - Limited offshore
    'HR': ['wind_offshore'],  # Croatia - Limited offshore
    'GR': ['wind_offshore'],  # Greece - Limited offshore
    'PL': ['wind_offshore'],  # Poland - Limited offshore
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
SUPPORTED_ALGORITHMS = ['xgboost', 'lightgbm', 'catboost']
DEFAULT_ALGORITHM = 'xgboost'

XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': 0,
    'early_stopping_rounds': 50,  # Stop if no improvement for 50 rounds
}

LIGHTGBM_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'regression',
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1,
}

CATBOOST_PARAMS = {
    'iterations': 500,
    'depth': 8,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': 0,
    'thread_count': -1,
}

# Early stopping configuration
EARLY_STOPPING_ROUNDS = 50  # Stop if no improvement for this many rounds

# Default grid search parameter spaces
GRID_SEARCH_PARAMS = {
    'xgboost': {
        'n_estimators': [300, 500, 700],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.1]
    },
    'lightgbm': {
        'n_estimators': [300, 500, 700],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.1]
    },
    'catboost': {
        'iterations': [300, 500, 700],
        'depth': [6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.1]
    }
}


def get_default_params(algorithm: str) -> dict:
    """Get default parameters for an algorithm."""
    params_map = {
        'xgboost': XGBOOST_PARAMS,
        'lightgbm': LIGHTGBM_PARAMS,
        'catboost': CATBOOST_PARAMS
    }
    return params_map.get(algorithm, XGBOOST_PARAMS).copy()


def get_grid_search_params(algorithm: str) -> dict:
    """Get default grid search parameter space for an algorithm."""
    return GRID_SEARCH_PARAMS.get(algorithm, GRID_SEARCH_PARAMS['xgboost']).copy()

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
# Lag features: hours back from the same hour
LAG_DAYS = [1, 7, 14]  # D-1, D-7, D-14 (same hour)

# Rolling window sizes in hours
ROLLING_WINDOWS = [24, 168]  # 24h and 1 week

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Train/validation/test split
VALIDATION_DAYS = 30
TEST_DAYS = 30

# Minimum data required for training (hours)
MIN_TRAINING_HOURS = 8760  # 1 year

# ============================================================================
# WEATHER FEATURES BY FORECAST TYPE
# ============================================================================
WEATHER_FEATURES = {
    'load': [
        'temperature_2m_k',
        'relative_humidity_2m_frac',
    ],
    'price': [
        'temperature_2m_k',
        'wind_speed_100m_ms',
        'shortwave_radiation_wm2',
    ],
    'renewable': [
        'shortwave_radiation_wm2',
        'direct_radiation_wm2',
        'diffuse_radiation_wm2',
        'wind_speed_100m_ms',
        'wind_speed_10m_ms',
    ],
    # Individual renewable type features
    'solar': [
        'shortwave_radiation_wm2',
        'direct_radiation_wm2',
        'diffuse_radiation_wm2',
    ],
    'wind_onshore': [
        'wind_speed_100m_ms',
        'wind_speed_10m_ms',
    ],
    'wind_offshore': [
        'wind_speed_100m_ms',
        'wind_speed_10m_ms',
    ],
    'hydro_total': [
        'temperature_2m_k',
    ],
    'biomass': [
        'temperature_2m_k',
    ],
    'net_position': [
        'temperature_2m_k',
        'wind_speed_100m_ms',
        'shortwave_radiation_wm2',
    ],
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_FILE = LOGS_DIR / "forecast.log"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration on startup"""
    errors = []

    # Check database exists
    if not DATABASE_PATH.exists():
        errors.append(f"Database not found at {DATABASE_PATH}")

    # Ensure directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True


# ============================================================================
# MODEL RUNNERS (multi-model forecast comparison)
# ============================================================================
# Each entry defines a model runner that can generate forecasts.
# - "builtin" runners use the Forecaster.load() pipeline (XGBoost/LightGBM/CatBoost)
# - "external" runners are invoked as subprocesses with their own Python environment
MODEL_RUNNERS = [
    {
        "name": "builtin",       # Uses whatever algo is saved in model.joblib
        "type": "builtin",       # Loaded via Forecaster.load()
        "enabled": True,
        "production": True,      # Designate as the production model
        "countries": "all",
        "forecast_types": "all",
    },
    {
        "name": "chronos-bolt-small",
        "type": "external",      # Runs as subprocess
        "enabled": True,
        "production": False,
        "countries": ["BE"],
        "forecast_types": ["price"],
        "python_executable": r"C:\Users\guill\.openclaw\workspace\experiments\chronos-venv\Scripts\python.exe",
        "script": "src/chronos_forecaster.py",
    },
    {
        "name": "tso-correction",
        "type": "external",
        "enabled": True,
        "production": False,     # tso_corrected is the challenger; tso_raw is also saved
        "countries": ["BE"],
        "forecast_types": ["solar", "wind_onshore", "wind_offshore"],
        "python_executable": r"C:\Users\guill\miniconda3\python.exe",
        "script": "src/tso_correction_forecaster.py",
    },
    {
        "name": "chronos-2",
        "type": "external",
        "enabled": False,  # Enable after fine-tuning
        "production": False,
        "countries": "all",
        "forecast_types": "all",
        "python_executable": None,  # Set to chronos-2 venv path
        "script": "scripts/forecast_chronos2.py",
    },
]


# ============================================================================
# CHRONOS-2 CONFIGURATION
# ============================================================================
CHRONOS2_MODEL_NAME = "amazon/chronos-2"  # 120M param foundation model
CHRONOS2_CONTEXT_LENGTH = 672  # 4 weeks of hourly data (netpredict2 best)
CHRONOS2_PREDICTION_LENGTH = 24  # 1 day ahead
CHRONOS2_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Experiment directory
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# ============================================================================
# BACKTEST WEEKS (held out from ALL model training for fair comparison)
# ============================================================================
BACKTEST_WEEKS = [
    ("W01", "2024-01-15", "2024-01-21"),   # Winter Y1
    ("W02", "2024-03-11", "2024-03-17"),   # Early spring Y1
    ("W03", "2024-04-22", "2024-04-28"),   # Spring Y1
    ("W04", "2024-07-15", "2024-07-21"),   # Summer Y1
    ("W05", "2024-09-09", "2024-09-15"),   # Late summer Y1
    ("W06", "2024-11-11", "2024-11-17"),   # Autumn Y1
    ("W07", "2025-01-13", "2025-01-19"),   # Winter Y2
    ("W08", "2025-04-07", "2025-04-13"),   # Spring Y2
    ("W09", "2025-06-16", "2025-06-22"),   # Summer Y2
    ("W10", "2025-10-06", "2025-10-12"),   # Autumn Y2
    ("W11", "2026-01-12", "2026-01-18"),   # Winter Y3
    ("W12", "2026-02-16", "2026-02-22"),   # Late winter Y3
]

def get_backtest_exclude_dates() -> list[tuple[str, str]]:
    """Return list of (start, end) date pairs for backtest exclusion."""
    return [(start, end) for _, start, end in BACKTEST_WEEKS]

def get_backtest_weeks(week_ids: list[str] | None = None) -> list[tuple[str, str, str]]:
    """Get backtest weeks, optionally filtered by week IDs.

    Args:
        week_ids: List of week IDs to include (e.g., ['W01', 'W03']).
                  None means all weeks.

    Returns:
        List of (week_id, start_date, end_date) tuples
    """
    if week_ids is None:
        return BACKTEST_WEEKS
    return [w for w in BACKTEST_WEEKS if w[0] in week_ids]

# ============================================================================
# GEOGRAPHIC NEIGHBORS (for cross-country features)
# ============================================================================
COUNTRY_NEIGHBORS = {
    "AT": ["DE", "CZ", "HU", "SI", "CH", "IT"],
    "BE": ["FR", "NL", "DE"],
    "BG": ["RO", "GR"],
    "CH": ["DE", "FR", "AT", "IT"],
    "CZ": ["DE", "PL", "SK", "AT"],
    "DE": ["FR", "NL", "PL", "CZ", "AT", "CH"],
    "EE": ["LT", "LV", "FI"],
    "ES": ["FR", "PT"],
    "FI": ["SE", "EE", "NO"],
    "FR": ["DE", "BE", "ES", "IT", "CH"],
    "GR": ["BG", "IT"],
    "HR": ["SI", "HU"],
    "HU": ["AT", "SK", "RO", "HR", "SI"],
    "IT": ["FR", "AT", "SI", "GR", "CH"],
    "LT": ["LV", "PL", "EE"],
    "LV": ["LT", "EE"],
    "NL": ["DE", "BE"],
    "NO": ["SE", "FI"],
    "PL": ["DE", "CZ", "SK", "LT"],
    "PT": ["ES"],
    "RO": ["BG", "HU"],
    "SE": ["NO", "FI"],
    "SI": ["AT", "IT", "HR", "HU"],
    "SK": ["CZ", "PL", "HU", "AT"],
}


if __name__ == "__main__":
    print("Energy Forecast Configuration")
    print(f"Database: {DATABASE_PATH}")
    print(f"Database exists: {DATABASE_PATH.exists()}")
    print(f"Supported countries: {len(SUPPORTED_COUNTRIES)}")
    print(f"Forecast types: {FORECAST_TYPES}")
    print(f"D+{FORECAST_TARGET_DAYS} daily at {FORECAST_HOUR}:00")

    try:
        validate_config()
        print("\n[OK] Configuration validation passed!")
    except ValueError as e:
        print(f"\n[FAIL] Configuration validation failed:\n{e}")
