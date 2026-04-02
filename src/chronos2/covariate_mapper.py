"""Covariate mapping for Chronos-2 forecasting.

Maps (country_code, forecast_type) to the available covariates from the
energy dashboard database. Replaces netpredict2's Meteologica-based
covariate mapping with ENTSO-E + Open-Meteo weather data.

Covariate suffix convention (from netpredict2):
- suffix-0: Available through D+2 (future-known) — weather forecasts, time features
- suffix-1: Available through D+1 only (past-only) — TSO forecasts, DA prices
"""

import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger("energy_forecast.chronos2")


# ============================================================================
# WEATHER FEATURE MAPPING BY FORECAST TYPE
# ============================================================================
# These are suffix-0 (future-known via Open-Meteo forecasts)
WEATHER_COVARIATES = {
    "load": [
        "temperature_2m_k",
        "relative_humidity_2m_frac",
    ],
    "price": [
        "temperature_2m_k",
        "wind_speed_100m_ms",
        "shortwave_radiation_wm2",
    ],
    "renewable": [
        "shortwave_radiation_wm2",
        "direct_radiation_wm2",
        "diffuse_radiation_wm2",
        "wind_speed_100m_ms",
        "wind_speed_10m_ms",
    ],
    "solar": [
        "shortwave_radiation_wm2",
        "direct_radiation_wm2",
        "diffuse_radiation_wm2",
    ],
    "wind_onshore": [
        "wind_speed_100m_ms",
        "wind_speed_10m_ms",
    ],
    "wind_offshore": [
        "wind_speed_100m_ms",
        "wind_speed_10m_ms",
    ],
    "hydro_total": [
        "temperature_2m_k",
    ],
    "biomass": [
        "temperature_2m_k",
    ],
    "net_position": [
        "temperature_2m_k",
        "wind_speed_100m_ms",
        "shortwave_radiation_wm2",
    ],
}


# ============================================================================
# TSO FORECAST MAPPING BY FORECAST TYPE
# ============================================================================
# These are suffix-1 (available through D+1 only)
TSO_COVARIATES = {
    "load": ["tso_load_forecast"],
    "price": ["tso_load_forecast", "tso_solar_forecast", "tso_wind_forecast"],
    "renewable": ["tso_solar_forecast", "tso_wind_forecast"],
    "solar": ["tso_solar_forecast"],
    "wind_onshore": ["tso_wind_forecast"],
    "wind_offshore": ["tso_wind_forecast"],
    "hydro_total": ["tso_load_forecast"],
    "biomass": ["tso_load_forecast"],
    "net_position": ["tso_load_forecast"],
}


# ============================================================================
# COVARIATE MAP BUILDER
# ============================================================================

def build_covariate_map(
    country_code: str,
    forecast_type: str,
    include_neighbors: bool = False,
    top_n_neighbors: int = 3,
) -> dict:
    """Build the covariate mapping for a (country, forecast_type) pair.

    Returns a dict describing which covariates to load and how to classify them
    (suffix-0 = future-known, suffix-1 = past-only).

    Args:
        country_code: ISO 2-letter country code
        forecast_type: One of load, price, renewable, solar, wind_onshore, etc.
        include_neighbors: Whether to include neighbor country features
        top_n_neighbors: Number of geographic neighbors to include

    Returns:
        dict with keys:
            "suffix_0": list of {source, column, cov_name} — future-known covariates
            "suffix_1": list of {source, column, cov_name} — past-only covariates
    """
    suffix_0 = []  # Future-known (weather + time)
    suffix_1 = []  # Past-only (TSO forecasts, DA prices)

    # --- Suffix-0: Weather features (from weather_data table) ---
    weather_cols = WEATHER_COVARIATES.get(forecast_type, ["temperature_2m_k"])
    for col in weather_cols:
        suffix_0.append({
            "source": "weather_data",
            "column": col,
            "cov_name": f"weather__{col}",
        })

    # --- Suffix-0: Calendar features (always available) ---
    for cal_feat in ["hour", "dayofweek", "month"]:
        suffix_0.append({
            "source": "calendar",
            "column": cal_feat,
            "cov_name": f"cal__{cal_feat}",
        })
    suffix_0.append({
        "source": "calendar",
        "column": "is_holiday",
        "cov_name": "cal__is_holiday",
    })

    # --- Suffix-1: TSO forecasts ---
    tso_covs = TSO_COVARIATES.get(forecast_type, [])
    for tso_name in tso_covs:
        if tso_name == "tso_load_forecast":
            suffix_1.append({
                "source": "energy_load_forecast",
                "column": "forecast_value_mw",
                "cov_name": "tso__load_forecast",
            })
        elif tso_name == "tso_solar_forecast":
            suffix_1.append({
                "source": "energy_generation_forecast",
                "column": "solar_mw",
                "cov_name": "tso__solar_forecast",
            })
        elif tso_name == "tso_wind_forecast":
            suffix_1.append({
                "source": "energy_generation_forecast",
                "column": "wind_onshore_mw",
                "cov_name": "tso__wind_onshore_forecast",
            })
            suffix_1.append({
                "source": "energy_generation_forecast",
                "column": "wind_offshore_mw",
                "cov_name": "tso__wind_offshore_forecast",
            })

    # --- Suffix-1: Day-ahead prices ---
    if forecast_type in ("load", "price", "renewable", "net_position"):
        suffix_1.append({
            "source": "energy_price",
            "column": "price_eur_mwh",
            "cov_name": "da__price",
        })

    # --- Suffix-1: Crossborder flows (net_position only) ---
    if forecast_type == "net_position":
        suffix_1.append({
            "source": "crossborder_flows",
            "column": "flow_mw",
            "cov_name": "crossborder_flows",
        })

    # --- Suffix-1: Neighbor features ---
    if include_neighbors:
        neighbors = config.COUNTRY_NEIGHBORS.get(country_code, [])[:top_n_neighbors]
        for neighbor in neighbors:
            if forecast_type == "net_position":
                suffix_1.append({
                    "source": "net_position",
                    "column": "net_position_mw",
                    "country_override": neighbor,
                    "cov_name": f"neighbor_np__{neighbor}",
                })
            else:
                suffix_1.append({
                    "source": "energy_load",
                    "column": "load_mw",
                    "country_override": neighbor,
                    "cov_name": f"neighbor__{neighbor}_load",
                })
                suffix_1.append({
                    "source": "energy_price",
                    "column": "price_eur_mwh",
                    "country_override": neighbor,
                    "cov_name": f"neighbor__{neighbor}_price",
                })

    return {
        "suffix_0": suffix_0,
        "suffix_1": suffix_1,
    }


def get_all_covariate_names(
    country_code: str,
    forecast_type: str,
    include_neighbors: bool = False,
) -> tuple[list[str], list[str]]:
    """Get flat lists of covariate names for a (country, forecast_type) pair.

    Returns:
        (suffix_0_names, suffix_1_names) — lists of covariate name strings
    """
    cov_map = build_covariate_map(country_code, forecast_type, include_neighbors)
    suffix_0_names = [c["cov_name"] for c in cov_map["suffix_0"]]
    suffix_1_names = [c["cov_name"] for c in cov_map["suffix_1"]]
    return suffix_0_names, suffix_1_names
