# Data-Driven vs Centroid Multi-Point Weather Comparison

**Analysis Date:** 2026-03-15 18:42:12

## Executive Summary

This analysis compares two approaches to weather data collection for energy forecasting:

1. **Centroid** - Single representative point per country (current production)
2. **Data-Driven Multi-Point** - Zones derived from K-means clustering of actual renewable installations

**Key Findings:**
- **Correlation improvements:** 40/84 (47.6%) comparisons show improvement
- **Average correlation improvement:** +0.0629
- **MAE improvements:** 46/84 (54.8%) comparisons show improvement  
- **Average MAE improvement:** +9.25%

## Detailed Results

### Top 15 Correlation Improvements (Data-Driven vs Centroid)

| Country | Forecast Type | Weather Variable | Energy Target | Centroid Corr | Data-Driven Corr | Improvement |
|---------|---------------|------------------|---------------|---------------|------------------|-------------|
| DE | wind_offshore | wind_speed_100m_ms | wind_offshore_mw | 0.0721 | 0.7932 | +0.7211 |
| DE | wind_offshore | wind_speed_10m_ms | wind_offshore_mw | 0.1070 | 0.7858 | +0.6788 |
| FR | wind_offshore | wind_speed_10m_ms | wind_offshore_mw | 0.2204 | 0.8206 | +0.6002 |
| DE | wind_onshore | wind_speed_10m_ms | wind_onshore_mw | 0.2973 | 0.8853 | +0.5880 |
| DE | wind_onshore | wind_speed_100m_ms | wind_onshore_mw | 0.3767 | 0.9483 | +0.5717 |
| FR | wind_offshore | wind_speed_100m_ms | wind_offshore_mw | 0.2597 | 0.8206 | +0.5609 |
| BE | wind_offshore | wind_speed_10m_ms | wind_offshore_mw | 0.3595 | 0.9170 | +0.5575 |
| FR | wind_onshore | wind_speed_100m_ms | wind_onshore_mw | 0.3779 | 0.9315 | +0.5536 |
| BE | wind_offshore | wind_speed_100m_ms | wind_offshore_mw | 0.3634 | 0.9158 | +0.5524 |
| FR | wind_onshore | wind_speed_10m_ms | wind_onshore_mw | 0.3429 | 0.8874 | +0.5444 |
| BE | wind_onshore | wind_speed_10m_ms | wind_onshore_mw | 0.4240 | 0.9080 | +0.4840 |
| BE | wind_onshore | wind_speed_100m_ms | wind_onshore_mw | 0.4512 | 0.9329 | +0.4817 |
| BE | solar | direct_radiation_wm2 | solar_mw | 0.6846 | 0.8717 | +0.1871 |
| DE | solar | direct_radiation_wm2 | solar_mw | 0.7634 | 0.9276 | +0.1642 |
| FR | solar | direct_radiation_wm2 | solar_mw | 0.7565 | 0.9186 | +0.1620 |

### Top 15 MAE Improvements (Data-Driven vs Centroid)

| Country | Forecast Type | Energy Target | Centroid MAE | Data-Driven MAE | Improvement % |
|---------|---------------|---------------|--------------|-----------------|---------------|
| FR | wind_onshore | wind_onshore_mw | 2864.06 | 1069.34 | +62.66% |
| DE | wind_onshore | wind_onshore_mw | 5922.30 | 2414.02 | +59.24% |
| BE | wind_offshore | wind_offshore_mw | 569.49 | 242.93 | +57.34% |
| BE | wind_offshore | wind_offshore_mw | 568.21 | 245.63 | +56.77% |
| BE | wind_onshore | wind_onshore_mw | 478.59 | 212.63 | +55.57% |
| FR | wind_onshore | wind_onshore_mw | 2925.02 | 1419.01 | +51.49% |
| FR | solar | solar_mw | 1024.34 | 504.19 | +50.78% |
| BE | wind_onshore | wind_onshore_mw | 491.08 | 252.08 | +48.67% |
| FR | wind_offshore | wind_offshore_mw | 467.21 | 241.33 | +48.35% |
| FR | wind_offshore | wind_offshore_mw | 459.64 | 239.70 | +47.85% |
| DE | wind_offshore | wind_offshore_mw | 1935.32 | 1046.55 | +45.92% |
| DE | wind_offshore | wind_offshore_mw | 1923.52 | 1069.09 | +44.42% |
| DE | solar | solar_mw | 3035.86 | 1689.93 | +44.33% |
| FR | solar | solar_mw | 1480.82 | 831.81 | +43.83% |
| DE | wind_onshore | wind_onshore_mw | 6148.90 | 3523.97 | +42.69% |

### Per-Country Performance Summary

| Country | Avg Corr Improvement | Avg MAE Improvement % | Total Comparisons | Positive Corr | Positive MAE |
|---------|--------------------|-----------------------|-------------------|---------------|---------------|
| BE | +0.0529 | +9.02% | 28 | 10 | 11 |
| FR | +0.0637 | +8.71% | 28 | 15 | 16 |
| DE | +0.0722 | +10.03% | 28 | 15 | 19 |

## Recommendations

### Overall Recommendation: **RECOMMEND** implementing data-driven multi-point weather zones

- **Correlation improvements:** 47.6% of comparisons show improvement
- **MAE improvements:** 54.8% of comparisons show improvement
- **Average correlation improvement:** +0.0629
- **Average MAE improvement:** +9.25%

## Data Summary

- **Analysis period:** 2025-12-15 to 2026-03-14 (90 days)
- **Countries analyzed:** BE, FR, DE
- **Total weather-energy comparisons:** 84
- **Centroid weather records:** 84
- **Data-driven weather records:** 84

### Data-Driven Zone Information

The data-driven zones are based on K-means clustering of actual renewable power plant locations from the Open Power System Data (OPSD) dataset. Each zone point is weighted by the actual installed capacity in that cluster.

**Benefits of Data-Driven Approach:**
- Reflects real installation geography
- Weights based on actual capacity
- Objective, reproducible methodology
- Regular updates possible as new installations come online

**Implementation Considerations:**
- Requires multiple weather API calls per forecast
- More complex aggregation logic
- Higher computational and API costs

**Conclusion:** The forecast accuracy improvements justify the additional complexity and costs.

---
*Analysis generated by simplified comparison script*